"""delegate_task(action='result', delegation_id=...): read one durable delegation of THIS conversation.

The recovery path for a completion whose message never reached the chat. Ownership is the caller's session
lineage; a missing or foreign id fails closed with one identical error, never another session's data.
"""

import json
from types import SimpleNamespace

import pytest

from tools import async_delegation as ad
from tools.delegate_tool import DELEGATE_TASK_SCHEMA, delegate_task


@pytest.fixture(autouse=True)
def _clean_state():
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


def _parent(session_id, *, lineage=None):
    db = None
    if lineage is not None:
        db = SimpleNamespace(resolve_resume_session_id=lambda sid: lineage.get(sid, sid))
    return SimpleNamespace(session_id=session_id, _session_db=db)


def _dispatch(delegation_id, parent_session_id, **extra):
    record = {"delegation_id": delegation_id, "dispatched_at": 10.0, "session_key": "agent:main:telegram:dm:1",
              "parent_session_id": parent_session_id, "goal": "g", **extra}
    ad._persist_dispatch(record)
    return record


def _result(parent, delegation_id):
    return json.loads(delegate_task(action="result", delegation_id=delegation_id, parent_agent=parent))


def test_schema_documents_result_action_and_delegation_id():
    props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    assert "result" in props["action"]["enum"]
    assert "delegation_id" in props and "result" in props["delegation_id"]["description"]


def test_own_delegation_returns_the_stored_result_and_delivery_facts():
    record = _dispatch("own-1", "parent-a")
    ad._persist_completion({**record, "status": "completed", "completed_at": 20.0},
                           {"status": "completed", "summary": "the answer", "api_calls": 3})
    payload = _result(_parent("parent-a"), "own-1")
    assert payload["action"] == "result" and payload["delegation_id"] == "own-1"
    assert payload["state"] == "completed" and payload["delivery_state"] == "pending"
    assert payload["delivery_attempts"] == 0
    assert (payload["dispatched_at"], payload["completed_at"]) == (10.0, 20.0)
    assert payload["result"]["summary"] == "the answer"


def test_running_unit_returns_per_child_partial_entries():
    _dispatch("partial-1", "parent-a")
    ad.record_unit_child("partial-1", {"task_index": 0, "status": "completed", "summary": "child zero"})
    ad.record_unit_child("partial-1", {"task_index": 1, "status": "failed", "error": "boom"})
    payload = _result(_parent("parent-a"), "partial-1")
    assert payload["state"] == "running" and payload["result"]["partial"] is True
    assert {r["task_index"]: r["status"] for r in payload["result"]["results"]} == {0: "completed", 1: "failed"}


def test_foreign_and_missing_ids_fail_closed_with_the_same_error():
    record = _dispatch("foreign-1", "parent-b")
    ad._persist_completion({**record, "status": "completed"}, {"status": "completed", "summary": "SECRET-OF-B"})
    foreign = delegate_task(action="result", delegation_id="foreign-1", parent_agent=_parent("parent-a"))
    missing = delegate_task(action="result", delegation_id="no-such-id", parent_agent=_parent("parent-a"))
    assert "SECRET-OF-B" not in foreign and "error" in json.loads(foreign)
    assert json.loads(foreign)["error"].replace("foreign-1", "X") == json.loads(missing)["error"].replace("no-such-id", "X")


def test_row_without_any_owner_is_never_readable():
    _dispatch("ownerless-1", "")
    assert "error" in _result(_parent("parent-a"), "ownerless-1")


def test_blank_delegation_id_is_a_tool_error():
    assert "requires delegation_id" in _result(_parent("parent-a"), "  ")["error"]


def test_compression_rotated_parent_still_owns_its_delegation():
    _dispatch("rotated-1", "parent-old")
    lineage = {"parent-old": "parent-tip"}
    assert _result(_parent("parent-tip", lineage=lineage), "rotated-1")["delegation_id"] == "rotated-1"
    assert "error" in _result(_parent("parent-unrelated", lineage=lineage), "rotated-1")


def test_result_never_spends_a_spawn_or_starts_a_child(monkeypatch):
    monkeypatch.setattr("tools.delegate_tool.is_spawn_paused", lambda: True)  # a paused spawn gate must not block reads
    _dispatch("paused-1", "parent-a")
    assert _result(_parent("parent-a"), "paused-1")["delegation_id"] == "paused-1"
