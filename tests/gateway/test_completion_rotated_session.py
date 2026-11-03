"""A completion whose parent session rotated away lands in the SAME route's live session, labelled.

Preflight called an idle/timeout-ended parent deliverable while the session authority admits only the
route's current session (or its compression successor), so the wake was refused forever. The retarget
needs proof (the parent row is stamped with this session_key and ended for a non-boundary reason); any
other route's completion is never moved.
"""

import dataclasses
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gateway import session_automation
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionSource
from hermes_state_runtime import RuntimeStoreError
from tools import async_delegation as ad

LABEL = "前セッションの委任結果"


def _rotate(store, entry, *, end_reason="idle", new_id="fresh-session"):
    """The route's session ended and a new one took the key (what an idle/daily reset does)."""
    db = cast(Any, store._db)
    db.end_session(entry.session_id, end_reason)
    db.create_session(new_id, source="telegram", session_key=entry.session_key)
    with store._lock:
        store._entries[entry.session_key] = dataclasses.replace(entry, session_id=new_id)
    return new_id


def _event(delegation_id, session_key, parent_session_id):
    event = {
        "type": "async_delegation", "delegation_id": delegation_id, "session_key": session_key,
        "parent_session_id": parent_session_id, "dispatched_at": 1.0, "status": "completed", "summary": "result",
    }
    ad._persist_dispatch(event)
    ad._persist_completion(event, {"status": "completed", "summary": "result"})
    return event


def _runner(tmp_path):
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    admitted: list[Any] = []

    async def accept(event):
        admitted.append(event)
        event._gateway_accepted = True

    runner.adapters[Platform.TELEGRAM] = cast(Any, SimpleNamespace(handle_message=accept))
    return runner, admitted


def _authority_owner(runner, event):
    """The session authority's own ownership gate for this wake (``_owner``), on the real store."""
    db = runner._session_db
    authority = SimpleNamespace(db=SimpleNamespace(get_compression_tip=lambda _sid: None), _db=db)
    return session_automation._owner(runner, event, authority)


@pytest.mark.asyncio
async def test_rotated_parent_delivers_to_the_same_route_with_previous_session_label(
    tmp_path, private_db_probe_cleanup,
):
    with _profile_runtime_scope(tmp_path):
        runner, admitted = _runner(tmp_path)
        store = runner.session_store
        try:
            entry = store.get_or_create_session(
                SessionSource(platform=Platform.TELEGRAM, chat_id="rot-chat", chat_type="dm"))
            old = entry.session_id
            assert cast(Any, store._db).get_session(old)["session_key"] == entry.session_key
            fresh = _rotate(store, entry)
            event = _event("rotated-1", entry.session_key, old)

            claim = await runner._preflight_completion_delivery(event)
            assert (claim.previous_session_id, claim.owner_session_id, claim.proven_session_id) == (old, fresh, fresh)
            ad.release_completion_delivery("rotated-1", claim.claim_id)

            assert await runner._deliver_completion_notification("[IMPORTANT: done]", event) is True
            woke = admitted[0]
            assert woke.metadata["gateway_session_id"] == fresh
            assert woke.text.startswith("[" + LABEL) and old in woke.text and "[IMPORTANT: done]" in woke.text
            assert _authority_owner(runner, woke).session_id == fresh  # what the authority admits
            row = ad.get_durable_delegation("rotated-1")
            assert row is not None and row["delivery_state"] == "delivered"
        finally:
            store.close_all_db_handles()
            runner.close_all_session_db_handles()


@pytest.mark.asyncio
async def test_other_session_key_is_never_retargeted(tmp_path, private_db_probe_cleanup):
    with _profile_runtime_scope(tmp_path):
        runner, admitted = _runner(tmp_path)
        store = runner.session_store
        try:
            mine = store.get_or_create_session(
                SessionSource(platform=Platform.TELEGRAM, chat_id="mine", chat_type="dm"))
            other = store.get_or_create_session(
                SessionSource(platform=Platform.TELEGRAM, chat_id="other", chat_type="dm"))
            old = mine.session_id
            _rotate(store, mine)
            # A completion stamped for ANOTHER route's key but pinned to my rotated-away session.
            event = _event("foreign-1", other.session_key, old)

            assert await runner._rotated_route_target(old, other.session_key) == ""
            claim = await runner._preflight_completion_delivery(event)
            assert claim.previous_session_id == "" and claim.owner_session_id == old
            ad.release_completion_delivery("foreign-1", claim.claim_id)

            assert await runner._deliver_completion_notification("[IMPORTANT: done]", event) is True
            woke = admitted[0]
            assert LABEL not in woke.text and woke.metadata["gateway_session_id"] == old
            with pytest.raises(RuntimeStoreError) as refused:  # the authority still refuses it
                _authority_owner(runner, woke)
            assert refused.value.reason == "admission_conflict"
        finally:
            store.close_all_db_handles()
            runner.close_all_session_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["user_boundary", "no_route", "live_session_ended", "compression"])
async def test_retarget_needs_proof(tmp_path, private_db_probe_cleanup, case):
    with _profile_runtime_scope(tmp_path):
        runner, _admitted = _runner(tmp_path)
        store = runner.session_store
        try:
            entry = store.get_or_create_session(
                SessionSource(platform=Platform.TELEGRAM, chat_id=f"proof-{case}", chat_type="dm"))
            old, key = entry.session_id, entry.session_key
            if case == "user_boundary":
                _rotate(store, entry, end_reason="session_reset")  # /new: a deliberate boundary
            elif case == "no_route":
                cast(Any, store._db).end_session(old, "idle")
                key = "agent:main:telegram:dm:no-such-route"
            elif case == "live_session_ended":
                fresh = _rotate(store, entry)
                cast(Any, store._db).end_session(fresh, "idle")
            else:
                _rotate(store, entry, end_reason="compression")  # the compression path owns this
            assert await runner._rotated_route_target(old, key) == ""
        finally:
            store.close_all_db_handles()
            runner.close_all_session_db_handles()
