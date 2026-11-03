"""A refused async-delegation completion is never silent: a bounded retry cap posts ONE notice.

The durable row stays ``pending`` (no drop, no ack) and keeps retrying; the notice is a plain chat message
(no model turn) deduped per delegation_id.
"""

import contextlib
import time
from typing import Any, cast

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.run_notifications import _STUCK_COMPLETION_NOTICE_REFUSALS
from gateway.session import SessionSource
from tools import async_delegation as ad

DELEGATION_ID = "stuck-delegation-1"


def _persist(entry) -> dict:
    event = {
        "type": "async_delegation", "delegation_id": DELEGATION_ID, "session_key": entry.session_key,
        "parent_session_id": entry.session_id, "dispatched_at": 1.0, "status": "completed",
        "goal": "summarise\nthe   quarterly report", "summary": "done",
    }
    ad._persist_dispatch(event)
    ad._persist_completion(event, {"status": "completed", "summary": "done"})
    return event


class _Adapter:
    """Refuses every wake until ``accepting``; records plain chat sends."""

    def __init__(self):
        self.accepting = False
        self.sent: list[tuple[Any, str, Any]] = []

    async def handle_message(self, event):
        event._gateway_accepted = self.accepting

    async def send(self, chat_id, text, metadata=None):
        self.sent.append((chat_id, text, metadata))


async def _pass(runner, evt) -> Any:
    """One watcher pass over a one-event group, as ``_async_delegation_watcher`` runs it."""
    delivered = await runner._deliver_async_delegation_group([evt])
    if delivered is False:
        runner._back_off_refused_completions([evt])
        await runner._notify_stuck_completions([evt])
    return delivered


@pytest.mark.asyncio
async def test_notice_after_n_refusals_exactly_once_and_row_stays_pending(tmp_path, private_db_probe_cleanup):
    with _profile_runtime_scope(tmp_path):
        runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
        store = runner.session_store
        try:
            entry = store.get_or_create_session(
                SessionSource(platform=Platform.TELEGRAM, chat_id="stuck-chat", chat_type="dm"))
            adapter = _Adapter()
            runner.adapters[Platform.TELEGRAM] = cast(Any, adapter)
            evt = _persist(entry)
            runner._enrich_async_delegation_routing(evt)

            for _ in range(_STUCK_COMPLETION_NOTICE_REFUSALS - 1):
                assert await _pass(runner, evt) is False
            assert adapter.sent == []  # below the cap: still quiet

            assert await _pass(runner, evt) is False
            assert len(adapter.sent) == 1
            chat_id, text, _metadata = adapter.sent[0]
            assert chat_id == "stuck-chat"
            assert DELEGATION_ID in text and "completed" in text
            assert "not accepted" in text  # the refusal reason
            assert "summarise the quarterly report" in text  # one-line goal
            assert str(ad._db_path()) in text  # where the result is kept
            assert f"delegate_task(action='result', delegation_id='{DELEGATION_ID}')" in text

            for _ in range(3):  # keeps retrying the real delivery, never spams
                assert await _pass(runner, evt) is False
            assert len(adapter.sent) == 1
            row = ad.get_durable_delegation(DELEGATION_ID)
            assert row is not None
            assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 0)

            # A fresh event copy of the same delegation (replay / restart) is still deduped.
            replay = {**evt, "_watch_retry_step": 0, "_watch_stuck_notified": False}
            replay["_watch_first_refused_at"] = time.monotonic() - 10_000
            await runner._notify_stuck_completions([replay])
            assert len(adapter.sent) == 1

            adapter.accepting = True  # the route recovers: the real delivery still lands
            assert await _pass(runner, evt) is True
            row = ad.get_durable_delegation(DELEGATION_ID)
            assert row is not None and row["delivery_state"] == "delivered"
        finally:
            store.close_all_db_handles()
            runner.close_all_session_db_handles()


@pytest.mark.asyncio
async def test_notice_after_five_minutes_of_continuous_deferral(tmp_path, private_db_probe_cleanup):
    with _profile_runtime_scope(tmp_path):
        runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
        store = runner.session_store
        try:
            entry = store.get_or_create_session(
                SessionSource(platform=Platform.TELEGRAM, chat_id="slow-chat", chat_type="dm"))
            adapter = _Adapter()
            runner.adapters[Platform.TELEGRAM] = cast(Any, adapter)
            evt = _persist(entry)
            runner._enrich_async_delegation_routing(evt)

            assert await _pass(runner, evt) is False
            assert adapter.sent == []
            evt["_watch_first_refused_at"] = time.monotonic() - 301  # deferred for over five minutes
            assert await _pass(runner, evt) is False
            assert len(adapter.sent) == 1
            row = ad.get_durable_delegation(DELEGATION_ID)
            assert row is not None and row["delivery_state"] == "pending"
        finally:
            store.close_all_db_handles()
            runner.close_all_session_db_handles()


@pytest.mark.asyncio
async def test_event_without_a_chat_route_is_logged_not_sent():
    runner = object.__new__(GatewayRunner)
    runner._build_process_event_source = lambda _evt: None  # type: ignore[method-assign]
    runner._completion_event_scope = lambda _evt: contextlib.nullcontext()  # type: ignore[method-assign]
    sent = []

    async def send(*args):
        sent.append(args)

    runner._send_watcher_message = send  # type: ignore[method-assign]
    await runner._send_stuck_completion_notice(
        {"delegation_id": "cli-1", "status": "completed", "_watch_retry_step": 8, "session_key": ""})
    assert sent == []
