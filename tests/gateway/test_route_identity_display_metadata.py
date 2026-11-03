"""A wake's route identity compares routing facts only; display metadata never invalidates a route.

Regression: an auto-created Discord thread's origin carries ``auto_thread_created`` and the thread's
original name, while the latest committed human envelope carries neither (and a renamed thread), so every
async-delegation wake from the 2nd human message onward was refused ``admission_conflict`` forever.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from gateway import session_automation
from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from hermes_state_runtime import RuntimeStoreError


def _thread_source(**overrides) -> SessionSource:
    base = SessionSource(
        platform=Platform.DISCORD, chat_id="thread-1", chat_type="thread", thread_id="thread-1",
        user_id="user-1", scope_id="guild-1", parent_chat_id="channel-1",
        chat_name="Original thread name", auto_thread_created=True, auto_thread_initial_name="Original thread name",
        message_id="wake-anchor",
    )
    return replace(base, **overrides)


def _committed_source(**overrides) -> SessionSource:
    """The last committed HUMAN message: renamed thread, no auto-thread bookkeeping, a real reply anchor."""
    return _thread_source(
        chat_name="Renamed by the user", auto_thread_created=False, auto_thread_initial_name=None,
        message_id="human-msg", chat_topic="a topic", **overrides,
    )


def test_route_identity_ignores_display_metadata():
    origin, latest = _thread_source(), _committed_source()
    assert session_automation._route_identity(origin) == session_automation._route_identity(latest)


@pytest.mark.parametrize("field", ["chat_name", "chat_topic", "auto_thread_created", "auto_thread_initial_name",
                                   "user_name", "message_id"])
def test_route_identity_excludes_display_fields(field):
    assert field not in session_automation._route_identity(_thread_source(user_name="Alice", chat_topic="t"))


def test_route_identity_normalises_scope_alias_and_absent_values():
    legacy = SessionSource(platform=Platform.DISCORD, chat_id="c", chat_type="group", guild_id="g")
    modern = SessionSource(platform=Platform.DISCORD, chat_id="c", chat_type="group", scope_id="g")
    identity = session_automation._route_identity(legacy)
    assert identity == session_automation._route_identity(modern)
    assert identity["scope_id"] == "g" and "guild_id" not in identity and "thread_id" not in identity


@pytest.mark.parametrize("change", [{"chat_id": "thread-2"}, {"thread_id": "thread-2"}, {"user_id": "user-2"},
                                    {"chat_type": "group"}, {"scope_id": "guild-2"}, {"parent_chat_id": "channel-2"},
                                    {"thread_id": None}, {"profile": "other"}])
def test_route_identity_still_distinguishes_real_route_changes(change):
    assert (session_automation._route_identity(_thread_source(**change))
            != session_automation._route_identity(_committed_source()))


def _snapshot(wake_source: SessionSource, committed: SessionSource, monkeypatch):
    """Run the real ``snapshot_automation`` route comparison with the store/connector edges faked."""
    adapter = object()
    entry = SimpleNamespace(session_id="s-1", session_key="agent:main:discord:thread:thread-1")
    envelope = {"source": committed.to_dict(), "provenance": {"connector": "discord"}, "media": []}
    runner = SimpleNamespace(_adapter_for_source=lambda _source: adapter)
    authority = SimpleNamespace(runner=runner, logical_owner=lambda sid: sid)
    monkeypatch.setattr(session_automation, "_owner", lambda *_args: entry)
    monkeypatch.setattr(session_automation, "committed_route_envelope", lambda *_args: envelope)
    monkeypatch.setattr("gateway.session_envelope.restore_native",
                        lambda payload, _runner: SimpleNamespace(source=committed))
    event = MessageEvent(text="[IMPORTANT: delegation done]", message_type=MessageType.TEXT, source=wake_source,
                         internal=True, metadata={"gateway_session_key": entry.session_key})
    return session_automation.snapshot_automation(authority, adapter, event, '["async_delegation","d-1",""]')


def test_auto_thread_wake_is_admitted_after_the_thread_was_renamed(monkeypatch):
    payload, entry = _snapshot(_thread_source(message_id=None), _committed_source(), monkeypatch)
    assert entry.session_id == "s-1"
    assert "message_id" not in payload["native_text_v1"]["source"]


@pytest.mark.parametrize("change", [{"chat_id": "elsewhere"}, {"thread_id": "elsewhere"}, {"user_id": "intruder"}])
def test_wake_for_a_different_route_is_still_refused(monkeypatch, change):
    with pytest.raises(RuntimeStoreError) as refused:
        _snapshot(_thread_source(message_id=None, **change), _committed_source(), monkeypatch)
    assert refused.value.reason == "admission_conflict"
