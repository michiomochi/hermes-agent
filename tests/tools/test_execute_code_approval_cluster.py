"""Regression tests for the execute_code approval-bypass cluster.

Covers the canonical fix for issues #4146, #27303, #30882, #33057:

  1. tools.thread_context.propagate_context_to_thread — propagates the agent
     turn's ContextVars AND thread-local approval/sudo callbacks into worker
     threads, and clears the callbacks on teardown.
  2. Local and remote file-RPC carry the cell's authority in real processes.
  3. tools.approval.check_execute_code_guard — the entry-point guard decision
     matrix (isolated backends, yolo/off, cron-deny, headless-local,
     gateway approve/deny/timeout/missing-notify, smart mode).
  4. tools.code_execution_env._scrub_child_env — broad HERMES_ prefix dropped,
     operational allowlist kept, DSN/WEBHOOK blocked, passthrough precedence.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import json
import threading
import time

import pytest

from tools import approval as A
from tools import approval_detection
from tools import approval_context
from tools import approval_smart
from tools.thread_context import propagate_context_to_thread
from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars
from tests.tools._child_env_fixtures import child_env


# ---------------------------------------------------------------------------
# 1. Context + callback propagation helper
# ---------------------------------------------------------------------------

def test_helper_propagates_contextvar_and_approval_callback():
    from tools import terminal_tool as TT

    probe: contextvars.ContextVar[str] = contextvars.ContextVar(
        "cluster_probe", default="unset"
    )
    probe.set("parent-value")
    sentinel = object()
    TT.set_approval_callback(sentinel)
    try:
        seen: dict = {}

        def worker():
            seen["probe"] = probe.get()
            seen["cb"] = TT._get_approval_callback()

        t = threading.Thread(target=propagate_context_to_thread(worker))
        t.start()
        t.join(timeout=5)

        assert seen["probe"] == "parent-value"  # ContextVar propagated
        assert seen["cb"] is sentinel            # thread-local callback propagated
    finally:
        TT.set_approval_callback(None)


def test_helper_clears_callbacks_on_teardown():
    """A recycled worker thread must not retain the propagated callback after
    the wrapped target finishes (mirrors the GHSA-qg5c-hvr5-hjgr teardown)."""
    from tools import terminal_tool as TT

    sentinel = object()
    TT.set_approval_callback(sentinel)
    try:
        seen: dict = {}

        def first():
            seen["during"] = TT._get_approval_callback()

        def second():  # NOT wrapped — runs on the same recycled worker thread
            seen["after"] = TT._get_approval_callback()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            ex.submit(propagate_context_to_thread(first)).result(timeout=5)
            ex.submit(second).result(timeout=5)

        assert seen["during"] is sentinel  # installed for the wrapped target
        assert seen["after"] is None       # cleared on teardown
    finally:
        TT.set_approval_callback(None)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("transport", ["local", "remote-file"])
def test_rpc_carries_authority_to_real_dispatch(child_env, monkeypatch, transport):
    import time
    import model_tools
    from tests.tools._child_env_fixtures import RunToCompletionEnv, run_code
    from tools import terminal_tool
    from tools.code_execution_tool import _run_remote_per_call

    witness = child_env / "authority.txt"

    marker = contextvars.ContextVar("rpc-authority", default="missing")
    seen = []
    original = model_tools.handle_function_call

    def dispatch(name, args, *pos, **kwargs):
        seen.append((marker.get(), terminal_tool._get_approval_callback()))
        return original(name, args, *pos, **kwargs)

    monkeypatch.setattr(model_tools, "handle_function_call", dispatch)
    code = f"from hermes_tools import read_file\nimport json\nprint(json.dumps(read_file({str(witness)!r})))"
    try:
        for turn in ("first", "second"):
            witness.write_text(f"real RPC payload {turn}\n", encoding="utf-8")
            callback = object()
            token = marker.set(turn)
            terminal_tool.set_approval_callback(callback)
            try:
                if transport == "local":
                    result = run_code(code, reset=(turn == "first"))
                else:
                    raw = json.loads(_run_remote_per_call(
                        RunToCompletionEnv(child_env), "ssh", code, "authority-test",
                        frozenset({"read_file"}), timeout=30, max_tool_calls=2, exec_start=time.monotonic()))
                    assert raw["status"] == "success", raw
                    assert raw["tool_calls_made"] == 1
                    result = json.loads(raw["output"])
                assert "content" in result, result
                assert f"real RPC payload {turn}" in result["content"]
                assert seen[-1] == (turn, callback)
            finally:
                marker.reset(token)
    finally:
        terminal_tool.set_approval_callback(None)
    assert len(seen) == 2


# ---------------------------------------------------------------------------
# 3. check_execute_code_guard decision matrix
# ---------------------------------------------------------------------------

@pytest.fixture
def gw_session(monkeypatch):
    """A clean gateway session: HERMES_GATEWAY_SESSION set, a bound session
    key, and isolated gateway queues/callbacks. Yields the session_key."""
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    # Force manual mode regardless of host config and disable any process-level
    # yolo inherited from the developer's live environment.
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)

    session_key = "cluster-test-session"
    token = approval_context.set_current_session_key(session_key)
    with A._lock:
        A._gateway_queues.pop(session_key, None)
        A._gateway_notify_cbs.pop(session_key, None)
        A._permanent_approved.discard("execute_code")
        A._session_approved.get(session_key, set()).discard("execute_code")
    try:
        yield session_key
    finally:
        approval_context.reset_current_session_key(token)
        with A._lock:
            A._gateway_queues.pop(session_key, None)
            A._gateway_notify_cbs.pop(session_key, None)


def _register_resolver(session_key: str, result):
    """Register a gateway notify callback that immediately resolves the most
    recent queued approval entry with *result* (simulating a user response)."""
    def cb(_approval_data):
        with A._lock:
            entries = A._gateway_queues.get(session_key, [])
            if entries:
                entry = entries[-1]
                entry.result = result
                entry.event.set()
    with A._lock:
        A._gateway_notify_cbs[session_key] = cb


def _register_capturing_resolver(session_key: str, result):
    """Resolve immediately and retain the exact approval payload shown."""
    seen = {}

    def cb(approval_data):
        seen["approval_data"] = approval_data
        with A._lock:
            entries = A._gateway_queues.get(session_key, [])
            if entries:
                entries[-1].result = result
                entries[-1].event.set()

    with A._lock:
        A._gateway_notify_cbs[session_key] = cb
    return seen


def test_guard_isolated_backend_approved():
    # Container backends already sandbox the child — no-op approve.
    assert A.check_execute_code_guard("import os", "docker")["approved"] is True


def test_guard_headless_local_approved(monkeypatch):
    # Documented #30882 limitation: no approval surface → preserve auto-run.
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    assert A.check_execute_code_guard("import os", "local")["approved"] is True


def test_guard_cron_deny_blocks(monkeypatch):
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval_context, "_get_cron_approval_mode", lambda: "deny")
    tokens = set_session_vars(cron_session="1")
    try:
        res = A.check_execute_code_guard("import os", "local")
    finally:
        clear_session_vars(tokens)
    assert res["approved"] is False
    assert res["outcome"] == "blocked"


def test_guard_explicit_non_cron_masks_leaked_env(monkeypatch):
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval_context, "_get_cron_approval_mode", lambda: "deny")
    tokens = set_session_vars(cron_session="")
    try:
        res = A.check_execute_code_guard("import os", "local")
    finally:
        clear_session_vars(tokens)
        reset_session_vars()
    assert res["approved"] is True


def test_guard_legacy_env_cron_still_blocks(monkeypatch):
    reset_session_vars()
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval_context, "_get_cron_approval_mode", lambda: "deny")
    res = A.check_execute_code_guard("import os", "local")
    assert res["approved"] is False
    assert res["outcome"] == "blocked"


def test_guard_gateway_user_approves_is_one_shot(gw_session):
    _register_resolver(gw_session, "once")
    res = A.check_execute_code_guard("import os; print(1)", "local")
    assert res["approved"] is True
    assert res.get("user_approved") is True
    # One-shot: approval must NOT persist to future scripts.
    assert A.is_approved(gw_session, "execute_code") is False


def test_guard_session_approval_short_circuits_prompt(gw_session):
    """Once session-approved, execute_code skips the approval prompt (#39275)."""
    # Manually set session approval.
    A.approve_session(gw_session, "execute_code")
    try:
        # Even with a denier registered, the is_approved check short-circuits.
        _register_resolver(gw_session, "deny")
        res = A.check_execute_code_guard("import os", "local")
        assert res["approved"] is True
    finally:
        with A._lock:
            s = A._session_approved.get(gw_session, set())
            s.discard("execute_code")


def test_guard_gateway_missing_notify_is_pending(gw_session):
    # No notify callback registered → backward-compat pending approval.
    res = A.check_execute_code_guard("import os", "local")
    assert res["approved"] is False
    assert res["status"] == "pending_approval"


def test_guard_smart_mode(gw_session, monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "smart")

    monkeypatch.setattr(approval_smart, "_smart_approve", lambda c, d, **__: "approve")
    res = A.check_execute_code_guard("import os", "local")
    assert res["approved"] is True and res.get("smart_approved") is True

    # Smart DENY on an interactive surface now asks the owner. With no bound
    # notifier it remains pending rather than being hard-denied.
    monkeypatch.setattr(approval_smart, "_smart_approve", lambda c, d, **__: "deny")
    res = A.check_execute_code_guard("import os", "local")
    assert res["approved"] is False and res["status"] == "pending_approval"

    # escalate → falls through to manual gateway approval
    monkeypatch.setattr(approval_smart, "_smart_approve", lambda c, d, **__: "escalate")
    _register_resolver(gw_session, "once")
    res = A.check_execute_code_guard("import os", "local")
    assert res["approved"] is True


def test_terminal_smart_deny_owner_override_is_one_operation(gw_session, monkeypatch):
    """A human may override DENY, but a broad UI choice must not be persisted."""
    with A._lock:
        A._permanent_approved.discard("owner-override-test-danger")
        A._session_approved.get(gw_session, set()).discard("owner-override-test-danger")
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(approval_smart, "_smart_approve", lambda _command, _description, **__: "deny")
    monkeypatch.setattr(
        A,
        "detect_dangerous_command",
        lambda command: (True, "owner-override-test-danger", f"risk:{command}"),
    )
    monkeypatch.setattr(
        approval_detection,
        "detect_dangerous_command",
        lambda command: (True, "owner-override-test-danger", f"risk:{command}"),
    )

    shown = _register_capturing_resolver(gw_session, "always")
    result = A.check_all_command_guards("dangerous /tmp/first", "local")

    assert result["approved"] is True
    assert result["user_approved"] is True
    assert shown["approval_data"]["smart_denied"] is True
    assert shown["approval_data"]["allow_permanent"] is False
    assert A.is_approved(gw_session, "owner-override-test-danger") is False

    _register_resolver(gw_session, "deny")
    changed = A.check_all_command_guards("dangerous /tmp/second", "local")
    assert changed["approved"] is False
    assert changed["outcome"] == "denied"


def test_execute_code_smart_deny_owner_override_is_one_operation(gw_session, monkeypatch):
    """Never persist the coarse execute_code key after overriding smart DENY."""
    with A._lock:
        A._permanent_approved.discard("execute_code")
        A._session_approved.get(gw_session, set()).discard("execute_code")
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(approval_smart, "_smart_approve", lambda _command, _description, **__: "deny")

    shown = _register_capturing_resolver(gw_session, "session")
    result = A.check_execute_code_guard("print('first')", "local")

    assert result["approved"] is True
    assert result["user_approved"] is True
    assert shown["approval_data"]["smart_denied"] is True
    assert shown["approval_data"]["allow_permanent"] is False
    assert A.is_approved(gw_session, "execute_code") is False

    _register_resolver(gw_session, "deny")
    changed = A.check_execute_code_guard("print('second')", "local")
    assert changed["approved"] is False
    assert changed["outcome"] == "denied"


def test_smart_escalate_still_persists_session_choice(gw_session, monkeypatch):
    """The DENY restriction must not alter Smart ESCALATE's manual choices."""
    key = "smart-escalate-persistence"
    with A._lock:
        A._session_approved.get(gw_session, set()).discard(key)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(approval_smart, "_smart_approve", lambda _command, _description, **__: "escalate")
    monkeypatch.setattr(
        A, "detect_dangerous_command",
        lambda command: (True, key, f"risk:{command}"),
    )
    monkeypatch.setattr(
        approval_detection, "detect_dangerous_command",
        lambda command: (True, key, f"risk:{command}"),
    )

    shown = _register_capturing_resolver(gw_session, "session")
    result = A.check_all_command_guards("dangerous escalate", "local")

    assert result["approved"] is True
    assert shown["approval_data"]["allow_permanent"] is True
    assert "smart_denied" not in shown["approval_data"]
    assert A.is_approved(gw_session, key) is True


def test_terminal_smart_deny_pending_payload_is_one_operation(gw_session, monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(approval_smart, "_smart_approve", lambda _command, _description, **__: "deny")
    monkeypatch.setattr(
        A, "detect_dangerous_command",
        lambda command: (True, "pending-smart-deny", f"risk:{command}"),
    )
    monkeypatch.setattr(
        approval_detection, "detect_dangerous_command",
        lambda command: (True, "pending-smart-deny", f"risk:{command}"),
    )

    result = A.check_all_command_guards("dangerous pending", "local")

    assert result["status"] == "pending_approval"
    assert result["smart_denied"] is True
    assert result["allow_permanent"] is False
    with A._lock:
        pending = dict(A._pending[gw_session])
    assert pending["smart_denied"] is True
    assert pending["allow_permanent"] is False


def test_execute_code_smart_deny_pending_payload_is_one_operation(gw_session, monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(approval_smart, "_smart_approve", lambda _command, _description, **__: "deny")

    result = A.check_execute_code_guard("print('pending')", "local")

    assert result["status"] == "pending_approval"
    assert result["smart_denied"] is True
    assert result["allow_permanent"] is False
    with A._lock:
        pending = dict(A._pending[gw_session])
    assert pending["smart_denied"] is True
    assert pending["allow_permanent"] is False


def test_terminal_serializes_smart_deny_pending_capabilities(monkeypatch):
    from tools import terminal_tool as terminal_module

    monkeypatch.setattr(
        terminal_module,
        "_check_all_guards",
        lambda *_args, **_kwargs: {
            "approved": False,
            "status": "pending_approval",
            "command": "rm -rf /tmp/example",
            "description": "recursive delete",
            "pattern_key": "rm-rf",
            "smart_denied": True,
            "allow_permanent": False,
        },
    )

    payload = json.loads(terminal_module.terminal_tool(command="rm -rf /tmp/example"))

    assert payload["smart_denied"] is True
    assert payload["allow_permanent"] is False


def test_guard_session_yolo_bypasses(gw_session):
    A.enable_session_yolo(gw_session)
    try:
        # Even with a denier registered, yolo short-circuits before the prompt.
        _register_resolver(gw_session, "deny")
        assert A.check_execute_code_guard("import os", "local")["approved"] is True
    finally:
        A.disable_session_yolo(gw_session)


# ---------------------------------------------------------------------------
# 4. Env scrubbing (#27303)
# ---------------------------------------------------------------------------

def test_env_scrub_hermes_allowlist_and_secret_blocks():
    from tools.code_execution_env import _scrub_child_env

    env = {
        # operational allowlist → kept
        "HERMES_HOME": "/h", "HERMES_PROFILE": "p",
        "HERMES_CONFIG": "/c.yaml", "HERMES_ENV": "/e",
        "HERMES_DELEGATED_CHILD_CONTEXT": "1",
        # other HERMES_* → dropped (broad prefix removed)
        "HERMES_BASE_URL": "https://x", "HERMES_INTERACTIVE": "1",
        "HERMES_KANBAN_TASK": "t_parent",
        # secret substrings (incl. new DSN/WEBHOOK) → dropped
        "SENTRY_DSN": "https://a@s.io/1", "SLACK_WEBHOOK": "https://h/x",
        "OPENAI_API_KEY": "sk", "GITHUB_TOKEN": "ghp",
        # safe prefix → kept; uncategorized → dropped
        "PATH": "/usr/bin", "RANDOM_X": "y",
    }
    out = _scrub_child_env(env, is_passthrough=lambda _: False, is_windows=False)

    for kept in (
        "HERMES_HOME", "HERMES_PROFILE", "HERMES_CONFIG", "HERMES_ENV",
        "HERMES_DELEGATED_CHILD_CONTEXT", "PATH",
    ):
        assert kept in out, f"{kept} should be kept"
    for dropped in (
        "HERMES_BASE_URL", "HERMES_INTERACTIVE", "HERMES_KANBAN_TASK",
        "SENTRY_DSN", "SLACK_WEBHOOK", "OPENAI_API_KEY", "GITHUB_TOKEN",
        "RANDOM_X",
    ):
        assert dropped not in out, f"{dropped} should be dropped"


def test_env_scrub_passthrough_overrides_secret_block():
    """A skill/config-declared passthrough var is an explicit user opt-in and
    passes even if it matches a secret substring (precedence is intentional)."""
    from tools.code_execution_env import _scrub_child_env

    env = {"MY_SERVICE_DSN": "value"}
    out = _scrub_child_env(env, is_passthrough=lambda k: k == "MY_SERVICE_DSN",
                           is_windows=False)
    assert out.get("MY_SERVICE_DSN") == "value"


# ---------------------------------------------------------------------------
# 5. File-tool sensitive-path refusal (security B1)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. Env-scrub diagnosability mitigation (#27303 follow-up)
# ---------------------------------------------------------------------------


def test_execute_code_guard_passes_the_script_to_the_guardian(gw_session, monkeypatch):
    """The gate must hand the raw script to the guardian: the whole point of the
    ``smart_code`` wiring is that the program is assessed as code, not re-lexed as a
    shell string. A stub that swallows ``**kwargs`` would hide a regression here."""
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "smart")
    seen = {}

    def fake_approve(command, description, *, code=None):
        seen["command"], seen["description"], seen["code"] = command, description, code
        return "approve"

    monkeypatch.setattr(approval_smart, "_smart_approve", fake_approve)
    script = "print('the-real-program')"
    result = A.check_execute_code_guard(script, "local")

    assert result["approved"] is True and result.get("smart_approved") is True
    assert seen["code"] == script
    assert "the-real-program" in seen["command"]


# ---------------------------------------------------------------------------
# 7. Static pre-screen (approvals.code_prescreen) + capability-scoped keys
# ---------------------------------------------------------------------------

REVIEW_PROGRAM = "import os\nprint(os.getcwd(), 1)"      # review: unvetted import, call:os.getcwd
SAME_CAPABILITY = "import os\nprint(os.getcwd(), 2)"
OTHER_CAPABILITY = "import os\nos.kill(1, 0)"
HIGH_PROGRAM = "print(open('.env').read())"
SAFE_PROGRAM = "print(sum([1, 2, 3]))"


@pytest.fixture
def prescreen(gw_session, monkeypatch, tmp_path):
    """Smart mode with the pre-screen set per test; scratch dir as the only $TMPDIR write root."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "smart")
    state = {"approvals": {"code_prescreen": "on"}}
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: state["approvals"])

    class Ctl:
        session = gw_session
        scratch_dir = scratch

        @staticmethod
        def mode(value):
            state["approvals"] = {**state["approvals"], "code_prescreen": value}

        @staticmethod
        def config(**kw):
            state["approvals"] = {**state["approvals"], **kw}

    def clean():
        with A._lock:
            A._session_approved.pop(gw_session, None)
            A._pending.pop(gw_session, None)
            A._permanent_approved.difference_update(
                {k for k in A._permanent_approved if k.startswith("execute_code")})

    clean()
    yield Ctl
    clean()


def _guardian(monkeypatch, verdict="escalate"):
    """Replace the guardian; records every call (kwargs included)."""
    calls = []

    def fake(command, description, *, code=None, static_summary=None, allowed_roots=None):
        calls.append({"code": code, "static_summary": static_summary, "allowed_roots": allowed_roots})
        return verdict

    monkeypatch.setattr(approval_smart, "_smart_approve", fake)
    return calls


def _cap_key(code, prescreen, monkeypatch):
    from tools.approval_code_scan import capability_key
    scan, _ = A._code_prescreen(code)
    return capability_key(scan, "execute_code")


def test_prescreen_safe_program_never_calls_the_guardian(prescreen, monkeypatch):
    calls = _guardian(monkeypatch, "deny")
    hooks = []
    monkeypatch.setattr(approval_context, "_fire_approval_hook", lambda name, **kw: hooks.append((name, kw)))

    result = A.check_execute_code_guard(SAFE_PROGRAM, "local")

    assert result["approved"] is True and result["static_approved"] is True
    assert "smart_approved" not in result and "user_approved" not in result
    assert calls == []
    assert [(n, kw["choice"], kw["decided_by"]) for n, kw in hooks] == [
        ("post_approval_response", "static_approve", "static_prescreen")
    ]
    # One-shot: nothing is remembered, neither for the capability nor as the legacy blanket key.
    key = _cap_key(SAFE_PROGRAM, prescreen, monkeypatch)
    assert not A.is_approved(prescreen.session, key) and not A.is_approved(prescreen.session, "execute_code")


def test_prescreen_shadow_changes_nothing(prescreen, monkeypatch, caplog):
    """Shadow logs the tier but the decision path is byte-for-byte the old one: the guardian is asked (even for a
    SAFE program), with the old call shape, and the old blanket key is what the person's answer is stored under."""
    prescreen.mode("shadow")
    calls = []

    # The stub has the pre-pre-screen signature on purpose: shadow must not pass ``static_summary`` / ``allowed_roots``
    # at all (a stub with defaults for them would hide a regression that sends ``static_summary=None``).
    def old_shape(command, description, *, code=None):
        calls.append({"code": code})
        return "escalate"

    monkeypatch.setattr(approval_smart, "_smart_approve", old_shape)
    _register_resolver(prescreen.session, "session")

    with caplog.at_level("INFO", logger="tools.approval"):
        result = A.check_execute_code_guard(SAFE_PROGRAM, "local")

    assert result["approved"] is True and result.get("user_approved") is True
    assert calls == [{"code": SAFE_PROGRAM}]
    assert A.is_approved(prescreen.session, "execute_code") is True      # legacy blanket key, as before
    assert any("code_prescreen tier=safe mode=shadow" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("value, expected", [
    ("on", "on"), ("ON", "on"), (True, "on"), ("shadow", "shadow"), ("off", "off"), (False, "off"),
    ("nonsense", "shadow"), (None, "shadow"),
])
def test_prescreen_mode_parsing(prescreen, value, expected):
    prescreen.mode(value)
    assert A._code_prescreen_mode() == expected


def test_prescreen_off_does_not_scan(prescreen, monkeypatch):
    from tools import approval_code_scan
    prescreen.mode("off")
    monkeypatch.setattr(approval_code_scan, "scan_python",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("scan must not run")))
    calls = _guardian(monkeypatch, "approve")
    result = A.check_execute_code_guard(SAFE_PROGRAM, "local")
    assert result["approved"] is True and result.get("smart_approved") is True
    assert len(calls) == 1


def test_prescreen_only_applies_in_smart_mode(prescreen, monkeypatch):
    """Manual mode keeps asking a person for every program: SAFE is not auto-approved there."""
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    _register_resolver(prescreen.session, "once")
    result = A.check_execute_code_guard(SAFE_PROGRAM, "local")
    assert result.get("user_approved") is True and "static_approved" not in result


def test_prescreen_scan_failure_falls_back_to_old_behaviour(prescreen, monkeypatch):
    from tools import approval_code_scan
    monkeypatch.setattr(approval_code_scan, "scan_python",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    calls = _guardian(monkeypatch, "approve")
    result = A.check_execute_code_guard(SAFE_PROGRAM, "local")
    assert result["approved"] is True and result.get("smart_approved") is True
    assert calls == [{"code": SAFE_PROGRAM, "static_summary": None, "allowed_roots": None}]


def test_prescreen_review_program_reaches_guardian_with_static_summary(prescreen, monkeypatch):
    calls = _guardian(monkeypatch, "approve")
    result = A.check_execute_code_guard(REVIEW_PROGRAM, "local")
    assert result["approved"] is True and result.get("smart_approved") is True
    (call,) = calls
    assert call["static_summary"]["tier"] == "review"
    assert "call:os.getcwd" in call["static_summary"]["features"]
    assert str(prescreen.scratch_dir.resolve()) in call["allowed_roots"]


def test_prescreen_high_program_goes_to_a_person_even_if_the_guardian_approves(prescreen, monkeypatch):
    calls = _guardian(monkeypatch, "approve")
    shown = _register_capturing_resolver(prescreen.session, "always")

    result = A.check_execute_code_guard(HIGH_PROGRAM, "local")

    assert len(calls) == 1 and calls[0]["static_summary"]["tier"] == "high"
    assert result["approved"] is True and result.get("user_approved") is True
    assert "smart_approved" not in result
    data = shown["approval_data"]
    assert data["allow_session"] is False and data["allow_permanent"] is False
    assert data["persist_allowed"] is False
    # An old client answering "always" anyway must not create a grant.
    key = _cap_key(HIGH_PROGRAM, prescreen, monkeypatch)
    assert not A.is_approved(prescreen.session, key)
    assert not A.is_approved(prescreen.session, "execute_code")
    # ... so the next HIGH program asks again (and a denial stands).
    _register_resolver(prescreen.session, "deny")
    again = A.check_execute_code_guard(HIGH_PROGRAM, "local")
    assert again["approved"] is False and again["outcome"] == "denied"


def test_prescreen_high_program_pending_payload_offers_once_only(prescreen, monkeypatch):
    _guardian(monkeypatch, "approve")
    result = A.check_execute_code_guard(HIGH_PROGRAM, "local")      # no notifier registered
    assert result["approved"] is False and result["status"] == "pending_approval"
    pending = A._pending[prescreen.session]
    assert pending["allow_session"] is False and pending["allow_permanent"] is False


def test_prescreen_high_program_ignores_the_legacy_blanket_grant(prescreen, monkeypatch):
    A.approve_session(prescreen.session, "execute_code")
    _guardian(monkeypatch, "approve")
    _register_resolver(prescreen.session, "deny")
    result = A.check_execute_code_guard(HIGH_PROGRAM, "local")
    assert result["approved"] is False and result["outcome"] == "denied"


def test_prescreen_high_program_guardian_deny_is_still_an_owner_override(prescreen, monkeypatch):
    _guardian(monkeypatch, "deny")
    shown = _register_capturing_resolver(prescreen.session, "session")
    result = A.check_execute_code_guard(HIGH_PROGRAM, "local")
    assert result["approved"] is True and result.get("user_approved") is True
    assert shown["approval_data"]["smart_denied"] is True and shown["approval_data"]["allow_session"] is False


def test_prescreen_session_grant_is_scoped_to_the_capability(prescreen, monkeypatch):
    _guardian(monkeypatch, "escalate")
    shown = _register_capturing_resolver(prescreen.session, "session")
    first = A.check_execute_code_guard(REVIEW_PROGRAM, "local")
    assert first["approved"] is True and shown["approval_data"]["allow_session"] is True
    key = shown["approval_data"]["pattern_key"]
    assert key.startswith("execute_code:cap:") and "call:os.getcwd" in key
    assert A.is_approved(prescreen.session, key) and not A.is_approved(prescreen.session, "execute_code")

    # Same capability, different script: no prompt (a denier is registered and would block a prompt).
    _register_resolver(prescreen.session, "deny")
    same = A.check_execute_code_guard(SAME_CAPABILITY, "local")
    assert same["approved"] is True and "user_approved" not in same

    # A different capability is a different key: it asks, and is denied here.
    other = A.check_execute_code_guard(OTHER_CAPABILITY, "local")
    assert other["approved"] is False and other["outcome"] == "denied"
    # The blanket key of old no longer leaks from a capability grant to plain scripts either.
    unrelated = A.check_execute_code_guard("import os\nprint(os.getpid())", "local")
    assert unrelated["approved"] is False


def test_prescreen_always_is_stored_under_the_capability_key(prescreen, monkeypatch):
    saved = {}
    monkeypatch.setattr(A, "save_permanent_allowlist", lambda patterns: saved.setdefault("p", set(patterns)))
    _guardian(monkeypatch, "escalate")
    shown = _register_capturing_resolver(prescreen.session, "always")
    try:
        A.check_execute_code_guard(REVIEW_PROGRAM, "local")
        key = shown["approval_data"]["pattern_key"]
        assert key in saved["p"] and "execute_code" not in saved["p"]
    finally:
        with A._lock:
            A._permanent_approved.clear()


def test_prescreen_legacy_blanket_key_still_covers_review_programs(prescreen, monkeypatch):
    A.approve_session(prescreen.session, "execute_code")
    calls = _guardian(monkeypatch, "deny")
    result = A.check_execute_code_guard(REVIEW_PROGRAM, "local")
    assert result["approved"] is True and calls == []


def test_prescreen_deny_rules_reach_script_bodies(prescreen, monkeypatch):
    """approvals.deny used to guard only terminal commands; a script reading a denied path is now HIGH."""
    prescreen.config(deny=["*notes-private*"])
    calls = _guardian(monkeypatch, "approve")
    shown = _register_capturing_resolver(prescreen.session, "once")
    result = A.check_execute_code_guard("print(open('notes-private.txt').read())", "local")
    assert result.get("user_approved") is True and calls[0]["static_summary"]["tier"] == "high"
    assert shown["approval_data"]["allow_session"] is False


def test_prescreen_extra_write_root_makes_a_write_safe(prescreen, monkeypatch, tmp_path):
    extra = tmp_path / "extra"
    extra.mkdir()
    program = f"open({str(extra / 'a.txt')!r}, 'w').write('x')"
    calls = _guardian(monkeypatch, "escalate")
    _register_resolver(prescreen.session, "once")
    assert A.check_execute_code_guard(program, "local").get("user_approved") is True     # not a root yet
    prescreen.config(code_write_roots=[str(extra)])
    calls.clear()
    assert A.check_execute_code_guard(program, "local").get("static_approved") is True
    assert calls == []


def test_prescreen_linked_git_worktree_of_cwd_is_a_write_root(prescreen, monkeypatch, tmp_path):
    wt = tmp_path / "wt"
    (wt / "sub").mkdir(parents=True)
    (wt / ".git").write_text("gitdir: /elsewhere\n")
    monkeypatch.setattr(A, "_code_scan_cwd", lambda: str(wt / "sub"))
    assert str(wt.resolve()) in A._code_write_roots(str(wt / "sub"))
    primary = tmp_path / "primary"
    (primary / ".git").mkdir(parents=True)
    assert str(primary.resolve()) not in A._code_write_roots(str(primary))


# --- review round 2 ---------------------------------------------------------------------------------------------

@pytest.fixture
def child_cwd(prescreen, monkeypatch, tmp_path):
    """A linked git worktree as TERMINAL_CWD (so the cwd is a write root) plus a separate $HOME; the session cwd
    record starts empty and is driven by ``child_cwd.cd``."""
    from tools import code_execution_tool, terminal_tool

    home, wt = tmp_path / "home", tmp_path / "wt"
    home.mkdir()
    wt.mkdir()
    (wt / ".git").write_text("gitdir: /elsewhere\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TERMINAL_CWD", str(wt))
    monkeypatch.setattr(code_execution_tool, "_get_execution_mode", lambda: "project")
    recorded = {}
    monkeypatch.setattr(terminal_tool, "get_session_cwd", lambda key: recorded.get(key))

    class Ctl:
        pass

    Ctl.home, Ctl.wt = home, wt
    Ctl.cd = staticmethod(lambda task, path: recorded.__setitem__(task, str(path)))
    return Ctl


def test_blocker6_scan_cwd_follows_the_sessions_recorded_cd_like_the_child(child_cwd):
    import os
    assert os.path.realpath(A._code_scan_cwd("t1")) == os.path.realpath(child_cwd.wt)
    child_cwd.cd("t1", child_cwd.home)              # `cd ~` in the session's terminal
    assert os.path.realpath(A._code_scan_cwd("t1")) == os.path.realpath(child_cwd.home)
    assert os.path.realpath(A._code_scan_cwd("other-task")) == os.path.realpath(child_cwd.wt)


def test_blocker6_relative_write_after_cd_home_is_not_safe(child_cwd):
    program = "open('.bashrc', 'a').write('x')"
    scan, _ = A._code_prescreen(program, "t1")
    assert scan.tier == "safe"                      # the worktree is the cwd: a root write
    child_cwd.cd("t1", child_cwd.home)
    scan, _ = A._code_prescreen(program, "t1")
    assert scan.tier != "safe" and "write:outside" in scan.features


def test_blocker6_strict_mode_scans_in_the_staging_dir_not_the_session_cwd(child_cwd, monkeypatch):
    import os
    import tempfile
    from tools import code_execution_tool
    monkeypatch.setattr(code_execution_tool, "_get_execution_mode", lambda: "strict")
    child_cwd.cd("t1", child_cwd.home)
    # (the staging dir is one level below the temp dir: see test_review3_s1_*)
    assert os.path.dirname(A._code_scan_cwd("t1")) == os.path.realpath(tempfile.gettempdir())


def test_blocker6_guard_hands_the_task_id_to_the_scan(prescreen, monkeypatch):
    _guardian(monkeypatch, "approve")
    seen = []
    real = A._code_prescreen
    monkeypatch.setattr(A, "_code_prescreen",
                        lambda code, task_id="", kernel_state=None: (seen.append(task_id), real(code, task_id, kernel_state))[1])
    assert A.check_execute_code_guard(SAFE_PROGRAM, "local", task_id="t1")["approved"] is True
    assert seen == ["t1"]


def test_blocker6_execute_code_passes_its_task_id_to_the_guard(monkeypatch):
    from tools.code_execution_tool import execute_code
    seen = {}

    def guard(code, env_type, has_host_access=False, task_id="", **_):
        seen["task_id"] = task_id
        return {"approved": False, "message": "stop"}

    monkeypatch.setattr("tools.approval.check_execute_code_guard", guard)
    execute_code(code="print(1)", task_id="abc-123")
    assert seen == {"task_id": "abc-123"}


@pytest.mark.parametrize("answer", ["session", "always"])
def test_coarse_capability_cannot_be_granted_for_session_or_always(prescreen, monkeypatch, tmp_path, answer):
    program = f"open({str(tmp_path / 'elsewhere.txt')!r}, 'w').write('x')"        # write:outside, REVIEW
    calls = _guardian(monkeypatch, "escalate")
    shown = _register_capturing_resolver(prescreen.session, answer)

    result = A.check_execute_code_guard(program, "local")

    assert result["approved"] is True and result.get("user_approved") is True
    assert calls[0]["static_summary"]["tier"] == "review"
    data = shown["approval_data"]
    assert data["allow_session"] is False and data["allow_permanent"] is False and data["persist_allowed"] is False
    key = _cap_key(program, prescreen, monkeypatch)
    assert not A.is_approved(prescreen.session, key) and not A.is_approved(prescreen.session, "execute_code")
    # ...so the very same program is asked about again.
    calls.clear()
    shown = _register_capturing_resolver(prescreen.session, "once")
    assert A.check_execute_code_guard(program, "local").get("user_approved") is True and len(calls) == 1


def test_coarse_capability_ignores_a_blanket_grant_made_earlier(prescreen, monkeypatch, tmp_path):
    program = f"open({str(tmp_path / 'elsewhere.txt')!r}, 'w').write('x')"
    A._persist_choice(prescreen.session, "session", ["execute_code"])
    assert A.is_approved(prescreen.session, "execute_code")
    calls = _guardian(monkeypatch, "escalate")
    _register_resolver(prescreen.session, "once")
    assert A.check_execute_code_guard(program, "local").get("user_approved") is True
    assert len(calls) == 1


def test_coarse_capability_lets_a_guardian_approve_end_the_gate(prescreen, monkeypatch, tmp_path):
    """Unlike HIGH, a coarse REVIEW program is still the guardian's call; only the grant scope is narrowed."""
    program = f"open({str(tmp_path / 'elsewhere.txt')!r}, 'w').write('x')"
    _guardian(monkeypatch, "approve")
    result = A.check_execute_code_guard(program, "local")
    assert result["approved"] is True and result.get("smart_approved") is True
    assert not A.is_approved(prescreen.session, _cap_key(program, prescreen, monkeypatch))


def test_fine_grained_capability_can_still_be_granted_for_the_session(prescreen, monkeypatch):
    calls = _guardian(monkeypatch, "escalate")
    shown = _register_capturing_resolver(prescreen.session, "session")
    assert A.check_execute_code_guard(REVIEW_PROGRAM, "local").get("user_approved") is True
    assert shown["approval_data"]["allow_session"] is True
    assert A.check_execute_code_guard(SAME_CAPABILITY, "local").get("approved") is True
    assert len(calls) == 1


# --- review round 3 ---------------------------------------------------------------------------------------------

@pytest.fixture
def real_kernel(prescreen, monkeypatch, tmp_path):
    """Smart mode + pre-screen ``on`` in front of a REAL session kernel (strict mode, staging dirs in the scratch
    dir, which is also the only write root). ``run(code, **kw)`` is ``execute_code`` for one conversation."""
    import tempfile
    from unittest.mock import patch

    from tools import code_kernel
    from tools.code_execution_tool import execute_code

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(prescreen.scratch_dir))
    code_kernel.shutdown_all_kernels()

    class Ctl:
        scratch = prescreen.scratch_dir
        home = tmp_path / "home"
        outside = tmp_path / "outside"

        @staticmethod
        def run(code, **kw):
            with patch("tools.code_execution_tool._load_config", return_value={"mode": "strict", "timeout": 30}):
                return json.loads(execute_code(code, task_id="review3", **kw))

    Ctl.outside.mkdir()
    yield Ctl
    code_kernel.shutdown_all_kernels()


def test_review3_b2_a_name_rebound_by_an_earlier_cell_cannot_launder_a_move(real_kernel, prescreen, monkeypatch):
    """cell 1: ``str = pathlib.Path`` (SAFE). cell 2: ``str(cfg).replace(dst)`` (SAFE in a fresh interpreter, because
    ``str`` reads as the string type). In ONE kernel namespace the second cell moved ~/victim to the scratch dir."""
    victim, dst = real_kernel.outside / "victim.txt", real_kernel.scratch / "moved.txt"
    victim.write_text("precious")
    calls = _guardian(monkeypatch, "escalate")
    _register_resolver(prescreen.session, "deny")
    cell2 = (f"import pathlib\ncfg = pathlib.Path({str(victim)!r})\ndst = {str(dst)!r}\n"
             "str(cfg).replace(dst)")

    first = real_kernel.run("import pathlib\nstr = pathlib.Path")
    assert first["status"] == "success" and not calls            # SAFE: statically approved, no guardian
    second = real_kernel.run(cell2)

    assert len(calls) == 1 and calls[0]["static_summary"]["tier"] != "safe"     # it reached a reviewer ...
    assert second["status"] == "error"                                         # ... who (here) said no
    assert victim.read_text() == "precious" and not dst.exists()


def test_review3_b2_the_same_cell_in_a_fresh_kernel_is_still_safe(real_kernel, prescreen, monkeypatch):
    calls = _guardian(monkeypatch, "escalate")
    _register_resolver(prescreen.session, "deny")
    assert real_kernel.run("y = 1")["status"] == "success"
    assert real_kernel.run("print(sum([1, 2]))")["status"] == "success"          # no shared names: still SAFE
    assert real_kernel.run("y = 2\nprint(y)")["status"] == "success"            # re-binding before reading: SAFE
    assert not calls


def test_review3_b2_a_cell_that_was_not_statically_safe_ends_safe_for_the_rest_of_that_kernel(
        real_kernel, prescreen, monkeypatch):
    calls = _guardian(monkeypatch, "approve")             # the reviewer lets the unvetted cell through
    assert real_kernel.run("import os\nos.chdir('/')")["status"] == "success"
    assert len(calls) == 1
    # Whatever it did (chdir, patching a module ...) is not something the scan can see, so a program that would
    # otherwise be SAFE now asks the reviewer too ...
    assert real_kernel.run("print(1)")["status"] == "success" and len(calls) == 2
    assert calls[1]["static_summary"]["tier"] == "review" and "state:chdir" in calls[1]["static_summary"]["features"]
    # ... until the state is thrown away: reset=true starts from an interpreter nobody has touched.
    assert real_kernel.run("print(2)", reset=True)["status"] == "success" and len(calls) == 2


def test_review3_b2_the_state_belongs_to_the_kernel_not_to_the_conversation(real_kernel, prescreen, monkeypatch):
    from tools import code_kernel
    calls = _guardian(monkeypatch, "approve")
    real_kernel.run("import os\nos.chdir('/')")
    assert len(calls) == 1
    code_kernel.shutdown_all_kernels()                    # the kernel dies (idle reaper, crash, /new ...)
    assert real_kernel.run("print(1)")["status"] == "success" and len(calls) == 1     # fresh kernel: SAFE again


def test_review3_b2_a_denied_cell_leaves_no_trace_in_the_state(real_kernel, prescreen, monkeypatch):
    _guardian(monkeypatch, "escalate")
    _register_resolver(prescreen.session, "deny")
    denied = real_kernel.run("import os\nos.chdir('/')")
    assert denied["status"] == "error"                    # never ran
    calls = _guardian(monkeypatch, "escalate")
    assert real_kernel.run("print(1)")["status"] == "success" and not calls


def test_review3_b2_cells_of_one_kernel_are_scanned_and_run_one_at_a_time(real_kernel, monkeypatch):
    """Cell A is scanned against the state before it; if cell B could be scanned meanwhile and then run first, A would
    run in a namespace its scan never saw. So scan -> approval -> run is one critical section per kernel."""
    from tools import code_kernel
    key = ("owner", "strict", "python", "", ())
    order, gate = [], threading.Event()

    def cell(name):
        with code_kernel.kernel_cell_section(key, is_interrupted=lambda: False):
            order.append(f"{name}-in")
            if name == "a":
                gate.wait(2)
            order.append(f"{name}-out")

    ta, tb = threading.Thread(target=cell, args=("a",)), threading.Thread(target=cell, args=("b",))
    ta.start()
    deadline = time.monotonic() + 5
    while "a-in" not in order and time.monotonic() < deadline:
        time.sleep(0.01)
    assert order == ["a-in"]
    tb.start()
    tb.join(0.3)
    assert order == ["a-in"]                              # b waits while a holds the section
    gate.set()
    ta.join(5)
    tb.join(5)
    assert order == ["a-in", "a-out", "b-in", "b-out"]
    with code_kernel.kernel_cell_section(("other",), is_interrupted=lambda: False):       # other kernels never wait
        pass


def test_review3_b2_waiting_for_the_section_honours_an_interrupt():
    from tools import code_kernel
    key = ("owner-i", "strict", "python", "", ())
    release, held = threading.Event(), threading.Event()

    def hold():
        with code_kernel.kernel_cell_section(key, is_interrupted=lambda: False):
            held.set()
            release.wait(5)

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait(2)
    try:
        with pytest.raises(code_kernel.CellSectionInterrupted):
            with code_kernel.kernel_cell_section(key, is_interrupted=lambda: True):
                pytest.fail("entered the section while another cell holds it")
    finally:
        release.set()
        holder.join(5)


def test_review3_b2_execute_code_hands_the_kernels_prior_state_to_the_guard(prescreen, monkeypatch):
    from tools.approval_code_scan import KernelScanState
    from tools.code_execution_tool import execute_code
    seen = []

    def guard(code, env_type, has_host_access=False, task_id="", kernel_state=None):
        seen.append(kernel_state)
        return {"approved": False, "message": "stop"}

    monkeypatch.setattr("tools.approval.check_execute_code_guard", guard)
    execute_code(code="print(1)", task_id="review3-state")
    assert seen and isinstance(seen[0], KernelScanState) and seen[0] == KernelScanState()   # no kernel yet: fresh


def test_review3_s1_strict_mode_scans_where_the_kernel_really_starts(child_cwd, monkeypatch):
    """The kernel's cwd is ``mkdtemp(prefix='hermes_kernel_')`` INSIDE the temp dir, one level below it."""
    import os
    import tempfile
    from tools import code_execution_tool
    monkeypatch.setattr(code_execution_tool, "_get_execution_mode", lambda: "strict")
    cwd = A._code_scan_cwd("t1")
    assert os.path.dirname(cwd) == os.path.realpath(tempfile.gettempdir())
    assert os.path.basename(cwd).startswith("hermes_kernel_")


def test_review3_s1_a_relative_path_climbing_out_of_the_staging_dir_is_not_safe(prescreen, monkeypatch, tmp_path):
    import os
    import tempfile
    from tools import code_execution_tool
    home = tmp_path / "home"
    scratch = home / "scratch"                              # $TMPDIR below $HOME, as in a profile's scratch dir
    scratch.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    monkeypatch.setattr(code_execution_tool, "_get_execution_mode", lambda: "strict")
    # staging dir = <scratch>/hermes_kernel_x  ->  '../..' is $HOME, so this reads ~/.bash_history
    program = f"print(open({'../' * 2 + '.bash_history'!r}).read())"
    scan, _ = A._code_prescreen(program, "t1")
    assert scan.tier != "safe" and "read:homedot" in scan.features, (scan.reasons, sorted(scan.features))
    # An ordinary relative file in the staging dir is still a root write.
    scan, _ = A._code_prescreen("open('notes.txt', 'w').write('x')", "t1")
    assert scan.tier == "safe", (scan.reasons, sorted(scan.features))
    assert os.path.realpath(A._code_scan_cwd("t1")).startswith(os.path.realpath(str(scratch)) + os.sep)


@pytest.mark.parametrize("first, second", [
    pytest.param("import sqlite3\nsqlite3.connect('@H@/.hermes/state.db').execute('DELETE FROM sessions')",
                 "import sqlite3\nsqlite3.connect('@H@/other.db').execute('DELETE FROM sessions')",
                 id="sqlite-write-outside-the-roots"),
    pytest.param("print(open('@H@/.bashrc').read())", "print(open('@H@/.gitconfig').read())", id="home-dotfile-read"),
    pytest.param("import subprocess\nsubprocess.run(['git', 'status'])",
                 "import subprocess\nsubprocess.run(['ls', '/etc'])", id="constant-process-command"),
])
def test_review3_s2_one_approval_never_covers_a_different_effect(prescreen, monkeypatch, tmp_path, first, second):
    """Each pair used to share ONE capability key, so "approve for the session" on the first passed the second."""
    monkeypatch.setenv("HOME", str(tmp_path))
    first, second = first.replace("@H@", str(tmp_path)), second.replace("@H@", str(tmp_path))
    calls = _guardian(monkeypatch, "escalate")
    shown = _register_capturing_resolver(prescreen.session, "session")
    assert A.check_execute_code_guard(first, "local").get("user_approved") is True
    assert shown["approval_data"]["allow_session"] is False and shown["approval_data"]["allow_permanent"] is False
    assert not A.is_approved(prescreen.session, _cap_key(first, prescreen, monkeypatch))
    calls.clear()
    _register_resolver(prescreen.session, "deny")                      # a person is asked again, and says no
    result = A.check_execute_code_guard(second, "local")
    assert result["approved"] is False and len(calls) == 1
