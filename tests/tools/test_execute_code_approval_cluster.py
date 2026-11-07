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
    calls = _guardian(monkeypatch, "escalate")
    _register_resolver(prescreen.session, "session")

    with caplog.at_level("INFO", logger="tools.approval"):
        result = A.check_execute_code_guard(SAFE_PROGRAM, "local")

    assert result["approved"] is True and result.get("user_approved") is True
    assert calls == [{"code": SAFE_PROGRAM, "static_summary": None, "allowed_roots": None}]
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
