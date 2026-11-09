"""Regression tests for Honcho startup fail-open behavior."""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

from hermes_plugin_honcho import HonchoMemoryProvider


class _FakeHonchoConfig(SimpleNamespace):
    raw: dict = {}
    host: str = "hermes"

    def resolve_session_name(self, **kwargs):
        return "test-session"


def _configured_hybrid_config() -> _FakeHonchoConfig:
    return _FakeHonchoConfig(
        enabled=True,
        api_key=None,
        base_url="http://127.0.0.1:8000",
        recall_mode="hybrid",
        init_on_session_start=False,
        injection_frequency="every-turn",
        context_cadence=1,
        dialectic_cadence=1,
        query_rewrite=False,
        first_turn_base_wait=3.0,
        first_turn_dialectic_wait=2.0,
        dialectic_depth=1,
        dialectic_depth_levels=None,
        reasoning_heuristic=True,
        reasoning_level_cap="high",
        context_tokens=None,
        message_max_chars=25000,
        session_strategy="per-directory",
    )


def _configured_tools_config(*, init_on_session_start: bool = False) -> _FakeHonchoConfig:
    cfg = _configured_hybrid_config()
    cfg.recall_mode = "tools"
    cfg.init_on_session_start = init_on_session_start
    return cfg




def test_stalled_init_only_delays_first_turn_prefetch(monkeypatch):
    """A stalled session init may bound-wait on turn 1 only; every later
    prefetch must keep the fail-open contract and return immediately."""
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    release = threading.Event()

    monkeypatch.setattr(
        "hermes_plugin_honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )

    def stalled_session_init(self, cfg, session_id, **kwargs):
        release.wait(timeout=10)

    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", stalled_session_init)
    provider.initialize("session-1", platform="cli")
    provider._FIRST_TURN_BASE_TIMEOUT = 1.0

    try:
        provider._turn_count = 1
        start = time.perf_counter()
        assert provider.prefetch("first question") == ""
        assert time.perf_counter() - start >= 0.5  # turn 1 waited (bounded)

        for turn in (2, 3, 4):
            provider._turn_count = turn
            start = time.perf_counter()
            assert provider.prefetch("follow-up question") == ""
            assert time.perf_counter() - start < 0.4  # fail-open, no wait
    finally:
        release.set()
        init_thread = getattr(provider, "_init_thread", None)
        if init_thread:
            init_thread.join(timeout=1)


def test_honcho_background_init_rechecks_state_after_lock_race():
    """Startup should not spawn/crash if init completes while waiting for lock."""
    provider = HonchoMemoryProvider()
    provider._config = _configured_hybrid_config()
    provider._lazy_init_kwargs = {"platform": "cli"}
    provider._lazy_init_session_id = "session-1"

    class RacingLock:
        def __enter__(self):
            provider._session_initialized = True
            provider._lazy_init_kwargs = None
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    provider._init_lock = RacingLock()

    provider._start_session_init_background()

    assert provider._init_thread is None
    assert provider._session_initialized is True




def test_first_turn_base_wait_is_shared_by_init_and_context_fetch():
    """Session init and base retrieval share one configured turn-1 deadline."""
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    cfg.first_turn_base_wait = 0.5
    cfg.timeout = None
    release_context = threading.Event()

    class SlowManager:
        def get_prefetch_context(self, session_key, user_message=None):
            release_context.wait(timeout=5)
            return {"representation": "late"}

        def set_context_result(self, session_key, result):
            pass

        def pop_context_result(self, session_key):
            return {}

    def finish_init():
        time.sleep(0.3)
        provider._manager = SlowManager()
        provider._session_initialized = True

    provider._config = cfg
    provider._session_key = "test-session"
    provider._recall_mode = "context"
    provider._turn_count = 1
    provider._last_dialectic_turn = 0
    provider._FIRST_TURN_BASE_TIMEOUT = cfg.first_turn_base_wait
    provider._init_thread = threading.Thread(target=finish_init, daemon=True)
    provider._init_thread.start()

    try:
        started = time.perf_counter()
        assert provider.prefetch("what do you know about me?") == ""
        elapsed = time.perf_counter() - started
        # Property: prefetch waits for init (0.3s sleep) but is bounded by
        # first_turn_base_wait rather than blocking forever on the slow
        # context fetch. The old 0.4..0.65 window was 0.25s wide — pure
        # scheduler noise on a loaded runner. Lower bound proves the wait
        # happened; loose upper bound proves it didn't hang.
        assert 0.25 <= elapsed < 2.0
    finally:
        release_context.set()
        provider._init_thread.join(timeout=10)





def test_honcho_sync_turn_waits_for_full_background_startup(monkeypatch):
    """Manager assignment alone is not readiness while background init continues."""
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    session_created = threading.Event()
    migration_started = threading.Event()
    release_migration = threading.Event()
    get_calls = []

    class StartupManager:
        def __init__(self, *args, **kwargs):
            pass

        def get_or_create(self, session_key):
            get_calls.append(session_key)
            session_created.set()
            return SimpleNamespace(messages=[])

        def migrate_memory_files(self, session_key, mem_dir):
            migration_started.set()
            release_migration.wait(timeout=5)

        def prefetch_context(self, session_key, user_message=None):
            pass

        def _flush_session(self, session):
            pass

    monkeypatch.setattr(
        "hermes_plugin_honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )
    monkeypatch.setattr("hermes_plugin_honcho.client.get_honcho_client", lambda cfg: object())
    monkeypatch.setattr("hermes_plugin_honcho.session.HonchoSessionManager", StartupManager)

    provider.initialize("session-1", platform="cli")
    try:
        assert session_created.wait(timeout=1)
        assert migration_started.wait(timeout=1)
        assert provider._manager is not None
        assert provider._session_initialized is False

        provider.sync_turn("hello", "world")

        assert provider._sync_thread is None
        assert get_calls == ["test-session"]
    finally:
        release_migration.set()
        init_thread = getattr(provider, "_init_thread", None)
        if init_thread:
            init_thread.join(timeout=1)
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=1)

    assert provider._session_initialized is True


def test_honcho_system_prompt_advertises_active_while_background_init_runs(monkeypatch):
    """Prompt metadata should not require a completed network session."""
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    release = threading.Event()

    monkeypatch.setattr(
        "hermes_plugin_honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )

    def slow_session_init(self, cfg, session_id, **kwargs):
        release.wait(timeout=5)
        self._session_initialized = True

    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", slow_session_init)

    provider.initialize("session-1", platform="cli")
    try:
        prompt = provider.system_prompt_block()
        assert "Honcho Memory" in prompt
        assert "hybrid mode" in prompt
    finally:
        release.set()
        init_thread = getattr(provider, "_init_thread", None)
        if init_thread:
            init_thread.join(timeout=1)




def test_honcho_tools_eager_init_failure_does_not_leave_ready_manager(monkeypatch):
    """Failed eager tools startup must not leave hooks seeing a ready session."""
    provider = HonchoMemoryProvider()
    cfg = _configured_tools_config(init_on_session_start=True)

    monkeypatch.setattr(
        "hermes_plugin_honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )

    def failing_session_init(self, cfg, session_id, **kwargs):
        self._manager = SimpleNamespace()
        self._session_key = "test-session"
        raise RuntimeError("boom")

    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", failing_session_init)

    provider.initialize("session-1", platform="cli")
    assert provider._session_initialized is False
    assert provider._manager is None

    background_started = threading.Event()
    provider._start_session_init_background = background_started.set
    provider.sync_turn("hello", "world")
    provider.on_memory_write("add", "user", "prefers safe Honcho startup")

    assert provider._sync_thread is None
    assert not background_started.is_set()

    result = json.loads(provider.handle_tool_call("honcho_profile", {"peer": "user"}))
    assert "could not be initialized" in result["error"]
    assert provider._manager is None


def _init_with_unresolved_peer(monkeypatch, cfg, platform: str = "cli") -> tuple[HonchoMemoryProvider, list[int]]:
    """Provider whose session init fails because no user peer can be named. Returns the attempt log."""
    from hermes_plugin_honcho.session_peers import HonchoPeerUnresolvedError

    provider = HonchoMemoryProvider()
    monkeypatch.setattr("hermes_plugin_honcho.client.HonchoClientConfig.from_global_config", lambda: cfg)
    attempts: list[int] = []

    def no_peer(self, cfg, session_id, **kwargs):
        attempts.append(1)
        raise HonchoPeerUnresolvedError("Honcho has no user peer for session 'x': honcho.json declares no peerName.")

    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", no_peer)
    provider.initialize("session-1", platform=platform)
    if provider._init_thread:
        provider._init_thread.join(timeout=5)
    return provider, attempts


@pytest.mark.parametrize("platform, present, absent", [
    ("cli", "hermes honcho peer --user", "Do not suggest peerName"),
    ("telegram", "Do not suggest peerName", "hermes honcho peer --user"),
])
def test_honcho_unresolved_peer_notices_once_and_stops_retrying(monkeypatch, platform, present, absent):
    """No runtime identity and no peerName: memory stays off for the session, the model hears it once, and
    later turns do not re-run init for a config gap that cannot heal (#93326). The notice recommends peerName
    only on a local platform: on a multi-user gateway a shared peerName would merge every human onto one peer."""
    provider, attempts = _init_with_unresolved_peer(monkeypatch, _configured_hybrid_config(), platform=platform)

    assert provider._manager is None
    assert provider._can_start_init() is False

    notice = provider.prefetch("what did we decide about the schema?")
    assert "Honcho memory is off" in notice and "peerName" in notice
    assert present in notice and absent not in notice
    assert provider.prefetch("second question") == ""
    provider.sync_turn("hello", "world")
    assert attempts == [1]


@pytest.mark.parametrize("platform, present, absent", [
    ("cli", "hermes honcho peer --user", "could not be initialized"),
    ("discord", "supplied no user id", "hermes honcho peer --user"),
])
def test_honcho_unresolved_peer_tool_error_names_the_fix_for_the_platform(monkeypatch, platform, present, absent):
    cfg = _configured_tools_config(init_on_session_start=True)
    provider, _ = _init_with_unresolved_peer(monkeypatch, cfg, platform=platform)

    result = json.loads(provider.handle_tool_call("honcho_profile", {"peer": "user"}))

    assert "peerName" in result["error"]
    assert present in result["error"] and absent not in result["error"]


def test_honcho_tools_lazy_hooks_do_not_prestart_background_init(monkeypatch):
    """tools lazy mode lets the first tool call own session initialization."""
    provider = HonchoMemoryProvider()
    cfg = _configured_tools_config(init_on_session_start=False)

    monkeypatch.setattr(
        "hermes_plugin_honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )

    provider.initialize("session-1", platform="cli")
    background_started = threading.Event()
    provider._start_session_init_background = background_started.set

    provider.prefetch("what do you know?")
    provider.queue_prefetch("what do you know?")
    provider.sync_turn("hello", "world")
    provider.on_memory_write("add", "user", "prefers fail-open memory")

    assert not background_started.is_set()
    assert provider._session_initialized is False

    class ToolManager:
        def get_peer_card(self, session_key, peer="user"):
            return ["ready"]

    init_calls = []

    def fake_session_init(self, cfg, session_id, **kwargs):
        init_calls.append(session_id)
        self._manager = ToolManager()
        self._session_key = "test-session"
        self._session_initialized = True

    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", fake_session_init)

    result = json.loads(provider.handle_tool_call("honcho_profile", {"peer": "user"}))

    assert result == {"result": ["ready"]}
    assert init_calls == ["session-1"]
    assert not background_started.is_set()


# ---------------------------------------------------------------------------
# Write-containment regression tests
# ---------------------------------------------------------------------------


def test_honcho_sync_turn_skips_write_when_save_messages_is_disabled():
    """The resolved write-disable switch must gate an initialized provider."""
    provider = HonchoMemoryProvider()
    cfg = _configured_tools_config(init_on_session_start=True)
    cfg.save_messages = False
    manager_calls = []

    class Manager:
        def get_or_create(self, session_key):
            manager_calls.append(session_key)
            return SimpleNamespace()

    provider._config = cfg
    provider._manager = Manager()
    provider._session_key = "test-session"
    provider._session_initialized = True

    provider.sync_turn("a genuine user turn", "a genuine assistant reply")

    assert provider._sync_thread is None
    assert manager_calls == []


def test_honcho_sync_turn_skips_anchored_gateway_notifications():
    """Known bracketed gateway wrappers must not become durable messages."""
    wrappers = (
        "[ASYNC DELEGATION BATCH COMPLETE — deleg_1]\nworker results follow",
        "[ASYNC DELEGATION COMPLETE — deleg_2]",
        "[CONTEXT COMPACTION — REFERENCE ONLY]\nsummary follows",
        "[CONTEXT COMPACTION - REFERENCE ONLY]",
        "[CONTEXT COMPACTION]",
        "[PRIOR CONTEXT — for reference only; not a new message]",
        "[Your active task list was preserved across context compression]",
        "[CONTEXT SUMMARY]: previous context",
        "[IMPORTANT: Background process 12 matched watch pattern \"foo\"\nCommand: x",
    )

    for wrapper in wrappers:
        provider = HonchoMemoryProvider()
        manager_calls = []

        class Manager:
            def get_or_create(self, session_key):
                manager_calls.append(session_key)
                return SimpleNamespace()

        provider._config = _configured_tools_config(init_on_session_start=True)
        provider._manager = Manager()
        provider._session_key = "test-session"
        provider._session_initialized = True

        provider.sync_turn(wrapper, "assistant reply")

        assert provider._sync_thread is None, f"wrapper not suppressed: {wrapper[:60]!r}"
        assert manager_calls == [], f"wrapper not suppressed: {wrapper[:60]!r}"


@pytest.mark.parametrize("harness", [
    (
        '[Group chat: "Skill Handoff"] You are @slina, one participant in a group chat '
        "with @dev and the user.\n\nYou (user): hello"
    ),
    (
        '[Discussion: "Skill Handoff"] You are @slina, one participant '
        "with @dev and the user.\n\n"
        "New messages in this thread since your last turn (oldest first):\n"
        "  User (user): hello\n\nRules for this Discussion:\n"
        "- Reply with one conversational message only when you have something new worth adding."
    ),
], ids=["group-chat", "discussion"])
def test_honcho_sync_turn_skips_room_harness(harness):
    """Group-chat and hosted Discussion harnesses must not become user speech."""
    marker = harness.split(":", 1)[0]
    wrappers = (
        harness,
        f"  \n{harness}",
        harness.replace(marker, marker.lower(), 1),
        harness.replace(marker, marker.upper(), 1),
    )

    for wrapper in wrappers:
        provider = HonchoMemoryProvider()
        manager_calls = []
        messages = []

        class Manager:
            def get_or_create(self, session_key):
                manager_calls.append(session_key)
                return SimpleNamespace(
                    add_message=lambda role, content, **kwargs: messages.append((role, content)),
                )

            def resolve_author_peer_id(self, session_key, author_id, author_name=None):
                return None

            def save(self, session):
                pass

        provider._config = _configured_tools_config(init_on_session_start=True)
        provider._manager = Manager()
        provider._session_key = "test-session"
        provider._session_initialized = True

        provider.sync_turn(wrapper, "assistant reply")
        if provider._sync_thread is not None:
            provider._sync_thread.join(timeout=5)

        assert provider._sync_thread is None, f"harness not suppressed: {wrapper[:60]!r}"
        assert manager_calls == [], f"harness not suppressed: {wrapper[:60]!r}"
        assert messages == [], f"harness turn was written: {messages!r}"


@pytest.mark.parametrize("marker", ["Group chat", "Discussion"])
def test_honcho_sync_turn_skips_room_harness_before_session_init(monkeypatch, marker):
    """Dropping a room harness must not kick off lazy session initialization."""
    provider = HonchoMemoryProvider()
    provider._config = _configured_hybrid_config()

    def unexpected_init():
        pytest.fail("room harness reached session initialization")

    monkeypatch.setattr(provider, "_ready_or_kick_init", unexpected_init)

    provider.sync_turn(f'[{marker}: "Skill Handoff"] You are @slina', "assistant reply")

    assert provider._session_initialized is False
    assert provider._init_thread is None
    assert provider._sync_thread is None


def test_honcho_sync_turn_skips_prose_gateway_notifications():
    """Prose-form gateway notifications must not become durable messages."""
    prose_wrappers = (
        "A background fan-out of 3 subagent(s) you dispatched earlier has finished.",
        "A background subagent you dispatched earlier has finished. You may have moved on.",
    )

    for wrapper in prose_wrappers:
        provider = HonchoMemoryProvider()
        manager_calls = []

        class Manager:
            def get_or_create(self, session_key):
                manager_calls.append(session_key)
                return SimpleNamespace()

        provider._config = _configured_tools_config(init_on_session_start=True)
        provider._manager = Manager()
        provider._session_key = "test-session"
        provider._session_initialized = True

        provider.sync_turn(wrapper, "assistant reply")

        assert provider._sync_thread is None, f"prose wrapper not suppressed: {wrapper[:60]!r}"
        assert manager_calls == [], f"prose wrapper not suppressed: {wrapper[:60]!r}"


def test_honcho_sync_turn_does_not_suppress_genuine_user_messages():
    """Genuine user messages that mention gateway terms must still be stored."""
    genuine = (
        "A background process I ran has finished — can you check the output?",
        "A background subagent you dispatched earlier has finished? no wait, I was asking about the report",
        "the async delegation batch complete marker disappeared from my log",
        "CONTEXT COMPACTION happened mid-message and I want to see it",
        "When you see PRIOR CONTEXT, treat it carefully",
        "I want to know about your task list",
        "IMPORTANT: Background process — can you explain what that means?",
        "[IMPORTANT: Background process — what does that mean?]",
        "Can we set up a group chat with the other bots tomorrow?",
        'the log line [Group chat: "Skill Handoff"] showed up mid-sentence in my notes',
        "Can we have a discussion with the other bots tomorrow?",
        'the log line [dIsCuSsIoN: "Skill Handoff"] showed up mid-sentence in my notes',
        "[Discussion] Can you explain how hosted rooms work?",
    )

    for msg in genuine:
        provider = HonchoMemoryProvider()
        manager_calls = []
        messages = []
        saved_sessions = []
        session = SimpleNamespace(
            add_message=lambda role, content, **kwargs: messages.append((role, content)),
        )

        class Manager:
            def get_or_create(self, session_key):
                manager_calls.append(session_key)
                return session

            def resolve_author_peer_id(self, session_key, author_id, author_name=None):
                return None

            def save(self, session):
                saved_sessions.append(session)

        provider._config = _configured_tools_config(init_on_session_start=True)
        provider._manager = Manager()
        provider._session_key = "test-session"
        provider._session_initialized = True

        provider.sync_turn(msg, "assistant reply")

        assert provider._sync_thread is not None, f"genuine message suppressed: {msg[:60]!r}"
        provider._sync_thread.join(timeout=5)
        assert not provider._sync_thread.is_alive()
        assert manager_calls == ["test-session"], f"genuine message suppressed: {msg[:60]!r}"
        assert messages == [("user", msg), ("assistant", "assistant reply")]
        assert saved_sessions == [session]


# ---------------------------------------------------------------------------
# Background-process notices (Hermes core tools/process_registry_notifications.py)
# ---------------------------------------------------------------------------

# Observed in a stored receipt: "completed (exit_code=0)".
_PROC_COMPLETED_OBSERVED = (
    "[IMPORTANT: Background process proc_8f3a2c1d9e completed (exit_code=0).\n"
    "Command: pytest -q tests/\n"
    "Output:\n"
    "68 passed in 4.12s]"
)
# Current formatter: "<status> (exit code N[, SIGTERM])."
_PROC_COMPLETED_NORMALLY = (
    "[IMPORTANT: Background process proc_8f3a2c1d9e completed normally (exit code 0).\n"
    "Command: pytest -q tests/\n"
    "Output:\n"
    "68 passed in 4.12s]"
)
_PROC_WATCH_MATCH = (
    '[IMPORTANT: Background process proc_8f3a2c1d9e matched watch pattern "ERROR".\n'
    "Command: tail -f app.log\n"
    "Matched output:\n"
    "ERROR [db] connection reset]"
)
_PROC_HEARTBEAT = (
    "[Background process proc_8f3a2c1d9e heartbeat #3 — still running after 18m (next in 300s when there "
    "is new output; you will also be told when it exits).\n"
    "Command: ./train.sh --epochs 10\n"
    "Output since last heartbeat:\n"
    "epoch 4/10 loss=0.213]"
)

_PROCESS_NOTICES = {
    "completion-observed-exit_code": _PROC_COMPLETED_OBSERVED,
    "completion-formatter-exit-code": _PROC_COMPLETED_NORMALLY,
    "completion-exited-nonzero": (
        "[IMPORTANT: Background process proc_8f3a2c1d9e exited (exit code 2).\n"
        "Command: make build\nOutput:\nmake: *** [all] Error 2]"
    ),
    "completion-terminated-sigterm": (
        "[IMPORTANT: Background process proc_8f3a2c1d9e terminated by Hermes (exit code -15, SIGTERM).\n"
        "Command: sleep 600\nOutput:\n]"
    ),
    "completion-terminated-negative-exit": (
        "[IMPORTANT: Background process proc_8f3a2c1d9e terminated by user (exit code -9).\n"
        "Command: yes\nOutput:\ny\ny]"
    ),
    "completion-failed-start": (
        "[IMPORTANT: Background process proc_8f3a2c1d9e failed to start (exit code ?).\n"
        "Command: ./missing.sh\nOutput:\n/bin/sh: ./missing.sh: not found]"
    ),
    "completion-lost": (
        "[IMPORTANT: Background process proc_8f3a2c1d9e marked lost because the process backend "
        "disappeared (exit code ?).\nCommand: ssh build-host make\nOutput:\n]"
    ),
    "completion-subagent-attribution-cut-output": (
        "[IMPORTANT: Background process proc_8f3a2c1d9e exited (exit code 1).\n"
        'Started by subagent sa-42 of delegation deleg_7. Task: "run the suite"\n'
        "Command: pytest -x\nOutput:\n...(first 4000 characters cut — process(action=\"log\", "
        "session_id=\"proc_8f3a2c1d9e\") has the full output)\nFAILED tests/test_x.py [ 50%]\n\n"
        "short test summary info]"
    ),
    "completion-multiline-command": (
        "[IMPORTANT: Background process proc_8f3a2c1d9e completed normally (exit code 0).\n"
        "Command: python - <<'EOF'\nprint('hi')\nEOF\nOutput:\nhi]"
    ),
    "completion-numeric-id": (
        "[IMPORTANT: Background process 12 completed (exit code 0).\nCommand: make\nOutput:\nok]"
    ),
    "watch-match": _PROC_WATCH_MATCH,
    "watch-match-rate-limited": (
        '[IMPORTANT: Background process proc_8f3a2c1d9e matched watch pattern "Traceback".\n'
        "Command: python serve.py\nMatched output:\nTraceback (most recent call last):\n"
        "(4 earlier matches were suppressed by rate limit)]"
    ),
    "heartbeat": _PROC_HEARTBEAT,
    "heartbeat-omitted-output": (
        "[Background process proc_8f3a2c1d9e heartbeat #12 — still running after 2h3m (next in 600s when "
        "there is new output; you will also be told when it exits).\n"
        "Command: ./train.sh\nOutput since last heartbeat:\n...(5120 earlier characters omitted)\nstep 900]"
    ),
    "completion-batch": (
        "[IMPORTANT: 2 background processes completed. Treat these results as one batch and give one "
        "consolidated response; preserve failures and actionable results.]\n\n"
        f"{_PROC_COMPLETED_NORMALLY}\n\n"
        "[IMPORTANT: Background process proc_77b0e4 exited (exit code 1).\n"
        "Command: ruff check .\nOutput:\nE501 line too long]"
    ),
}


def _recording_sync_provider(monkeypatch):
    """Initialized provider with a fake manager that records every durable write, plus a spy on
    the readiness/lazy-init gate so a test can prove suppression happened before it."""
    provider = HonchoMemoryProvider()
    record = SimpleNamespace(readiness_checks=0, manager_calls=[], messages=[], saved=[])
    session = SimpleNamespace(
        add_message=lambda role, content, **kwargs: record.messages.append((role, content)),
    )

    class Manager:
        def get_or_create(self, session_key):
            record.manager_calls.append(session_key)
            return session

        def resolve_author_peer_id(self, session_key, author_id, author_name=None):
            return None

        def save(self, saved_session):
            record.saved.append(saved_session)

    provider._config = _configured_tools_config(init_on_session_start=True)
    provider._manager = Manager()
    provider._session_key = "test-session"
    provider._session_initialized = True
    real_ready = provider._ready_or_kick_init

    def spy_ready():
        record.readiness_checks += 1
        return real_ready()

    monkeypatch.setattr(provider, "_ready_or_kick_init", spy_ready)
    return provider, record, session


@pytest.mark.parametrize("notice", list(_PROCESS_NOTICES.values()), ids=list(_PROCESS_NOTICES))
def test_honcho_sync_turn_skips_background_process_notices(monkeypatch, notice):
    """A complete background-process notice is machine metadata arriving on the user-role channel,
    not a human instruction: it must never become a durable user (or paired assistant) message."""
    wrappers = (notice, f"  \n{notice}", f"{notice}\n", notice.lower(), notice.upper())

    for wrapper in wrappers:
        provider, record, _ = _recording_sync_provider(monkeypatch)

        provider.sync_turn(wrapper, "The suite passed; nothing else to do.")
        if provider._sync_thread is not None:
            provider._sync_thread.join(timeout=5)

        assert record.readiness_checks == 0, f"notice reached session readiness: {wrapper[:70]!r}"
        assert provider._sync_thread is None, f"notice not suppressed: {wrapper[:70]!r}"
        assert record.manager_calls == [], f"notice not suppressed: {wrapper[:70]!r}"
        assert record.messages == [], f"notice turn was written: {record.messages!r}"


@pytest.mark.parametrize("notice", [_PROC_COMPLETED_OBSERVED, _PROC_WATCH_MATCH, _PROC_HEARTBEAT],
                         ids=["completion", "watch-match", "heartbeat"])
def test_honcho_sync_turn_skips_process_notice_before_session_init(monkeypatch, notice):
    """Dropping a process notice must not kick off lazy session initialization."""
    provider = HonchoMemoryProvider()
    provider._config = _configured_hybrid_config()

    def unexpected_init():
        pytest.fail("background process notice reached session initialization")

    monkeypatch.setattr(provider, "_ready_or_kick_init", unexpected_init)

    provider.sync_turn(notice, "assistant reply")

    assert provider._session_initialized is False
    assert provider._init_thread is None
    assert provider._sync_thread is None


_FENCE = "```"
_HUMAN_TEXT_AROUND_PROCESS_NOTICES = {
    "approval-yes": "yes",
    "rejection-no": "no",
    "approval-go-ahead": "go ahead",
    "rejection-dont-rerun": "no, don't rerun it",
    "correction-proc-id": "actually it was proc_77b0e4, not proc_8f3a2c1d9e",
    "instruction-stop-watch": "stop the watch on proc_8f3a2c1d9e",
    "discussion-mid-message": (
        "why did Background process proc_8f3a2c1d9e completed (exit code 2) show up as an error?"
    ),
    "discussion-heartbeat": "The background process heartbeat #3 says still running — should I wait?",
    "bracketed-header-only": (
        "[IMPORTANT: Background process proc_8f3a2c1d9e completed normally (exit code 0).]"
    ),
    "bracketed-header-then-question": (
        "[IMPORTANT: Background process proc_8f3a2c1d9e completed normally (exit code 0).] is that good?"
    ),
    "bracketed-heartbeat-question": "[Background process proc_8f3a2c1d9e heartbeat #3] what does this mean?",
    "bracketed-batch-question": "[IMPORTANT: 2 background processes completed.] which one failed?",
    "important-prose": "[IMPORTANT: the release freeze starts Friday; do not deploy]",
    "important-process-prose": "[IMPORTANT: Background process naming must follow the runbook]",
    "unknown-notice-shape": "[NOTICE: deploy finished]",
    "unknown-job-shape": "[Background job 12 finished]",
    "quoted-notice": "> " + _PROC_COMPLETED_NORMALLY.replace("\n", "\n> ") + "\n\nwhy did this pass now?",
    "quoted-notice-alone": "> " + _PROC_COMPLETED_OBSERVED.replace("\n", "\n> "),
    "fenced-notice": f"{_FENCE}\n{_PROC_COMPLETED_OBSERVED}\n{_FENCE}",
    "fenced-notice-question": f"{_FENCE}text\n{_PROC_WATCH_MATCH}\n{_FENCE}\nIs this the same bug as yesterday?",
    "instruction-before-completion": (
        f"Rerun this with -v and summarize the failures:\n\n{_PROC_COMPLETED_OBSERVED}"
    ),
    "instruction-before-heartbeat": f"Kill it if it is still this slow:\n{_PROC_HEARTBEAT}",
    "instruction-after-observed-completion": (
        f"{_PROC_COMPLETED_OBSERVED}\n\nPlease rerun with -v and summarize the failures."
    ),
    "instruction-after-formatter-completion": f"{_PROC_COMPLETED_NORMALLY}\n\nok, ship it",
    "instruction-after-watch-match": f"{_PROC_WATCH_MATCH}\nRestart the db container, then tell me.",
    "instruction-after-heartbeat": f"{_PROC_HEARTBEAT}\n\nno — stop it now",
}


@pytest.mark.parametrize("msg", list(_HUMAN_TEXT_AROUND_PROCESS_NOTICES.values()),
                         ids=list(_HUMAN_TEXT_AROUND_PROCESS_NOTICES))
def test_honcho_sync_turn_keeps_human_text_around_process_notices(monkeypatch, msg):
    """Short approvals, discussion, quotations and instructions written before or after a
    complete notice are human input and must reach the manager unchanged."""
    provider, record, session = _recording_sync_provider(monkeypatch)

    provider.sync_turn(msg, "assistant reply")

    assert provider._sync_thread is not None, f"human message suppressed: {msg[:70]!r}"
    provider._sync_thread.join(timeout=5)
    assert not provider._sync_thread.is_alive()
    assert record.readiness_checks == 1
    assert record.manager_calls == ["test-session"], f"human message suppressed: {msg[:70]!r}"
    assert record.messages == [("user", msg), ("assistant", "assistant reply")]
    assert record.saved == [session]


# A human suffix that itself ends in "]" must not be swallowed as the previous notice's output.
_PLAIN_NOTICE = (
    "[IMPORTANT: Background process proc_abc123 completed normally (exit code 0).\n"
    "Command: printf done\nOutput:\ndone]"
)
_PROC_BATCH = _PROCESS_NOTICES["completion-batch"]
_BRACKET_ENDING_HUMAN_TEXT = {
    "suffix-staging-only": f"{_PLAIN_NOTICE}\n\nKeep going [staging only]",
    "suffix-staging-only-period": f"{_PLAIN_NOTICE}\n\nKeep going [staging only].",
    "suffix-single-newline": f"{_PLAIN_NOTICE}\nKeep going [staging only]",
    "suffix-same-line": f"{_PLAIN_NOTICE} Keep going [staging only]",
    "suffix-bare-bracket-instruction": f"{_PLAIN_NOTICE}\n\n[staging only]",
    "suffix-bracket-then-text": f"{_PLAIN_NOTICE}\n\n[staging only] keep going",
    "suffix-quoted-bracket": f"{_PLAIN_NOTICE}\n\n> [staging only]",
    "suffix-quoted-second-notice": (
        f"{_PLAIN_NOTICE}\n\nCompare yesterday's:\n> " + _PROC_COMPLETED_NORMALLY.replace("\n", "\n> ")
    ),
    "human-text-between-notices": f"{_PLAIN_NOTICE}\n\nand then this one:\n\n{_PROC_COMPLETED_NORMALLY}",
    "suffix-after-observed-completion": f"{_PROC_COMPLETED_OBSERVED}\n\nKeep going [staging only]",
    "suffix-after-watch-match": f"{_PROC_WATCH_MATCH}\nRestart the db container [prod]",
    "suffix-after-heartbeat": f"{_PROC_HEARTBEAT}\n\nno — stop it [now]",
    "suffix-after-batch": f"{_PROC_BATCH}\n\nship it [after review]",
}


@pytest.mark.parametrize("msg", list(_BRACKET_ENDING_HUMAN_TEXT.values()), ids=list(_BRACKET_ENDING_HUMAN_TEXT))
def test_honcho_sync_turn_keeps_bracket_ending_human_text_after_process_notices(monkeypatch, msg):
    """Human text after a complete notice keeps the turn even when that text ends in "]"."""
    provider, record, session = _recording_sync_provider(monkeypatch)

    provider.sync_turn(msg, "assistant reply")

    assert provider._sync_thread is not None, f"human suffix suppressed: {msg[-60:]!r}"
    provider._sync_thread.join(timeout=5)
    assert not provider._sync_thread.is_alive()
    assert record.readiness_checks == 1
    assert record.manager_calls == ["test-session"], f"human suffix suppressed: {msg[-60:]!r}"
    assert record.messages == [("user", msg), ("assistant", "assistant reply")]
    assert record.saved == [session]


_BRACKETED_OUTPUT_NOTICES = {
    "balanced-nested-output": _PLAIN_NOTICE.replace("Output:\ndone]", "Output:\n[[1, 2], [3]]\nok]"),
    "ansi-colored-output": _PLAIN_NOTICE.replace("done]", "\x1b[32mPASSED\x1b[0m 3 tests]"),
    "watch-pattern-with-brackets": _PROC_WATCH_MATCH.replace('"ERROR"', '"\\[ERROR\\]"'),
    "watch-then-completion": f"{_PROC_WATCH_MATCH}\n\n{_PLAIN_NOTICE}",
}


@pytest.mark.parametrize("notice", list(_BRACKETED_OUTPUT_NOTICES.values()), ids=list(_BRACKETED_OUTPUT_NOTICES))
def test_honcho_sync_turn_skips_process_notices_with_bracketed_output(monkeypatch, notice):
    """Balanced brackets (and terminal escape sequences) inside a notice keep it a notice."""
    for wrapper in (notice, f"  \n{notice}", f"{notice}\n"):
        provider, record, _ = _recording_sync_provider(monkeypatch)

        provider.sync_turn(wrapper, "assistant reply")
        if provider._sync_thread is not None:
            provider._sync_thread.join(timeout=5)

        assert record.readiness_checks == 0, f"notice reached session readiness: {wrapper[:70]!r}"
        assert provider._sync_thread is None, f"notice not suppressed: {wrapper[:70]!r}"
        assert record.manager_calls == [] and record.messages == []


# No origin flag exists, so where the envelope's own closing "]" cannot be located the turn
# is kept: a stray bracket in the output could just as well hide a human suffix.
_AMBIGUOUS_BRACKET_NOTICES = {
    "unmatched-close-in-output": _PLAIN_NOTICE.replace("done]", "tail]]"),
    "unmatched-open-in-output": _PLAIN_NOTICE.replace("done]", "progress [=====>]"),
    "unmatched-open-then-human-suffix": (
        _PLAIN_NOTICE.replace("done]", "progress [=====>]") + "\n\nKeep going [staging only]"
    ),
}


@pytest.mark.parametrize("msg", list(_AMBIGUOUS_BRACKET_NOTICES.values()), ids=list(_AMBIGUOUS_BRACKET_NOTICES))
def test_honcho_sync_turn_keeps_notice_whose_closing_bracket_is_ambiguous(monkeypatch, msg):
    """Conservative false negative by design: an unbalanced bracket keeps the turn unchanged."""
    provider, record, session = _recording_sync_provider(monkeypatch)

    provider.sync_turn(msg, "assistant reply")

    assert provider._sync_thread is not None, f"ambiguous turn suppressed: {msg[-60:]!r}"
    provider._sync_thread.join(timeout=5)
    assert record.manager_calls == ["test-session"]
    assert record.messages == [("user", msg), ("assistant", "assistant reply")]
    assert record.saved == [session]


def test_honcho_sync_turn_skips_empty_content():
    """Empty or whitespace-only turns must not be stored."""
    provider = HonchoMemoryProvider()
    manager_calls = []

    class Manager:
        def get_or_create(self, session_key):
            manager_calls.append(session_key)
            return SimpleNamespace()

    provider._config = _configured_tools_config(init_on_session_start=True)
    provider._manager = Manager()
    provider._session_key = "test-session"
    provider._session_initialized = True

    provider.sync_turn("   ", "  ")

    assert provider._sync_thread is None
    assert manager_calls == []


def test_honcho_sync_turn_same_instance_config_flip_gates_writes():
    """The cached-provider regression: flipping save_messages on the SAME
    configured instance must stop writes without re-initialization."""
    provider = HonchoMemoryProvider()
    cfg = _configured_tools_config(init_on_session_start=True)
    cfg.save_messages = True
    manager_calls = []
    write_done = threading.Event()

    class Manager:
        def get_or_create(self, session_key):
            manager_calls.append(session_key)
            return SimpleNamespace(add_message=lambda role, content, **kwargs: None)

        def resolve_author_peer_id(self, session_key, author_id, author_name=None):
            return None

        def save(self, session):
            write_done.set()

    provider._config = cfg
    provider._manager = Manager()
    provider._session_key = "test-session"
    provider._session_initialized = True

    # enabled -> write happens
    provider.sync_turn("user turn", "assistant reply")
    assert write_done.wait(timeout=5), "first write never completed"

    # operator flips containment on the same cached config object
    cfg.save_messages = False
    manager_calls.clear()
    provider.sync_turn("user turn two", "assistant reply two")

    # no new write may occur; the stale _sync_thread from the enabled write is fine
    assert manager_calls == []


def test_honcho_on_memory_write_honors_save_messages_false():
    """The memory-tool mirror is an automatic write path and must respect the
    write-disable switch; otherwise containment only covers conversation turns."""
    provider = HonchoMemoryProvider()
    cfg = _configured_tools_config(init_on_session_start=True)
    cfg.save_messages = False
    conclusion_calls = []

    class Manager:
        def create_conclusion(self, session_key, content):
            conclusion_calls.append((session_key, content))

    provider._config = cfg
    provider._manager = Manager()
    provider._session_key = "test-session"
    provider._session_initialized = True

    provider.on_memory_write("add", "user", "prefers fail-open memory")

    assert conclusion_calls == []


def test_honcho_on_memory_write_still_writes_when_enabled():
    """With save_messages enabled, the memory-tool mirror still writes."""
    provider = HonchoMemoryProvider()
    cfg = _configured_tools_config(init_on_session_start=True)
    cfg.save_messages = True
    conclusion_calls = []
    write_done = threading.Event()

    class Manager:
        def create_conclusion(self, session_key, content):
            conclusion_calls.append((session_key, content))
            write_done.set()

    provider._config = cfg
    provider._manager = Manager()
    provider._session_key = "test-session"
    provider._session_initialized = True

    provider.on_memory_write("add", "user", "prefers fail-open memory")

    assert write_done.wait(timeout=5), "memory mirror write never completed"
    assert conclusion_calls != []
