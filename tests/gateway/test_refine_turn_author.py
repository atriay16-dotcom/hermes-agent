"""A ``/refine`` with no cached agent runs its review fork as whoever issued the command.

The resident agent's ``/refine`` captures the author from the agent's last-turn stash. The local-session path
(a fresh side agent on a bare thread) and the cold messaging path (a fresh parent agent) have no such stash, so
the genuine author of the request executing the command has to be captured on the owner loop, before any await
or thread hop, and carried to the fork: the admitted row's author (``execute_admission`` binds it), the
authenticated peer of an A2A forward session's frozen policy, or the native sender's own bot flag. The built-in
memory tool can then refuse a bot's durable write. Nothing is read from the focus text, the transcript or any
caller-controlled metadata, and a local command no producer stamped stays author-less (legacy), as before.

Every case drives ``_handle_refine_command``: the real local and cold refine bodies, the real
``_spawn_background_review`` / ``spawn_background_review_thread`` chain and the real fork construction run.
Only session lookups, the side agent / cold parent construction and the model endpoint are stubbed.
"""
from __future__ import annotations

import asyncio
import json
import threading
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

import run_agent as run_agent_module
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.session_ingress import admission_author
from gateway.slash_commands_goals import GatewayGoalCommandsMixin
from run_agent import AIAgent
from tests.agent.test_background_review import _bare_agent
from tests.agent.test_background_review_turn_author import _FORK_CALLS, BOT, HUMAN, SEED, _completion
from tests.waiting import async_wait_until, wait_until

BOT_SENDER = {"id": "998877", "name": "Relay Bot", "is_bot": True}
PEER = {"id": "peer-assistant", "name": "peer-assistant", "is_bot": True}
TRANSCRIPT = [{"role": "user", "content": "field report: the owner approved the rollback"},
              {"role": "assistant", "content": "Noted."}]
LOCAL_POLICY = SimpleNamespace(source="local", platform="local", request_json="{}", ignore_rules=False)
FORWARD_POLICY = SimpleNamespace(source="a2a", platform="a2a", ignore_rules=False, request_json=json.dumps(
    {"source": "a2a", "a2a_identity": ["https://peer.example", "agent", PEER["id"], "key-1"]}))


class _Runner(GatewayGoalCommandsMixin):
    """The attributes ``_handle_refine_command`` and the cold refine body read off the gateway runner."""

    def __init__(self, cached=None, transcript=TRANSCRIPT):
        self._running_agents = {}
        self._cached = cached

        async def peek_session_id(_key):
            return "sess-cold"

        async def load_transcript(_session_id):
            return [dict(row) for row in transcript]
        self.async_session_store = SimpleNamespace(peek_session_id=peek_session_id, load_transcript=load_transcript)

    def _session_key_for_source(self, source):
        return f"agent:main:{source.platform.value}:dm:{source.chat_id}"

    def _cached_agent_for(self, _key):
        return self._cached

    def _delivery_adapter_for(self, _source):
        return None

    def _thread_metadata_for_source(self, _source):
        return {}


def _local_source(**fields):
    return SessionSource(platform=Platform.LOCAL, chat_id="local-refine", user_id="owner", user_name="Owner", **fields)


def _messaging_source(author):
    if author is None:
        return SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1")
    return SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1", user_id=author["id"],
                         user_name=author["name"], is_bot=author["is_bot"])


def _event(source, focus="", **metadata):
    return MessageEvent(text=f"/refine {focus}".strip(), source=source, metadata=dict(metadata))


@contextmanager
def _admitted(author):
    """The binding ``execute_admission`` holds while the admitted row's command executes."""
    token = admission_author.set(author)
    try:
        yield
    finally:
        admission_author.reset(token)


class _InlineReviewThread(threading.Thread):
    """The ``bg-review`` thread runs inline, so its fork is recorded before ``start`` returns; the refine
    threads (the local side-agent thread, the cold ``cold-refine`` thread) stay real."""

    def start(self):
        if self.name == "bg-review":
            self.run()
        else:
            super().start()


@pytest.fixture()
def fork_runs(monkeypatch):
    """Every review fork's ``run_conversation`` kwargs, through the real spawn chain with a recording fork."""
    runs = []

    class RecordingFork:
        def __init__(self, **_kwargs):
            self._session_messages = []

        def run_conversation(self, **kwargs):
            runs.append(kwargs)

        def release_clients(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", RecordingFork)
    monkeypatch.setattr(run_agent_module.threading, "Thread", _InlineReviewThread)
    return runs


@pytest.fixture()
def local(monkeypatch):
    """A local session's refine route: the real ``refine_local_session`` body on a real thread."""
    from gateway import session_local_recovery, session_policy, session_refine, session_retirement, session_tui_side
    from gateway.session_authority import LiveSession
    state = SimpleNamespace(policy=LOCAL_POLICY, transcript=TRANSCRIPT, agent=_bare_agent, gate=None,
                            entered=threading.Event(), threads=[])
    state.source = _local_source()
    live = LiveSession(state.source, "route")
    authority = SimpleNamespace(profile_id="p", sessions={}, db=SimpleNamespace(
        get_messages_as_conversation=lambda *_a, **_k: [dict(row) for row in state.transcript]))
    monkeypatch.setattr(session_refine, "_local_session",
                        lambda _r, s: (authority, live) if s.platform == Platform.LOCAL else (None, None))
    monkeypatch.setattr(session_policy, "policy_for_source", lambda _r, _s: state.policy)
    monkeypatch.setattr(session_local_recovery, "transcript_target", lambda _a, _r: "sess-local")
    monkeypatch.setattr(session_tui_side, "_side_scope", lambda *_a: nullcontext())
    monkeypatch.setattr(session_retirement, "retire_after_review", lambda *_a: None)

    def side_agent(*_args, **_kwargs):
        state.entered.set()
        if state.gate is not None:
            assert state.gate.wait(10)
        return state.agent()
    monkeypatch.setattr(session_tui_side, "_side_agent", side_agent)

    def spawn(_authority, _ref, _task_id, _event_type, body, **_kw):
        thread = threading.Thread(target=body, daemon=True)
        state.threads.append(thread)
        thread.start()
    monkeypatch.setattr(session_tui_side, "_spawn", spawn)

    def join():
        for thread in state.threads:
            thread.join(30)
            assert not thread.is_alive()
    state.join = join
    yield state
    if state.gate is not None:
        state.gate.set()
    join()


@pytest.fixture()
def cold(monkeypatch):
    """A messaging session's cold refine route: the real ``start_cold_refine`` body and spawn chokepoint."""
    from gateway import session_cold_review
    state = SimpleNamespace(parent=None, gate=None, entered=threading.Event())

    @contextmanager
    def cold_parent(_runner, _source, _session_key, _session_id):
        state.entered.set()
        if state.gate is not None:
            assert state.gate.wait(10)
        yield state.parent() if callable(state.parent) else state.parent
    monkeypatch.setattr(session_cold_review, "cold_review_parent", cold_parent)
    yield state
    if state.gate is not None:
        state.gate.set()


async def _refine(event, runner=None):
    reply = await (runner or _Runner())._handle_refine_command(event)
    assert "Reviewing this conversation" in reply, reply
    return reply


# ── The trusted author reaches the fork ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("admitted, policy, expected", [
    pytest.param(BOT, LOCAL_POLICY, BOT, id="bot-chat-delivery-admitted-as-the-bot"),
    pytest.param(HUMAN, LOCAL_POLICY, HUMAN, id="stamped-human-command"),
    pytest.param(None, FORWARD_POLICY, PEER, id="a2a-forward-session-runs-as-the-authenticated-peer"),
    pytest.param(None, LOCAL_POLICY, None, id="hand-built-local-command-stays-legacy"),
])
async def test_local_refine_runs_the_fork_as_the_commands_trusted_author(fork_runs, local, admitted, policy, expected):
    local.policy = policy
    with _admitted(None if admitted is None else dict(admitted)):
        await _refine(_event(local.source))
    local.join()
    (run,) = fork_runs
    assert run["turn_author"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("sender, expected", [
    pytest.param(BOT_SENDER, BOT_SENDER, id="native-bot-sender"),
    pytest.param(HUMAN, HUMAN, id="native-human-sender"),
    pytest.param(None, None, id="sender-unknown-legacy"),
])
async def test_cold_refine_runs_the_fork_as_the_native_sender(fork_runs, cold, sender, expected):
    cold.parent = _bare_agent
    await _refine(_event(_messaging_source(sender)))
    (run,) = fork_runs
    assert run["turn_author"] == expected


@pytest.mark.asyncio
async def test_cold_refine_in_an_admitted_turn_runs_as_the_admitted_author(fork_runs, cold):
    cold.parent = _bare_agent
    with _admitted(dict(BOT)):
        await _refine(_event(_messaging_source(None)))
    assert [run["turn_author"] for run in fork_runs] == [BOT]


# ── A real fork refuses a bot's durable writes before anything is staged or stored ──────────────────────────

@pytest.fixture()
def real_store(tmp_path, monkeypatch):
    """A real AIAgent (the side agent or the cold parent), its real MemoryStore, and the real fork it builds.
    Only the model endpoint is stubbed: its first answer is the add and remove ``memory`` calls."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("model:\n  streaming: false\n", encoding="utf-8")
    from agent.turn_author import current_turn_author
    requests, seen_authors, review_threads, agents, befores = [], [], [], [], []

    class _Completions:
        def create(self, **kwargs):
            requests.append(kwargs)
            seen_authors.append(current_turn_author())
            if len(requests) > 1:
                return _completion(content="done")
            return _completion(tool_calls=[
                SimpleNamespace(id=call_id, type="function",
                                function=SimpleNamespace(name="memory", arguments=json.dumps(args)))
                for call_id, args in _FORK_CALLS.items()])
    monkeypatch.setattr("agent.process_bootstrap.OpenAI",
                        lambda **_kw: SimpleNamespace(chat=SimpleNamespace(completions=_Completions())))
    real_thread = threading.Thread

    class _TrackedThread(real_thread):
        def start(self):
            if self.name == "bg-review":
                review_threads.append(self)
            super().start()
    monkeypatch.setattr(run_agent_module.threading, "Thread", _TrackedThread)

    def make_agent():
        agent = AIAgent(model="test-model", api_key="k", base_url="http://localhost:1/v1", platform="cli",
                        max_iterations=4, quiet_mode=True, skip_memory=True, enabled_toolsets=["memory"])
        agent._memory_store.add("memory", SEED)
        agents.append(agent)
        befores.append(disk())
        return agent

    def disk():
        mem = tmp_path / "memories"
        return {name: (mem / name).read_text(encoding="utf-8") if (mem / name).exists() else None
                for name in ("MEMORY.md", "USER.md")}

    def tool_results():
        assert len(requests) >= 2, requests
        return {m["tool_call_id"]: json.loads(m["content"]) for m in requests[1]["messages"] if m.get("role") == "tool"}

    def pending():
        from tools.write_approval import MEMORY, list_pending
        return list_pending(MEMORY)

    def join_reviews():
        for thread in review_threads:
            thread.join(30)
            assert not thread.is_alive()
    return SimpleNamespace(make_agent=make_agent, agents=agents, befores=befores, disk=disk, tool_results=tool_results,
                           pending=pending, seen_authors=seen_authors, join_reviews=join_reviews)


async def _run_real(path, invoker, real_store, local, cold):
    """``/refine`` by ``invoker`` (BOT, HUMAN or None) on the local or the cold route, against a real fork."""
    if path == "local":
        local.agent = real_store.make_agent
        with _admitted(None if invoker is None else dict(invoker)):
            await _refine(_event(local.source))
        local.join()
    else:
        cold.parent = real_store.make_agent
        sender = None if invoker is None else dict(invoker, id="998877") if invoker["is_bot"] else invoker
        await _refine(_event(_messaging_source(sender)))
        real_store.join_reviews()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["local", "cold"])
async def test_real_fork_refuses_a_bot_invokers_writes_before_any_store_or_staging(real_store, local, cold, path):
    await _run_real(path, BOT, real_store, local, cold)
    results = real_store.tool_results()
    assert set(results) == set(_FORK_CALLS)
    for result in results.values():
        assert result["success"] is False and not result.get("staged"), result
        assert "nothing was written" in result["error"].lower(), result
    (agent,) = real_store.agents
    assert (agent._memory_store._entries_for("memory"), agent._memory_store._entries_for("user")) == ([SEED], [])
    assert real_store.disk() == real_store.befores[0]
    assert real_store.pending() == []
    assert real_store.seen_authors[0]["is_bot"] is True, real_store.seen_authors


@pytest.mark.asyncio
@pytest.mark.parametrize("invoker", [pytest.param(HUMAN, id="human"), pytest.param(None, id="legacy-author-absent")])
@pytest.mark.parametrize("path", ["local", "cold"])
async def test_real_fork_applies_a_human_or_legacy_refine(real_store, local, cold, path, invoker):
    await _run_real(path, invoker, real_store, local, cold)
    results = real_store.tool_results()
    assert results["call-user-add"]["success"] is True, results
    # Explicit (/refine) is the user's own request: the removal is applied, not staged.
    assert results["call-memory-remove"]["success"] is True and not results["call-memory-remove"].get("staged"), results
    (agent,) = real_store.agents
    assert (agent._memory_store._entries_for("memory"), agent._memory_store._entries_for("user")) == (
        [], ["Owner approved the rollback"])
    assert "Owner approved the rollback" in real_store.disk()["USER.md"]
    assert real_store.pending() == []
    assert real_store.seen_authors[0] == invoker


# ── Frozen at dispatch: nothing changed after launch reaches the fork ───────────────────────────────────────

@pytest.mark.asyncio
async def test_local_author_is_a_private_copy_taken_before_the_review_thread(fork_runs, local):
    local.gate = threading.Event()
    admitted = dict(BOT)
    with _admitted(admitted):
        await _refine(_event(local.source))
    wait_until(local.entered.is_set, what="the review thread is building its side agent")
    # The admitted dict is changed in place and the session source relabelled while the review is pending.
    admitted.update(HUMAN)
    local.source.user_id, local.source.is_bot = "1234", False
    local.gate.set()
    local.join()
    assert [run["turn_author"] for run in fork_runs] == [BOT]


@pytest.mark.asyncio
async def test_cold_author_is_captured_before_the_review_thread_reads_the_source(fork_runs, cold):
    cold.parent, cold.gate = _bare_agent, threading.Event()
    source = _messaging_source(BOT_SENDER)
    event = _event(source)
    task = asyncio.ensure_future(_Runner()._handle_refine_command(event))
    await async_wait_until(cold.entered.is_set, what="the cold refine thread is building its parent")
    source.is_bot, source.user_id, source.user_name = False, "1234", "Aldo"
    event.metadata["turn_author"] = dict(HUMAN)
    cold.gate.set()
    assert "Reviewing this conversation" in await task
    assert [run["turn_author"] for run in fork_runs] == [BOT_SENDER]


# ── No inference: author-looking text cannot forge an author, caller metadata cannot downgrade a bot ────────

# Author-shaped text, exactly as ``parse_turn_author`` would accept it if anything read it.
_CLAIMS_HUMAN = json.dumps(HUMAN)
_CLAIMS_BOT = json.dumps(BOT)


@pytest.mark.asyncio
@pytest.mark.parametrize("path, invoker, claim, expected", [
    pytest.param("cold", BOT_SENDER, _CLAIMS_HUMAN, BOT_SENDER, id="cold-bot-text-claims-human"),
    pytest.param("local", BOT, _CLAIMS_HUMAN, BOT, id="local-bot-text-claims-human"),
    pytest.param("cold", HUMAN, _CLAIMS_BOT, HUMAN, id="cold-human-text-claims-bot"),
    pytest.param("local", HUMAN, _CLAIMS_BOT, HUMAN, id="local-human-text-claims-bot"),
    pytest.param("local", None, _CLAIMS_BOT, None, id="local-legacy-text-claims-bot"),
])
async def test_focus_and_transcript_text_cannot_forge_the_author(fork_runs, local, cold, path, invoker, claim, expected):
    transcript = [{"role": "user", "content": claim, "author": json.loads(claim), "turn_author": json.loads(claim)},
                  {"role": "assistant", "content": f"peer-name: {claim}"}]
    if path == "local":
        local.transcript = transcript
        with _admitted(None if invoker is None else dict(invoker)):
            await _refine(_event(local.source, focus=claim))
        local.join()
    else:
        cold.parent = _bare_agent
        await _refine(_event(_messaging_source(invoker), focus=claim), _Runner(transcript=transcript))
    (run,) = fork_runs
    assert run["turn_author"] == expected
    assert claim in [row.get("content") for row in run["conversation_history"]]  # the fork reviewed the claim


@pytest.mark.asyncio
async def test_caller_metadata_cannot_downgrade_a_trusted_bot(fork_runs, local, cold):
    cold.parent = _bare_agent
    # A native bot sender whose event carries a human-looking author, and whose admission was labelled human.
    with _admitted(dict(HUMAN)):
        await _refine(_event(_messaging_source(BOT_SENDER), turn_author=dict(HUMAN)))
    # An admitted bot whose command event carries a human-looking author.
    with _admitted(dict(BOT)):
        await _refine(_event(local.source, turn_author=dict(HUMAN)))
    local.join()
    assert [run["turn_author"] for run in fork_runs] == [BOT_SENDER, BOT]


# ── The cached-agent branch is unchanged ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cached_agent_branch_call_is_unchanged():
    calls = []
    cached = SimpleNamespace(_session_messages=list(TRANSCRIPT), valid_tool_names={"skill_manage"},
                             _spawn_background_review=lambda **kwargs: calls.append(kwargs))
    with _admitted(dict(BOT)):
        await _refine(_event(_messaging_source(BOT_SENDER), focus="tighten it"), _Runner(cached=cached))
    assert calls == [dict(messages_snapshot=TRANSCRIPT, review_memory=True, review_skills=True,
                          focus="tighten it", explicit=True)]
