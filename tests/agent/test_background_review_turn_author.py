"""The review fork runs as the author of the turn that spawned it.

The fork is built with ``skip_memory=True``, so no external provider is there to refuse a bot-authored
write; the built-in memory tool can only refuse it if the fork's own turn carries the originating author.
That author is frozen when the review is spawned: a deferred review dispatched after the next turn, a
requeued one, or one whose parent stash dict is later changed in place must not pick up a different author.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import agent.background_review as bg
import agent.review_idle_queue as riq
import run_agent as run_agent_module
from run_agent import AIAgent
from tests.agent.test_background_review import ImmediateThread, _bare_agent

BOT = {"id": "bot:field-foreman", "name": "Field Foreman", "is_bot": True}
HUMAN = {"id": "1234", "name": "Aldo", "is_bot": False}
SNAPSHOT = [{"role": "user", "content": "field report: the owner approved the rollback"}]


@pytest.fixture()
def fork_runs(monkeypatch):
    """Every fork ``run_conversation`` call, through the real spawn chain with a recording fork."""
    runs = []

    class RecordingFork:
        def __init__(self, **_kwargs):
            self._session_messages = []

        def run_conversation(self, **kwargs):
            runs.append(kwargs)

        def release_clients(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", RecordingFork)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)
    return runs


@pytest.fixture()
def deferred(monkeypatch):
    """Route automatic reviews into a captured idle queue instead of spawning them."""
    from agent.review_idle_queue import QUEUE
    queued = []
    monkeypatch.setattr(run_agent_module, "_review_should_defer", lambda _agent, _cfg: True)
    monkeypatch.setattr(QUEUE, "enqueue", lambda _agent, _key, kwargs: queued.append(kwargs))
    return queued


def _agent_with_stash(author):
    agent = _bare_agent()
    agent._turn_author = None if author is None else dict(author)
    return agent


@pytest.mark.parametrize("spawning, later", [
    pytest.param(BOT, HUMAN, id="bot-turn-review-dispatched-after-human-turn"),
    pytest.param(None, BOT, id="human-turn-review-dispatched-after-bot-turn"),
])
def test_deferred_review_keeps_the_spawning_turns_author(fork_runs, deferred, spawning, later):
    agent = _agent_with_stash(spawning)
    AIAgent._spawn_background_review(agent, SNAPSHOT, review_memory=True)
    (kwargs,) = deferred
    assert fork_runs == []

    # The next turn by the other author replaces the stash before the idle dispatcher runs the review.
    agent._turn_author = None if later is None else dict(later)
    agent._spawn_background_review_now(**kwargs)

    (run,) = fork_runs
    assert run["turn_author"] == spawning


def test_spawn_snapshot_is_not_aliased_to_the_parent_stash(fork_runs, deferred):
    agent = _agent_with_stash(BOT)
    stash = agent._turn_author
    AIAgent._spawn_background_review(agent, SNAPSHOT, review_memory=True)
    stash.update(id="1234", name="Aldo", is_bot=False)  # in-place change of the shared parent dict
    agent._spawn_background_review_now(**deferred[0])
    assert fork_runs[0]["turn_author"] == BOT


def test_immediate_review_carries_the_bot_author(fork_runs):
    agent = _agent_with_stash(BOT)
    AIAgent._spawn_background_review(agent, SNAPSHOT, review_memory=True)
    (run,) = fork_runs
    assert run["turn_author"] == BOT


def test_review_spawned_inside_a_turn_uses_that_turns_binding(fork_runs, deferred):
    """Inside a running turn the bound author wins over the agent attribute, which another turn may own."""
    from agent.turn_author import bound_turn_author
    agent = _agent_with_stash(None)
    with bound_turn_author(dict(BOT)):
        AIAgent._spawn_background_review(agent, SNAPSHOT, review_memory=True)
    with bound_turn_author(dict(HUMAN)):
        agent._spawn_background_review_now(**deferred[0])
    assert fork_runs[0]["turn_author"] == BOT


def test_requeued_review_keeps_the_author(fork_runs, deferred):
    agent = _agent_with_stash(BOT)
    requeued = []
    agent._maybe_requeue_preempted_review = lambda _run, kwargs: requeued.append(kwargs)
    AIAgent._spawn_background_review(agent, SNAPSHOT, review_memory=True)
    agent._turn_author = None
    agent._spawn_background_review_now(**deferred[0])
    (again,) = requeued
    assert again["turn_author"] == BOT
    agent._spawn_background_review_now(**again)
    assert [run["turn_author"] for run in fork_runs] == [BOT, BOT]


@pytest.mark.parametrize("stash, expected", [
    pytest.param(BOT, BOT, id="after-bot-delivery"),
    pytest.param(HUMAN, HUMAN, id="after-human-turn"),
    pytest.param(None, None, id="author-absent-legacy"),
])
def test_explicit_refine_call_site_is_unchanged_and_carries_the_last_turns_author(fork_runs, stash, expected):
    """``/refine`` callers pass no author; the fork reviews the last turn, so it runs as that turn's author."""
    agent = _agent_with_stash(stash)
    AIAgent._spawn_background_review(agent, SNAPSHOT, review_memory=True, explicit=True)
    (run,) = fork_runs
    assert run["turn_author"] == expected


def test_explicit_fork_passes_the_author_and_stays_attended():
    runs, forks = [], []

    def fake_build(agent, task_cfg=None, *, max_iterations, write_origin="background_review"):
        fork = SimpleNamespace(_memory_enabled=True, _user_profile_enabled=False,
                               run_conversation=lambda **kw: runs.append(kw), _session_messages=[])
        forks.append(fork)
        return fork, {}, False

    noop = lambda *a, **k: None
    with patch.object(bg, "build_cache_parity_fork", fake_build), \
            patch.object(bg, "_track_review_fork", noop), \
            patch.object(bg, "_snapshot_review_usage", lambda a: {}), \
            patch.object(bg, "_record_review_usage_to_parent", noop), \
            patch.object(bg, "finish_background_review_run", noop), \
            patch.object(bg, "_release_fork_clients", noop):
        st = bg._ReviewForkState()
        bg._run_review_fork(SimpleNamespace(), [], "p", None, None, st, True, True, turn_author=dict(BOT))
    (run,), (fork,) = runs, forks
    assert run["turn_author"] == BOT
    assert fork._review_attended is True


# ── The real idle queue: enqueue, the dispatcher loop, and the preemption requeue ─────────────────────────────
# A private ReviewIdleQueue replaces the process singleton (nothing else is swapped on it). Only the machine
# probes are stubbed: whether the review targets the managed local server, and whether that server is idle.
# The test thread runs the real dispatcher loop itself, so no dispatcher thread is started.


class _LoopOnce:
    """Wake event handed to the real ``ReviewIdleQueue._run``: lets one loop iteration through, then stops it."""

    class Stop(BaseException):
        pass

    def __init__(self):
        self._open = True

    def wait(self, timeout=None):
        if not self._open:
            raise self.Stop
        self._open = False
        return True

    def set(self):  # enqueue, including a requeue from inside a dispatch, wakes the dispatcher
        pass

    def clear(self):
        pass


@pytest.fixture()
def idle_queue(monkeypatch):
    queue = riq.ReviewIdleQueue()
    clock = {"t": 0.0}
    managed = {"on": True}
    queue._now = lambda: clock["t"]
    queue._server_idle = lambda: True
    queue._ensure_thread = lambda: None
    monkeypatch.setattr(riq, "QUEUE", queue)
    monkeypatch.setattr(riq, "review_targets_managed_local", lambda _agent, _cfg: managed["on"])
    monkeypatch.setattr(riq, "_POLL_INTERVAL_S", 0.0)  # an undispatchable queue fails the test, never sleeps

    def dispatch_next():
        """The next live turn finishes, the machine stays quiet past the settle window, the dispatcher runs."""
        queue.note_turn_started()
        queue.note_turn_finished()
        clock["t"] += riq._IDLE_SETTLE_S + 1
        queue._wake = _LoopOnce()
        with pytest.raises(_LoopOnce.Stop):
            queue._run()

    def pending():
        with queue._lock:
            return [item.kwargs for item in queue._pending.values()]

    return SimpleNamespace(queue=queue, managed=managed, dispatch_next=dispatch_next, pending=pending)


@pytest.mark.parametrize("where", ["stash", "turn-binding"])
@pytest.mark.parametrize("spawning, later", [
    pytest.param(BOT, HUMAN, id="bot-turn-review-dispatched-after-human-turn"),
    pytest.param(HUMAN, BOT, id="human-turn-review-dispatched-after-bot-turn"),
    pytest.param(None, BOT, id="authorless-turn-review-dispatched-after-bot-turn"),
])
def test_real_idle_queue_dispatches_the_review_as_the_spawning_turns_author(
        fork_runs, idle_queue, spawning, later, where):
    """``stash``: spawned between turns from ``_turn_author`` (no binding rides in the queued context).
    ``turn-binding``: spawned by ``finalize_turn`` inside the turn, with an empty stash."""
    from agent.turn_author import bound_turn_author
    if where == "stash":
        agent = _agent_with_stash(spawning)
        AIAgent._spawn_background_review(agent, SNAPSHOT, review_memory=True)
    else:
        agent = _agent_with_stash(None)
        with bound_turn_author(spawning):
            AIAgent._spawn_background_review(agent, SNAPSHOT, review_memory=True)
    (queued,) = idle_queue.pending()
    assert queued["turn_author"] == spawning
    assert fork_runs == []

    # Turn N+1 by the other author replaces the stash and is bound on the thread that runs the dispatcher.
    agent._turn_author = None if later is None else dict(later)
    with bound_turn_author(later):
        idle_queue.dispatch_next()

    assert idle_queue.pending() == []
    (run,) = fork_runs
    assert run["turn_author"] == spawning
    assert run["conversation_history"] == SNAPSHOT


@pytest.fixture()
def preempting_fork_runs(fork_runs, monkeypatch):
    """``fork_runs``, but the first fork that starts is preempted by a live turn: the running review's own
    ``_BackgroundReviewRun.cancel()`` fence, the one the live-turn canceller trips, fired from inside the fork."""
    state = {"parent": None, "left": 1}
    recording = run_agent_module.AIAgent

    class PreemptedOnce(recording):
        def run_conversation(self, **kwargs):
            super().run_conversation(**kwargs)
            if state["left"]:
                state["left"] -= 1
                state["parent"]._background_review_run.cancel()

    monkeypatch.setattr(run_agent_module, "AIAgent", PreemptedOnce)
    return fork_runs, state


@pytest.mark.parametrize("spawning, later", [
    pytest.param(BOT, HUMAN, id="bot-review-preempted-by-human-turn"),
    pytest.param(HUMAN, BOT, id="human-review-preempted-by-bot-turn"),
])
def test_preempted_review_requeued_by_the_real_method_keeps_its_author_and_snapshot(
        preempting_fork_runs, idle_queue, spawning, later):
    runs, preempt = preempting_fork_runs
    agent = preempt["parent"] = _agent_with_stash(spawning)
    AIAgent._spawn_background_review(agent, SNAPSHOT, review_memory=True)
    agent._turn_author = dict(later)  # turn N+1 starts, preempts the running review, and owns the stash

    idle_queue.dispatch_next()  # dispatched, preempted, requeued by _maybe_requeue_preempted_review
    (requeued,) = idle_queue.pending()
    assert requeued["_requeue_attempts"] == 1
    assert requeued["turn_author"] == spawning
    assert requeued["messages_snapshot"] == SNAPSHOT

    idle_queue.dispatch_next()  # not preempted this time: it completes and is not requeued
    assert idle_queue.pending() == []
    assert [run["turn_author"] for run in runs] == [spawning, spawning]
    assert [run["conversation_history"] for run in runs] == [SNAPSHOT, SNAPSHOT]


def test_next_turns_review_coalesces_over_a_requeued_one_with_its_own_author(preempting_fork_runs, idle_queue):
    """One slot per session, newest wins. The human turn's own review replaces the requeued bot review, and the
    author always travels with its own snapshot: the slot never pairs the human snapshot with the bot."""
    runs, preempt = preempting_fork_runs
    agent = preempt["parent"] = _agent_with_stash(BOT)
    AIAgent._spawn_background_review(agent, SNAPSHOT, review_memory=True)
    agent._turn_author = dict(HUMAN)
    idle_queue.dispatch_next()  # the bot review is preempted by the human turn and requeued
    assert [kw["turn_author"] for kw in idle_queue.pending()] == [BOT]

    human_snapshot = SNAPSHOT + [{"role": "user", "content": "keep the rollback note out of memory"}]
    AIAgent._spawn_background_review(agent, human_snapshot, review_memory=True)  # the human turn's own review
    (slot,) = idle_queue.pending()
    assert (slot["turn_author"], slot["messages_snapshot"]) == (HUMAN, human_snapshot)
    assert slot.get("_requeue_attempts", 0) == 0

    idle_queue.dispatch_next()
    assert idle_queue.pending() == []
    assert [(run["turn_author"], run["conversation_history"]) for run in runs] == [
        (BOT, SNAPSHOT), (HUMAN, human_snapshot)]


# ── A real review fork, end to end, through the built-in memory tool ──────────────────────────────────────────
# A real parent AIAgent, the real spawn chain and the real fork it builds (``build_cache_parity_fork``,
# ``skip_memory=True``), running a real ``memory`` tool call against the parent's real MemoryStore. Only the
# model endpoint is stubbed. The review thread is a real thread, joined before anything is asserted.

SEED = "Seeded entry from the owner"
_FORK_CALLS = {
    "call-user-add": {"target": "user", "operations": [{"action": "add", "content": "Owner approved the rollback"}]},
    "call-memory-remove": {"action": "remove", "target": "memory", "old_text": "Seeded"},
}


def _completion(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=tool_calls, reasoning=None, reasoning_content=None),
        finish_reason="tool_calls" if tool_calls else "stop")], usage=None, model="test-model")


@pytest.fixture()
def real_review(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # This test's own HERMES_HOME: the stub endpoint answers non-streaming requests only.
    (tmp_path / "config.yaml").write_text("model:\n  streaming: false\n", encoding="utf-8")
    try:
        from agent.turn_author import current_turn_author as probe
    except ImportError:  # unmodified base: keep the stub endpoint from raising, let the assertions decide
        probe = lambda: "unbound"
    requests, seen_authors = [], []

    class _Completions:
        def create(self, **kwargs):
            requests.append(kwargs)
            seen_authors.append(probe())
            if len(requests) > 1:
                return _completion(content="done")
            return _completion(tool_calls=[
                SimpleNamespace(id=call_id, type="function",
                                function=SimpleNamespace(name="memory", arguments=json.dumps(args)))
                for call_id, args in _FORK_CALLS.items()])

    monkeypatch.setattr("agent.process_bootstrap.OpenAI",
                        lambda **_kw: SimpleNamespace(chat=SimpleNamespace(completions=_Completions())))

    review_threads = []
    real_thread = threading.Thread

    class _TrackedThread(real_thread):
        def start(self):
            if self.name == "bg-review":
                review_threads.append(self)
            super().start()

    monkeypatch.setattr(run_agent_module.threading, "Thread", _TrackedThread)

    parent = AIAgent(model="test-model", api_key="k", base_url="http://localhost:1/v1", platform="cli",
                     max_iterations=4, quiet_mode=True, skip_memory=True, enabled_toolsets=["memory"])
    assert parent._disable_streaming is True and parent._memory_store is not None
    parent._memory_store.add("memory", SEED)

    def join():
        assert len(review_threads) == 1, review_threads
        review_threads[0].join(30)
        assert not review_threads[0].is_alive()

    def disk():
        mem = tmp_path / "memories"
        return {name: (mem / name).read_text(encoding="utf-8") if (mem / name).exists() else None
                for name in ("MEMORY.md", "USER.md")}

    def tool_results():
        assert len(requests) >= 2, requests
        return {m["tool_call_id"]: json.loads(m["content"])
                for m in requests[1]["messages"] if m.get("role") == "tool"}

    def pending():
        from tools.write_approval import MEMORY, list_pending
        return list_pending(MEMORY)

    return SimpleNamespace(parent=parent, join=join, disk=disk, tool_results=tool_results, pending=pending,
                           seen_authors=seen_authors)


@pytest.mark.parametrize("deferred", [False, True], ids=["immediate", "deferred-past-the-next-turn"])
@pytest.mark.parametrize("spawning, later", [
    pytest.param(BOT, HUMAN, id="bot-turn"),
    pytest.param(HUMAN, BOT, id="human-turn-control"),
])
def test_real_review_fork_runs_the_memory_tool_as_the_spawning_turns_author(
        real_review, idle_queue, spawning, later, deferred):
    parent = real_review.parent
    idle_queue.managed["on"] = deferred
    before = real_review.disk()
    parent._turn_author = dict(spawning)
    AIAgent._spawn_background_review(parent, SNAPSHOT, review_memory=True)
    if deferred:
        assert len(idle_queue.pending()) == 1
        parent._turn_author = dict(later)  # turn N+1 by the other author runs before the idle dispatch
        idle_queue.dispatch_next()
        assert idle_queue.pending() == []
    real_review.join()

    results = real_review.tool_results()
    assert set(results) == set(_FORK_CALLS)
    store = parent._memory_store
    if spawning is BOT:
        for result in results.values():
            assert result["success"] is False and not result.get("staged"), result
            assert "nothing was written" in result["error"].lower(), result
        assert real_review.disk() == before
        assert (store._entries_for("memory"), store._entries_for("user")) == ([SEED], [])
        assert real_review.pending() == []
    else:
        assert results["call-user-add"]["success"] is True, results
        # The unattended fork's remove is held by the existing delete gate: staged for the owner, not applied.
        assert results["call-memory-remove"]["staged"] is True, results
        assert "Owner approved the rollback" in real_review.disk()["USER.md"]
        assert real_review.disk()["MEMORY.md"] == before["MEMORY.md"]
        assert (store._entries_for("memory"), store._entries_for("user")) == ([SEED], ["Owner approved the rollback"])
        (record,) = real_review.pending()
        assert record["origin"] == "background_review"
    # Inside the fork's own model calls the bound author is the spawning turn's, never turn N+1's.
    assert real_review.seen_authors[:2] == [spawning, spawning]
