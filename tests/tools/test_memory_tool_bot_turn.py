"""A bot-authored turn cannot change the built-in MEMORY.md / USER.md stores.

Both stores are injected into every future turn as the owner's own facts and carry no provenance, so a
bot's report written there would read as something the owner said. The external provider already refuses
bot-authored mirrors, but only AFTER the local write has landed; the refusal has to happen first, before
the store, the pending-approval store, or the mirror is touched. The author comes from the runtime binding
of the turn (never from tool arguments) and survives a worker that runs after the parent turn moved on.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from tools.memory_tool import memory_tool
from tools.memory_tool_store import MemoryStore

BOT = {"id": "bot:field-foreman", "name": "Field Foreman", "is_bot": True}
HUMAN = {"id": "1234", "name": "Aldo", "is_bot": False}
SEED = "Seeded entry from the owner"


def _bound(author):
    from agent.turn_author import bound_turn_author
    return bound_turn_author(author)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    s = MemoryStore(memory_char_limit=2000, user_char_limit=1000)
    s.load_from_disk()
    return s


def _disk(home):
    mem = home / "memories"
    return {name: (mem / name).read_text(encoding="utf-8") if (mem / name).exists() else None
            for name in ("MEMORY.md", "USER.md")}


def _pending():
    from tools.write_approval import MEMORY, list_pending
    return list_pending(MEMORY)


def _call(store, **args):
    return json.loads(memory_tool(store=store, **args))


def _assert_refused(result):
    assert result["success"] is False, result
    assert not result.get("staged"), result
    assert "nothing was written" in result["error"].lower(), result


SINGLE_MUTATIONS = [
    pytest.param(dict(action="add", content="Owner approved the rollback"), id="add"),
    pytest.param(dict(action="add", new_text="Owner approved the rollback"), id="add-new_text-alias"),
    pytest.param(dict(action="patch", old_text="Seeded entry", content="Bot entry"), id="patch"),
    pytest.param(dict(action="patch", old_text="Seeded entry", new_text="Bot entry"), id="patch-new_text-alias"),
    pytest.param(dict(action="remove", old_text="Seeded entry"), id="remove"),
]


@pytest.mark.parametrize("target", ["memory", "user"])
@pytest.mark.parametrize("call", SINGLE_MUTATIONS)
def test_bot_turn_single_mutation_is_refused_before_the_store(store, tmp_path, target, call):
    store.add(target, SEED)
    before = _disk(tmp_path)
    with _bound(BOT):
        result = _call(store, target=target, **call)
    _assert_refused(result)
    assert _disk(tmp_path) == before
    assert store._entries_for(target) == [SEED]
    assert _pending() == []


@pytest.mark.parametrize("target", ["memory", "user"])
def test_bot_turn_batch_is_refused_whole(store, tmp_path, target):
    store.add(target, SEED)
    before = _disk(tmp_path)
    operations = [
        {"action": "add", "content": "Bot fact one"},
        {"action": "patch", "old_text": "Seeded entry", "new_text": "Bot rewrite"},
        {"action": "remove", "old_text": "Seeded"},
    ]
    with _bound(BOT):
        result = _call(store, target=target, operations=operations)
    _assert_refused(result)
    assert _disk(tmp_path) == before
    assert store._entries_for(target) == [SEED]
    assert _pending() == []


@pytest.mark.parametrize("attended", [False, True], ids=["unattended-review", "explicit-refine"])
def test_bot_authored_review_fork_neither_writes_nor_stages(store, tmp_path, attended):
    """The review fork runs with ``skip_memory=True`` (no external provider guard); the unattended
    delete gate would otherwise STAGE a pending record, itself a write the owner is asked to approve."""
    from tools.skill_provenance import (
        reset_current_write_origin, reset_review_attended, set_current_write_origin, set_review_attended)
    store.add("memory", SEED)
    before = _disk(tmp_path)
    origin, attend = set_current_write_origin("background_review"), set_review_attended(attended)
    try:
        with _bound(BOT):
            removed = _call(store, action="remove", target="memory", old_text="Seeded")
            added = _call(store, target="memory", operations=[{"action": "add", "content": "Bot fact"}])
    finally:
        reset_review_attended(attend)
        reset_current_write_origin(origin)
    _assert_refused(removed)
    _assert_refused(added)
    assert _disk(tmp_path) == before
    assert _pending() == []


@pytest.mark.parametrize("call, expected", [
    pytest.param(dict(action="replace", target="memory", old_text="Seeded", content="x"),
                 "'replace' is not available", id="legacy-replace"),
    pytest.param(dict(action="frobnicate", target="memory", content="x"), "Unknown action", id="unknown-action"),
    pytest.param(dict(action="add", target=None, content="x"), "Missing required target", id="missing-target"),
    pytest.param(dict(action="add", target="memory"), "Content is required", id="add-without-content"),
    pytest.param(dict(target="memory", operations="not-a-list"), "operations must be a list", id="bad-operations"),
])
def test_bot_turn_keeps_the_existing_malformed_call_errors(store, tmp_path, call, expected):
    before = _disk(tmp_path)
    with _bound(BOT):
        result = _call(store, **call)
    assert result.get("success") is False, result
    assert expected in result["error"], result
    assert _disk(tmp_path) == before


def test_bot_turn_keeps_the_non_mutating_inventory_read(store):
    """A remove without old_text answers with the current entries; it reads, it changes nothing."""
    store.add("user", SEED)
    with _bound(BOT):
        result = _call(store, action="remove", target="user")
    assert result["success"] is False
    assert result["current_entries"] == [SEED]


SHORT_DECISIONS = ["Yes", "No", "Stop", "Proceed", "ok"]


@pytest.mark.parametrize("content", SHORT_DECISIONS)
def test_unbound_legacy_call_writes_short_decisions(store, tmp_path, content):
    assert _call(store, action="add", target="user", content=f"{content}")["success"] is True
    assert content in _disk(tmp_path)["USER.md"]


@pytest.mark.parametrize("author", [None, HUMAN, {"id": "1234", "name": "Aldo"}],
                         ids=["no-author", "human", "human-without-is_bot"])
@pytest.mark.parametrize("content", SHORT_DECISIONS)
def test_human_turn_writes_short_decisions_and_corrections(store, tmp_path, author, content):
    with _bound(author):
        assert _call(store, action="add", target="user", content=content)["success"] is True
        assert _call(store, action="patch", target="user", old_text=content,
                     content=f"{content}, confirmed")["success"] is True
        batch = _call(store, target="memory", operations=[{"action": "add", "content": f"Decision: {content}"}])
        assert batch["success"] is True
    disk = _disk(tmp_path)
    assert f"{content}, confirmed" in disk["USER.md"]
    assert f"Decision: {content}" in disk["MEMORY.md"]


def test_bound_author_is_a_private_snapshot(store, tmp_path):
    """Neither the caller's dict (the agent stash) nor a value read back can flip the bound author."""
    from agent.turn_author import current_turn_author
    raw = dict(BOT)
    with _bound(raw):
        raw["is_bot"] = False
        read_back = current_turn_author()
        read_back["is_bot"] = False
        assert current_turn_author()["is_bot"] is True
        result = _call(store, action="add", target="user", content="Bot fact")
    _assert_refused(result)
    assert _disk(tmp_path)["USER.md"] is None


def test_nested_turn_restores_the_enclosing_author():
    """A child turn on the same thread (it binds its own author) must not clear the parent's."""
    from agent.turn_author import current_turn_author
    with _bound(BOT):
        with _bound(None):
            assert current_turn_author() is None
        assert current_turn_author() == BOT
    assert current_turn_author() is None


@pytest.mark.parametrize("worker_author, later_parent_author, writes", [
    pytest.param(BOT, HUMAN, False, id="delayed-bot-worker-after-human-turn"),
    pytest.param(HUMAN, BOT, True, id="delayed-human-worker-after-bot-turn"),
])
def test_delayed_worker_keeps_its_own_turn_author(store, tmp_path, worker_author, later_parent_author, writes):
    """The tool executor submits workers through ``propagate_context_to_thread``; a worker that runs
    after the parent turn moved on still acts as its own turn's author."""
    from tools.thread_context import propagate_context_to_thread
    release = threading.Event()

    def _worker():
        assert release.wait(10)
        return _call(store, action="add", target="user", content="Fact from the delayed worker")

    with ThreadPoolExecutor(max_workers=1) as pool:
        with _bound(worker_author):
            future = pool.submit(propagate_context_to_thread(_worker))
        with _bound(later_parent_author):
            release.set()
            result = future.result(timeout=10)
    assert result["success"] is writes, result
    assert (_disk(tmp_path)["USER.md"] is not None) is writes


# ── The ordinary write-approval gate (memory.write_approval: true) and the human approval replay ─────────────
# Configured in each test's own HERMES_HOME the way tests/tools/test_write_approval.py does it. With no inline
# prompt channel a gated write is STAGED as a pending record: a bot's request must be refused before that.


@pytest.fixture()
def approval_on(store, tmp_path):
    from tests.tools.test_write_approval import _set_approval
    from tools import write_approval as wa
    from tools.terminal_tool import set_approval_callback
    _set_approval("memory", True)
    assert (tmp_path / "config.yaml").is_file() and wa.write_approval_enabled(wa.MEMORY)
    set_approval_callback(None)  # no inline prompt on this thread: a gated write stages
    yield store
    set_approval_callback(None)


@contextmanager
def _origin(kind):
    """``foreground``: an ordinary turn. ``attended-review``: an explicit /refine fork, which the unattended
    delete gate leaves alone, so the ordinary approval gate is the one that would stage its patch/remove."""
    if kind == "foreground":
        yield
        return
    from tools.skill_provenance import (
        reset_current_write_origin, reset_review_attended, set_current_write_origin, set_review_attended)
    origin, attend = set_current_write_origin("background_review"), set_review_attended(True)
    try:
        yield
    finally:
        reset_review_attended(attend)
        reset_current_write_origin(origin)


def _approve(store, pending_id):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools.write_approval import MEMORY
    return handle_pending_subcommand(MEMORY, ["approve", pending_id], memory_store=store)


GATED_CALLS = [
    pytest.param(dict(action="add", content="Owner approved the rollback"), id="single-add"),
    pytest.param(dict(action="patch", old_text="Seeded entry", content="Reviewed entry"), id="single-patch"),
    pytest.param(dict(action="remove", old_text="Seeded entry"), id="single-remove"),
    pytest.param(dict(operations=[{"action": "add", "content": "Owner approved the rollback"},
                                  {"action": "patch", "old_text": "Seeded entry", "new_text": "Reviewed entry"}]),
                 id="batch"),
]
ORIGINS = ["foreground", "attended-review"]


@pytest.mark.parametrize("origin", ORIGINS)
@pytest.mark.parametrize("call", GATED_CALLS)
def test_bot_turn_is_refused_before_the_write_approval_gate_stages(approval_on, tmp_path, origin, call):
    store = approval_on
    store.add("memory", SEED)
    before = _disk(tmp_path)
    with _origin(origin), _bound(BOT):
        result = _call(store, target="memory", **call)
    _assert_refused(result)
    assert _pending() == []
    assert _disk(tmp_path) == before
    assert store._entries_for("memory") == [SEED]


@pytest.mark.parametrize("author", [HUMAN, None], ids=["human", "no-author"])
@pytest.mark.parametrize("origin", ORIGINS)
@pytest.mark.parametrize("call", GATED_CALLS)
def test_human_turn_still_stages_through_the_write_approval_gate(approval_on, tmp_path, origin, call, author):
    store = approval_on
    store.add("memory", SEED)
    before = _disk(tmp_path)
    with _origin(origin), _bound(author):
        result = _call(store, target="memory", **call)
    assert result["success"] is True and result["staged"] is True, result
    (record,) = _pending()
    assert record["id"] == result["pending_id"]
    assert record["payload"]["action"] == ("batch" if "operations" in call else call["action"])
    assert record["payload"]["target"] == "memory"
    assert (record["origin"] == "background_review") is (origin == "attended-review")
    assert _disk(tmp_path) == before  # staged, not yet applied
    assert store._entries_for("memory") == [SEED]

    out = _approve(store, record["id"])
    assert out.startswith("Approved 1 memory write(s)."), out
    assert "Failed" not in out
    assert _pending() == []
    assert _disk(tmp_path) != before


@pytest.mark.parametrize("author, prompted", [(BOT, False), (HUMAN, True), (None, True)],
                         ids=["bot", "human", "no-author"])
def test_bot_turn_never_reaches_the_owners_inline_approval_prompt(approval_on, tmp_path, author, prompted):
    """On an interactive surface the gate prompts the owner inline. A bot's write must not even ask: an
    approval the owner gives there would be granted to the bot's report."""
    from tools.terminal_tool import set_approval_callback
    store = approval_on
    prompts = []
    set_approval_callback(lambda command, description, **kw: prompts.append(command) or "once")
    with _bound(author):
        result = _call(store, action="add", target="user", content="Owner approved the rollback")
    assert len(prompts) == (1 if prompted else 0)
    if prompted:
        assert result["success"] is True and not result.get("staged"), result
        assert "Owner approved the rollback" in _disk(tmp_path)["USER.md"]
    else:
        _assert_refused(result)
        assert _disk(tmp_path)["USER.md"] is None
    assert _pending() == []


@pytest.mark.parametrize("replay_on", ["same-thread", "worker-with-copied-context"])
def test_human_approval_staged_before_a_bot_turn_replays_after_it(approval_on, tmp_path, replay_on):
    """The human's staged write survives a bot's turn untouched and is approved through the real
    ``/memory approve`` replay afterwards. The bot's binding ended with its turn: nothing of it is left on the
    thread (or in a context copied from it) to refuse the human's next write."""
    from agent.turn_author import current_turn_author
    from tools.thread_context import propagate_context_to_thread
    store = approval_on
    store.add("memory", SEED)
    with _bound(HUMAN):
        staged = _call(store, target="memory", operations=[
            {"action": "add", "content": "Owner approved the rollback"},
            {"action": "patch", "old_text": "Seeded entry", "new_text": "Owner-confirmed entry"}])
    assert staged["staged"] is True, staged

    with _bound(BOT):  # the bot's turn
        _assert_refused(_call(store, target="memory", operations=[{"action": "add", "content": "Bot fact"}]))
        _assert_refused(_call(store, action="remove", target="memory", old_text="Seeded"))
    assert current_turn_author() is None
    assert [record["id"] for record in _pending()] == [staged["pending_id"]]

    if replay_on == "same-thread":
        out = _approve(store, staged["pending_id"])
    else:
        with ThreadPoolExecutor(max_workers=1) as pool:
            out = pool.submit(propagate_context_to_thread(lambda: _approve(store, staged["pending_id"]))).result(10)
    assert out.startswith("Approved 1 memory write(s)."), out
    assert "Failed" not in out
    assert store._entries_for("memory") == ["Owner-confirmed entry from the owner", "Owner approved the rollback"]
    persisted = _disk(tmp_path)["MEMORY.md"]
    assert "Owner-confirmed entry from the owner" in persisted and "Bot fact" not in persisted
    assert _pending() == []

    # The human's next write, bound or not, still goes to the gate (and stages) rather than being refused.
    with _bound(HUMAN):
        assert _call(store, action="add", target="user", content="Proceed")["staged"] is True
    assert _call(store, action="add", target="user", content="Stop")["staged"] is True
    assert len(_pending()) == 2


def test_human_approval_replays_after_a_real_bot_authored_agent_turn(approval_on, tmp_path, monkeypatch):
    """The same, with the bot's turn run by a real AIAgent (``run_conversation(turn_author=bot)``, stubbed
    endpoint) whose memory call is refused before the approval gate stages it."""
    from agent.turn_author import current_turn_author
    from tests.agent.test_turn_author_memory_guard import _live_agent
    store = approval_on
    staged = _call(store, action="add", target="user", content="Owner approved the rollback")
    assert staged["staged"] is True, staged

    agent, _seen = _live_agent(monkeypatch, [
        {"target": "user", "operations": [{"action": "add", "content": "Bot says the owner approved"}]}])
    bot_turn = agent.run_conversation("report from the field bot", turn_author=dict(BOT))
    (bot_result,) = [json.loads(m["content"]) for m in bot_turn["messages"] if m.get("role") == "tool"]
    _assert_refused(bot_result)
    assert current_turn_author() is None
    assert [record["id"] for record in _pending()] == [staged["pending_id"]]

    out = _approve(store, staged["pending_id"])
    assert out.startswith("Approved 1 memory write(s)."), out
    persisted = _disk(tmp_path)["USER.md"]
    assert "Owner approved the rollback" in persisted and "Bot says" not in persisted
    assert _pending() == []


# ── A trusted producer's ID-less, name-less bot (a Slack workflow post) is a bot, not a legacy author ────────

IDLESS_BOTS = [
    pytest.param({"id": None, "name": None, "is_bot": True}, id="turn-runner-shape"),
    pytest.param({"is_bot": True}, id="bare-flag"),
    pytest.param({"id": "  ", "name": "\x00\t", "is_bot": "true"}, id="cleaned-empty-fields-string-flag"),
]


@pytest.mark.parametrize("author", IDLESS_BOTS)
@pytest.mark.parametrize("target", ["memory", "user"])
def test_idless_bot_turn_is_refused_before_the_store_and_keeps_the_inventory_read(store, tmp_path, author, target):
    store.add(target, SEED)
    before = _disk(tmp_path)
    with _bound(author):
        single = _call(store, action="add", target=target, content="Owner approved the rollback")
        batch = _call(store, target=target, operations=[{"action": "add", "content": "Bot fact"},
                                                        {"action": "remove", "old_text": "Seeded"}])
        inventory = _call(store, action="remove", target=target)
    _assert_refused(single)
    _assert_refused(batch)
    assert inventory["current_entries"] == [SEED]
    assert _disk(tmp_path) == before
    assert store._entries_for(target) == [SEED]
    assert _pending() == []


@pytest.mark.parametrize("author", [
    pytest.param({"is_bot": False}, id="flag-false"),
    pytest.param({"id": None, "name": None, "is_bot": "maybe"}, id="invalid-flag"),
    pytest.param({"id": "", "name": " ", "is_bot": 0}, id="cleaned-empty-fields-zero-flag"),
])
def test_idless_author_without_a_true_bot_flag_stays_legacy_and_writes(store, tmp_path, author):
    with _bound(author):
        assert _call(store, action="add", target="user", content="Proceed")["success"] is True
    assert "Proceed" in _disk(tmp_path)["USER.md"]


def test_idless_bot_binding_resets_after_nesting_and_follows_its_worker(store, tmp_path):
    import contextvars
    with _bound(HUMAN):
        with _bound({"is_bot": True}):
            worker = contextvars.copy_context()
            _assert_refused(_call(store, action="add", target="user", content="Inner bot fact"))
        assert _call(store, action="add", target="user", content="Outer human fact")["success"] is True
    # A tool worker copied from the bot's turn keeps that turn's author after the turn has moved on.
    _assert_refused(worker.run(_call, store, action="add", target="user", content="Late bot fact"))
    persisted = _disk(tmp_path)["USER.md"]
    assert "Outer human fact" in persisted and "bot fact" not in persisted.lower()
