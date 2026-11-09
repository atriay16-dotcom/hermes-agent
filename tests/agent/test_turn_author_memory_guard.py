"""The turn's author reaches the built-in memory tool through the real agent paths.

``AIAgent.run_conversation(turn_author=...)`` binds the normalized author for exactly that turn; the
memory tool refuses a bot-authored turn's mutations before the local store (or the external mirror) is
touched, and the next turn of the same cached agent is not affected. The live-loop agent is built with
``skip_memory=True``, the review fork's shape: no external provider is there to catch the write.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
from tools.memory_tool_store import MemoryStore

BOT = {"id": "bot:field-foreman", "name": "Field Foreman", "is_bot": True}


def _bound(author):
    from agent.turn_author import bound_turn_author
    return bound_turn_author(author)


class _Sink:
    name = "test-sink"

    def __init__(self):
        self.writes = []

    def get_tool_schemas(self):
        return []

    def on_memory_write(self, action, target, content, metadata=None):
        self.writes.append((action, target, content))


def _inline_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.memory_manager import MemoryManager
    sink, manager = _Sink(), MemoryManager()
    manager.add_provider(sink)
    agent = SimpleNamespace(_memory_store=MemoryStore(memory_char_limit=500, user_char_limit=300),
                            _memory_manager=manager, _build_memory_write_metadata=lambda **kw: kw)
    return agent, sink


def _inline_call(agent, **args):
    ctx = InlineToolContext(effective_task_id="task-1", tool_call_id="call-1")
    return json.loads(INLINE_TOOL_EXECUTORS["memory"](agent, args, ctx))


def test_inline_bot_turn_write_reaches_neither_store_nor_mirror(tmp_path, monkeypatch):
    agent, sink = _inline_agent(tmp_path, monkeypatch)
    with _bound(BOT):
        single = _inline_call(agent, action="add", target="user", content="Owner prefers rollbacks")
        batch = _inline_call(agent, target="memory", operations=[{"action": "add", "content": "Bot fact"}])
    assert single["success"] is False and batch["success"] is False
    assert sink.writes == []
    assert not (tmp_path / "memories" / "USER.md").exists()
    assert not (tmp_path / "memories" / "MEMORY.md").exists()


# What the turn runner builds for a native sender with the bot flag and no user id or name (a Slack workflow post).
IDLESS_BOT = {"id": None, "name": None, "is_bot": True}


def test_inline_idless_bot_turn_write_reaches_neither_store_nor_mirror(tmp_path, monkeypatch):
    agent, sink = _inline_agent(tmp_path, monkeypatch)
    with _bound(dict(IDLESS_BOT)):
        single = _inline_call(agent, action="add", target="user", content="Owner prefers rollbacks")
        batch = _inline_call(agent, target="memory", operations=[{"action": "add", "content": "Bot fact"}])
    assert single["success"] is False and batch["success"] is False
    assert sink.writes == []
    assert not (tmp_path / "memories" / "USER.md").exists()
    assert not (tmp_path / "memories" / "MEMORY.md").exists()


def test_tool_arguments_cannot_choose_the_author(tmp_path, monkeypatch):
    """Author-looking keys in the model's arguments are ignored in both directions."""
    agent, sink = _inline_agent(tmp_path, monkeypatch)
    spoof_human = {"turn_author": {"id": "1234", "name": "Aldo", "is_bot": False}, "is_bot": False,
                   "author": "Aldo"}
    with _bound(BOT):
        refused = _inline_call(agent, action="add", target="user", content="Bot fact", **spoof_human)
    assert refused["success"] is False
    assert sink.writes == []

    spoof_bot = {"turn_author": dict(BOT), "is_bot": True}
    written = _inline_call(agent, action="add", target="user", content="Owner fact", **spoof_bot)
    assert written["success"] is True
    assert sink.writes == [("add", "user", "Owner fact")]


def test_inline_human_turn_still_writes_and_mirrors(tmp_path, monkeypatch):
    agent, sink = _inline_agent(tmp_path, monkeypatch)
    with _bound({"id": "1234", "name": "Aldo", "is_bot": False}):
        assert _inline_call(agent, action="add", target="user", content="Proceed")["success"] is True
    assert sink.writes == [("add", "user", "Proceed")]
    assert "Proceed" in (tmp_path / "memories" / "USER.md").read_text(encoding="utf-8")


def _live_agent(monkeypatch, tool_args_by_turn, probe=lambda: None):
    """A real AIAgent whose completion stub emits one memory call per turn, then a final answer.
    ``probe`` runs inside every model call, i.e. on the turn's own thread mid-turn."""
    from run_agent import AIAgent

    seen_authors = []
    state = {"pending": None}

    def _message(content=None, tool_calls=None):
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=tool_calls, reasoning=None, reasoning_content=None),
            finish_reason="tool_calls" if tool_calls else "stop")], usage=None, model="test-model")

    class _Completions:
        def create(self, **_kw):
            seen_authors.append(probe())
            if state["pending"] is None:
                state["pending"] = tool_args_by_turn.pop(0)
                call = SimpleNamespace(id=f"call-{len(seen_authors)}", type="function",
                                       function=SimpleNamespace(name="memory",
                                                                arguments=json.dumps(state["pending"])))
                return _message(tool_calls=[call])
            state["pending"] = None
            return _message(content="done")

    monkeypatch.setattr("agent.process_bootstrap.OpenAI",
                        lambda **_kw: SimpleNamespace(chat=SimpleNamespace(completions=_Completions())))
    agent = AIAgent(model="test-model", api_key="k", base_url="http://localhost:1/v1", platform="cli",
                    max_iterations=4, quiet_mode=True, skip_memory=True, enabled_toolsets=["memory"])
    agent._disable_streaming = True
    assert agent._memory_store is not None and agent._memory_manager is None
    return agent, seen_authors


def _memory_results(result):
    return [json.loads(m["content"]) for m in result["messages"] if m.get("role") == "tool"]


_TURNS = [{"target": "user", "operations": [{"action": "add", "content": "Bot says the owner approved"}]},
          {"target": "user", "operations": [{"action": "add", "content": "Owner approved the rollback"}]}]


def test_live_bot_turn_cannot_write_and_next_human_turn_can(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    memories = tmp_path / "memories"
    agent, _seen = _live_agent(monkeypatch, [dict(t) for t in _TURNS])

    bot_turn = agent.run_conversation("report from the field bot", turn_author=dict(BOT))
    (bot_result,) = _memory_results(bot_turn)
    assert bot_result["success"] is False, bot_result
    assert "nothing was written" in bot_result["error"].lower()
    assert not (memories / "USER.md").exists()

    human_turn = agent.run_conversation("remember that I approved the rollback")
    (human_result,) = _memory_results(human_turn)
    assert human_result["success"] is True, human_result
    persisted = (memories / "USER.md").read_text(encoding="utf-8")
    assert "Owner approved the rollback" in persisted
    assert "Bot says" not in persisted


def test_live_turn_binds_its_author_only_for_its_own_duration(monkeypatch):
    from agent.turn_author import current_turn_author
    agent, seen_authors = _live_agent(monkeypatch, [dict(t) for t in _TURNS], probe=current_turn_author)

    agent.run_conversation("report from the field bot", turn_author=dict(BOT))
    assert current_turn_author() is None
    agent.run_conversation("remember that I approved the rollback")
    assert seen_authors == [BOT, BOT, None, None]


def test_live_idless_bot_turn_cannot_write_and_next_human_turn_can(tmp_path, monkeypatch):
    """The actual bound turn: ``run_conversation(turn_author=...)`` with the turn runner's ID-less bot shape."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    memories = tmp_path / "memories"
    agent, _seen = _live_agent(monkeypatch, [dict(t) for t in _TURNS])

    bot_turn = agent.run_conversation("workflow post", turn_author=dict(IDLESS_BOT))
    (bot_result,) = _memory_results(bot_turn)
    assert bot_result["success"] is False, bot_result
    assert "nothing was written" in bot_result["error"].lower()
    assert not (memories / "USER.md").exists()

    human_turn = agent.run_conversation("remember that I approved the rollback",
                                        turn_author={"id": "1234", "name": "Aldo", "is_bot": False})
    (human_result,) = _memory_results(human_turn)
    assert human_result["success"] is True, human_result
    persisted = (memories / "USER.md").read_text(encoding="utf-8")
    assert "Owner approved the rollback" in persisted
    assert "Bot says" not in persisted
