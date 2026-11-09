"""Producer-owned memory turn source, forwarded through the existing memory sync path.

The prologue stamps a fresh, content-independent per-invocation marker plus the native turn id on the
actual admitted user row (``display_metadata["memory_turn_source"]``, a producer-only lane every wire
copy drops), and the finalizer forwards the same primitive pair through the existing mirror ->
``AIAgent._sync_external_memory_for_turn`` -> ``MemoryManager.sync_all`` -> signature-filtered
``provider.sync_turn(turn_source=...)``. Everything queued for the worker is a structural snapshot.

Only the model client is faked; the agent, prologue, finalizer, manager and redactor are real.
"""

from __future__ import annotations

import copy
import json
import threading
import uuid
from types import SimpleNamespace

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider

SOURCE_KEY = "memory_turn_source"


class _Provider(MemoryProvider):
    """Records what ``sync_turn`` was handed (a current provider that accepts ``turn_source``)."""

    def __init__(self, name: str = "recorder"):
        self._name = name
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return self._name

    def initialize(self, session_id: str = "", **kwargs) -> None:
        pass

    def is_available(self) -> bool:
        return True

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query, *, session_id: str = "") -> str:
        return ""

    def queue_prefetch(self, query, *, session_id: str = "") -> None:
        pass

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs) -> str:
        return ""

    def sync_turn(self, user_content, assistant_content, *, session_id: str = "", messages=None, turn_source=None):
        self.calls.append({"user": user_content, "assistant": assistant_content,
                           "messages": messages, "turn_source": turn_source})


class _LegacyProvider(_Provider):
    """Written before ``turn_source`` existed: the manager's signature filter must not pass it."""

    def sync_turn(self, user_content, assistant_content, *, session_id: str = "", messages=None):
        self.calls.append({"user": user_content, "assistant": assistant_content, "messages": messages})


class _BlockingProvider(_Provider):
    """Holds the worker inside ``sync_turn`` until released, then serializes what it was handed."""

    def __init__(self):
        super().__init__("blocking")
        self.release = threading.Event()
        self.seen = None

    def sync_turn(self, user_content, assistant_content, *, session_id: str = "", messages=None,
                  turn_author=None, turn_source=None):
        assert self.release.wait(10)
        self.seen = json.dumps({"messages": messages, "author": turn_author, "source": turn_source}, sort_keys=True)


def _agent(monkeypatch, provider, *, on_request=None):
    """A real ``AIAgent`` whose only fake is the model client; ``provider`` sits behind a real manager."""
    from run_agent import AIAgent

    wire: list[str] = []
    holder: dict = {}

    class _Completions:
        def create(self, **kw):
            wire.append(json.dumps(kw.get("messages"), default=str))
            if on_request is not None:
                on_request(holder["agent"])
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None, reasoning=None, reasoning_content=None),
                finish_reason="stop")], usage=None, model="test-model")

    monkeypatch.setattr("agent.process_bootstrap.OpenAI",
                        lambda **_kw: SimpleNamespace(chat=SimpleNamespace(completions=_Completions())))
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *a, **k: [])
    agent = AIAgent(model="test-model", api_key="k", base_url="http://localhost:1/v1", platform="cli",
                    max_iterations=2, quiet_mode=True, skip_memory=True)
    agent._disable_streaming = True
    manager = MemoryManager()
    manager.add_provider(provider)
    agent._memory_manager = manager
    holder["agent"] = agent
    return agent, manager, wire


def _run(agent, manager, text, **kwargs):
    result = agent.run_conversation(text, **kwargs)
    assert manager.flush_pending(timeout=10)
    return result


def _marked(rows, source):
    return [r for r in rows if isinstance(r, dict) and isinstance(r.get("display_metadata"), dict)
            and r["display_metadata"].get(SOURCE_KEY) == source]


def _is_source(value) -> bool:
    return (isinstance(value, dict) and set(value) == {"turn_id", "marker"}
            and all(isinstance(v, str) and v for v in value.values()))


def test_identical_inputs_get_distinct_markers_forwarded_with_the_admitted_row(monkeypatch):
    provider = _Provider()
    agent, manager, wire = _agent(monkeypatch, provider)
    first = _run(agent, manager, "same words")
    second = _run(agent, manager, "same words", conversation_history=first["messages"])

    sources = [call["turn_source"] for call in provider.calls]
    assert len(sources) == 2 and all(_is_source(s) for s in sources), sources
    assert sources[0]["marker"] != sources[1]["marker"]
    assert sources[1]["turn_id"] == agent._current_turn_id
    for call, source in zip(provider.calls, sources):
        (row,) = _marked(call["messages"], source)
        assert (row["role"], row["content"]) == ("user", "same words")
    # The historical row keeps its own marker; nothing re-tags history with the new invocation's source.
    users = [r for r in second["messages"] if r.get("role") == "user"]
    assert [r["display_metadata"][SOURCE_KEY] for r in users] == sources
    # Producer-only: no request the model received carries the lane or the marker.
    assert wire and not any(SOURCE_KEY in body or s["marker"] in body for body in wire for s in sources)


def test_reused_native_turn_id_and_message_uid_still_mint_a_fresh_marker(monkeypatch):
    """A retry can reuse the native turn id (the facade's relay id is pinned here) and the CLI handoff's
    logical ``message_uid``; neither is the execution marker, so each invocation still gets its own."""
    provider = _Provider()
    agent, manager, _ = _agent(monkeypatch, provider)
    monkeypatch.setattr("agent.turn_facade.uuid", SimpleNamespace(uuid4=lambda: uuid.UUID(int=7)))
    rows = []
    for _ in range(2):
        rows.append({"role": "user", "content": "retry me", "message_uid": "uid-reused"})
        agent._pending_cli_user_message = rows[-1]
        _run(agent, manager, "retry me", task_id="task-1")
    first, second = (call["turn_source"] for call in provider.calls)
    assert _is_source(first) and _is_source(second)
    assert first["turn_id"] == second["turn_id"] and first["marker"] != second["marker"]
    assert [r["message_uid"] for r in rows] == ["uid-reused", "uid-reused"]
    assert [r["display_metadata"][SOURCE_KEY] for r in rows] == [first, second]


def test_pending_cli_handoff_is_tagged_in_place_without_touching_shared_metadata(monkeypatch):
    provider = _Provider()
    agent, manager, _ = _agent(monkeypatch, provider)
    shared = {"attempt": 1}
    old_row = {"role": "user", "content": "earlier", "display_metadata": shared}
    pending = {"role": "user", "content": "staged words", "display_metadata": shared}
    agent._pending_cli_user_message = pending
    result = _run(agent, manager, "staged words",
                  conversation_history=[old_row, {"role": "assistant", "content": "earlier reply"}])

    (current,) = [r for r in result["messages"] if r is pending]  # the handoff dict itself, never a clone
    assert shared == {"attempt": 1} and old_row["display_metadata"] is shared
    source = provider.calls[0]["turn_source"]
    assert _is_source(source)
    assert current["display_metadata"] is not shared
    assert current["display_metadata"] == {"attempt": 1, SOURCE_KEY: source}


@pytest.mark.parametrize("consumer", ["no_manager", "empty_manager", "legacy_only_provider"])
def test_without_a_source_aware_consumer_rows_and_metadata_are_unchanged(monkeypatch, consumer):
    """Nobody consumes the source: no marker, no stash, typed/plain rows and a CLI handoff exactly as before."""
    legacy = _LegacyProvider("legacy")
    manager = None if consumer == "no_manager" else MemoryManager()
    if consumer == "legacy_only_provider":
        manager.add_provider(legacy)
    agent, _, _ = _agent(monkeypatch, _Provider())
    agent._memory_manager = manager
    supplied = {"attempt": 2}
    typed = agent.run_conversation("continue please", persist_user_display_kind="auto_continue",
                                   persist_user_display_metadata=supplied)
    pending = {"role": "user", "content": "plain words"}
    agent._pending_cli_user_message = pending
    plain = agent.run_conversation("plain words", conversation_history=typed["messages"])
    if manager is not None:
        assert manager.flush_pending(timeout=10)

    (typed_row,) = [r for r in typed["messages"] if r.get("role") == "user"]
    assert typed_row.get("display_kind") == "auto_continue"
    assert typed_row.get("display_metadata") == {"attempt": 2} and supplied == {"attempt": 2}
    (current,) = [r for r in plain["messages"] if r is pending]  # the handoff dict itself
    assert "display_metadata" not in current
    assert getattr(agent, "_memory_turn_source", None) is None
    if consumer == "legacy_only_provider":
        assert [c["user"] for c in legacy.calls] == ["continue please", "plain words"]  # still synced


def test_source_aware_consumer_keeps_supplied_metadata_and_adds_only_its_marker(monkeypatch):
    provider = _Provider()
    agent, manager, _ = _agent(monkeypatch, provider)
    supplied = {"attempt": 2}
    result = _run(agent, manager, "continue please", persist_user_display_kind="auto_continue",
                  persist_user_display_metadata=supplied)
    (row,) = [r for r in result["messages"] if r.get("role") == "user"]
    (call,) = provider.calls
    assert _is_source(call["turn_source"]) and row["display_kind"] == "auto_continue"
    assert row["display_metadata"] == {"attempt": 2, SOURCE_KEY: call["turn_source"]}
    assert supplied == {"attempt": 2}  # the caller's container is never the one stamped


def test_staged_row_aliasing_history_is_not_tagged_and_forwards_no_source(monkeypatch):
    provider = _Provider()
    agent, manager, _ = _agent(monkeypatch, provider)
    history_row = {"role": "user", "content": "same words", "display_metadata": {"attempt": 1}}
    agent._pending_cli_user_message = history_row  # the staged dict IS a historical row
    _run(agent, manager, "same words",
         conversation_history=[history_row, {"role": "assistant", "content": "earlier reply"}])

    assert history_row["display_metadata"] == {"attempt": 1}
    (call,) = provider.calls
    assert call["turn_source"] is None and (call["user"], call["assistant"]) == ("same words", "ok")


def test_finalizer_forwards_only_the_source_bound_to_its_own_turn_id(monkeypatch):
    def _foreign_turn_overwrites_the_stash(agent):
        agent._memory_turn_source = {"turn_id": "another-turn", "marker": "f" * 32}

    provider = _Provider()
    agent, manager, _ = _agent(monkeypatch, provider, on_request=_foreign_turn_overwrites_the_stash)
    _run(agent, manager, "hello there")
    (call,) = provider.calls
    assert call["turn_source"] is None and call["user"] == "hello there"


def _transcript():
    return [{"role": "user", "content": "decide", "display_metadata": {SOURCE_KEY: {"turn_id": "t1", "marker": "m1"}}},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "clarify", "arguments": "{\"q\": 1}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "answer A"},
            {"role": "assistant", "content": "done"}]


def _mutate(messages, author, source=None):
    messages.append({"role": "user", "content": "a later turn"})
    messages[2]["content"] = "HIJACKED"
    messages[1]["tool_calls"][0]["function"]["arguments"] = "{}"
    messages[0]["display_metadata"][SOURCE_KEY]["marker"] = "forged"
    author["is_bot"] = True
    if source is not None:
        source["marker"] = "forged"


def test_queued_sync_transcript_and_author_are_frozen_before_the_worker():
    provider = _BlockingProvider()
    manager = MemoryManager()
    manager.add_provider(provider)
    messages, author = _transcript(), {"id": "u1", "name": "Ann", "is_bot": False}
    expected = json.dumps({"messages": copy.deepcopy(messages), "author": dict(author), "source": None}, sort_keys=True)

    manager.sync_all("decide", "done", session_id="s", messages=messages, turn_author=author)
    _mutate(messages, author)
    provider.release.set()
    assert manager.flush_pending(timeout=10)
    assert provider.seen == expected


def test_queued_turn_source_is_frozen_before_the_worker():
    provider = _BlockingProvider()
    manager = MemoryManager()
    manager.add_provider(provider)
    messages, author = _transcript(), {"id": "u1", "name": "Ann", "is_bot": False}
    source = {"turn_id": "t1", "marker": "m1"}
    expected = json.dumps({"messages": copy.deepcopy(messages), "author": dict(author), "source": dict(source)},
                          sort_keys=True)
    try:
        manager.sync_all("decide", "done", session_id="s", messages=messages, turn_author=author, turn_source=source)
    except TypeError as exc:
        pytest.fail(f"sync_all rejected the optional turn_source keyword: {exc}")
    _mutate(messages, author, source)
    provider.release.set()
    assert manager.flush_pending(timeout=10)
    assert provider.seen == expected


@pytest.mark.parametrize("supplied,forwarded", [
    ({"turn_id": "t1", "marker": "m1"}, {"turn_id": "t1", "marker": "m1"}),
    ({"turn_id": "t1", "marker": 7}, None),
    ({"turn_id": "", "marker": "m1"}, None),
    ("m1", None),
    ({"turn_id": "t1", "marker": "m1", "extra": "x"}, {"turn_id": "t1", "marker": "m1"}),
])
def test_only_a_primitive_source_reaches_an_accepting_provider(supplied, forwarded):
    provider = _Provider()
    manager = MemoryManager()
    manager.add_provider(provider)
    try:
        manager.sync_all("hi", "ok", session_id="s", messages=_transcript(), turn_source=supplied)
    except TypeError as exc:
        pytest.fail(f"sync_all rejected the optional turn_source keyword: {exc}")
    assert manager.flush_pending(timeout=10)
    (call,) = provider.calls
    assert call["turn_source"] == forwarded


def test_legacy_provider_signature_is_filtered_and_still_synced():
    legacy = _LegacyProvider("legacy")
    manager = MemoryManager()
    manager.add_provider(legacy)
    try:
        manager.sync_all("hi", "ok", session_id="s", messages=_transcript(), turn_source={"turn_id": "t", "marker": "m"})
    except TypeError as exc:
        pytest.fail(f"sync_all rejected the optional turn_source keyword: {exc}")
    assert manager.flush_pending(timeout=10)
    (call,) = legacy.calls
    assert (call["user"], call["assistant"]) == ("hi", "ok") and call["messages"] == _transcript()


# -- Composition with the upgraded core: fork suppression, author guard forwarding, declared-keyword opt-in --


@pytest.mark.parametrize("agent_context", ["primary", "subagent"])
def test_forked_agent_context_is_not_a_source_consumer(monkeypatch, agent_context):
    """A fork ("subagent") manager holds its provider only for the checkpoint gate and never writes external
    memory, so nothing would ever read a source there: its rows stay unstamped and nothing is stashed. The
    primary control proves the same wiring stamps and forwards when a real consumer exists."""
    provider = _Provider()
    agent, manager, _ = _agent(monkeypatch, provider)
    manager.initialize_all(session_id="s", agent_context=agent_context)
    result = _run(agent, manager, "fork words")

    (row,) = [r for r in result["messages"] if r.get("role") == "user"]
    if agent_context == "subagent":
        assert provider.calls == []  # upstream fork write suppression still holds
        assert "display_metadata" not in row
        assert getattr(agent, "_memory_turn_source", None) is None
        assert manager.accepts_turn_source() is False
    else:
        (call,) = provider.calls
        assert _is_source(call["turn_source"]) and call["user"] == "fork words"
        assert row["display_metadata"] == {SOURCE_KEY: call["turn_source"]}


class _AuthorSourceProvider(_Provider):
    def sync_turn(self, user_content, assistant_content, *, session_id: str = "", messages=None,
                  turn_author=None, turn_source=None):
        self.calls.append({"user": user_content, "author": turn_author, "turn_source": turn_source,
                           "messages": messages})


class _AuthorOnlyProvider(_Provider):
    """An existing guard-era provider: declares ``turn_author`` but not ``turn_source``."""

    def sync_turn(self, user_content, assistant_content, *, session_id: str = "", messages=None, turn_author=None):
        self.calls.append({"user": user_content, "author": turn_author, "messages": messages})


@pytest.mark.parametrize("provider_cls", [_AuthorSourceProvider, _AuthorOnlyProvider])
def test_turn_author_and_turn_source_reach_sync_together_per_turn(monkeypatch, provider_cls):
    """The finalizer's sync carries this turn's author (the existing guard forwarding) AND its own source; the
    next human turn (no author) gets neither the earlier author nor the earlier source."""
    provider = provider_cls()
    agent, manager, _ = _agent(monkeypatch, provider)
    first = _run(agent, manager, "from ann", turn_author={"id": "U123", "name": "Ann", "is_bot": False})
    second = _run(agent, manager, "no author", conversation_history=first["messages"])

    authored, plain = provider.calls
    assert authored["author"] == {"id": "U123", "name": "Ann", "is_bot": False}
    assert plain["author"] is None and plain["user"] == "no author"
    if provider_cls is _AuthorOnlyProvider:
        assert "turn_source" not in authored and "turn_source" not in plain
        assert all("display_metadata" not in r for r in second["messages"] if r.get("role") == "user")
        return
    assert _is_source(authored["turn_source"]) and _is_source(plain["turn_source"])
    assert authored["turn_source"]["marker"] != plain["turn_source"]["marker"]
    (row,) = _marked(authored["messages"], authored["turn_source"])
    assert (row["role"], row["content"]) == ("user", "from ann")


class _CatchAllProvider(_Provider):
    """A signature proxy / catch-all: forwards anything, declares nothing."""

    def sync_turn(self, user_content, assistant_content, **kwargs):
        self.calls.append({"user": user_content, "kwargs": dict(kwargs)})


class _UninspectableProvider(_CatchAllProvider):
    def sync_turn(self, user_content, assistant_content, **kwargs):
        super().sync_turn(user_content, assistant_content, **kwargs)

    sync_turn.__signature__ = "not a signature"  # inspect.signature raises TypeError: unreadable


@pytest.mark.parametrize("provider_cls", [_CatchAllProvider, _UninspectableProvider])
def test_catch_all_or_uninspectable_sync_turn_is_not_a_declared_source_consumer(monkeypatch, provider_cls):
    """``**kwargs``-only and unreadable signatures still receive the turn (legacy acceptance) but never opt the
    producer into stamping: only an explicitly declared ``turn_source`` keyword does."""
    provider = provider_cls()
    agent, manager, _ = _agent(monkeypatch, provider)
    pending = {"role": "user", "content": "catch all"}
    agent._pending_cli_user_message = pending
    result = _run(agent, manager, "catch all")

    assert manager.accepts_turn_source() is False
    (call,) = provider.calls
    assert call["user"] == "catch all" and "turn_source" not in call["kwargs"]
    (current,) = [r for r in result["messages"] if r is pending]
    assert "display_metadata" not in current
    assert getattr(agent, "_memory_turn_source", None) is None


# -- Parameter kind: only a POSITIONAL_OR_KEYWORD or KEYWORD_ONLY ``turn_source`` declares consumption --


class _PositionalOrKeywordSourceProvider(_Provider):
    def sync_turn(self, user_content, assistant_content, session_id="", messages=None, turn_source=None):
        self.calls.append({"user": user_content, "messages": messages, "turn_source": turn_source})


class _DeclaredSourceAndCatchAllProvider(_Provider):
    def sync_turn(self, user_content, assistant_content, *, turn_source=None, **kwargs):
        self.calls.append({"user": user_content, "messages": kwargs.get("messages"), "turn_source": turn_source})


class _PositionalOnlySourceProvider(_Provider):
    """``turn_source`` binds only positionally: passing ``turn_source=...`` would raise TypeError."""

    def sync_turn(self, user_content, assistant_content, turn_source=None, /, *, session_id="", messages=None,
                  turn_author=None):
        self.calls.append({"user": user_content, "slot": turn_source,
                           "kwargs": {"session_id": session_id, "messages": messages, "turn_author": turn_author}})


class _VarPositionalSourceProvider(_Provider):
    def sync_turn(self, user_content, assistant_content, *turn_source, session_id="", messages=None,
                  turn_author=None):
        self.calls.append({"user": user_content, "slot": turn_source,
                           "kwargs": {"session_id": session_id, "messages": messages, "turn_author": turn_author}})


class _NamedVarKeywordSourceProvider(_Provider):
    """A catch-all that happens to be spelled ``**turn_source``: still a catch-all, not a declaration."""

    def sync_turn(self, user_content, assistant_content, *, session_id="", **turn_source):
        self.calls.append({"user": user_content, "kwargs": {"session_id": session_id, **turn_source}})


class _NonCallableSyncProvider(_Provider):
    sync_turn = None


_WRONG_KINDS = [_PositionalOnlySourceProvider, _VarPositionalSourceProvider, _NamedVarKeywordSourceProvider]


@pytest.mark.parametrize("provider_cls,declared", [
    (_Provider, True),
    (_PositionalOrKeywordSourceProvider, True),
    (_DeclaredSourceAndCatchAllProvider, True),
    (_PositionalOnlySourceProvider, False),
    (_VarPositionalSourceProvider, False),
    (_NamedVarKeywordSourceProvider, False),
    (_CatchAllProvider, False),
    (_UninspectableProvider, False),
    (_NonCallableSyncProvider, False),
])
def test_only_a_keyword_bindable_turn_source_parameter_is_a_declared_consumer(provider_cls, declared):
    manager = MemoryManager()
    manager.add_provider(provider_cls())
    assert manager.accepts_turn_source() is declared
    manager.initialize_all(session_id="s", agent_context="subagent")
    assert manager.accepts_turn_source() is False  # a fork reads no source, whatever the signature


@pytest.mark.parametrize("other_cls", [*_WRONG_KINDS, _CatchAllProvider, _UninspectableProvider])
def test_aware_provider_beside_an_undeclared_one_syncs_both_and_leaks_no_source(other_cls):
    """Mixed manager: the declared consumer gets the source; the other provider still gets the whole turn through
    the legacy filter (``messages``/``turn_author`` by name, catch-all or unreadable signature) but never a
    ``turn_source`` it did not declare. Both see the snapshot taken before the worker ran."""
    aware, other = _BlockingProvider(), other_cls("builtin")  # the manager admits one external plus builtin
    manager = MemoryManager()
    manager.add_provider(aware)
    manager.add_provider(other)
    assert manager.accepts_turn_source() is True
    messages, author = _transcript(), {"id": "u1", "name": "Ann", "is_bot": False}
    source = {"turn_id": "t1", "marker": "m1"}
    frozen_messages, frozen_author = copy.deepcopy(messages), dict(author)

    manager.sync_all("decide", "done", session_id="s", messages=messages, turn_author=author, turn_source=source)
    _mutate(messages, author, source)
    aware.release.set()
    assert manager.flush_pending(timeout=10)

    assert aware.seen == json.dumps({"messages": frozen_messages, "author": frozen_author,
                                     "source": {"turn_id": "t1", "marker": "m1"}}, sort_keys=True)
    (call,) = other.calls  # its completed-turn sync is not lost to a swallowed TypeError
    assert call["user"] == "decide"
    assert call["kwargs"] == {"session_id": "s", "messages": frozen_messages, "turn_author": frozen_author}
    assert call.get("slot") in (None, ())  # nothing reached a positional-only or *args slot either


@pytest.mark.parametrize("provider_cls", _WRONG_KINDS)
def test_wrong_kind_turn_source_alone_is_not_stamped_and_keeps_its_author_sync(monkeypatch, provider_cls):
    provider = provider_cls()
    agent, manager, _ = _agent(monkeypatch, provider)
    pending = {"role": "user", "content": "wrong kind"}
    agent._pending_cli_user_message = pending
    author = {"id": "U123", "name": "Ann", "is_bot": False}
    result = _run(agent, manager, "wrong kind", turn_author=author)

    assert manager.accepts_turn_source() is False
    (current,) = [r for r in result["messages"] if r is pending]
    assert "display_metadata" not in current
    assert getattr(agent, "_memory_turn_source", None) is None
    (call,) = provider.calls
    assert call["user"] == "wrong kind" and call["kwargs"]["turn_author"] == author
    assert "turn_source" not in call["kwargs"] and call.get("slot") in (None, ())


@pytest.mark.parametrize("provider_cls", [_PositionalOrKeywordSourceProvider, _DeclaredSourceAndCatchAllProvider])
def test_either_keyword_bindable_kind_is_stamped_and_forwarded_beside_a_legacy_provider(monkeypatch, provider_cls):
    provider, legacy = provider_cls(), _LegacyProvider("builtin")
    agent, manager, _ = _agent(monkeypatch, provider)
    manager.add_provider(legacy)
    result = _run(agent, manager, "declared words")

    (call,) = provider.calls
    assert _is_source(call["turn_source"]) and call["user"] == "declared words"
    (row,) = [r for r in result["messages"] if r.get("role") == "user"]
    assert row["display_metadata"] == {SOURCE_KEY: call["turn_source"]}
    (marked,) = _marked(call["messages"], call["turn_source"])
    assert (marked["role"], marked["content"]) == ("user", "declared words")
    (legacy_call,) = legacy.calls
    assert legacy_call["user"] == "declared words" and legacy_call["messages"] == call["messages"]
