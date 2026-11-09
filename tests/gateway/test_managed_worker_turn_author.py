"""A managed worker's turn carries its author so memory providers attribute it like the owner path."""
import json
import os
from types import SimpleNamespace

import pytest


def _policy(tmp_path):
    from gateway.session_local import _bypass_policy
    return _bypass_policy({'cwd': str(tmp_path), 'model': 'safe-fixture', 'provider': 'custom',
                           'base_url': 'http://127.0.0.1:9/v1', 'ignore_user_config': True}, private_secrets={})


def test_bootstrap_frame_carries_the_admitted_author(tmp_path, monkeypatch):
    from gateway import session_managed_worker as smw
    from agent.managed_worker import validate_bootstrap

    monkeypatch.setattr('gateway.session_policy.launch_key', lambda authority, policy: None)
    live = SimpleNamespace(source=SimpleNamespace(user_id='u', chat_id='c'), route='route')
    authority = SimpleNamespace(sessions={'sid': live}, runner=None, profile_id=str(tmp_path))
    ref = SimpleNamespace(session_id='sid')
    scope = {'profile_id': str(tmp_path), 'session_id': 'sid', 'execution_id': 'x', 'generation': 1,
             'pid': os.getpid(), 'birth': 0, 'secret': 's', 'epoch': 1}
    author = {'id': 'bot-7', 'name': 'Seven', 'is_bot': True}
    row = {'payload': {'text': 'hi', 'local_automation_v1': {'turn_author': author}}}
    frame = smw._bootstrap(authority, ref, row, _policy(tmp_path), scope)
    assert frame['turn_author'] == author
    wire = json.loads(json.dumps(frame))
    assert validate_bootstrap(wire) is wire
    plain = smw._bootstrap(authority, ref, {'payload': {'text': 'hi'}}, _policy(tmp_path), scope)
    assert plain['turn_author'] is None
    assert validate_bootstrap(json.loads(json.dumps(plain)))['turn_author'] is None
    # A bare true bot flag is no longer malformed: parse_turn_author keeps it as the unnamed bot
    # (a Slack workflow post carries neither id nor name), so it is asserted as accepted below.
    # Everything without a usable id, name or true bot flag stays invalid, a non-bot included.
    for bad in ('bot-7', {}, {'is_bot': False}, {'is_bot': 'maybe'}, {'id': None, 'name': None, 'is_bot': False}):
        with pytest.raises(ValueError):
            validate_bootstrap({**wire, 'turn_author': bad})


def test_bootstrap_accepts_the_idless_bot_as_a_restricted_bot(tmp_path, monkeypatch):
    """The owner forwards an id-less bot author verbatim and the worker accepts it: no identity is made up."""
    from gateway import session_managed_worker as smw
    from agent.managed_worker import validate_bootstrap
    from agent.turn_author import parse_turn_author

    monkeypatch.setattr('gateway.session_policy.launch_key', lambda authority, policy: None)
    live = SimpleNamespace(source=SimpleNamespace(user_id='u', chat_id='c'), route='route')
    authority = SimpleNamespace(sessions={'sid': live}, runner=None, profile_id=str(tmp_path))
    scope = {'profile_id': str(tmp_path), 'session_id': 'sid', 'execution_id': 'x', 'generation': 1,
             'pid': os.getpid(), 'birth': 0, 'secret': 's', 'epoch': 1}
    row = {'payload': {'text': 'hi', 'local_automation_v1': {'turn_author': {'is_bot': True}}}}
    frame = smw._bootstrap(authority, SimpleNamespace(session_id='sid'), row, _policy(tmp_path), scope)
    assert frame['turn_author'] == {'is_bot': True}
    wire = json.loads(json.dumps(frame))
    assert validate_bootstrap(wire) is wire
    assert wire['turn_author'] == {'is_bot': True}
    assert parse_turn_author(wire['turn_author']) == {'id': None, 'name': None, 'is_bot': True}


def test_idless_bot_worker_turn_cannot_write_memory_and_legacy_turn_can(tmp_path, monkeypatch):
    """The worker's real bound turn: run_worker_turns -> AIAgent.run_conversation(turn_author=...), with the
    live guard fixture of tests/agent/test_turn_author_memory_guard.py (real agent, stubbed completions)."""
    from agent.turn_author import current_turn_author
    from gateway.session_kanban import run_worker_turns
    from tests.agent.test_turn_author_memory_guard import _TURNS, _live_agent, _memory_results

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    memories = tmp_path / 'memories'
    agent, seen_authors = _live_agent(monkeypatch, [dict(t) for t in _TURNS], probe=current_turn_author)

    bot_turn = run_worker_turns(agent, {'policy': {'kanban_json': None}, 'text': 'workflow post',
                                        'turn_author': {'is_bot': True}}, [])
    (bot_result,) = _memory_results(bot_turn)
    assert bot_result['success'] is False, bot_result
    assert 'nothing was written' in bot_result['error'].lower()
    assert not (memories / 'USER.md').exists()

    legacy_turn = run_worker_turns(agent, {'policy': {'kanban_json': None}, 'text': 'remember the approval',
                                           'turn_author': None}, [])
    (legacy_result,) = _memory_results(legacy_turn)
    assert legacy_result['success'] is True, legacy_result
    persisted = (memories / 'USER.md').read_text(encoding='utf-8')
    assert 'Owner approved the rollback' in persisted
    assert 'Bot says' not in persisted
    unnamed_bot = {'id': None, 'name': None, 'is_bot': True}
    assert seen_authors == [unnamed_bot, unnamed_bot, None, None]


def test_worker_turn_passes_the_author_to_run_conversation():
    from gateway.session_kanban import run_worker_turns

    seen = []
    agent = SimpleNamespace(run_conversation=lambda text, **kw: seen.append((text, kw)) or {'final_response': 'ok'})
    author = {'id': 'alice', 'name': 'alice', 'is_bot': True}
    run_worker_turns(agent, {'policy': {'kanban_json': None}, 'text': 'hello', 'turn_author': author}, [])
    run_worker_turns(agent, {'policy': {'kanban_json': None}, 'text': 'again', 'turn_author': None}, [])
    assert seen == [('hello', {'conversation_history': [], 'turn_author': author}),
                    ('again', {'conversation_history': []})]
