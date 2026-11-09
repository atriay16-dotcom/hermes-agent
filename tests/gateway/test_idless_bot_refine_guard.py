"""A trusted ID-less, name-less bot sender issuing ``/refine`` runs the review fork as a bot, not as legacy.

A Slack workflow post reaches the gateway with the bot flag set and no user id or user name. Both trusted
producers (the turn runner and ``command_turn_author``) hand the author parser exactly that shape; before the
fix it normalized to None, the "unknown / legacy" author, so the memory tool let the bot's turn write the owner's
memory. The cases drive the real ``_handle_refine_command`` cold route, the real spawn chain and a real fork,
reusing the route fixtures of ``test_refine_turn_author``; only the cold parent construction and the model
endpoint are stubbed.
"""
from __future__ import annotations

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_ingress import command_turn_author
from tests.gateway import test_refine_turn_author as _refine_fixtures
from tests.gateway.test_refine_turn_author import _admitted, _event, _refine
from tests.agent.test_background_review import _bare_agent
from tests.agent.test_background_review_turn_author import _FORK_CALLS, HUMAN, SEED

# Re-export the original pytest fixture objects without unused-import/shadow diagnostics.
cold = _refine_fixtures.cold
fork_runs = _refine_fixtures.fork_runs
real_store = _refine_fixtures.real_store

UNNAMED_BOT = {"id": None, "name": None, "is_bot": True}

EMPTY_FIELDS = [
    pytest.param(None, None, id="missing"),
    pytest.param("", "", id="empty"),
    pytest.param("  ", "\x00\t", id="cleaned-empty"),
]


def _slack_workflow_post(user_id, user_name, is_bot=True):
    return SessionSource(platform=Platform.SLACK, chat_id="C0WORKFLOW", user_id=user_id, user_name=user_name,
                         is_bot=is_bot)


@pytest.mark.parametrize("user_id, user_name", EMPTY_FIELDS)
def test_command_author_of_an_idless_slack_bot_is_an_unnamed_bot(user_id, user_name):
    with _admitted(None):
        assert command_turn_author(None, _slack_workflow_post(user_id, user_name)) == UNNAMED_BOT
        # The same transport facts without the bot flag stay unknown (legacy), never a fabricated human.
        assert command_turn_author(None, _slack_workflow_post(user_id, user_name, is_bot=False)) is None


def test_an_admitted_human_still_wins_over_an_idless_bot_flag():
    with _admitted(dict(HUMAN)):
        assert command_turn_author(None, _slack_workflow_post(None, None)) == HUMAN


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id, user_name", EMPTY_FIELDS)
async def test_cold_refine_runs_the_fork_as_the_unnamed_bot(fork_runs, cold, user_id, user_name):
    cold.parent = _bare_agent
    await _refine(_event(_slack_workflow_post(user_id, user_name)))
    (run,) = fork_runs
    assert run["turn_author"] == UNNAMED_BOT


@pytest.mark.asyncio
async def test_real_cold_fork_refuses_an_idless_bots_writes_before_any_store_or_staging(real_store, cold):
    cold.parent = real_store.make_agent
    await _refine(_event(_slack_workflow_post(None, None)))
    real_store.join_reviews()
    results = real_store.tool_results()
    assert set(results) == set(_FORK_CALLS)
    for result in results.values():
        assert result["success"] is False and not result.get("staged"), result
        assert "nothing was written" in result["error"].lower(), result
    (agent,) = real_store.agents
    assert (agent._memory_store._entries_for("memory"), agent._memory_store._entries_for("user")) == ([SEED], [])
    assert real_store.disk() == real_store.befores[0]
    assert real_store.pending() == []
    assert real_store.seen_authors[0] == UNNAMED_BOT


@pytest.mark.asyncio
async def test_real_cold_fork_of_an_idless_non_bot_sender_stays_legacy_and_applies(real_store, cold):
    cold.parent = real_store.make_agent
    await _refine(_event(_slack_workflow_post(None, None, is_bot=False)))
    real_store.join_reviews()
    results = real_store.tool_results()
    assert results["call-user-add"]["success"] is True, results
    assert "Owner approved the rollback" in real_store.disk()["USER.md"]
    assert real_store.seen_authors[0] is None
