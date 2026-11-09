"""Tests for agent.turn_author: the per-turn author carried from a dispatcher to memory hooks."""


import pytest

from agent.turn_author import (
    TURN_AUTHOR_ENV,
    parse_turn_author,
    take_turn_author_from_env,
    turn_author_env,
    turn_author_from_env,
)

FAMILY = "\U0001F468\u200D\U0001F469\u200D\U0001F467"


class TestParseTurnAuthor:

    @pytest.mark.parametrize("raw", [
        None, 42, [], ["bot:alpha"], "not json", '"a string"', "[1, 2]", b"\xff",
        {}, {"id": "", "name": "   "},
        # Without id and name only a true bot flag is information; anything else stays unknown (legacy).
        {"is_bot": False}, {"is_bot": None}, {"is_bot": 0}, {"is_bot": "maybe"}, {"is_bot": "false"},
        {"is_bot": 1.0}, {"is_bot": [True]}, {"id": "\x00", "name": "\t", "is_bot": "no"},
    ])
    def test_junk_and_authors_without_id_or_name_return_none(self, raw):
        assert parse_turn_author(raw) is None

    @pytest.mark.parametrize("raw", [
        {"is_bot": True}, {"id": None, "name": None, "is_bot": True}, {"is_bot": "true"}, {"is_bot": " YES "},
        {"is_bot": 1}, {"id": "", "name": "   ", "is_bot": True}, {"id": "\x00\x07", "name": "\n\t", "is_bot": True},
        '{"id": null, "name": null, "is_bot": true}',
    ])
    def test_a_bot_flag_without_id_or_name_is_an_unnamed_bot_not_unknown(self, raw):
        """A trusted producer that knows the sender is a bot (a Slack workflow post carries no user id or
        name) is not unknown: dropping it to None would read as legacy and let the bot's turn write memory."""
        assert parse_turn_author(raw) == {"id": None, "name": None, "is_bot": True}

    def test_an_unnamed_bot_gets_no_fabricated_identity(self):
        from agent.turn_author import a2a_key
        author = parse_turn_author({"is_bot": True, "origin": "peer-host", "hops": 2})
        assert author == {"id": None, "name": None, "is_bot": True, "hops": 2}
        assert a2a_key(author) is None


class TestEnvCarrier:
    def test_round_trip_through_env(self):
        author = {"id": "bot:alpha", "name": "Alpha", "is_bot": True}
        env = turn_author_env(author)
        assert set(env) == {TURN_AUTHOR_ENV}
        assert turn_author_from_env(env) == author


    def test_take_removes_the_variable(self):
        author = {"id": "bot:alpha", "name": "Alpha", "is_bot": True}
        env = dict(turn_author_env(author), OTHER="kept")
        assert take_turn_author_from_env(env) == author
        assert env == {"OTHER": "kept"}
        assert take_turn_author_from_env(env) is None
