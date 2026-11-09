"""Who wrote the user side of a turn, carried from the dispatcher into the recipient's turn.

The dispatcher sets ``HERMES_TURN_AUTHOR`` on the recipient's one-shot subprocess only. A cached
gateway agent sees several authors over its lifetime, so the author is read per turn, never per agent.
"""

from __future__ import annotations

import contextvars
import json
import os
import socket
import unicodedata
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Optional

TURN_AUTHOR_ENV = "HERMES_TURN_AUTHOR"

# The author of the turn running in this context, as a read-only private copy; unset outside any turn. Tool
# workers and the review thread run in copies of the turn's context, so one that outlives its turn keeps that
# turn's author while the agent's own ``_turn_author`` moves on to the next one.
_NO_TURN = object()
_bound_author: contextvars.ContextVar[Any] = contextvars.ContextVar("turn_author", default=_NO_TURN)

_MAX_FIELD_LEN = 200
_MAX_CHAIN_HOPS = 1000


_DROPPED_CATEGORIES = frozenset({"Cc", "Cs", "Cn", "Co"})
_TRUTHY = frozenset({"true", "1", "yes"})


def _clean_text(value: Any) -> Optional[str]:
    """Strip whitespace, control and unassigned characters, then cap the length. None when nothing is left.
    Format characters and non-breaking spaces stay so emoji sequences and display names survive."""
    if not isinstance(value, str):
        return None
    text = "".join(ch for ch in value if unicodedata.category(ch) not in _DROPPED_CATEGORIES).strip()
    if not text:
        return None
    return text[:_MAX_FIELD_LEN]


def _bot_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY
    return isinstance(value, (bool, int)) and bool(value)


def bot_author_id(profile: str, origin: Optional[str] = None) -> str:
    """``bot:<profile>`` for a bot on the recipient's own install, ``bot:<origin>/<profile>`` for one on another
    install. The origin is the Desktop's connection id on a relayed dm and the sender's hostname on a peer dm."""
    origin = (origin or "").strip()
    return f"bot:{origin}/{profile}" if origin else f"bot:{profile}"


def local_origin() -> str:
    """This machine's hostname as an author-id origin, cleaned like any author field. Empty when unknown."""
    try:
        host = socket.gethostname()
    except Exception:
        return ""
    # A slash would split the id into a different origin and profile at parse time.
    return (_clean_text(host) or "").replace("/", "")


def parse_turn_author(raw: Any) -> Optional[dict[str, Any]]:
    """Normalize a dict or JSON string into ``{"id", "name", "is_bot"}``; None for anything else, or for a non-bot
    without id and name (unknown, legacy). A bot without either stays a bot with no id: its producer knows that
    much (a Slack workflow post carries neither), and no identity is made up for it.
    The id is whatever the transport knows the sender by: ``bot:<profile>`` on a bot-mode delivery inside one
    install, ``bot:<connection>/<profile>`` when the Desktop relayed it from another machine,
    ``bot:<hostname>/<profile>`` on a peer dm, the platform user id elsewhere.
    An ``origin`` field qualifies a bare ``bot:<profile>`` id the same way."""
    try:
        if isinstance(raw, (str, bytes)):
            raw = json.loads(raw)
        if not isinstance(raw, Mapping):
            return None
        author = {
            "id": _clean_text(raw.get("id")),
            "name": _clean_text(raw.get("name")),
            "is_bot": _bot_flag(raw.get("is_bot")),
        }
        if author["id"] is None and author["name"] is None and not author["is_bot"]:
            return None
        # Runtime-written chain fields of a bot-to-bot delivery (agent/bot_chain.py); never model input.
        hops = raw.get("hops")
        if isinstance(hops, int) and not isinstance(hops, bool) and 0 < hops <= _MAX_CHAIN_HOPS:
            author["hops"] = hops
        origin = _clean_text(raw.get("origin"))
        if origin and author["id"] and author["id"].startswith("bot:") and "/" not in author["id"]:
            author["id"] = _clean_text(bot_author_id(author["id"][len("bot:"):], origin))
        return author
    except Exception:
        return None


@contextmanager
def bound_turn_author(raw: Any) -> Iterator[None]:
    """Bind the normalized author for one turn, then restore the enclosing binding, so a child turn run on
    the same thread cannot leave its author (or its absence) behind in the parent's turn."""
    author = parse_turn_author(raw)
    token = _bound_author.set(None if author is None else MappingProxyType(author))
    try:
        yield
    finally:
        _bound_author.reset(token)


def current_turn_author(fallback: Any = None) -> Optional[dict[str, Any]]:
    """A fresh copy of the author bound for the turn running in this context. Outside any turn, ``fallback``
    normalized: the caller's best record of who wrote what it is acting on."""
    author = _bound_author.get()
    if author is _NO_TURN:
        return parse_turn_author(fallback)
    return None if author is None else dict(author)


def turn_author_from_env(environ: Mapping[str, str] = os.environ) -> Optional[dict[str, Any]]:
    """The author the dispatcher placed in ``HERMES_TURN_AUTHOR``, or None."""
    return parse_turn_author(environ.get(TURN_AUTHOR_ENV))


def take_turn_author_from_env(environ: MutableMapping[str, str] = os.environ) -> Optional[dict[str, Any]]:
    """Read and remove ``HERMES_TURN_AUTHOR`` so subprocesses started during the turn do not inherit it."""
    return parse_turn_author(environ.pop(TURN_AUTHOR_ENV, None))


def turn_author_env(author: dict[str, Any]) -> dict[str, str]:
    """The environment entry a dispatcher merges into a child's env."""
    return {TURN_AUTHOR_ENV: json.dumps(author, separators=(",", ":"))}


def a2a_key(author: Optional[dict[str, Any]]) -> Optional[str]:
    """``a2a:<bot id>``, the shared name for a bot author's turns. None for a human or an id-less bot."""
    if not isinstance(author, dict) or not author.get("is_bot") or not author.get("id"):
        return None
    return f"a2a:{author['id']}"
