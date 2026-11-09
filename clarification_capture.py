"""Framed observations of this turn's submitted native ``clarify`` forms.

Core's ``MemoryManager.sync_all`` hands ``sync_turn`` the whole redacted transcript, not a turn
slice. This module picks out, purely and fail-closed, the answered questions of ``clarify`` calls
the current turn's assistant declared and whose one native result says ``submitted``, and renders
each as an explicitly framed observation for the Honcho write path. Nothing else of the transcript
is serialized, and the transcript is never retained.

Conservative limits, by design:

- The clarify callback and result carry no responder identity. A record is the assistant's
  observation of a tool submission: never the assistant's own assertion, never a user utterance,
  never authenticated owner approval. Every chunk says the responder is not verified.
- The current turn is anchored ONLY by core's producer-owned turn source (``sync_turn(turn_source=...)``
  plus the same pair on the admitted user row's ``display_metadata["memory_turn_source"]``). A missing,
  malformed, unmatched or duplicated source, one lost through compaction or a merge, or an old core that
  never supplies it means no records; text is only a verification of the marked row, never a search.
  New capture therefore needs the paired core producer; with an older core the two text writes stay as-is.
- Only an exact ``{"type": "function", "function": {"name": "clarify"}}`` call is a native declaration.
- Ids are bound with core's own bridge normalization; duplicate, colliding, orphan or out-of-order
  ids yield nothing rather than a guessed correlation.
- There is no seen-ledger: delivery keeps the write path's existing at-least-once semantics.
- Fail-closed paths log content-free debug notices only (no question, answer, id or payload text).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

logger = logging.getLogger(__name__)

CLARIFY_TOOL_NAME = "clarify"
OBSERVATION_KIND = "clarify_submission_observation"
# Core's producer-only lane on the admitted user row (``agent.turn_context.MEMORY_TURN_SOURCE_KEY``).
TURN_SOURCE_METADATA_KEY = "memory_turn_source"
_SUBMITTED = "submitted"
_RESPONSE_STATUSES = frozenset({"answered", "skipped", "unanswered"})
# Below this many body characters per chunk, framing would dominate; the record is omitted instead.
_MIN_BODY_PER_CHUNK = 40

Answer = Union[str, tuple]


@dataclass(frozen=True)
class ClarifyObservation:
    """One answered question of a submitted native clarify form, bound to its declaring call."""

    declaring_index: int  # transcript row of the assistant tool call
    call_position: int  # position of the call within that row's ``tool_calls``
    result_index: int  # transcript row of the single matching tool result
    call_id: str  # canonical id (core's bridge normalization)
    declared_id: str  # raw id as declared
    result_id: str  # raw id as on the tool result row
    question_number: int  # 1-based position in the form
    question_count: int
    answered_count: int
    question: str
    choices_offered: Optional[tuple]  # tuple of str, None for an open-ended question
    multi_select: bool
    answer: Answer  # str, or tuple of str for multi-select


@dataclass(frozen=True)
class ObservationRecord:
    """Rendered, immutable chunks of one observation, each carrying the full framing."""

    call_id: str
    question_number: int
    chunks: tuple


def _core_semantics():
    """Core's own normalization helpers, or None (fail closed) when an interface moved."""
    try:
        from agent.codex_responses_adapter import _summarize_user_message_for_log
        from agent.skill_commands import extract_user_instruction_from_skill_message
        from agent.tool_dispatch_helpers import _normalize_tool_call_id
        from tools.clarify_tool import _normalize_questions
    except Exception:
        return None
    return (_summarize_user_message_for_log, extract_user_instruction_from_skill_message,
            _normalize_tool_call_id, _normalize_questions)


def _valid_turn_source(turn_source: Any) -> Optional[tuple]:
    """``(turn_id, marker)`` of a well-formed producer source, or None (never coerced)."""
    if not isinstance(turn_source, dict):
        return None
    turn_id, marker = turn_source.get("turn_id"), turn_source.get("marker")
    if not (isinstance(turn_id, str) and turn_id.strip() and isinstance(marker, str) and marker.strip()):
        return None
    return turn_id, marker


def _carries_source(row: dict, source: tuple) -> bool:
    metadata = row.get("display_metadata")
    tagged = metadata.get(TURN_SOURCE_METADATA_KEY) if isinstance(metadata, dict) else None
    return isinstance(tagged, dict) and (tagged.get("turn_id"), tagged.get("marker")) == source


def find_current_turn_anchor(messages: Sequence[Any], user_content: str, turn_source: Any, *,
                             is_internal: Callable[[str], bool], flatten: Callable[..., str],
                             strip_skill: Callable[[Any], Optional[str]]) -> Optional[int]:
    """Index of the user row core's producer marked as this invocation's admitted input, or None.

    ``turn_source`` is core's optional ``{"turn_id", "marker"}`` (two non-empty strings, see
    ``agent.turn_context.stamp_memory_turn_source``). EXACTLY one transcript row must carry the same pair
    at ``display_metadata["memory_turn_source"]``; it must be a user row whose text, after the same
    multimodal flattening and skill-scaffold strip core applies before ``sync_all``, equals
    ``user_content`` and is not an internal machine notice (an assertion of the producer's claim, never a
    search); and no later user-originated row may follow it (internal notices are skipped). Anything else
    (absent, malformed, foreign or duplicated source, a marker lost through compaction or a merge, a
    non-dict row) proves nothing: there is no fallback to text, native turn id, ``message_uid`` or index."""
    source = _valid_turn_source(turn_source)
    if source is None or not isinstance(user_content, str) or not user_content.strip():
        return None

    def _user_text(row: dict) -> tuple:
        text = flatten(row.get("content"), sep="\n")
        instruction = strip_skill(text)
        internal = is_internal(text) or (isinstance(instruction, str) and is_internal(instruction))
        return instruction, internal

    if not all(isinstance(row, dict) for row in messages):
        return None
    marked = [index for index, row in enumerate(messages) if _carries_source(row, source)]
    if len(marked) != 1:
        return None
    anchor = marked[0]
    if messages[anchor].get("role") != "user":
        return None
    instruction, internal = _user_text(messages[anchor])
    if internal or instruction != user_content:
        return None
    if any(row.get("role") == "user" and not _user_text(row)[1] for row in messages[anchor + 1:]):
        return None
    return anchor


def _canonical_id(raw: Any, normalize: Callable[[Any], Any]) -> Optional[str]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    canonical = normalize(raw)
    return canonical if isinstance(canonical, str) and canonical else None


def _result_name_matches(row: dict) -> bool:
    """A result row naming a tool must name ``clarify``; a nameless (Codex-projected) row binds via its call."""
    return all(row.get(key) in (None, CLARIFY_TOOL_NAME) for key in ("name", "tool_name"))


def _declared_questions(arguments: Any, normalize_questions: Callable) -> Optional[list]:
    """The declaring call's questions exactly as core's clarify tool normalized them, or None."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError):
            return None
    if not isinstance(arguments, dict):
        return None
    normalized, error = normalize_questions(arguments.get("questions"))
    return None if error or not normalized else normalized


def _valid_answer(value: Any, multi_select: bool) -> Optional[Answer]:
    if multi_select:
        if isinstance(value, list) and value and all(isinstance(v, str) and v.strip() for v in value):
            return tuple(value)
        return None
    return value if isinstance(value, str) and value.strip() else None


def _answered_responses(content: Any, declared: list) -> tuple:
    """``(number, declared_entry, answer)`` per answered question of a submitted, well-formed result
    that is bound question-by-question to the declaring call; () for anything else."""
    if not isinstance(content, str):
        return ()
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return ()
    if not isinstance(parsed, dict) or "error" in parsed or parsed.get("outcome") != _SUBMITTED:
        return ()
    responses = parsed.get("responses")
    if not isinstance(responses, list) or len(responses) != len(declared):
        return ()
    answered = []
    for number, (entry, response) in enumerate(zip(declared, responses), 1):
        if (not isinstance(response, dict) or response.get("question") != entry["question"]
                or response.get("choices_offered") != entry["choices_offered"]
                or response.get("status") not in _RESPONSE_STATUSES):
            return ()
        if response["status"] != "answered":
            continue  # a skipped/unanswered question never invalidates its answered siblings
        answer = _valid_answer(response.get("user_response"), entry["multi_select"])
        if answer is None:
            return ()
        answered.append((number, entry, answer))
    return tuple(answered)


def extract_submitted_clarify_observations(messages: Any, user_content: str, *,
                                           is_internal: Callable[[str], bool], turn_source: Any = None) -> tuple:
    """Answered questions of this turn's submitted native clarify forms, in chronological order.

    Only forms declared after the row ``turn_source`` anchors (``find_current_turn_anchor``) count."""
    if not isinstance(messages, (list, tuple)) or not messages:
        return ()
    if _valid_turn_source(turn_source) is None:
        logger.debug("Honcho clarify observations skipped: no valid producer turn source")
        return ()
    core = _core_semantics()
    if core is None:
        logger.debug("Honcho clarify observations skipped: core interfaces unavailable")
        return ()
    flatten, strip_skill, normalize_id, normalize_questions = core
    anchor = find_current_turn_anchor(messages, user_content, turn_source, is_internal=is_internal,
                                      flatten=flatten, strip_skill=strip_skill)
    if anchor is None:
        logger.debug("Honcho clarify observations skipped: current turn source not uniquely verified")
        return ()

    # Counted over the WHOLE transcript, so a replayed or colliding id is never bound.
    declared_counts: dict[str, int] = {}
    result_rows: dict[str, list] = {}
    calls = []
    for index, row in enumerate(messages):
        if not isinstance(row, dict):
            continue
        if row.get("role") == "assistant" and isinstance(row.get("tool_calls"), list):
            for position, call in enumerate(row["tool_calls"]):
                raw_id = call.get("id") if isinstance(call, dict) else None
                call_id = _canonical_id(raw_id, normalize_id)
                if call_id is None:
                    continue
                declared_counts[call_id] = declared_counts.get(call_id, 0) + 1
                function = call.get("function")
                # Exactly core's native shape: a missing or foreign ``type`` is never a clarify declaration.
                if (index > anchor and call.get("type") == "function" and isinstance(function, dict)
                        and function.get("name") == CLARIFY_TOOL_NAME):
                    calls.append((index, position, raw_id, call_id, function.get("arguments")))
        elif row.get("role") == "tool":
            call_id = _canonical_id(row.get("tool_call_id"), normalize_id)
            if call_id is not None:
                result_rows.setdefault(call_id, []).append(index)

    observations = []
    for index, position, raw_id, call_id, arguments in calls:
        rows = result_rows.get(call_id, [])
        if declared_counts.get(call_id) != 1 or len(rows) != 1 or rows[0] <= index:
            logger.debug("Honcho clarify observation skipped: call/result correlation not unique and ordered")
            continue
        result = messages[rows[0]]
        if not _result_name_matches(result):
            logger.debug("Honcho clarify observation skipped: result names a different tool")
            continue
        declared = _declared_questions(arguments, normalize_questions)
        answered = _answered_responses(result.get("content"), declared) if declared else ()
        if not answered:
            continue
        for number, entry, answer in answered:
            offered = entry["choices_offered"]
            observations.append(ClarifyObservation(
                declaring_index=index, call_position=position, result_index=rows[0], call_id=call_id,
                declared_id=raw_id, result_id=result["tool_call_id"], question_number=number,
                question_count=len(declared), answered_count=len(answered), question=entry["question"],
                choices_offered=tuple(offered) if offered else None, multi_select=bool(entry["multi_select"]),
                answer=answer))
    return tuple(observations)


# ----- rendering -----

def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _header(part: Optional[tuple]) -> str:
    scope = f" (part {part[0]} of {part[1]} of one record)" if part else ""
    return (f"Assistant observation of a native clarify form submission{scope}: the responder is not "
            "verified by the tool; this is neither the assistant's own statement nor authenticated approval.")


def _source_line(obs: ClarifyObservation) -> str:
    raw = [f"declared as {_quote(obs.declared_id)}"] if obs.declared_id != obs.call_id else []
    if obs.result_id != obs.call_id:
        raw.append(f"result as {_quote(obs.result_id)}")
    aliases = f" ({', '.join(raw)})" if raw else ""
    return (f"Source: clarify tool call {_quote(obs.call_id)}{aliases} in the current turn; form outcome "
            f"submitted; {obs.answered_count} of {obs.question_count} questions answered; skipped or "
            "unanswered questions are not recorded.")


def _body(obs: ClarifyObservation, sanitize: Callable[[str], str]) -> Optional[str]:
    question = sanitize(obs.question).strip()
    offered = tuple(sanitize(c).strip() for c in obs.choices_offered) if obs.choices_offered else None
    if obs.multi_select:
        answer: Answer = tuple(sanitize(v).strip() for v in obs.answer)
        if not answer or not all(answer):
            return None
    else:
        answer = sanitize(obs.answer).strip()
    if not question or not answer or (offered is not None and not all(offered)):
        return None
    lines = [f"Question {obs.question_number} of {obs.question_count}, asked by the assistant: {_quote(question)}"]
    if offered is None:
        lines.append("Choices offered by the assistant: none (open-ended question)")
    else:
        mode = "multi-select" if obs.multi_select else "single-select"
        lines.append(f"Choices offered by the assistant ({mode}): "
                     + json.dumps(list(offered), ensure_ascii=False))
    if isinstance(answer, tuple):
        lines.append("Response submitted through the clarify tool (unverified responder): "
                     + json.dumps(list(answer), ensure_ascii=False))
        typed = [v for v in answer if v not in offered]
        if typed:
            lines.append("Entries typed as free text, not offered choices: " + json.dumps(typed, ensure_ascii=False))
    else:
        typed = "" if offered is None or answer in offered else " (typed free text, not an offered choice)"
        lines.append(f"Response submitted through the clarify tool (unverified responder){typed}: {_quote(answer)}")
    return "\n".join(lines)


def _split_literal(body: str, budget: int) -> list:
    """Split ``body`` into pieces of at most ``budget`` characters whose concatenation is ``body``."""
    pieces, rest = [], body
    while len(rest) > budget:
        cut = max(rest.rfind("\n", 0, budget + 1), rest.rfind(" ", 0, budget + 1))
        if cut < budget // 2:
            cut = budget
        pieces.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        pieces.append(rest)
    return pieces


def render_observation_chunks(obs: ClarifyObservation, limit: int,
                              sanitize: Callable[[str], str]) -> Optional[tuple]:
    """Chunks within ``limit`` characters, EVERY one framed by the observation header and source line,
    or None when the record cannot be framed safely (the caller omits only this record)."""
    body = _body(obs, sanitize)
    if body is None or not isinstance(limit, int) or limit <= 0:
        return None
    source = _source_line(obs)
    whole = "\n".join((_header(None), body, source))
    if len(whole) <= limit:
        return (whole,)
    total = 2
    while True:
        overhead = len(_header((total, total))) + len(source) + 2
        budget = limit - overhead
        if budget < _MIN_BODY_PER_CHUNK:
            return None
        pieces = _split_literal(body, budget)
        if len(pieces) <= total:
            count = len(pieces)
            chunks = tuple("\n".join((_header((k, count)), piece, source)) for k, piece in enumerate(pieces, 1))
            return chunks if all(len(c) <= limit for c in chunks) else None
        total = len(pieces)


def observation_records(messages: Any, user_content: str, *, limit: int,
                        is_internal: Callable[[str], bool],
                        sanitize: Callable[[str], str], turn_source: Any = None) -> tuple:
    """Immutable rendered records for this turn's submitted clarify answers; () on any doubt, including
    an absent or unverifiable ``turn_source``.

    Computed synchronously by the caller, so a queued write never sees later transcript mutation."""
    try:
        observations = extract_submitted_clarify_observations(messages, user_content, is_internal=is_internal,
                                                              turn_source=turn_source)
    except Exception as exc:
        logger.debug("Honcho clarify observations skipped: extraction failed (%s)", type(exc).__name__)
        return ()
    records = []
    for obs in observations:
        try:
            chunks = render_observation_chunks(obs, limit, sanitize)
        except Exception as exc:
            logger.debug("Honcho clarify observation omitted: rendering failed (%s)", type(exc).__name__)
            continue
        if chunks is None:
            logger.debug("Honcho clarify observation omitted: it cannot be framed within messageMaxChars")
            continue
        records.append(ObservationRecord(call_id=obs.call_id, question_number=obs.question_number, chunks=chunks))
    return tuple(records)
