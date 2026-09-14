"""The single, lossless model-context accounting entry point.

This module deliberately does *not* shorten messages. Semantic session
compaction lives in ``session_compaction``; provider paging and ``ResourceRef``
navigation live at the capability boundary. Keeping this module small prevents
a second hidden result-projection contract from growing back into the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from src.services.context_ledger import DEFAULT_CONTEXT_WINDOW_TOKENS, estimate_tokens


MINIMUM_CONTEXT_WINDOW_TOKENS = 200_000
DEFAULT_CONTEXT_MESSAGE_TOKEN_BUDGET = max(
    MINIMUM_CONTEXT_WINDOW_TOKENS,
    int(DEFAULT_CONTEXT_WINDOW_TOKENS or 0),
)


def configured_context_message_token_budget() -> int:
    """Return the one declared context window, never a second Composer cap."""

    return DEFAULT_CONTEXT_MESSAGE_TOKEN_BUDGET


@dataclass(frozen=True)
class ContextComposition:
    messages: tuple[BaseMessage, ...]
    stats: dict[str, Any]


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _message_transport_payload(message: BaseMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": str(getattr(message, "type", message.__class__.__name__)),
        "content": getattr(message, "content", ""),
    }
    if isinstance(message, AIMessage):
        payload["tool_calls"] = list(getattr(message, "tool_calls", None) or [])
        payload["invalid_tool_calls"] = list(
            getattr(message, "invalid_tool_calls", None) or []
        )
        additional_kwargs = dict(getattr(message, "additional_kwargs", None) or {})
        if additional_kwargs:
            payload["additional_kwargs"] = additional_kwargs
    if isinstance(message, ToolMessage):
        payload["tool_call_id"] = str(getattr(message, "tool_call_id", "") or "")
        payload["name"] = str(getattr(message, "name", "") or "")
        payload["status"] = str(getattr(message, "status", "") or "")
    return payload


def _fast_token_estimate(serialized: str) -> int:
    """Avoid tokenizer pathological cases without changing the source text."""

    text = str(serialized or "")
    if len(text) <= 16_000:
        return estimate_tokens(text)
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    other = len(text) - cjk
    return max(1, cjk + (other + 3) // 4)


def estimate_message_tokens(messages: Iterable[BaseMessage]) -> int:
    """Estimate model-visible tokens without mutating the messages."""

    return sum(
        _fast_token_estimate(_json_text(_message_transport_payload(message)))
        for message in messages
        if isinstance(message, BaseMessage)
    )


def compose_model_messages(
    messages: Iterable[BaseMessage],
    *,
    message_token_budget: int | None = None,
) -> ContextComposition:
    """Return the exact model message sequence plus budget telemetry.

    ``message_token_budget`` is observational. Going over it is a signal for
    the session compactor; it is never permission to slice strings, keep only
    the first N records, project ToolMessages, or delete protocol groups.
    There are intentionally no legacy projection arguments: any attempt to
    restore per-message slicing or first-N retention must fail at the call
    boundary instead of silently growing a second policy.
    """

    original = tuple(
        message for message in messages if isinstance(message, BaseMessage)
    )
    budget = max(
        MINIMUM_CONTEXT_WINDOW_TOKENS,
        int(
            message_token_budget
            if message_token_budget is not None
            else configured_context_message_token_budget()
        ),
    )
    tokens = estimate_message_tokens(original)
    serialized = [
        _json_text(_message_transport_payload(message)) for message in original
    ]
    stats = {
        "schema_version": "evoengine.context-composition/v2",
        "policy": "lossless_pass_through",
        "message_token_budget": budget,
        "original_message_count": len(original),
        "composed_message_count": len(original),
        "original_tokens_estimated": tokens,
        "composed_tokens_estimated": tokens,
        "original_chars": sum(len(item) for item in serialized),
        "composed_chars": sum(len(item) for item in serialized),
        "within_budget": tokens <= budget,
        "over_budget_tokens": max(0, tokens - budget),
    }
    return ContextComposition(messages=original, stats=stats)
