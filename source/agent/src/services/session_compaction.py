"""Cumulative semantic compaction for long Agent sessions.

Compaction is separate from tool-result projection. It replaces only complete,
closed historical message units with a validated cumulative checkpoint. The
authoritative transcript and durable machine results remain unchanged. If
generation or validation fails, no message is removed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
from typing import Any, Iterable, Protocol, Sequence

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, ValidationError

from src.agents.canonical_protocol_ledger import (
    server_task_completion_seal_from_tool_result,
)
from src.services.context_composer import estimate_message_tokens


SESSION_COMPACT_SCHEMA = "evoengine.session-compact-checkpoint/v1"
SESSION_COMPACTION_MODEL_TAG = "evoengine:session_compaction"
REQUIRED_SECTIONS = (
    "session_intent",
    "user_constraints",
    "current_task_state",
    "completed_work_and_evidence",
    "key_decisions_and_rationale",
    "rejected_options",
    "resource_map",
    "citation_bindings",
    "side_effect_receipts",
    "open_questions_hitl",
    "remaining_work",
    "next_safe_action",
)
_CHECKPOINT_MESSAGE_KIND = "session_compact_checkpoint"
_RUNTIME_CONTROL_KEY = "runtime_control"
_COMPLETION_AUTHORITY_KEY = "completion_authority"
_TASK_AUTHORITY_KEY = "task_authority"


def _validated_task_authority_control(
    value: Any,
    *,
    transport_request_id: str,
    completion_authority: dict[str, Any] | None = None,
    fill_run_from_receipt: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SessionCompactionError("session compact task authority is missing")
    expected_keys = {
        "schema_version",
        "transport_request_id",
        "task_authority_request_id",
        "task_run_id",
        "continuation_parent_request_id",
    }
    if set(value) != expected_keys:
        raise SessionCompactionError("session compact task authority fields are invalid")
    control = dict(value)
    transport = str(control.get("transport_request_id") or "").strip()
    authority = str(control.get("task_authority_request_id") or "").strip()
    run_id = str(control.get("task_run_id") or "").strip()
    parent = str(control.get("continuation_parent_request_id") or "").strip()
    if (
        control.get("schema_version") != "evoengine.task-authority-control/v1"
        or not transport_request_id
        or transport != str(transport_request_id)
        or not authority
        or (not parent and authority != transport)
        or parent == transport
        or (authority != transport and not run_id)
    ):
        raise SessionCompactionError("session compact task authority is invalid")
    if completion_authority is not None:
        receipt = completion_authority.get("completion_receipt")
        receipt_request_id = (
            str(receipt.get("request_id") or "").strip()
            if isinstance(receipt, dict)
            else ""
        )
        receipt_run_id = (
            str(receipt.get("run_id") or "").strip()
            if isinstance(receipt, dict)
            else ""
        )
        if receipt_request_id != authority or not receipt_run_id:
            raise SessionCompactionError(
                "SESSION_COMPACT_COMPLETION_RECEIPT_TASK_AUTHORITY_MISMATCH"
            )
        if run_id and run_id != receipt_run_id:
            raise SessionCompactionError(
                "SESSION_COMPACT_COMPLETION_RECEIPT_RUN_MISMATCH"
            )
        if not run_id and fill_run_from_receipt:
            control["task_run_id"] = receipt_run_id
            run_id = receipt_run_id
        if run_id != receipt_run_id:
            raise SessionCompactionError(
                "SESSION_COMPACT_COMPLETION_RECEIPT_RUN_MISMATCH"
            )
    return control


class SessionCompactionError(RuntimeError):
    """The active context cannot be compacted without losing its contract."""


class SessionCompactionFormatError(SessionCompactionError):
    """The provider completed a response that did not satisfy the schema."""


class SessionCompactionTransportError(SessionCompactionError):
    """The provider transport failed after the bounded reconnect budget."""


class SessionCompactionOutputLimitError(SessionCompactionError):
    """The provider explicitly stopped because its output limit was reached."""


class SessionCompactionNotApplicableError(SessionCompactionError):
    """The lossless transcript currently has no safely compactable prefix."""


class SubmitSessionCheckpoint(BaseModel):
    """Strict semantic portion of one cumulative session checkpoint.

    Stable resource, citation and side-effect identities remain mechanically
    extracted.  Narrative sections intentionally use strings/lists of strings
    so DeepSeek strict function calling receives a closed JSON Schema rather
    than arbitrary nested objects with unconstrained properties.
    """

    model_config = ConfigDict(extra="forbid")

    session_intent: list[str]
    user_constraints: list[str]
    current_task_state: list[str]
    completed_work_and_evidence: list[str]
    key_decisions_and_rationale: list[str]
    rejected_options: list[str]
    resource_map: list[str]
    citation_bindings: list[str]
    side_effect_receipts: list[str]
    open_questions_hitl: list[str]
    remaining_work: list[str]
    next_safe_action: str


@dataclass(frozen=True)
class SessionCompactionProtocolResult:
    sections: dict[str, Any]
    stats: dict[str, Any]


class SessionCompactionProtocol(Protocol):
    """Explicit adapter boundary between production strict calls and test fakes."""

    def invoke(
        self,
        model: Any,
        prompt: Sequence[BaseMessage],
    ) -> SessionCompactionProtocolResult: ...

    async def ainvoke(
        self,
        model: Any,
        prompt: Sequence[BaseMessage],
    ) -> SessionCompactionProtocolResult: ...


@dataclass(frozen=True)
class SessionCompactionPlan:
    source_messages: tuple[BaseMessage, ...]
    compacted_prefix: tuple[BaseMessage, ...]
    retained_messages: tuple[BaseMessage, ...]
    already_covered_count: int


@dataclass(frozen=True)
class SessionCompactionResult:
    messages: tuple[BaseMessage, ...]
    checkpoint: dict[str, Any] | None
    stats: dict[str, Any]


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _message_payload(message: BaseMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": str(getattr(message, "type", message.__class__.__name__)),
        "content": getattr(message, "content", ""),
    }
    additional_kwargs = dict(getattr(message, "additional_kwargs", None) or {})
    if additional_kwargs:
        payload["additional_kwargs"] = additional_kwargs
    message_id = str(getattr(message, "id", "") or "").strip()
    if message_id:
        payload["id"] = message_id
    if isinstance(message, AIMessage):
        payload["tool_calls"] = list(getattr(message, "tool_calls", None) or [])
        payload["invalid_tool_calls"] = list(
            getattr(message, "invalid_tool_calls", None) or []
        )
    if isinstance(message, ToolMessage):
        payload["tool_call_id"] = str(getattr(message, "tool_call_id", "") or "")
        payload["name"] = str(getattr(message, "name", "") or "")
        payload["status"] = str(getattr(message, "status", "") or "")
        artifact = getattr(message, "artifact", None)
        if isinstance(artifact, dict):
            bounded_artifact = {
                key: artifact[key]
                for key in (
                    "schema_version",
                    "control_projection",
                    "runtime_control",
                )
                if key in artifact
            }
            if bounded_artifact:
                payload["artifact"] = bounded_artifact
    return payload


def _messages_digest(messages: Sequence[BaseMessage]) -> str:
    return hashlib.sha256(
        _json_text([_message_payload(message) for message in messages]).encode("utf-8")
    ).hexdigest()


def _is_checkpoint_message(message: BaseMessage) -> bool:
    metadata = dict(getattr(message, "additional_kwargs", None) or {})
    return metadata.get("evo_context_kind") == _CHECKPOINT_MESSAGE_KIND


def _parse_tool_payload(message: ToolMessage) -> Any:
    content = getattr(message, "content", "")
    if isinstance(content, (dict, list)):
        return content
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"model_text": content}


def _walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_dicts(item)


def _dedupe_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        digest = hashlib.sha256(_json_text(row).encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        result.append(row)
    return result


def _stable_resource_rows(payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _walk_dicts(payload):
        resource_id = str(item.get("resource_id") or "").strip()
        if not resource_id:
            continue
        row = {"resource_id": resource_id}
        for key in (
            "uri",
            "kind",
            "media_type",
            "mime_type",
            "sha256",
            "size_bytes",
            "complete",
            "has_more",
            "cursor",
        ):
            if key in item:
                row[key] = item[key]
        rows.append(row)
    return _dedupe_rows(rows)


def _stable_citation_rows(payload: Any) -> list[dict[str, Any]]:
    identity_keys = {
        "citation_key",
        "cite_key",
        "claim_id",
        "evidence_id",
        "source_id",
        "doi",
        "pmid",
        "locator",
    }
    rows: list[dict[str, Any]] = []
    for item in _walk_dicts(payload):
        if not identity_keys.intersection(item):
            continue
        row = {key: item[key] for key in identity_keys if key in item}
        if row:
            rows.append(row)
    return _dedupe_rows(rows)


def _tool_call_arguments(messages: Sequence[BaseMessage]) -> dict[str, dict[str, Any]]:
    calls: dict[str, dict[str, Any]] = {}
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in [
            *list(getattr(message, "tool_calls", None) or []),
            *list(getattr(message, "invalid_tool_calls", None) or []),
        ]:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or "").strip()
            if call_id:
                calls[call_id] = {
                    "tool": str(call.get("name") or ""),
                    "arguments": dict(call.get("args") or {})
                    if isinstance(call.get("args"), dict)
                    else {},
                }
    return calls


def _extract_completion_authority(
    messages: Sequence[BaseMessage],
) -> dict[str, Any] | None:
    calls = _tool_call_arguments(messages)
    authority: dict[str, Any] | None = None
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        call_id = str(getattr(message, "tool_call_id", "") or "").strip()
        call = calls.get(call_id, {})
        seal = server_task_completion_seal_from_tool_result(
            tool_name=str(call.get("tool") or getattr(message, "name", "") or ""),
            call_id=call_id,
            tool_input=call.get("arguments") or {},
            result=message,
        )
        if seal is None:
            continue
        if authority is not None and _json_text(authority) != _json_text(seal):
            raise SessionCompactionError(
                "conflicting Server CompletionReceipts in one compact checkpoint"
            )
        authority = seal
    return authority


def completion_authority_from_checkpoint(
    checkpoint: Any,
) -> dict[str, Any] | None:
    """Return the private, exact Server completion seal from a checkpoint."""

    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("schema_version") != SESSION_COMPACT_SCHEMA
    ):
        return None
    runtime_control = checkpoint.get(_RUNTIME_CONTROL_KEY)
    if not isinstance(runtime_control, dict):
        return None
    authority = runtime_control.get(_COMPLETION_AUTHORITY_KEY)
    if authority is None:
        return None
    if not isinstance(authority, dict):
        raise SessionCompactionError(
            "session compact completion authority must be an object"
        )
    receipt = authority.get("completion_receipt")
    if (
        str(authority.get("schema_version") or "")
        != "evoengine.agent-terminal-seal/v1"
        or str(authority.get("tool_name") or "") != "task_complete"
        or not str(authority.get("tool_call_id") or "").strip()
        or not str(authority.get("summary") or "").strip()
        or not isinstance(receipt, dict)
        or str(receipt.get("schema_version") or "")
        != "evoengine.task-completion-receipt/v1"
        or str(receipt.get("authority") or "") != "task_tree_internal_api"
        or not str(receipt.get("receipt_id") or "").strip()
        or not str(receipt.get("run_id") or "").strip()
    ):
        raise SessionCompactionError(
            "session compact completion authority is not an exact Server receipt"
        )
    return json.loads(_json_text(authority))


def completion_authority_for_request(
    checkpoint: Any,
    *,
    request_id: str,
    task_authority: dict[str, Any],
) -> dict[str, Any] | None:
    """Resolve a completion seal only from a valid same-request checkpoint."""

    authority = completion_authority_from_checkpoint(checkpoint)
    if authority is None:
        return None
    exact_request_id = str(request_id or "")
    if (
        not exact_request_id
        or str((checkpoint or {}).get("created_by_request_id") or "")
        != exact_request_id
    ):
        raise SessionCompactionError(
            "SESSION_COMPACT_COMPLETION_AUTHORITY_REQUEST_MISMATCH"
        )
    coverage = (checkpoint or {}).get("active_request_coverage")
    try:
        covered_count = int(
            coverage.get("message_count") or 0
            if isinstance(coverage, dict)
            else 0
        )
    except (TypeError, ValueError):
        covered_count = 0
    prefix_digest = (
        str(coverage.get("prefix_digest") or "").strip().lower()
        if isinstance(coverage, dict)
        else ""
    )
    if (
        not isinstance(coverage, dict)
        or str(coverage.get("request_id") or "") != exact_request_id
        or covered_count <= 0
        or len(prefix_digest) != 64
        or any(character not in "0123456789abcdef" for character in prefix_digest)
    ):
        raise SessionCompactionError(
            "SESSION_COMPACT_COMPLETION_AUTHORITY_COVERAGE_INVALID"
        )
    runtime_control = (checkpoint or {}).get(_RUNTIME_CONTROL_KEY)
    checkpoint_task_authority = (
        runtime_control.get(_TASK_AUTHORITY_KEY)
        if isinstance(runtime_control, dict)
        else None
    )
    expected_task_authority = _validated_task_authority_control(
        task_authority,
        transport_request_id=exact_request_id,
        completion_authority=authority,
    )
    observed_task_authority = _validated_task_authority_control(
        checkpoint_task_authority,
        transport_request_id=exact_request_id,
        completion_authority=authority,
    )
    if _json_text(expected_task_authority) != _json_text(observed_task_authority):
        raise SessionCompactionError(
            "SESSION_COMPACT_TASK_AUTHORITY_MISMATCH"
        )
    return authority


def attach_completion_authority_to_checkpoint(
    checkpoint: Any,
    *,
    completion_authority: dict[str, Any],
    request_id: str,
    task_authority: dict[str, Any],
) -> dict[str, Any] | None:
    """Attach an exact terminal seal to an existing semantic checkpoint.

    TaskComplete is terminal, so there is no later model call at which the
    compactor could capture its receipt.  This helper performs only a
    mechanical merge; it never fabricates semantic coverage or invokes a
    model.  A turn without an existing valid compact checkpoint simply relies
    on the normal terminal CompletionReceipt event.
    """

    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("schema_version") != SESSION_COMPACT_SCHEMA
        or not str(checkpoint.get("checkpoint_id") or "").strip()
    ):
        return None
    exact_request_id = str(request_id or "")
    coverage = checkpoint.get("active_request_coverage")
    try:
        covered_count = int(
            coverage.get("message_count") or 0
            if isinstance(coverage, dict)
            else 0
        )
    except (TypeError, ValueError):
        covered_count = 0
    prefix_digest = (
        str(coverage.get("prefix_digest") or "").strip().lower()
        if isinstance(coverage, dict)
        else ""
    )
    if not exact_request_id or str(
        checkpoint.get("created_by_request_id") or ""
    ) != exact_request_id:
        raise SessionCompactionError(
            "cannot attach completion authority to a foreign checkpoint"
        )
    if (
        not isinstance(coverage, dict)
        or str(coverage.get("request_id") or "") != exact_request_id
        or covered_count <= 0
        or len(prefix_digest) != 64
        or any(character not in "0123456789abcdef" for character in prefix_digest)
    ):
        return None
    candidate = dict(checkpoint)
    runtime_control = dict(candidate.get(_RUNTIME_CONTROL_KEY) or {})
    runtime_control[_TASK_AUTHORITY_KEY] = _validated_task_authority_control(
        task_authority,
        transport_request_id=exact_request_id,
        completion_authority=completion_authority,
        fill_run_from_receipt=True,
    )
    runtime_control[_COMPLETION_AUTHORITY_KEY] = completion_authority
    candidate[_RUNTIME_CONTROL_KEY] = runtime_control
    validated = completion_authority_from_checkpoint(candidate)
    if validated is None:
        raise SessionCompactionError("completion authority attachment is invalid")
    existing = completion_authority_from_checkpoint(checkpoint)
    if existing is not None and _json_text(existing) != _json_text(validated):
        raise SessionCompactionError(
            "Server CompletionReceipt conflicts with compact checkpoint authority"
        )
    id_payload = {
        key: value
        for key, value in candidate.items()
        if key != "checkpoint_id" and not str(key).startswith("_runtime_")
    }
    candidate["checkpoint_id"] = "session_checkpoint_" + hashlib.sha256(
        _json_text(id_payload).encode("utf-8")
    ).hexdigest()[:24]
    return candidate


def _model_visible_mechanical_state(value: Any) -> dict[str, Any]:
    mechanical = dict(value) if isinstance(value, dict) else {}
    visible_receipts: list[dict[str, Any]] = []
    for raw in list(mechanical.get("tool_receipts") or []):
        if not isinstance(raw, dict):
            continue
        if str(raw.get("tool_id") or "") != "task_complete":
            visible_receipts.append(dict(raw))
            continue
        visible_receipts.append(
            {
                key: raw[key]
                for key in (
                    "tool_call_id",
                    "tool_id",
                    "status",
                    "complete",
                    "has_more",
                    "error_kind",
                    "error_code",
                    "completion_authority",
                )
                if key in raw
            }
        )
    mechanical["tool_receipts"] = visible_receipts
    return mechanical


def _model_visible_checkpoint(checkpoint: Any) -> dict[str, Any] | None:
    if not isinstance(checkpoint, dict):
        return None
    visible = dict(checkpoint)
    visible.pop(_RUNTIME_CONTROL_KEY, None)
    visible["mechanical_state"] = _model_visible_mechanical_state(
        visible.get("mechanical_state")
    )
    return visible


def _summary_message_payloads(
    messages: Sequence[BaseMessage],
) -> list[dict[str, Any]]:
    """Project sealed completion protocol without its receipt or long summary."""

    calls = _tool_call_arguments(messages)
    task_complete_ids = {
        call_id
        for call_id, call in calls.items()
        if str(call.get("tool") or "") == "task_complete"
    }
    rows: list[dict[str, Any]] = []
    for message in messages:
        payload = _message_payload(message)
        if isinstance(message, AIMessage):
            sanitized_calls: list[dict[str, Any]] = []
            for call in list(payload.get("tool_calls") or []):
                row = dict(call) if isinstance(call, dict) else {}
                if str(row.get("id") or "") in task_complete_ids:
                    row["args"] = {"summary": "[sealed runtime-only completion summary]"}
                sanitized_calls.append(row)
            payload["tool_calls"] = sanitized_calls
            if task_complete_ids.intersection(
                str(item.get("id") or "")
                for item in sanitized_calls
                if isinstance(item, dict)
            ):
                payload.pop("additional_kwargs", None)
        elif isinstance(message, ToolMessage) and str(
            getattr(message, "tool_call_id", "") or ""
        ) in task_complete_ids:
            parsed = _parse_tool_payload(message)
            parsed_dict = parsed if isinstance(parsed, dict) else {}
            payload["content"] = _json_text(
                {
                    "ok": parsed_dict.get("ok") is not False,
                    "accepted": bool(parsed_dict.get("accepted", True)),
                    "status": str(parsed_dict.get("status") or "succeeded"),
                    "completion_authority": "sealed_runtime_only",
                }
            )
            payload.pop("additional_kwargs", None)
            payload.pop("artifact", None)
        rows.append(payload)
    return rows


def extract_mechanical_state(messages: Sequence[BaseMessage]) -> dict[str, Any]:
    """Extract stable receipts without asking the summary model to invent them."""

    calls = _tool_call_arguments(messages)
    receipts: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    side_effects: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        call_id = str(getattr(message, "tool_call_id", "") or "").strip()
        call = calls.get(call_id, {})
        payload = _parse_tool_payload(message)
        payload_dict = payload if isinstance(payload, dict) else {}
        tool_id = str(
            call.get("tool") or getattr(message, "name", "") or ""
        )
        task_complete = tool_id == "task_complete"
        # The exact TaskComplete summary and Server CompletionReceipt live in
        # runtime_control.completion_authority.  Neither the summary model nor
        # the later main model needs their args, IDs, deliverables or nested
        # resource rows.
        resource_rows = [] if task_complete else _stable_resource_rows(payload)
        citation_rows = [] if task_complete else _stable_citation_rows(payload)
        receipt: dict[str, Any] = {
            "tool_call_id": call_id,
            "tool_id": tool_id,
            "status": str(
                payload_dict.get("status")
                or getattr(message, "status", "")
                or ("succeeded" if payload_dict.get("ok", True) else "failed")
            ),
            "scope": {} if task_complete else dict(call.get("arguments") or {}),
            "resource_refs": resource_rows,
            "citation_bindings": citation_rows,
        }
        if task_complete:
            receipt["completion_authority"] = "sealed_runtime_only"
        for key in ("complete", "has_more", "cursor", "error_kind", "error_code"):
            if key in payload_dict:
                receipt[key] = payload_dict[key]
        for key in (() if task_complete else (
            "side_effect_receipt",
            "side_effect",
            "idempotency_receipt",
        )):
            value = payload_dict.get(key)
            if value not in (None, "", [], {}):
                receipt["side_effect_receipt"] = value
                side_effects.append(
                    {
                        "tool_call_id": call_id,
                        "tool_id": receipt["tool_id"],
                        "receipt": value,
                    }
                )
                break
        receipts.append(receipt)
        resources.extend(resource_rows)
        citations.extend(citation_rows)
    return {
        "tool_receipts": _dedupe_rows(receipts),
        "resource_refs": _dedupe_rows(resources),
        "citation_bindings": _dedupe_rows(citations),
        "side_effect_receipts": _dedupe_rows(side_effects),
    }


def _message_units(messages: Sequence[BaseMessage]) -> list[tuple[int, int, bool]]:
    """Return ``(start, end, closed)`` units; AI/tool protocols stay atomic."""

    units: list[tuple[int, int, bool]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        calls = (
            [
                *list(getattr(message, "tool_calls", None) or []),
                *list(getattr(message, "invalid_tool_calls", None) or []),
            ]
            if isinstance(message, AIMessage)
            else []
        )
        if not calls:
            units.append((index, index + 1, True))
            index += 1
            continue
        expected = {
            str(call.get("id") or "").strip()
            for call in calls
            if isinstance(call, dict) and str(call.get("id") or "").strip()
        }
        cursor = index + 1
        observed: set[str] = set()
        while cursor < len(messages) and isinstance(messages[cursor], ToolMessage):
            observed.add(str(getattr(messages[cursor], "tool_call_id", "") or ""))
            cursor += 1
        units.append((index, cursor, bool(expected) and expected.issubset(observed)))
        index = cursor
    return units


def _covered_prefix_count(
    messages: Sequence[BaseMessage],
    checkpoint: dict[str, Any] | None,
    *,
    request_id: str,
) -> int:
    if not isinstance(checkpoint, dict):
        return 0
    coverage = checkpoint.get("active_request_coverage")
    if not isinstance(coverage, dict):
        return 0
    if str(coverage.get("request_id") or "") != str(request_id or ""):
        return 0
    try:
        count = int(coverage.get("message_count") or 0)
    except (TypeError, ValueError):
        return 0
    if count <= 0 or count > len(messages):
        return 0
    expected = str(coverage.get("prefix_digest") or "")
    return count if expected and _messages_digest(messages[:count]) == expected else 0


def active_messages_with_checkpoint(
    messages: Sequence[BaseMessage],
    *,
    checkpoint: dict[str, Any] | None,
    request_id: str,
) -> tuple[BaseMessage, ...]:
    """Inject one checkpoint and suppress only its verified covered prefix."""

    source = tuple(message for message in messages if not _is_checkpoint_message(message))
    covered = _covered_prefix_count(source, checkpoint, request_id=request_id)
    active = source[covered:]
    if (
        covered > 0
        and isinstance(checkpoint, dict)
        and checkpoint.get("schema_version") == SESSION_COMPACT_SCHEMA
    ):
        return (checkpoint_model_message(checkpoint), *active)
    return active


def plan_session_compaction(
    messages: Sequence[BaseMessage],
    *,
    checkpoint: dict[str, Any] | None,
    request_id: str,
    retain_recent_tokens: int,
) -> SessionCompactionPlan:
    source = tuple(message for message in messages if not _is_checkpoint_message(message))
    covered = _covered_prefix_count(source, checkpoint, request_id=request_id)
    active = source[covered:]
    units = _message_units(active)
    if len(units) < 2:
        raise SessionCompactionNotApplicableError(
            "context is over the compaction trigger but has no closed historical prefix"
        )

    retained_tokens = 0
    retained_unit_start = len(units) - 1
    for unit_index in range(len(units) - 1, -1, -1):
        start, end, _closed = units[unit_index]
        retained_tokens += estimate_message_tokens(active[start:end])
        retained_unit_start = unit_index
        if retained_tokens >= max(1, int(retain_recent_tokens)):
            break

    open_units = [
        index for index, (_start, _end, closed) in enumerate(units) if not closed
    ]
    if open_units:
        earliest_open = min(open_units)
        # The latest HumanMessage before an open call is the exact request that
        # caused that unfinished protocol. Keep it and the whole dependent
        # suffix verbatim; do not rely on a semantic checkpoint mid-action.
        dependency_start = earliest_open
        for unit_index in range(earliest_open - 1, -1, -1):
            start, _end, _closed = units[unit_index]
            if isinstance(active[start], HumanMessage):
                dependency_start = unit_index
                break
        retained_unit_start = min(retained_unit_start, dependency_start)

    prefix_end = units[retained_unit_start][0]
    if prefix_end <= 0:
        raise SessionCompactionNotApplicableError(
            "context is over the compaction trigger but safe protocol retention leaves no compactable prefix"
        )
    return SessionCompactionPlan(
        source_messages=source,
        compacted_prefix=tuple(active[:prefix_end]),
        retained_messages=tuple(active[prefix_end:]),
        already_covered_count=covered,
    )


def session_compaction_input_signature(
    messages: Sequence[BaseMessage],
    *,
    checkpoint: dict[str, Any] | None,
    request_id: str,
    retain_recent_tokens: int,
) -> dict[str, Any]:
    """Return a deterministic identity for one safe compaction candidate.

    The signature contains no message content.  It lets the caller persist a
    failed-attempt backoff across graph rebuilds and sandbox/HITL checkpoints
    without changing the lossless transcript or treating failure as coverage.
    """

    plan = plan_session_compaction(
        messages,
        checkpoint=checkpoint,
        request_id=request_id,
        retain_recent_tokens=retain_recent_tokens,
    )
    compacted_tokens = estimate_message_tokens(plan.compacted_prefix)
    payload = {
        "request_id": str(request_id or ""),
        "previous_checkpoint_id": (
            str((checkpoint or {}).get("checkpoint_id") or "")
            if plan.already_covered_count > 0
            else ""
        ),
        "already_covered_count": plan.already_covered_count,
        "compacted_message_count": len(plan.compacted_prefix),
        "compacted_prefix_digest": _messages_digest(plan.compacted_prefix),
        "compacted_prefix_tokens_estimated": compacted_tokens,
        "retain_recent_tokens": max(1, int(retain_recent_tokens)),
    }
    return {
        **payload,
        "fingerprint": hashlib.sha256(
            _json_text(payload).encode("utf-8")
        ).hexdigest(),
    }


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text") or item.get("content") or "")
            if isinstance(item, dict)
            else str(item)
            for item in content
        )
    return str(content or "")


def _validated_sections(payload: Any) -> dict[str, Any]:
    if isinstance(payload, SubmitSessionCheckpoint):
        parsed = payload
    else:
        try:
            parsed = SubmitSessionCheckpoint.model_validate(payload)
        except ValidationError as exc:
            raise SessionCompactionFormatError(
                "session compact checkpoint did not satisfy the required schema"
            ) from exc
    return {key: value for key, value in parsed.model_dump().items() if key in REQUIRED_SECTIONS}


def _parse_text_sections(response: Any) -> dict[str, Any]:
    """Parse the explicit test-only free-text adapter response.

    Production compaction never calls this parser; it uses a forced strict
    tool call and validates the decoded tool arguments directly.
    """

    text = _response_text(response).strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1 :] if first_newline >= 0 else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise SessionCompactionFormatError(
            "session compactor did not return a JSON object"
        )
    try:
        payload = json.loads(text[start : end + 1])
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SessionCompactionFormatError(
            "session compactor returned invalid JSON"
        ) from exc
    if isinstance(payload, dict) and isinstance(payload.get("sections"), dict):
        payload = payload["sections"]
    return _validated_sections(payload)


def _turn_id(message: BaseMessage) -> int:
    metadata = dict(getattr(message, "additional_kwargs", None) or {})
    raw = metadata.get("turn_id")
    if raw is None and isinstance(metadata.get("evo_history"), dict):
        raw = metadata["evo_history"].get("turn_id")
    text = str(raw or "").strip()
    if text.startswith("turn_"):
        text = text[5:]
    try:
        return max(0, int(text))
    except (TypeError, ValueError):
        return 0


def _summary_prompt(
    *,
    previous_checkpoint: dict[str, Any] | None,
    compacted_messages: Sequence[BaseMessage],
    mechanical_state: dict[str, Any],
) -> list[BaseMessage]:
    instruction = HumanMessage(
        content=(
            "Create a cumulative EvoEngine session handoff as one strict JSON object. "
            "Preserve the user's goal and hard constraints, completed facts with their "
            "evidence/resource identifiers, decisions and reasons, rejected options, "
            "side effects, HITL state, remaining work, and the next safe action. Merge "
            "the previous checkpoint instead of summarizing only the newest messages. "
            "Do not infer root-task completion from a tool/provider outcome. Do not "
            "issue instructions to the user. Return exactly these keys: "
            + ", ".join(REQUIRED_SECTIONS)
            + ". Use concise string-list items for every section except "
            "next_safe_action, which is one string. Empty list sections must be "
            "[] rather than omitted."
        )
    )
    body = {
        "previous_checkpoint": _model_visible_checkpoint(previous_checkpoint),
        "closed_history_to_compact": _summary_message_payloads(
            compacted_messages
        ),
        "mechanically_extracted_state": _model_visible_mechanical_state(
            mechanical_state
        ),
    }
    return [instruction, HumanMessage(content=_json_text(body))]


_FORMAT_CORRECTION_MAX_RETRIES = 1
_TRANSPORT_MAX_RECONNECTS = 3
_COMPACTION_REQUEST_TIMEOUT_DEFAULT_SECONDS = 60.0
_COMPACTION_REQUEST_TIMEOUT_MIN_SECONDS = 0.05
_COMPACTION_REQUEST_TIMEOUT_MAX_SECONDS = 120.0
_FORMAT_CORRECTION = HumanMessage(
    content=(
        "The previous generation did not produce exactly one valid "
        "SubmitSessionCheckpoint call. Submit that required function once with "
        "all schema fields populated. Do not return ordinary text."
    )
)


def _compaction_request_timeout_seconds() -> float:
    raw = str(
        os.environ.get(
            "EVO_SESSION_COMPACTION_REQUEST_TIMEOUT_SEC",
            _COMPACTION_REQUEST_TIMEOUT_DEFAULT_SECONDS,
        )
    ).strip()
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        parsed = _COMPACTION_REQUEST_TIMEOUT_DEFAULT_SECONDS
    return min(
        _COMPACTION_REQUEST_TIMEOUT_MAX_SECONDS,
        max(_COMPACTION_REQUEST_TIMEOUT_MIN_SECONDS, parsed),
    )


def _finish_reason(response: Any) -> str:
    for container in (
        getattr(response, "response_metadata", None),
        getattr(response, "additional_kwargs", None),
    ):
        if not isinstance(container, dict):
            continue
        value = container.get("finish_reason") or container.get("stop_reason")
        if value:
            return str(value).strip().lower()
    return ""


def _strict_tool_sections(response: Any) -> dict[str, Any]:
    finish_reason = _finish_reason(response)
    if finish_reason in {"length", "max_tokens", "max_output_tokens"}:
        raise SessionCompactionOutputLimitError(
            "session compactor reached the provider output limit"
        )
    tool_calls = list(getattr(response, "tool_calls", None) or [])
    if len(tool_calls) != 1:
        raise SessionCompactionFormatError(
            "session compactor did not return exactly one checkpoint tool call"
        )
    call = tool_calls[0]
    if not isinstance(call, dict) or str(call.get("name") or "") != (
        SubmitSessionCheckpoint.__name__
    ):
        raise SessionCompactionFormatError(
            "session compactor returned the wrong checkpoint tool call"
        )
    arguments = call.get("args")
    if not isinstance(arguments, dict):
        raise SessionCompactionFormatError(
            "session compactor returned undecodable checkpoint arguments"
        )
    return _validated_sections(arguments)


def _retryable_transport_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, (asyncio.TimeoutError, TimeoutError, httpx.TransportError)):
            return True
        if current.__class__.__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "ServiceUnavailableError",
        }:
            return True
        try:
            status_code = int(getattr(current, "status_code", 0) or 0)
        except (TypeError, ValueError):
            status_code = 0
        if status_code == 408 or status_code >= 500:
            return True
        current = current.__cause__ or current.__context__
    return False


def _disabled_thinking_extra_body(model: Any) -> dict[str, Any]:
    current = getattr(model, "extra_body", None)
    payload = dict(current) if isinstance(current, dict) else {}
    payload["thinking"] = {"type": "disabled"}
    return payload


def _fresh_strict_model(model: Any) -> tuple[Any, bool]:
    """Return an independently owned non-streaming strict-call transport."""

    fresh_copy = getattr(model, "fresh_transport_copy", None)
    if not callable(fresh_copy):
        return model, False
    updates: dict[str, Any] = {
        "extra_body": _disabled_thinking_extra_body(model),
        "max_retries": 0,
        "streaming": False,
        "disable_streaming": True,
    }
    try:
        from langchain_deepseek.chat_models import (
            DEFAULT_API_BASE,
            DEFAULT_BETA_API_BASE,
        )

        api_base = str(getattr(model, "api_base", "") or "").rstrip("/")
        if api_base == str(DEFAULT_API_BASE).rstrip("/"):
            updates["api_base"] = DEFAULT_BETA_API_BASE
    except (ImportError, AttributeError):
        pass
    return fresh_copy(**updates), True


def _bind_strict_checkpoint_tool(model: Any) -> Any:
    bind_tools = getattr(model, "bind_tools", None)
    if not callable(bind_tools):
        raise SessionCompactionError(
            "production session compaction requires strict tool-calling support"
        )
    kwargs: dict[str, Any] = {
        "tool_choice": SubmitSessionCheckpoint.__name__,
        "strict": True,
        "parallel_tool_calls": False,
    }
    if not callable(getattr(model, "fresh_transport_copy", None)):
        kwargs["extra_body"] = _disabled_thinking_extra_body(model)
    return bind_tools([SubmitSessionCheckpoint], **kwargs)


def _invoke_bound(bound: Any, prompt: Sequence[BaseMessage]) -> Any:
    return bound.invoke(
        list(prompt),
        config={"callbacks": [], "tags": [SESSION_COMPACTION_MODEL_TAG]},
    )


async def _ainvoke_bound(bound: Any, prompt: Sequence[BaseMessage]) -> Any:
    return await bound.ainvoke(
        list(prompt),
        config={"callbacks": [], "tags": [SESSION_COMPACTION_MODEL_TAG]},
    )


def _close_sync_owned_transport(model: Any) -> None:
    from src.services.model_transport import close_model_transport_sync

    close_model_transport_sync(model)


class StrictToolSessionCompactionProtocol:
    """DeepSeek strict function protocol with independent retry budgets."""

    def invoke(
        self,
        model: Any,
        prompt: Sequence[BaseMessage],
    ) -> SessionCompactionProtocolResult:
        template = model
        current, owned = _fresh_strict_model(template)
        format_retries = 0
        reconnects = 0
        active_prompt = list(prompt)
        try:
            while True:
                bound = _bind_strict_checkpoint_tool(current)
                try:
                    response = _invoke_bound(bound, active_prompt)
                except Exception as exc:
                    if not _retryable_transport_error(exc):
                        raise SessionCompactionError(
                            "session compactor provider request failed: "
                            + exc.__class__.__name__
                        ) from exc
                    if reconnects >= _TRANSPORT_MAX_RECONNECTS:
                        raise SessionCompactionTransportError(
                            "session compactor transport reconnect budget exhausted: "
                            + exc.__class__.__name__
                        ) from exc
                    if owned:
                        _close_sync_owned_transport(current)
                    replacement, replacement_owned = _fresh_strict_model(template)
                    if not replacement_owned or replacement is current:
                        raise SessionCompactionTransportError(
                            "session compactor cannot create a fresh transport"
                        ) from exc
                    current, owned = replacement, replacement_owned
                    reconnects += 1
                    continue
                try:
                    sections = _strict_tool_sections(response)
                except SessionCompactionOutputLimitError:
                    raise
                except SessionCompactionFormatError:
                    if format_retries >= _FORMAT_CORRECTION_MAX_RETRIES:
                        raise
                    format_retries += 1
                    active_prompt = [*prompt, _FORMAT_CORRECTION]
                    continue
                return SessionCompactionProtocolResult(
                    sections=sections,
                    stats={
                        "compaction_protocol": "strict_tool_call",
                        "compaction_format_retry_count": format_retries,
                        "compaction_transport_reconnect_count": reconnects,
                    },
                )
        finally:
            if owned:
                _close_sync_owned_transport(current)

    async def ainvoke(
        self,
        model: Any,
        prompt: Sequence[BaseMessage],
    ) -> SessionCompactionProtocolResult:
        from src.services.model_transport import close_model_transport

        template = model
        current, owned = _fresh_strict_model(template)
        format_retries = 0
        reconnects = 0
        active_prompt = list(prompt)
        try:
            while True:
                bound = _bind_strict_checkpoint_tool(current)
                try:
                    response = await asyncio.wait_for(
                        _ainvoke_bound(bound, active_prompt),
                        timeout=_compaction_request_timeout_seconds(),
                    )
                except Exception as exc:
                    if not _retryable_transport_error(exc):
                        raise SessionCompactionError(
                            "session compactor provider request failed: "
                            + exc.__class__.__name__
                        ) from exc
                    if reconnects >= _TRANSPORT_MAX_RECONNECTS:
                        raise SessionCompactionTransportError(
                            "session compactor transport reconnect budget exhausted: "
                            + exc.__class__.__name__
                        ) from exc
                    if owned:
                        await close_model_transport(current)
                    replacement, replacement_owned = _fresh_strict_model(template)
                    if not replacement_owned or replacement is current:
                        raise SessionCompactionTransportError(
                            "session compactor cannot create a fresh transport"
                        ) from exc
                    current, owned = replacement, replacement_owned
                    reconnects += 1
                    continue
                try:
                    sections = _strict_tool_sections(response)
                except SessionCompactionOutputLimitError:
                    raise
                except SessionCompactionFormatError:
                    if format_retries >= _FORMAT_CORRECTION_MAX_RETRIES:
                        raise
                    format_retries += 1
                    active_prompt = [*prompt, _FORMAT_CORRECTION]
                    continue
                return SessionCompactionProtocolResult(
                    sections=sections,
                    stats={
                        "compaction_protocol": "strict_tool_call",
                        "compaction_format_retry_count": format_retries,
                        "compaction_transport_reconnect_count": reconnects,
                    },
                )
        finally:
            if owned:
                await close_model_transport(current)


class JsonTextSessionCompactionTestProtocol:
    """Explicit adapter for deterministic unit fakes; never selected by production."""

    def invoke(
        self,
        model: Any,
        prompt: Sequence[BaseMessage],
    ) -> SessionCompactionProtocolResult:
        try:
            response = model.invoke(list(prompt), config={"callbacks": []})
        except TypeError:
            response = model.invoke(list(prompt))
        return SessionCompactionProtocolResult(
            sections=_parse_text_sections(response),
            stats={
                "compaction_protocol": "explicit_test_json_text",
                "compaction_format_retry_count": 0,
                "compaction_transport_reconnect_count": 0,
            },
        )

    async def ainvoke(
        self,
        model: Any,
        prompt: Sequence[BaseMessage],
    ) -> SessionCompactionProtocolResult:
        try:
            response = await model.ainvoke(list(prompt), config={"callbacks": []})
        except TypeError:
            response = await model.ainvoke(list(prompt))
        return SessionCompactionProtocolResult(
            sections=_parse_text_sections(response),
            stats={
                "compaction_protocol": "explicit_test_json_text",
                "compaction_format_retry_count": 0,
                "compaction_transport_reconnect_count": 0,
            },
        )


def _build_checkpoint(
    *,
    sections: dict[str, Any],
    mechanical_state: dict[str, Any],
    previous_checkpoint: dict[str, Any] | None,
    plan: SessionCompactionPlan,
    request_id: str,
    completion_authority: dict[str, Any] | None,
) -> dict[str, Any]:
    previous = previous_checkpoint if isinstance(previous_checkpoint, dict) else {}
    newly_compacted_count = len(plan.compacted_prefix)
    total_covered_count = plan.already_covered_count + newly_compacted_count
    active_prefix = plan.source_messages[:total_covered_count]
    previous_turn_id = int(previous.get("covered_through_turn_id") or 0)
    covered_turn_id = max(
        [previous_turn_id, *[_turn_id(message) for message in plan.compacted_prefix]]
    )
    generation = max(0, int(previous.get("generation") or 0)) + 1
    previous_mechanical = _model_visible_mechanical_state(
        previous.get("mechanical_state")
    )
    current_mechanical = _model_visible_mechanical_state(
        mechanical_state
    )
    cumulative_mechanical = {
        key: _dedupe_rows(
            [
                *list(previous_mechanical.get(key) or []),
                *list(current_mechanical.get(key) or []),
            ]
        )
        for key in (
            "tool_receipts",
            "resource_refs",
            "citation_bindings",
            "side_effect_receipts",
        )
    }
    previous_authority = completion_authority_from_checkpoint(previous)
    if (
        previous_authority is not None
        and completion_authority is not None
        and _json_text(previous_authority) != _json_text(completion_authority)
    ):
        raise SessionCompactionError(
            "Server CompletionReceipt changed across compact generations"
        )
    cumulative_completion_authority = completion_authority or previous_authority
    payload: dict[str, Any] = {
        "schema_version": SESSION_COMPACT_SCHEMA,
        "generation": generation,
        "previous_checkpoint_id": str(previous.get("checkpoint_id") or "") or None,
        "created_by_request_id": str(request_id or ""),
        "covered_through_turn_id": covered_turn_id,
        "compacted_message_count": int(previous.get("compacted_message_count") or 0)
        + newly_compacted_count,
        "newly_compacted_message_count": newly_compacted_count,
        "compacted_prefix_digest": _messages_digest(plan.compacted_prefix),
        "active_request_coverage": {
            "request_id": str(request_id or ""),
            "message_count": total_covered_count,
            "prefix_digest": _messages_digest(active_prefix),
        },
        "mechanical_state": cumulative_mechanical,
        **sections,
    }
    if cumulative_completion_authority is not None:
        previous_runtime_control = (
            previous.get(_RUNTIME_CONTROL_KEY)
            if isinstance(previous.get(_RUNTIME_CONTROL_KEY), dict)
            else {}
        )
        previous_task_authority = previous_runtime_control.get(
            _TASK_AUTHORITY_KEY
        )
        if isinstance(previous_task_authority, dict):
            checkpoint_task_authority = _validated_task_authority_control(
                previous_task_authority,
                transport_request_id=str(request_id or ""),
                completion_authority=cumulative_completion_authority,
            )
        else:
            receipt = cumulative_completion_authority.get("completion_receipt")
            receipt_request_id = (
                str(receipt.get("request_id") or "").strip()
                if isinstance(receipt, dict)
                else ""
            )
            receipt_run_id = (
                str(receipt.get("run_id") or "").strip()
                if isinstance(receipt, dict)
                else ""
            )
            if (
                not receipt_request_id
                or receipt_request_id != str(request_id or "")
                or not receipt_run_id
            ):
                raise SessionCompactionError(
                    "cross-request completion authority requires explicit task authority"
                )
            checkpoint_task_authority = {
                "schema_version": "evoengine.task-authority-control/v1",
                "transport_request_id": str(request_id or ""),
                "task_authority_request_id": receipt_request_id,
                "task_run_id": receipt_run_id,
                "continuation_parent_request_id": None,
            }
        payload[_RUNTIME_CONTROL_KEY] = {
            _TASK_AUTHORITY_KEY: checkpoint_task_authority,
            _COMPLETION_AUTHORITY_KEY: cumulative_completion_authority,
        }
    id_payload = dict(payload)
    payload["checkpoint_id"] = "session_checkpoint_" + hashlib.sha256(
        _json_text(id_payload).encode("utf-8")
    ).hexdigest()[:24]
    return payload


def checkpoint_model_message(checkpoint: dict[str, Any]) -> HumanMessage:
    if checkpoint.get("schema_version") != SESSION_COMPACT_SCHEMA:
        raise SessionCompactionError("unsupported session compact checkpoint schema")
    model_checkpoint = _model_visible_checkpoint(checkpoint)
    if model_checkpoint is None:
        raise SessionCompactionError("invalid session compact checkpoint")
    return HumanMessage(
        content=(
            "[SESSION_COMPACT_CHECKPOINT]\n"
            "This is a validated cumulative handoff for older closed history. "
            "Stable IDs in mechanical_state are machine-derived; narrative sections "
            "are semantic continuity, not proof that the root task is complete.\n"
            + _json_text(model_checkpoint)
            + "\n[/SESSION_COMPACT_CHECKPOINT]"
        ),
        additional_kwargs={
            "evo_context_kind": _CHECKPOINT_MESSAGE_KIND,
            "checkpoint_id": str(checkpoint.get("checkpoint_id") or ""),
        },
    )


class SessionCompactor:
    """Generate validated cumulative checkpoints with the configured model."""

    def __init__(
        self,
        model: Any,
        *,
        request_id: str = "",
        protocol: SessionCompactionProtocol | None = None,
    ) -> None:
        self._model = model
        self._request_id = str(request_id or "")
        self._protocol = protocol or StrictToolSessionCompactionProtocol()

    def _prepare(
        self,
        messages: Sequence[BaseMessage],
        *,
        previous_checkpoint: dict[str, Any] | None,
        retain_recent_tokens: int,
    ) -> tuple[
        SessionCompactionPlan,
        dict[str, Any],
        dict[str, Any] | None,
        list[BaseMessage],
        dict[str, Any] | None,
    ]:
        if self._model is None:
            raise SessionCompactionError("session compaction requires a summary model")
        source = tuple(
            message for message in messages if not _is_checkpoint_message(message)
        )
        effective_previous = (
            previous_checkpoint
            if _covered_prefix_count(
                source,
                previous_checkpoint,
                request_id=self._request_id,
            )
            > 0
            else None
        )
        plan = plan_session_compaction(
            messages,
            checkpoint=effective_previous,
            request_id=self._request_id,
            retain_recent_tokens=retain_recent_tokens,
        )
        mechanical = extract_mechanical_state(plan.compacted_prefix)
        completion_authority = _extract_completion_authority(plan.source_messages)
        prompt = _summary_prompt(
            previous_checkpoint=effective_previous,
            compacted_messages=plan.compacted_prefix,
            mechanical_state=mechanical,
        )
        return plan, mechanical, completion_authority, prompt, effective_previous

    def compact(
        self,
        messages: Sequence[BaseMessage],
        *,
        previous_checkpoint: dict[str, Any] | None,
        retain_recent_tokens: int,
    ) -> SessionCompactionResult:
        plan, mechanical, completion_authority, prompt, effective_previous = self._prepare(
            messages,
            previous_checkpoint=previous_checkpoint,
            retain_recent_tokens=retain_recent_tokens,
        )
        protocol_result = self._protocol.invoke(self._model, prompt)
        checkpoint = _build_checkpoint(
            sections=protocol_result.sections,
            mechanical_state=mechanical,
            previous_checkpoint=effective_previous,
            plan=plan,
            request_id=self._request_id,
            completion_authority=completion_authority,
        )
        result_messages = (
            checkpoint_model_message(checkpoint),
            *plan.retained_messages,
        )
        return SessionCompactionResult(
            messages=tuple(result_messages),
            checkpoint=checkpoint,
            stats={
                **_compaction_stats(plan, result_messages, checkpoint),
                **protocol_result.stats,
            },
        )

    async def acompact(
        self,
        messages: Sequence[BaseMessage],
        *,
        previous_checkpoint: dict[str, Any] | None,
        retain_recent_tokens: int,
    ) -> SessionCompactionResult:
        plan, mechanical, completion_authority, prompt, effective_previous = self._prepare(
            messages,
            previous_checkpoint=previous_checkpoint,
            retain_recent_tokens=retain_recent_tokens,
        )
        protocol_result = await self._protocol.ainvoke(self._model, prompt)
        checkpoint = _build_checkpoint(
            sections=protocol_result.sections,
            mechanical_state=mechanical,
            previous_checkpoint=effective_previous,
            plan=plan,
            request_id=self._request_id,
            completion_authority=completion_authority,
        )
        result_messages = (
            checkpoint_model_message(checkpoint),
            *plan.retained_messages,
        )
        return SessionCompactionResult(
            messages=tuple(result_messages),
            checkpoint=checkpoint,
            stats={
                **_compaction_stats(plan, result_messages, checkpoint),
                **protocol_result.stats,
            },
        )


def _compaction_stats(
    plan: SessionCompactionPlan,
    messages: Sequence[BaseMessage],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    return {
        "compaction_triggered": True,
        "checkpoint_id": str(checkpoint.get("checkpoint_id") or ""),
        "checkpoint_generation": int(checkpoint.get("generation") or 0),
        "compacted_message_count": len(plan.compacted_prefix),
        "already_covered_message_count": plan.already_covered_count,
        "retained_message_count": len(plan.retained_messages),
        "source_tokens_estimated": estimate_message_tokens(plan.source_messages),
        "compacted_prefix_tokens_estimated": estimate_message_tokens(
            plan.compacted_prefix
        ),
        "result_tokens_estimated": estimate_message_tokens(messages),
    }
