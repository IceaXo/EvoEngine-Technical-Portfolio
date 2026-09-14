from __future__ import annotations

import asyncio
import ast
import hashlib
import inspect
import json
import logging
import os
import re
import sys
import time
import traceback
import uuid
from typing import Any, AsyncIterator, Callable, TypedDict

import httpx

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from src.agents.canonical_protocol_ledger import (
    CanonicalProtocolLedger,
    CanonicalRuntimeTransition,
)
from src.agents.lead_agent import (
    _CAPABILITY_PREFETCH_CLOSE,
    _CAPABILITY_PREFETCH_OPEN,
    EVO_AGENT_RECURSION_LIMIT,
    _EVO_MARKER_RE,
    _StreamMarkerStripper,
    build_citations_from_tool_output,
    verified_sources_from_tool_output,
    build_dynamic_agent,
    build_tool_input_preview,
    build_tool_output_preview,
    extract_tool_items,
    get_runtime_cache_policy,
    get_runtime_dynamic_entry_ids,
    get_runtime_resident_tool_ids,
    summarize_tool_done,
    summarize_tool_start,
    _expand_loaded_tool_ids,
)
from src.agents.timeline_graph import CitationRegistry, TimelineEventAdapter
from src.config.settings import Settings, is_native_reasoning_model
from src.capabilities.models import (
    CapabilityCall,
    CapabilityNeedsInput,
    CapabilityRegistrySnapshot,
)
from src.context.task_state import TaskPhase, TaskState, transition_task_state
from src.schemas.api import AgentInvokeRequest
from src.schemas.hitl import HumanQuestionBundle
from src.services.context_ledger import (
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    build_context_plan,
    build_context_usage,
    build_tool_result_context,
    estimate_tokens,
)
from src.services.resource_access_ledger import (
    normalize_resource_access_ledger,
    update_resource_access_ledger_from_messages,
)
from src.services.model_usage_ledger import (
    append_model_call,
    build_context_snapshot_id,
    build_model_call_record,
    extract_model_usage,
    next_model_call_index,
    normalize_usage_ledger,
)
from src.services.session_compaction import (
    SESSION_COMPACTION_MODEL_TAG,
    attach_completion_authority_to_checkpoint,
    completion_authority_for_request,
)
from src.services.orchestrator import (
    compose_user_prompt,
    compose_user_prompt_ledger_parts,
    history_protocol_messages,
)
from src.services.tool_result_envelope import (
    infer_tool_completion_signal,
    normalize_capability_outcome,
    normalize_tool_completion_signal,
)
from src.services.task_tree_client import (
    build_task_tree_snapshot_brief,
    build_task_tree_summary,
    get_latest_snapshot,
    is_terminal_snapshot,
    reconcile_sandbox_nodes_from_authoritative_jobs,
)
from src.services.tool_schema_ledger import (
    build_initial_tool_load_metrics,
    normalize_tool_load_metrics,
    record_tool_load_request,
    record_tool_schema_snapshot,
)
from src.skills.registry import get_skill_registry
from src.runtime.worker_diagnostics import emit_worker_checkpoint as _worker_checkpoint

logger = logging.getLogger(__name__)


class NativeModelStreamStall(TimeoutError):
    def __init__(
        self,
        *,
        call_index: int,
        phase: str,
        timeout_s: float,
        elapsed_ms: int | None,
        since_last_progress_ms: int | None,
    ) -> None:
        self.call_index = int(call_index or 0)
        self.phase = str(phase or "")
        self.timeout_s = float(timeout_s or 0)
        self.elapsed_ms = elapsed_ms
        self.since_last_progress_ms = since_last_progress_ms
        detail = f"model stream stalled after {self.timeout_s:g}s"
        if self.call_index:
            detail += f" on call_index={self.call_index}"
        if self.phase:
            detail += f", phase={self.phase}"
        if elapsed_ms is not None:
            detail += f", elapsed_ms={elapsed_ms}"
        if since_last_progress_ms is not None:
            detail += f", since_last_progress_ms={since_last_progress_ms}"
        super().__init__(detail)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return float(default)


def _native_stream_first_chunk_timeout_seconds() -> float:
    return max(0.0, _float_env("EVO_NATIVE_STREAM_FIRST_CHUNK_TIMEOUT_SEC", 15.0))


def _native_stream_stall_timeout_seconds() -> float:
    return max(0.0, _float_env("EVO_NATIVE_STREAM_STALL_TIMEOUT_SEC", 12.0))


def _native_stream_recovery_max_attempts() -> int:
    raw = os.getenv("EVO_NATIVE_STREAM_RECOVERY_MAX_ATTEMPTS", "3")
    try:
        return max(0, int(str(raw).strip()))
    except (TypeError, ValueError):
        return 3


def _native_stream_recovery_total_max_attempts() -> int:
    """Return the hard whole-turn cap while per-episode retries stay bounded.

    A successful model call resets the consecutive counter.  Long tasks can
    therefore survive independent provider stalls without losing the
    historical counter used for observability and the absolute safety cap.
    """

    per_episode = _native_stream_recovery_max_attempts()
    raw = os.getenv(
        "EVO_NATIVE_STREAM_RECOVERY_TOTAL_MAX_ATTEMPTS",
        str(max(6, per_episode * 4)) if per_episode > 0 else "0",
    )
    try:
        return max(0, int(str(raw).strip()))
    except (TypeError, ValueError):
        return max(6, per_episode * 4) if per_episode > 0 else 0


def _native_stream_recovery_timeout_seconds() -> float:
    return max(0.0, _float_env("EVO_NATIVE_STREAM_RECOVERY_TIMEOUT_SEC", 15.0))


def _native_stream_transport_close_timeout_seconds() -> float:
    return max(0.1, _float_env("EVO_NATIVE_STREAM_TRANSPORT_CLOSE_TIMEOUT_SEC", 2.0))


def _consume_background_task_result(task: asyncio.Future[Any]) -> None:
    try:
        task.exception()
    except BaseException:  # cancelled/failed cleanup must not leak a warning
        pass


async def _close_event_stream_bounded(event_stream: Any) -> dict[str, Any]:
    """Best-effort close of one stalled iterator without blocking recovery."""

    close_stream = getattr(event_stream, "aclose", None)
    if not callable(close_stream):
        return {
            "attempted": False,
            "closed": False,
            "error_type": "CloseMethodUnavailable",
        }
    try:
        maybe_awaitable = close_stream()
    except Exception as exc:  # noqa: BLE001
        return {
            "attempted": True,
            "closed": False,
            "error_type": exc.__class__.__name__,
        }
    if not inspect.isawaitable(maybe_awaitable):
        return {"attempted": True, "closed": True, "error_type": ""}

    close_task = asyncio.ensure_future(maybe_awaitable)
    try:
        done, _pending = await asyncio.wait(
            {close_task},
            timeout=_native_stream_transport_close_timeout_seconds(),
        )
    except asyncio.CancelledError:
        close_task.cancel()
        close_task.add_done_callback(_consume_background_task_result)
        raise
    if close_task not in done:
        close_task.cancel()
        close_task.add_done_callback(_consume_background_task_result)
        return {"attempted": True, "closed": False, "error_type": "TimeoutError"}
    try:
        close_task.result()
    except BaseException as exc:  # cancellation/error is observable but non-blocking
        return {
            "attempted": True,
            "closed": False,
            "error_type": exc.__class__.__name__,
        }
    return {"attempted": True, "closed": True, "error_type": ""}


def _retryable_model_transport_error(exc: BaseException) -> bool:
    """Classify transient model transport/provider failures, not task errors."""

    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, (asyncio.TimeoutError, httpx.TransportError)):
            return True
        error_name = current.__class__.__name__
        if error_name in {
            "APIConnectionError",
            "APITimeoutError",
            "RateLimitError",
            "InternalServerError",
            "ServiceUnavailableError",
        }:
            return True
        status_code = getattr(current, "status_code", None)
        try:
            status = int(status_code)
        except (TypeError, ValueError):
            status = 0
        if status in {408, 409, 429} or status >= 500:
            return True
        current = current.__cause__ or current.__context__
    return False


def _native_fake_stream_delay_seconds() -> float:
    return max(0.0, _float_env("EVO_NATIVE_FAKE_STREAM_DELAY_SEC", 0.022))


def _native_fake_stream_chunk_chars(total_chars: int) -> int:
    raw = os.getenv("EVO_NATIVE_FAKE_STREAM_CHUNK_CHARS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            pass
    if total_chars >= 6000:
        return 24
    if total_chars >= 2500:
        return 16
    if total_chars >= 1000:
        return 10
    return 6




class NativeAgentContext(TypedDict, total=False):
    settings: Settings
    llm_base: Any
    llm_factory: Callable[[], Any]
    llm_transport_generation: int
    llm_transport_owner: dict[str, Any]
    llm_protected_transport_ids: frozenset[int]
    feature_flags: dict[str, Any]
    project_id: str | None
    conversation_id: str | None
    user_id: int | None
    # Immutable request/turn-local catalog decision.  It intentionally lives
    # outside NativeAgentState so it is neither serialized nor exposed in
    # timeline/HITL payloads, while shallow recovery copies preserve identity.
    sandbox_exposure_context: Any


class NativeAgentState(TypedDict, total=False):
    request_id: str
    task_authority: dict[str, Any]
    task_authority_request_id: str
    task_authority_run_id: str | None
    user_text: str
    project_summary: str
    system_prompt: str
    history: list[dict[str, Any]]
    knowledge: dict[str, Any]
    database: dict[str, Any]
    attachments: list[dict[str, Any]]
    conversation_files: list[dict[str, Any]]
    reference_context: dict[str, Any] | None
    project_id: str | None
    conversation_id: str | None
    user_id: int | None
    pending_question_bundle: dict[str, Any] | None
    pending_sandbox_jobs: list[dict[str, Any]]
    sandbox_resume_payload: dict[str, Any] | None
    sandbox_promotion_results: list[dict[str, Any]]
    human_answer_bundle: dict[str, Any] | None
    capability_resume_control: dict[str, Any] | None
    pending_capability_authorization: dict[str, Any] | None
    turn_checkpoint: dict[str, Any] | None
    session_compact_checkpoint: dict[str, Any]
    resource_access_ledger: dict[str, Any]
    timeline_blocks: list[dict[str, Any]]
    usage: dict[str, Any]
    citations: list[dict[str, Any]]
    final_reply_text: str
    completion_receipt: dict[str, Any]
    reply_incomplete: bool
    error_message: str
    resume_count: int
    # Consecutive stalls are reset by a completed model call.  The cumulative
    # counter is request-local and never resets, so recursive recovery cannot
    # silently renew the configured recovery budget.
    stream_recovery_attempts: int
    stream_recovery_total_attempts: int
    stream_recovery_mode: bool
    stream_recovery_replay_prefix: str
    recovery_state_envelope: dict[str, Any] | None
    truncated_answer_continuation_attempts: int
    final_answer_recovery_attempts: int
    tool_outcomes: list[dict[str, Any]]
    side_effect_ledger: dict[str, Any]
    continuation_messages: list[Any]
    hitl_resume_messages: list[Any]
    loaded_tool_ids: list[str]
    loaded_tool_entries: list[str]
    activated_skills: list[dict[str, Any]]
    tool_cache_scope: str
    tool_cache_turn: int
    task_phase: str
    task_state: dict[str, Any]
    tool_schema_reload_count: int
    tool_schema_snapshot: dict[str, Any]
    tool_load_metrics: dict[str, Any]
    max_turns: int | None
    model_turns_used: int
    status: str
    # In-process only.  The serializable protocol checkpoint is always the
    # ledger snapshot, never this lock-bearing runtime object.
    _canonical_protocol_ledger: Any
    _completion_receipt_seal_emitted: bool
    _terminal_completion_emitted: bool
    _canonical_any_tool_called: bool


_DYNAMIC_TOOL_CACHE: dict[str, dict[str, Any]] = {}
_DYNAMIC_TOOL_CACHE_TURN = 0


def _public_task_completion_receipt(value: Any) -> dict[str, Any] | None:
    """Expose only the Server-issued TaskTree receipt on terminal events.

    A generic accepted control envelope may seal an ordinary no-tree answer
    inside the runtime, but it is not a Server authority and must not be
    projected as an immutable CompletionReceipt.
    """

    if not isinstance(value, dict):
        return None
    if (
        str(value.get("schema_version") or "")
        != "evoengine.task-completion-receipt/v1"
        or str(value.get("authority") or "") != "task_tree_internal_api"
    ):
        return None
    return dict(value)


def _task_terminal_status_from_receipt(value: Any) -> str:
    receipt = value if isinstance(value, dict) else {}
    status = str(receipt.get("terminal_status") or "completed").strip().lower()
    return "blocked" if status == "blocked" else "completed"


def _next_dynamic_tool_cache_turn() -> int:
    global _DYNAMIC_TOOL_CACHE_TURN
    _DYNAMIC_TOOL_CACHE_TURN += 1
    return _DYNAMIC_TOOL_CACHE_TURN


def _tool_cache_scope(request: AgentInvokeRequest | Any) -> str:
    return "|".join(
        [
            str(getattr(request, "project_id", None) or ""),
            str(getattr(request, "conversation_id", None) or ""),
            str(getattr(request, "user_id", None) or ""),
        ]
    )


def _dynamic_cache_entry_expanded_tools(entry: str) -> set[str]:
    return set(_expand_loaded_tool_ids([str(entry or "").strip()]))


def _prune_dynamic_tool_cache(scope: str, *, turn: int) -> list[str]:
    policy = get_runtime_cache_policy()
    ttl_turns = int(policy.get("ttl_turns") or 2)
    max_entries = int(policy.get("max_entries") or 8)
    bucket = _DYNAMIC_TOOL_CACHE.setdefault(scope, {"entries": {}})
    entries = bucket.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        bucket["entries"] = entries
    pruned: dict[str, dict[str, Any]] = {}
    for raw_entry, raw_meta in entries.items():
        entry = str(raw_entry or "").strip()
        if not entry:
            continue
        meta = raw_meta if isinstance(raw_meta, dict) else {}
        try:
            last_used = int(meta.get("last_used_turn") or meta.get("loaded_turn") or 0)
        except (TypeError, ValueError):
            last_used = 0
        if turn - last_used > ttl_turns:
            continue
        pruned[entry] = {
            "loaded_turn": int(meta.get("loaded_turn") or last_used or turn),
            "last_used_turn": last_used or turn,
        }
    ordered = sorted(
        pruned.items(),
        key=lambda item: (
            int(item[1].get("last_used_turn") or 0),
            int(item[1].get("loaded_turn") or 0),
        ),
        reverse=True,
    )[:max_entries]
    bucket["entries"] = {entry: meta for entry, meta in ordered}
    return [entry for entry, _meta in ordered]


def _build_initial_tool_cache_state(request: AgentInvokeRequest | Any) -> dict[str, Any]:
    scope = _tool_cache_scope(request)
    turn = _next_dynamic_tool_cache_turn()
    entries = _prune_dynamic_tool_cache(scope, turn=turn)
    loaded_tool_ids = _expand_loaded_tool_ids(entries)
    return {
        "tool_cache_scope": scope,
        "tool_cache_turn": turn,
        "loaded_tool_entries": entries,
        "loaded_tool_ids": loaded_tool_ids,
        "tool_schema_snapshot": {},
        "tool_load_metrics": build_initial_tool_load_metrics(cache_hit=bool(entries)),
    }


def _store_dynamic_tool_cache_entries(
    *,
    scope: str,
    turn: int,
    entries: list[str],
) -> None:
    if not scope or not turn:
        return
    policy = get_runtime_cache_policy()
    max_entries = int(policy.get("max_entries") or 8)
    bucket = _DYNAMIC_TOOL_CACHE.setdefault(scope, {"entries": {}})
    cached = bucket.setdefault("entries", {})
    if not isinstance(cached, dict):
        cached = {}
        bucket["entries"] = cached
    for entry in get_runtime_dynamic_entry_ids(entries):
        cached[str(entry)] = {
            "loaded_turn": int(turn),
            "last_used_turn": int(turn),
        }
    ordered = sorted(
        cached.items(),
        key=lambda item: (
            int((item[1] if isinstance(item[1], dict) else {}).get("last_used_turn") or 0),
            int((item[1] if isinstance(item[1], dict) else {}).get("loaded_turn") or 0),
        ),
        reverse=True,
    )[:max_entries]
    bucket["entries"] = {entry: meta for entry, meta in ordered}


def _evict_dynamic_tool_cache_ids(scope: str, tool_ids: set[str]) -> list[str]:
    """Drop cached entries whose expanded tool set contains a rejected ID."""

    if not scope or not tool_ids:
        return []
    bucket = _DYNAMIC_TOOL_CACHE.get(scope)
    if not isinstance(bucket, dict):
        return []
    cached = bucket.get("entries")
    if not isinstance(cached, dict):
        return []
    removed: list[str] = []
    for raw_entry in list(cached):
        entry = str(raw_entry or "").strip()
        if entry in tool_ids or _dynamic_cache_entry_expanded_tools(entry).intersection(tool_ids):
            cached.pop(raw_entry, None)
            if entry:
                removed.append(entry)
    return removed


def _mark_dynamic_tool_cache_used(
    *,
    scope: str,
    turn: int,
    loaded_entries: list[str],
    tool_name: str,
) -> None:
    if not scope or not turn or not tool_name:
        return
    bucket = _DYNAMIC_TOOL_CACHE.setdefault(scope, {"entries": {}})
    cached = bucket.setdefault("entries", {})
    if not isinstance(cached, dict):
        return
    changed = False
    for entry in get_runtime_dynamic_entry_ids(loaded_entries):
        meta = cached.get(entry)
        if not isinstance(meta, dict):
            continue
        if str(tool_name) == str(entry) or str(tool_name) in _dynamic_cache_entry_expanded_tools(entry):
            meta["last_used_turn"] = int(turn)
            changed = True
    if changed:
        _prune_dynamic_tool_cache(scope, turn=turn)


def build_runtime_context(
    settings: Settings,
    llm_base: Any,
    *,
    llm_factory: Callable[[], Any] | None = None,
    project_id: str | None = None,
    conversation_id: str | None = None,
    user_id: int | None = None,
    feature_flags: dict[str, Any] | None = None,
) -> NativeAgentContext:
    context: NativeAgentContext = {
        "settings": settings,
        "llm_base": llm_base,
        "feature_flags": dict(feature_flags or {}),
        "llm_transport_generation": 0,
    }
    from src.services.model_transport import model_transport_identity

    context["llm_protected_transport_ids"] = model_transport_identity(llm_base)
    if callable(llm_factory):
        context["llm_factory"] = llm_factory
    if project_id is not None:
        context["project_id"] = str(project_id)
    if conversation_id is not None:
        context["conversation_id"] = str(conversation_id)
    if user_id is not None:
        context["user_id"] = int(user_id)
    return context


async def _close_owned_llm_transport(
    context: NativeAgentContext,
) -> dict[str, Any]:
    """Close a request-owned recovery client at most once.

    The worker's generation-0 model may be shared by concurrent requests and
    intentionally has no owner record, so this helper can never close it.
    """

    owner = context.get("llm_transport_owner")
    if not isinstance(owner, dict) or owner.get("state") == "closed":
        return {"owned": False, "attempted": 0, "closed": 0, "errors": []}
    if owner.get("state") == "closing":
        return {"owned": False, "attempted": 0, "closed": 0, "errors": []}
    owner["state"] = "closing"
    from src.services.model_transport import close_model_transport

    try:
        result = await asyncio.wait_for(
            close_model_transport(owner.get("model")),
            timeout=_native_stream_transport_close_timeout_seconds(),
        )
    except asyncio.CancelledError:
        owner["state"] = "open"
        raise
    except Exception as exc:  # noqa: BLE001
        result = {
            "attempted": 0,
            "closed": 0,
            "errors": [exc.__class__.__name__],
        }
    owner["close_result"] = dict(result)
    if not result.get("errors") and int(result.get("attempted") or 0) == int(
        result.get("closed") or 0
    ):
        owner["state"] = "closed"
    else:
        owner["state"] = "open"
    return {"owned": True, **result}


def _fresh_recovery_context(
    context: NativeAgentContext,
) -> tuple[NativeAgentContext, dict[str, Any] | None]:
    """Create the next request-local model transport when a factory exists."""

    factory = context.get("llm_factory")
    if not callable(factory):
        raise RuntimeError("model recovery factory is unavailable")
    current_model = context.get("llm_base")
    fresh_model = factory()
    if fresh_model is None:
        raise RuntimeError("model recovery factory returned no client")
    if fresh_model is current_model:
        raise RuntimeError("model recovery factory reused the stalled client")
    if getattr(fresh_model, "bound", None) is not None:
        raise RuntimeError("model recovery factory must return a direct model client")
    from src.services.model_transport import model_transport_identity

    fresh_transport_ids = model_transport_identity(fresh_model)
    current_transport_ids = model_transport_identity(current_model)
    protected_transport_ids = frozenset(
        context.get("llm_protected_transport_ids") or ()
    )
    if not fresh_transport_ids:
        raise RuntimeError("model recovery factory returned no transport identity")
    if fresh_transport_ids.intersection(current_transport_ids):
        raise RuntimeError("model recovery factory reused the current transport")
    if fresh_transport_ids.intersection(protected_transport_ids):
        raise RuntimeError("model recovery factory reused a protected shared transport")
    generation = int(context.get("llm_transport_generation") or 0) + 1
    owner: dict[str, Any] = {
        "model": fresh_model,
        "generation": generation,
        "state": "open",
    }
    recovered: NativeAgentContext = dict(context)
    recovered["llm_base"] = fresh_model
    recovered["llm_transport_generation"] = generation
    recovered["llm_transport_owner"] = owner
    return recovered, owner


def _adapter_from_state(state: NativeAgentState) -> TimelineEventAdapter:
    registry = CitationRegistry()
    for citation in state.get("citations") or []:
        if isinstance(citation, dict):
            registry.add(citation)
    return TimelineEventAdapter(
        str(state.get("request_id") or ""),
        registry,
        existing_blocks=list(state.get("timeline_blocks") or []),
        usage=dict(state.get("usage") or {}),
        reply_incomplete=bool(state.get("reply_incomplete")),
    )


def _hydrate_citation_registry_from_reference_store(
    adapter: TimelineEventAdapter,
    *,
    request_id: str,
    project_id: str | None,
    conversation_id: str | None,
) -> int:
    """Restore publishable citations for this request before finalization.

    Tool blocks are an event projection and may be rebuilt across schema reloads
    or stream recovery.  The request-scoped reference store is the durable
    source authority; loading it here only makes explicit ``[[cite:...]]``
    markers resolvable and does not attach unused references to the answer.
    """

    from src.agents.citation_settings import citations_enabled

    active_request = str(request_id or "").strip()
    active_project = str(project_id or "").strip()
    active_conversation = str(conversation_id or "").strip()
    if (
        not citations_enabled()
        or not active_request
        or not active_project
        or not active_conversation
    ):
        return 0

    from src.subagents.citation_policy import sanitize_citation_candidate
    from src.subagents.reference_store import load_reference_records
    from src.subagents.source_ledger import normalize_source_record

    added = 0
    records = load_reference_records(
        request_id=active_request,
        project_id=active_project,
        conversation_id=active_conversation,
    )
    for index, record in enumerate(records, 1):
        normalized = normalize_source_record(record, fallback_index=index)
        if normalized is None:
            continue
        candidate = sanitize_citation_candidate(normalized)
        if candidate is None:
            continue
        before = len(adapter.registry.citations)
        adapter.registry.add(candidate)
        if len(adapter.registry.citations) > before:
            added += 1
    return added


def _state_from_adapter(state: NativeAgentState, adapter: TimelineEventAdapter) -> dict[str, Any]:
    return {
        "timeline_blocks": adapter.snapshot_blocks(),
        "usage": dict(adapter.usage),
        "citations": list(adapter.registry.citations),
        "final_reply_text": adapter.final_text,
        "completion_receipt": dict(state.get("completion_receipt") or {}),
        "reply_incomplete": bool(adapter.reply_incomplete),
        "error_message": str(state.get("error_message") or ""),
        "task_phase": str(state.get("task_phase") or TaskPhase.INTAKE.value),
        "task_state": dict(state.get("task_state") or {}),
        "stream_recovery_attempts": int(state.get("stream_recovery_attempts") or 0),
        "stream_recovery_total_attempts": int(
            state.get("stream_recovery_total_attempts") or 0
        ),
        "stream_recovery_mode": bool(state.get("stream_recovery_mode")),
        "side_effect_ledger": _normalize_side_effect_ledger(
            state.get("side_effect_ledger")
        ),
        "resource_access_ledger": normalize_resource_access_ledger(
            state.get("resource_access_ledger")
        ),
        "reference_context": _normalize_reference_context_payload(
            state.get("reference_context")
        ),
        "model_turns_used": max(0, int(state.get("model_turns_used") or 0)),
    }


def _record_model_turn(state: NativeAgentState) -> None:
    """Record root-model turns for telemetry without imposing a call budget."""
    used = max(0, int(state.get("model_turns_used") or 0))
    state["model_turns_used"] = used + 1


def _execution_budget_prompt(state: NativeAgentState) -> str:
    # Kept as a compatibility hook for checkpoint readers.  Model-call counts
    # are telemetry only; timeout/cancellation own execution termination.
    return ""


def _question_bundle_reply_text(bundle: dict[str, Any] | None) -> str:
    """Render a bounded user-facing clarification without protocol JSON."""

    if not isinstance(bundle, dict):
        return "需要你补充关键信息后我才能继续。"
    lines = [
        str(bundle.get("bundle_title") or "需要你补充信息").strip(),
        str(bundle.get("bundle_summary") or "").strip(),
    ]
    for index, raw_question in enumerate((bundle.get("questions") or [])[:5], start=1):
        if not isinstance(raw_question, dict):
            continue
        label = str(
            raw_question.get("label") or raw_question.get("description") or ""
        ).strip()
        if not label:
            continue
        options = [
            str(option.get("label") or "").strip()
            for option in (raw_question.get("options") or [])[:8]
            if isinstance(option, dict) and str(option.get("label") or "").strip()
        ]
        lines.append(
            f"{index}. {label}"
            + (f"（可选：{'、'.join(options)}）" if options else "")
        )
    return "\n".join(line for line in lines if line)[:8000]


def _normalize_reference_context_payload(value: Any) -> dict[str, Any] | None:
    """Keep one canonical absence value across rebuild/checkpoint boundaries."""

    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("reference_context must be a mapping or None")
    return dict(value) or None


def _transition_native_task_state(
    state: NativeAgentState,
    phase: TaskPhase,
    **updates: Any,
) -> dict[str, Any]:
    current = state.get("task_state")
    if not isinstance(current, dict):
        current = TaskState(
            task_id=str(state.get("conversation_id") or state.get("request_id") or ""),
            request_id=str(state.get("request_id") or ""),
        ).model_dump(mode="json")
    task_state = transition_task_state(current, phase, **updates)
    payload = task_state.model_dump(mode="json")
    state["task_phase"] = task_state.phase.value
    state["task_state"] = payload
    return {"task_phase": task_state.phase.value, "task_state": payload}


def _task_state_prompt(state: NativeAgentState) -> str:
    try:
        task_state = TaskState.model_validate(state.get("task_state") or {})
    except Exception:
        return ""
    phase_objectives = {
        TaskPhase.INTAKE: "建立任务合同、选择主泳道并识别依赖",
        TaskPhase.CLARIFY: "等待或处理会改变结果的用户输入",
        TaskPhase.PLAN: "形成目标与验收契约，并把已知依赖映射到任务 DAG",
        TaskPhase.EXECUTE: "推进当前 ready frontier 并记录真实结果",
        TaskPhase.VERIFY: "对照根目标、验收字段和现有交付物检查缺项",
        TaskPhase.DELIVER: "交付已通过验收的结论、来源和结果文件",
        TaskPhase.SUSPENDED: "保留当前事实，等待用户输入或外部任务结果",
        TaskPhase.COMPLETED: "根目标已经通过验收",
        TaskPhase.FAILED: "记录失败事实并判断可重试依赖或可交付限制",
    }
    return (
        "[TASK_STATE]\n"
        f"phase={task_state.phase.value}\n"
        f"phase_objective={phase_objectives.get(task_state.phase, '')}\n"
        f"plan_ref={task_state.plan_ref or ''}\n"
        f"resume_reason={task_state.resume_reason}\n"
        "role=current_lifecycle_observation\n"
        "任务阶段描述当前执行位置；下一步由任务合同、任务树 frontier、工具结果和验收缺项共同决定。\n"
        "[/TASK_STATE]"
    )


def _split_trailing_json_candidate(text: str) -> tuple[str, str]:
    """Delay a possible tool-argument object until the next protocol event.

    The JSON is never classified from field names.  It is suppressed only if a
    subsequent real ``on_tool_start`` proves that the object contains that
    tool's actual input; otherwise it is emitted unchanged as user-visible
    answer text.
    """

    raw = str(text or "")
    stripped_offset = len(raw) - len(raw.lstrip())
    starts = [stripped_offset] if raw[stripped_offset:].startswith("{") else []
    newline_start = raw.rfind("\n{")
    if newline_start >= 0:
        starts.append(newline_start + 1)
    if not starts:
        return raw, ""
    start = min(starts)
    return raw[:start], raw[start:]


def _json_contains(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _json_contains(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _json_contains(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    return actual == expected


def _held_json_matches_tool_input(text: str, tool_input: Any) -> bool:
    try:
        candidate = json.loads(str(text or "").strip())
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(candidate, dict) or not isinstance(tool_input, dict) or not tool_input:
        return False
    return _json_contains(candidate, tool_input)


def _context_project_id(
    context: NativeAgentContext, state: NativeAgentState
) -> str | None:
    project_id = context.get("project_id")
    if project_id is None:
        project_id = state.get("project_id")
    value = str(project_id or "").strip()
    return value or None


def _context_user_id(
    context: NativeAgentContext, state: NativeAgentState
) -> int | None:
    user_id = context.get("user_id")
    if user_id is None:
        user_id = state.get("user_id")
    try:
        return int(user_id) if user_id is not None else None
    except (TypeError, ValueError):
        return None


def _context_conversation_id(
    context: NativeAgentContext, state: NativeAgentState
) -> str | None:
    conversation_id = context.get("conversation_id")
    if conversation_id is None:
        conversation_id = state.get("conversation_id")
    value = str(conversation_id or "").strip()
    return value or None


def _extract_full_output_text(output_obj: Any) -> str:
    if output_obj is None:
        return ""
    raw_content = getattr(output_obj, "content", None)
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts: list[str] = []
        for block in raw_content:
            if isinstance(block, dict):
                text = str(block.get("text") or "")
                if text:
                    parts.append(text)
        return "".join(parts)
    return ""


def _extract_usage(output_obj: Any) -> tuple[int, int, int, str]:
    usage = extract_model_usage(output_obj)
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    return input_tokens, output_tokens, input_tokens + output_tokens, str(usage.get("model") or "")


def _emit_events(writer, events: list[dict[str, Any]]) -> None:
    for event in events:
        writer(event)


def _has_answer_content(adapter: TimelineEventAdapter) -> bool:
    for block in adapter.snapshot_blocks():
        if str(block.get("kind") or "") != "answer":
            continue
        if str(block.get("status") or "") in {"discarded", "superseded"}:
            continue
        payload = block.get("payload") if isinstance(block.get("payload"), dict) else {}
        if str(payload.get("content") or "").strip():
            return True
    return False


def _adapter_answer_text(adapter: TimelineEventAdapter) -> str:
    parts: list[str] = []
    for block in adapter.snapshot_blocks():
        if str(block.get("kind") or "") != "answer":
            continue
        if str(block.get("status") or "") in {"discarded", "superseded"}:
            continue
        payload = block.get("payload") if isinstance(block.get("payload"), dict) else {}
        content = str(payload.get("content") or "")
        if content:
            parts.append(content)
    return "".join(parts)


def _answer_text_since(adapter: TimelineEventAdapter, start_chars: int) -> str:
    text = _adapter_answer_text(adapter)
    return text[max(0, int(start_chars or 0)):].strip()


def _finish_reason_is_truncated(finish_reason: str | None) -> bool:
    return str(finish_reason or "").strip().lower() in {
        "length",
        "max_tokens",
        "max_output_tokens",
    }


class _RecoveryPrefixSuppressor:
    """Suppress a replayed recovery prefix across arbitrary stream chunks.

    A recovery response can replay all (or the tail) of text that is already in
    the timeline.  Comparing each provider delta independently is incorrect:
    ``"abcdef"`` replayed as ``"abc"`` + ``"def"`` has no suffix overlap on
    the first delta and is therefore duplicated.  Keep the ambiguous recovery
    prefix buffered until the longest possible overlap is known.

    If a partial match later diverges, the complete buffer is released.  On an
    incomplete final match, :meth:`flush` also releases the buffer instead of
    guessing that valid new text was a replay.
    """

    def __init__(
        self,
        existing: str,
        *,
        max_tail_overlap: int = 2000,
        min_tail_overlap: int = 8,
    ) -> None:
        self._existing = str(existing or "")
        self._buffer = ""
        self._resolved = not bool(self._existing)
        self._received_text = False
        self._matched_overlap = 0

        bounded_tail = min(
            len(self._existing),
            max(0, int(max_tail_overlap or 0)),
        )
        credible_tail_start = max(2, int(min_tail_overlap or 0))
        overlaps = set(range(credible_tail_start, bounded_tail + 1))
        # Exact full-prefix replay remains valid even when the interrupted
        # delta is shorter than the credible suffix threshold.  Short suffixes
        # are deliberately excluded: a shared trailing character is not
        # enough evidence to discard new user-visible text.
        if self._existing:
            overlaps.add(len(self._existing))
        self._possible_overlaps = sorted(overlaps)

    @property
    def received_text(self) -> bool:
        return self._received_text

    @property
    def recovered_existing_prefix(self) -> str:
        """Return the already-emitted prefix only after replay is proven.

        The recovery stream emits only its novel suffix to the timeline. The
        terminal reply still needs the prefix that survived the interrupted
        stream; exposing it here keeps display de-duplication and authoritative
        answer reconstruction consistent.
        """

        return self._existing if self._matched_overlap > 0 else ""

    def feed(self, chunk: str) -> str:
        text = str(chunk or "")
        if not text:
            return ""
        self._received_text = True
        if self._resolved:
            return text

        previous_len = len(self._buffer)
        self._buffer = f"{self._buffer}{text}"
        current_len = len(self._buffer)
        compatible: list[int] = []
        for overlap in self._possible_overlaps:
            # Once an overlap is fully matched, later characters are the new
            # suffix and cannot invalidate that candidate.
            if previous_len >= overlap:
                compatible.append(overlap)
                continue
            compare_end = min(current_len, overlap)
            existing_start = len(self._existing) - overlap
            if self._buffer[previous_len:compare_end] == self._existing[
                existing_start + previous_len : existing_start + compare_end
            ]:
                compatible.append(overlap)
        self._possible_overlaps = compatible

        if not compatible:
            return self._resolve_without_overlap()

        pending_longer_match = any(overlap > current_len for overlap in compatible)
        if pending_longer_match:
            return ""

        longest_overlap = max(compatible)
        return self._resolve_with_overlap(longest_overlap)

    def flush(self) -> str:
        if self._resolved:
            return ""
        complete_overlaps = [
            overlap
            for overlap in self._possible_overlaps
            if overlap <= len(self._buffer)
        ]
        if complete_overlaps:
            return self._resolve_with_overlap(max(complete_overlaps))
        # The stream ended while only part of an old prefix matched.  That is
        # not enough evidence to discard user-visible text.
        return self._resolve_without_overlap()

    def _resolve_with_overlap(self, overlap: int) -> str:
        self._matched_overlap = max(0, int(overlap or 0))
        output = self._buffer[max(0, int(overlap or 0)) :]
        self._buffer = ""
        self._possible_overlaps = []
        self._resolved = True
        return output

    def _resolve_without_overlap(self) -> str:
        self._matched_overlap = 0
        output = self._buffer
        self._buffer = ""
        self._possible_overlaps = []
        self._resolved = True
        return output


def _clone_llm_disable_streaming(llm_base: Any) -> Any:
    """Best-effort clone for recovery calls that should not rely on SSE chunks."""
    if llm_base is None:
        return llm_base
    for method_name in ("model_copy", "copy"):
        method = getattr(llm_base, method_name, None)
        if not callable(method):
            continue
        try:
            return method(update={"disable_streaming": True})
        except TypeError:
            try:
                cloned = method()
                try:
                    setattr(cloned, "disable_streaming", True)
                except Exception:  # noqa: BLE001
                    pass
                return cloned
            except Exception:  # noqa: BLE001
                continue
        except Exception:  # noqa: BLE001
            continue
    bind = getattr(llm_base, "bind", None)
    if callable(bind):
        try:
            return bind(disable_streaming=True)
        except Exception:  # noqa: BLE001
            pass
    try:
        setattr(llm_base, "disable_streaming", True)
    except Exception:  # noqa: BLE001
        pass
    return llm_base




def _normalize_tool_outcomes(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "").strip()
        if not tool:
            continue
        (
            preview_result_ref,
            preview_artifact_ids,
            preview_job_id,
            preview_result_status,
        ) = _recovery_result_refs(
            _parse_tool_message_content(item.get("resume_preview"))
        )
        outcome: dict[str, Any] = {
            "tool": tool,
            "ok": bool(item.get("ok", True)),
            "summary": str(item.get("summary") or "").strip(),
            "error_kind": str(item.get("error_kind") or "").strip(),
            "severity": str(item.get("severity") or "").strip().lower(),
        }
        for key in (
            "skill_id",
            "script_name",
            "tool_call_id",
            "runtime_run_id",
            "arguments_digest",
            "result_status",
            "result_ref",
            "raw_ref",
            "job_id",
            "cursor",
        ):
            fallback = {
                "result_status": preview_result_status,
                "result_ref": preview_result_ref,
                "raw_ref": preview_result_ref,
                "job_id": preview_job_id,
            }.get(key, "")
            value = str(item.get(key) or fallback or "").strip()
            if value:
                outcome[key] = value
        for key in ("complete", "has_more"):
            if isinstance(item.get(key), bool):
                outcome[key] = item[key]
        for key in ("resources", "artifacts", "artifact_ids"):
            if isinstance(item.get(key), list):
                outcome[key] = list(item[key])
        if preview_artifact_ids:
            artifact_ids = [
                str(value or "").strip()
                for value in outcome.get("artifact_ids") or []
                if str(value or "").strip()
            ]
            for artifact_id in preview_artifact_ids:
                if artifact_id and artifact_id not in artifact_ids:
                    artifact_ids.append(artifact_id)
            outcome["artifact_ids"] = artifact_ids
        completion_signal = normalize_tool_completion_signal(item.get("completion_signal"))
        if completion_signal:
            outcome["completion_signal"] = completion_signal
        capability_outcome = _compact_recovery_capability_outcome(
            item.get("capability_outcome")
        )
        if capability_outcome:
            outcome["capability_outcome"] = capability_outcome
        normalized.append(outcome)
    return normalized


def _append_tool_outcome(existing: list[dict[str, Any]], outcome: dict[str, Any]) -> list[dict[str, Any]]:
    return _normalize_tool_outcomes([*existing, outcome])


_SIDE_EFFECT_TOOL_RE = re.compile(
    r"(?:^|_)(?:submit|create|add|update|delete|remove|register|save|write|"
    r"upload|finalize|generate|execute|run|prepare|ensure|complete|download|"
    r"publish|commit|cancel)(?:_|$)",
    flags=re.IGNORECASE,
)


def _is_side_effect_tool(tool_name: Any) -> bool:
    """Classify mutation/execution tools by stable tool-name action verbs."""
    return bool(_SIDE_EFFECT_TOOL_RE.search(str(tool_name or "").strip()))


_CHECKPOINT_REPLAY_SAFE_READ_TOOLS = frozenset()


def _is_checkpoint_replay_tool(tool_name: Any) -> bool:
    normalized = str(tool_name or "").strip()
    return _is_side_effect_tool(normalized) or normalized in _CHECKPOINT_REPLAY_SAFE_READ_TOOLS


def _normalize_side_effect_ledger(raw: Any) -> dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}
    raw_items = source.get("items") if isinstance(source.get("items"), list) else []
    dropped_count = max(0, int(source.get("dropped_count") or 0))
    ordered: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        tool = _safe_recovery_identifier(raw_item.get("tool"))
        arguments_digest = _safe_recovery_identifier(
            raw_item.get("arguments_digest")
        )
        tool_call_id = _safe_recovery_identifier(raw_item.get("tool_call_id"))
        runtime_run_id = _safe_recovery_identifier(raw_item.get("runtime_run_id"))
        if not tool:
            continue
        effect_identity = _safe_recovery_identifier(
            raw_item.get("effect_identity")
        ) or _recovery_digest(
            {
                "tool": tool,
                "arguments_digest": arguments_digest,
                "fallback_call_id": (
                    "" if arguments_digest else (tool_call_id or runtime_run_id)
                ),
            }
        )
        item = {
            "effect_identity": effect_identity,
            "tool": tool,
            "tool_call_id": tool_call_id,
            "runtime_run_id": runtime_run_id,
            "arguments_digest": arguments_digest,
            "result_status": _safe_recovery_enum(
                raw_item.get("result_status"),
                _RECOVERY_RESULT_STATUSES,
                default="succeeded" if raw_item.get("ok", True) else "failed",
            ),
            "ok": bool(raw_item.get("ok", True)),
            "identity_complete": bool(tool_call_id and arguments_digest),
        }
        job_id = _safe_recovery_identifier(raw_item.get("job_id"))
        result_ref = _safe_recovery_identifier(raw_item.get("result_ref"))
        if job_id:
            item["job_id"] = job_id
        if result_ref:
            item["result_ref"] = result_ref
        existing_position = positions.get(effect_identity)
        if existing_position is None:
            positions[effect_identity] = len(ordered)
            ordered.append(item)
            continue
        previous = ordered[existing_position]
        # Preserve the first real model call identity while refreshing the
        # latest observed status/reference for an idempotent duplicate.
        item["tool_call_id"] = previous.get("tool_call_id") or item["tool_call_id"]
        item["runtime_run_id"] = previous.get("runtime_run_id") or item["runtime_run_id"]
        item["identity_complete"] = bool(
            item.get("tool_call_id") and item.get("arguments_digest")
        )
        ordered[existing_position] = item

    return {
        "items": ordered,
        "dropped_count": dropped_count,
        "total_count": len(ordered) + dropped_count,
        "truncated": dropped_count > 0,
        "digest": _recovery_digest(ordered),
    }


def _append_side_effect_identity(raw: Any, outcome: dict[str, Any]) -> dict[str, Any]:
    ledger = _normalize_side_effect_ledger(raw)
    return _normalize_side_effect_ledger(
        {
            "items": [*ledger["items"], outcome],
            "dropped_count": ledger["dropped_count"],
        }
    )


def _tool_message_call_id(output: Any) -> str:
    if not isinstance(output, ToolMessage):
        return ""
    return str(getattr(output, "tool_call_id", "") or "").strip()


def _normalize_continuation_messages(raw: Any) -> list[BaseMessage]:
    if not isinstance(raw, list):
        return []
    return [message for message in raw if isinstance(message, BaseMessage)]


def _capability_prefetch_messages(
    rendered: Any,
    *,
    current_user_message: str,
    seen_requirement_ids: set[str] | None = None,
) -> list[BaseMessage]:
    """Represent each staged prefetch requirement as its matcher Tool result."""

    text = str(rendered or "")
    if (
        _CAPABILITY_PREFETCH_OPEN not in text
        or _CAPABILITY_PREFETCH_CLOSE not in text
    ):
        return []
    raw_payload = (
        text.split(_CAPABILITY_PREFETCH_OPEN, 1)[1]
        .split(_CAPABILITY_PREFETCH_CLOSE, 1)[0]
        .strip()
    )
    try:
        payload = json.loads(raw_payload)
    except Exception:
        return []
    if (
        not isinstance(payload, dict)
        or payload.get("source") != "match_capability"
        or payload.get("requires_agent_choice") is not True
    ):
        return []
    seen = set(seen_requirement_ids or set())
    requirements = payload.get("requirements")
    if not isinstance(requirements, list):
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return []
        requirements = [
            {
                "requirement_id": "legacy_user_goal_"
                + hashlib.sha256(
                    str(current_user_message or "").strip().encode("utf-8")
                ).hexdigest()[:20],
                "source": "current_user_goal",
                "summary": str(current_user_message or "").strip()[:600],
                "query": "",
                "decision": str(payload.get("decision") or "candidates"),
                "candidates": candidates,
            }
        ]

    messages: list[BaseMessage] = []
    wave_id = str(payload.get("wave_id") or "").strip()
    for requirement in requirements:
        if not isinstance(requirement, dict):
            continue
        candidates = requirement.get("candidates")
        requirement_id = str(
            requirement.get("requirement_id") or ""
        ).strip()
        if (
            not requirement_id
            or requirement_id in seen
            or not isinstance(candidates, list)
            or not candidates
        ):
            continue
        args: dict[str, Any] = {
            "query": str(requirement.get("query") or ""),
            "max_results": max(1, min(10, len(candidates))),
        }
        for facet in ("operations", "source_types", "object_types", "input_types"):
            values = requirement.get(facet)
            if isinstance(values, list) and values:
                args[facet] = values[:4]
        result_payload = {
            "result_type": "capability_candidates",
            "decision": str(requirement.get("decision") or "candidates"),
            "candidates": candidates,
            "candidate_count": len(candidates),
            "requires_agent_choice": True,
            "side_effects": False,
            "discovery_wave_id": wave_id,
            "discovery_requirement_id": requirement_id,
            "discovery_requirement": {
                "source": str(requirement.get("source") or ""),
                "summary": str(requirement.get("summary") or "")[:600],
            },
        }
        call_id = "capability_prefetch_" + hashlib.sha256(
            (
                requirement_id
                + "\n"
                + json.dumps(result_payload, ensure_ascii=False, sort_keys=True)
            ).encode("utf-8")
        ).hexdigest()[:16]
        messages.extend(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": call_id,
                            "name": "match_capability",
                            "args": args,
                        }
                    ],
                ),
                ToolMessage(
                    content=json.dumps(
                        result_payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    tool_call_id=call_id,
                    name="match_capability",
                ),
            ]
        )
    return messages


def _observed_capability_discovery_requirement_ids(
    messages: list[BaseMessage],
) -> set[str]:
    observed: set[str] = set()
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if str(getattr(message, "name", "") or "") != "match_capability":
            continue
        payload = _parse_tool_message_content(getattr(message, "content", None))
        if not isinstance(payload, dict):
            continue
        requirement_id = str(
            payload.get("discovery_requirement_id") or ""
        ).strip()
        if requirement_id:
            observed.add(requirement_id)
    return observed


def _normalize_tool_message_protocol(
    raw: Any,
) -> tuple[list[BaseMessage], list[str]]:
    """Normalize only identities that are provably unambiguous.

    A valid call without a result is protocol corruption.  It must remain
    missing so the caller can fail closed; fabricating a failure result here
    would create a second protocol authority outside actual tool execution.
    Invalid model calls are different: the model itself proves no tool ran, so
    their deterministic rejection message is a faithful protocol result.
    """
    messages = _normalize_continuation_messages(raw)
    normalized: list[BaseMessage] = []
    repairs: list[str] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        valid_calls = (
            list(getattr(message, "tool_calls", None) or [])
            if isinstance(message, AIMessage)
            else []
        )
        invalid_calls = (
            list(getattr(message, "invalid_tool_calls", None) or [])
            if isinstance(message, AIMessage)
            else []
        )
        if not isinstance(message, AIMessage) or not (valid_calls or invalid_calls):
            if isinstance(message, ToolMessage):
                repairs.append(f"orphan_tool_message:{getattr(message, 'name', '') or '?'}")
            else:
                normalized.append(message)
            index += 1
            continue

        normalized.append(message)
        expected_calls = [
            (call, False) for call in valid_calls if isinstance(call, dict)
        ] + [
            (call, True) for call in invalid_calls if isinstance(call, dict)
        ]
        index += 1
        candidates: list[ToolMessage] = []
        while index < len(messages) and isinstance(messages[index], ToolMessage):
            candidates.append(messages[index])
            index += 1

        unused = list(candidates)
        for position, (call, call_was_invalid) in enumerate(expected_calls):
            expected_id = str(call.get("id") or "").strip() or f"runtime_tool_call_{position}"
            expected_name = str(call.get("name") or "").strip()
            matched: ToolMessage | None = None
            for candidate in unused:
                if str(getattr(candidate, "tool_call_id", "") or "").strip() == expected_id:
                    matched = candidate
                    break
            if matched is not None:
                unused.remove(matched)
                normalized.append(matched)
                continue
            repairs.append(
                f"{'invalid_tool_call_result' if call_was_invalid else 'missing_tool_result'}:"
                f"{expected_name or expected_id}"
            )
            if call_was_invalid:
                normalized.append(
                    ToolMessage(
                        content=json.dumps(
                            {
                                "ok": False,
                                "error_kind": "runtime_tool_call_invalid",
                                "summary": (
                                    "模型生成的工具参数无法解析，工具未执行。请重新调用同一工具，"
                                    "只填写 schema 要求的字段并缩短自由文本；批量参数过长时拆成更小批次。"
                                ),
                            },
                            ensure_ascii=False,
                        ),
                        tool_call_id=expected_id,
                        name=expected_name or None,
                        status="error",
                    )
                )
        for candidate in unused:
            repairs.append(f"unmatched_tool_message:{getattr(candidate, 'name', '') or '?'}")
    return normalized, repairs


def _build_plan_execution_contract_prompt(
    conversation_files: list[dict[str, Any]] | None,
) -> str:
    plan_file = next(
        (
            item
            for item in conversation_files or []
            if isinstance(item, dict)
            and str(item.get("drawer_section") or "").strip().lower() == "plan_file"
        ),
        None,
    )
    if plan_file is None:
        return ""
    plan_name = str(plan_file.get("name") or plan_file.get("file_name") or "current plan").strip()
    plan_content = str(plan_file.get("text") or plan_file.get("content_text") or "").strip()
    raw_contract = plan_file.get("plan_contract")
    contract = dict(raw_contract) if isinstance(raw_contract, dict) else {}

    if not contract and plan_content:
        sections: dict[str, list[str]] = {}
        current = ""
        title = plan_name.rsplit(".", 1)[0]
        for raw_line in plan_content.splitlines():
            line = raw_line.strip()
            if line.startswith("# ") and not line.startswith("## "):
                title = line[2:].strip() or title
                continue
            if line.startswith("## "):
                current = line[3:].strip()
                sections.setdefault(current, [])
                continue
            if current and line:
                sections.setdefault(current, []).append(line)

        def _items(section: str) -> list[str]:
            return [re.sub(r"^(?:[-*]\s+|\d+[.)]\s*)", "", line).strip() for line in sections.get(section, []) if line]

        steps = _items("执行步骤")
        contract = {
            "schema_version": "plan_contract_v1_legacy_projection",
            "title": title,
            "goal": " ".join(sections.get("目标", [])),
            "confirmed_scope": " ".join(sections.get("已确认范围", [])),
            "steps": [
                {
                    "step_id": f"P{index}",
                    "description": item,
                    "depends_on_step_ids": [],
                }
                for index, item in enumerate(steps, start=1)
            ],
            "expected_outputs": _items("预期产物"),
            "assumptions": _items("当前假设"),
            "notes": _items("注意事项"),
        }
        if not any(
            contract.get(key)
            for key in ("goal", "confirmed_scope", "steps", "expected_outputs", "assumptions", "notes")
        ):
            # Preserve a legacy plan even when it predates the canonical
            # Markdown section names. New plans always carry plan_contract.
            contract["notes"] = [plan_content]

    def _plan_text(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    compact_steps: list[dict[str, Any]] = []
    for index, item in enumerate(contract.get("steps") or [], start=1):
        if isinstance(item, dict):
            compact_steps.append(
                {
                    "step_id": _plan_text(item.get("step_id") or f"P{index}"),
                    "description": _plan_text(item.get("description")),
                    "depends_on_step_ids": [
                        _plan_text(dep)
                        for dep in (item.get("depends_on_step_ids") or [])
                        if str(dep or "").strip()
                    ],
                }
            )
        elif str(item or "").strip():
            compact_steps.append(
                {
                    "step_id": f"P{index}",
                    "description": _plan_text(item),
                    "depends_on_step_ids": [],
                }
            )

    envelope = {
        "schema_version": "active_plan_envelope_v1",
        "role": "goal_scope_acceptance_contract",
        "progress_authority": "task_tree_current_frontier",
        "action_relation": "serve_unmet_plan_step_or_required_dependency",
        "revision_condition": "goal_scope_or_acceptance_changed",
        "plan_ref": str(plan_file.get("file_id") or plan_file.get("conversation_file_id") or plan_name),
        "plan_file": plan_name,
        "revision": _plan_text(contract.get("revision")),
        "goal": _plan_text(contract.get("goal")),
        "confirmed_scope": _plan_text(contract.get("confirmed_scope")),
        "steps": compact_steps,
        "expected_outputs": [_plan_text(item) for item in (contract.get("expected_outputs") or [])],
        "assumptions": [_plan_text(item) for item in (contract.get("assumptions") or [])],
        "notes": [_plan_text(item) for item in (contract.get("notes") or [])],
    }
    if not envelope["revision"]:
        envelope["revision"] = hashlib.sha256(
            json.dumps(envelope, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
    return "[ACTIVE_PLAN_ENVELOPE]\n" + json.dumps(envelope, ensure_ascii=False, separators=(",", ":")) + "\n[/ACTIVE_PLAN_ENVELOPE]"


def _trusted_plan_contract(
    conversation_files: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Return the current runtime-created plan contract, never free-form text."""

    for item in conversation_files or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("drawer_section") or "").strip().lower() != "plan_file":
            continue
        contract = item.get("plan_contract")
        if not isinstance(contract, dict):
            continue
        schema_version = str(contract.get("schema_version") or "").strip()
        if schema_version.startswith("plan_contract_v1"):
            return dict(contract)
    return {}


def _capability_discovery_requirement_id(payload: dict[str, Any]) -> str:
    return "capability_requirement_" + hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:20]


def _bounded_discovery_text(value: Any, *, limit: int = 1200) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _build_staged_capability_discovery_requirements(
    *,
    current_user_message: str,
    conversation_files: list[dict[str, Any]] | None,
    task_tree_snapshot: dict[str, Any] | None,
    loaded_tool_ids: list[str] | None,
) -> list[dict[str, Any]]:
    """Build the next discovery wave from trusted, unsatisfied obligations.

    Task-tree status is the progress authority.  The plan contract contributes
    goal/scope context, but is never treated as proof that a step completed.
    When no live tree exists, the dependency-free plan steps are the only safe
    initial wave.  Ordinary requests retain one user-goal discovery wave.
    """

    loaded = set(_normalize_loaded_tool_ids(loaded_tool_ids))
    plan_contract = _trusted_plan_contract(conversation_files)
    goal_context = _bounded_discovery_text(plan_contract.get("goal"), limit=500)
    scope_context = _bounded_discovery_text(
        plan_contract.get("confirmed_scope"),
        limit=400,
    )

    if isinstance(task_tree_snapshot, dict) and not is_terminal_snapshot(
        task_tree_snapshot
    ):
        brief = build_task_tree_snapshot_brief(task_tree_snapshot)
        raw_frontier = list(brief.get("current_frontier") or [])
        # ``partial`` deliberately stays non-terminal and carries a concrete
        # missing-requirement/next-action summary, even though it is not part
        # of the task-tree service's ready-pending frontier projection.
        partial_nodes = [
            item
            for item in (brief.get("nodes") or [])
            if isinstance(item, dict)
            and str(item.get("status") or "").strip().lower() == "partial"
        ]
        all_nodes_by_id = {
            str(item.get("node_id") or "").strip(): item
            for item in (brief.get("nodes") or [])
            if isinstance(item, dict) and str(item.get("node_id") or "").strip()
        }
        obligation_by_id: dict[str, dict[str, Any]] = {}
        for item in [*raw_frontier, *partial_nodes]:
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("node_id") or "").strip()
            if node_id:
                obligation_by_id[node_id] = {
                    **dict(all_nodes_by_id.get(node_id) or {}),
                    **dict(item),
                }

        requirements: list[dict[str, Any]] = []
        run_id = str(brief.get("run_id") or "").strip()
        for node in obligation_by_id.values():
            status = str(node.get("status") or "pending").strip().lower()
            tool_name = str(node.get("tool_name") or "").strip()
            sandbox_job_ids = [
                str(item or "").strip()
                for item in (node.get("sandbox_job_ids") or [])
                if str(item or "").strip()
            ]
            if status in {"running", "waiting", "blocked"} and sandbox_job_ids:
                # This obligation is waiting for a known job, not a missing
                # execution capability.  Sandbox resume owns the next action.
                continue
            if status != "partial" and tool_name:
                expanded = set(_expand_loaded_tool_ids([tool_name]))
                if tool_name in loaded or (expanded and expanded.issubset(loaded)):
                    continue

            title = _bounded_discovery_text(node.get("title"), limit=500)
            result_summary = _bounded_discovery_text(
                node.get("result_summary"),
                limit=700,
            )
            query_parts = [title]
            if status == "partial" and result_summary:
                query_parts.append("Unmet result requirements: " + result_summary)
            if tool_name and status != "partial":
                query_parts.append(
                    "Declared execution entry when applicable: " + tool_name
                )
            if goal_context:
                query_parts.append("Parent goal: " + goal_context)
            if scope_context:
                query_parts.append("Confirmed scope: " + scope_context)
            query = "\n".join(part for part in query_parts if part)[:1600]
            identity = {
                "source": "task_tree_current_frontier",
                "run_id": run_id,
                "node_id": str(node.get("node_id") or ""),
                "status": status,
                "tool_name": tool_name,
                "title": title,
                "result_summary": result_summary if status == "partial" else "",
            }
            requirements.append(
                {
                    "requirement_id": _capability_discovery_requirement_id(identity),
                    "source": "task_tree_current_frontier",
                    "summary": (
                        title
                        + (
                            "；当前缺口：" + result_summary
                            if status == "partial" and result_summary
                            else ""
                        )
                    )[:600],
                    "query": query,
                    "task_node_id": str(node.get("node_id") or ""),
                    "status": status,
                }
            )
        # A live tree is the progress authority even when the current wave is
        # already covered.  Falling back to the entire user request here would
        # reintroduce the global, repeated matcher query this function removes.
        return requirements

    raw_requirements = plan_contract.get("execution_requirements")
    if not isinstance(raw_requirements, list) or not raw_requirements:
        raw_requirements = plan_contract.get("steps")
    if isinstance(raw_requirements, list) and raw_requirements:
        normalized_rows: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_requirements, start=1):
            if isinstance(raw, dict):
                row = dict(raw)
            elif str(raw or "").strip():
                row = {"description": str(raw).strip()}
            else:
                continue
            row.setdefault("step_id", f"P{index}")
            normalized_rows.append(row)
        ready_rows = [
            row
            for row in normalized_rows
            if not (
                row.get("depends_on_requirement_ids")
                or row.get("depends_on_step_ids")
            )
        ]
        requirements = []
        revision = str(plan_contract.get("revision") or "").strip()
        for row in ready_rows:
            description = _bounded_discovery_text(
                row.get("description") or row.get("requirement"),
                limit=900,
            )
            if not description:
                continue
            raw_load_tool_ids = row.get("load_tool_ids")
            declared_tool_ids = _normalize_loaded_tool_ids(
                [
                    row.get("tool_id"),
                    row.get("tool_name"),
                    *(
                        raw_load_tool_ids
                        if isinstance(raw_load_tool_ids, list)
                        else []
                    ),
                ]
            )
            if declared_tool_ids and set(declared_tool_ids).issubset(loaded):
                continue
            query = "\n".join(
                part
                for part in (
                    description,
                    "Parent goal: " + goal_context if goal_context else "",
                    "Confirmed scope: " + scope_context if scope_context else "",
                )
                if part
            )[:1600]
            identity = {
                "source": "trusted_plan_contract",
                "revision": revision,
                "step_id": str(row.get("step_id") or ""),
                "description": description,
            }
            requirement = {
                "requirement_id": _capability_discovery_requirement_id(identity),
                "source": "trusted_plan_contract",
                "summary": description[:600],
                "query": query,
            }
            for facet in (
                "operations",
                "source_types",
                "object_types",
                "input_types",
            ):
                values = row.get(facet)
                if isinstance(values, list) and values:
                    requirement[facet] = values[:4]
            requirements.append(requirement)
        return requirements

    user_goal = str(current_user_message or "").strip()
    if not user_goal:
        return []
    return [
        {
            "requirement_id": _capability_discovery_requirement_id(
                {"source": "current_user_goal", "goal": user_goal}
            ),
            "source": "current_user_goal",
            "summary": _bounded_discovery_text(user_goal, limit=600),
            "query": "",
        }
    ]


def _build_current_turn_tool_state_prompt(
    *,
    tool_outcomes: list[dict[str, Any]],
    pending_sandbox_jobs: list[dict[str, Any]],
) -> str:
    outcomes = _normalize_tool_outcomes(tool_outcomes)
    if not outcomes and not pending_sandbox_jobs:
        return ""
    compact_outcomes: list[dict[str, Any]] = []
    for outcome in outcomes:
        parsed_preview = _parse_tool_message_content(outcome.get("resume_preview"))
        (
            preview_result_ref,
            preview_artifact_ids,
            preview_job_id,
            preview_result_status,
        ) = _recovery_result_refs(parsed_preview)
        result_ref = _safe_recovery_identifier(
            outcome.get("result_ref")
            or outcome.get("raw_ref")
            or preview_result_ref
        )
        artifact_ids = [
            _safe_recovery_identifier(item)
            for item in outcome.get("artifact_ids") or []
            if str(item or "").strip()
        ]
        for artifact_id in preview_artifact_ids:
            if artifact_id and artifact_id not in artifact_ids:
                artifact_ids.append(artifact_id)
        job_id = _safe_recovery_identifier(
            outcome.get("job_id") or preview_job_id
        )
        result_status = _safe_recovery_enum(
            outcome.get("result_status") or preview_result_status,
            _RECOVERY_RESULT_STATUSES,
            default="succeeded" if outcome.get("ok", True) else "failed",
        )
        completion = _recovery_completion_projection(outcome.get("completion_signal"))
        tool_id = _safe_recovery_identifier(outcome.get("tool"))
        item: dict[str, Any] = {
            "tool": tool_id,
            "ok": bool(outcome.get("ok", True)),
            "result_status": result_status,
            "artifact_count": len(artifact_ids),
            "outcome_digest": _recovery_digest(
                {
                    "tool": tool_id,
                    "ok": bool(outcome.get("ok", True)),
                    "completion": completion,
                    "result_ref": result_ref,
                    "job_id": job_id,
                    "artifact_ids": artifact_ids,
                    "source_digest": _recovery_digest(outcome),
                }
            ),
        }
        if completion:
            item["completion"] = completion
        if result_ref:
            item["result_ref"] = result_ref
        if job_id:
            item["job_id"] = job_id
        if artifact_ids:
            item["artifact_ids_digest"] = _recovery_digest(artifact_ids)
        compact_outcomes.append(item)
    compact_jobs = [
        {
            "app_id": _safe_recovery_identifier(job.get("app_id")),
            "job_id": _safe_recovery_identifier(job.get("job_id")),
            "status": _safe_recovery_enum(
                job.get("status"),
                _RECOVERY_RESULT_STATUSES,
                default="unknown",
            ),
        }
        for job in pending_sandbox_jobs
        if isinstance(job, dict) and str(job.get("job_id") or "").strip()
    ]
    return (
        "[CURRENT_TURN_TOOL_STATE]\n"
        + json.dumps(
            {
                "schema_version": "current_turn_tool_state_v2",
                "authority": "persisted_timeline_projection",
                "role": "read_only_observation",
                "completed_actions": compact_outcomes,
                "pending_sandbox_jobs": compact_jobs,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n[/CURRENT_TURN_TOOL_STATE]"
    )


_RECOVERY_ENVELOPE_OPEN = "[RECOVERY_STATE_ENVELOPE]"
_RECOVERY_ENVELOPE_CLOSE = "[/RECOVERY_STATE_ENVELOPE]"
_LEGACY_STREAM_RECOVERY_MARKER = "\n\n【系统恢复说明】\n"


def _recovery_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


_RECOVERY_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,159}$")
_RECOVERY_RESULT_STATUSES = {
    "unknown",
    "pending",
    "queued",
    "submitted",
    "running",
    "success",
    "succeeded",
    "completed",
    "failed",
    "error",
    "timeout",
    "timed_out",
    "cancelled",
    "canceled",
}
_SANDBOX_TERMINAL_STATUSES = {
    "succeeded",
    "success",
    "completed",
    "failed",
    "error",
    "timeout",
    "timed_out",
    "cancelled",
    "canceled",
}
_RECOVERY_EXECUTION_MODES = {
    "",
    "immediate",
    "blocking_short",
    "nonblocking_long",
    "async",
    "synchronous",
}
_RECOVERY_COMPLETION_STATUSES = {
    "unknown",
    "in_progress",
    "continue",
    "ready_to_answer",
    "answer_ready",
    "completed",
    "complete",
    "task_complete",
    "failed",
    "retrieval_complete",
    "retrieval_run_completed",
    "result_available",
    "blocked_needs_user_action",
    "answer_candidate_locked",
    "answer_candidate_ready",
    "requires_next_tool",
    "plan_saved",
    "deliverable_node_complete",
    "artifact_node_complete",
    "capability_not_applicable",
    "capability_scope_exhausted",
    "capability_retry_advised",
    "capability_input_refinement_advised",
    "capability_alternative_advised",
    "capability_scope_partial",
}


def _safe_recovery_identifier(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if _RECOVERY_IDENTIFIER_RE.fullmatch(text):
        return text
    return "sha256:" + _recovery_digest(text)


def _safe_recovery_enum(value: Any, allowed: set[str], *, default: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else default


def _compact_recovery_capability_outcome(value: Any) -> dict[str, Any]:
    if isinstance(value, dict) and value.get("outcome_digest"):
        allowed = {
            "applicability": {
                "unknown",
                "applicable",
                "partially_applicable",
                "not_applicable",
            },
            "evidence": {"unknown", "available", "partial", "none"},
            "coverage": {"unknown", "more_available", "exhausted", "not_applicable"},
            "recovery": {
                "unknown",
                "retry_same_call",
                "refine_input",
                "switch_capability",
                "do_not_retry",
                "wait",
            },
        }
        compact = {
            key: _safe_recovery_enum(value.get(key), choices, default="unknown")
            for key, choices in allowed.items()
        }
        scope = " ".join(str(value.get("coverage_scope") or "").split())
        if scope:
            compact["coverage_scope"] = scope
        for key in ("basis_digest", "outcome_digest"):
            digest = _safe_recovery_identifier(value.get(key))
            if digest:
                compact[key] = digest
        return compact
    normalized = normalize_capability_outcome(value)
    if not normalized:
        return {}
    compact = {
        key: normalized[key]
        for key in ("applicability", "evidence", "coverage", "recovery")
    }
    scope = " ".join(str(normalized.get("coverage_scope") or "").split())
    if scope:
        compact["coverage_scope"] = scope
    bases = {
        key: normalized.get(key)
        for key in (
            "applicability_basis",
            "evidence_basis",
            "coverage_basis",
            "recovery_basis",
        )
        if normalized.get(key)
    }
    if bases:
        compact["basis_digest"] = _recovery_digest(bases)
    compact["outcome_digest"] = _recovery_digest(normalized)
    return compact


def _strip_legacy_recovery_state(system_prompt: Any) -> str:
    """Return the immutable base prompt used before legacy recursive notes."""
    text = str(system_prompt or "")
    legacy_index = text.find(_LEGACY_STREAM_RECOVERY_MARKER)
    if legacy_index >= 0:
        text = text[:legacy_index]
    tagged_pattern = re.compile(
        re.escape(_RECOVERY_ENVELOPE_OPEN)
        + r".*?"
        + re.escape(_RECOVERY_ENVELOPE_CLOSE),
        flags=re.DOTALL,
    )
    return tagged_pattern.sub("", text).rstrip()


def _recovery_goal_projection(conversation_files: Any) -> dict[str, Any]:
    plan_file = next(
        (
            item
            for item in reversed(list(conversation_files or []))
            if isinstance(item, dict)
            and str(item.get("drawer_section") or "").strip().lower() == "plan_file"
        ),
        None,
    )
    if not isinstance(plan_file, dict):
        return {
            "plan_ref": "",
            "plan_revision": "",
            "acceptance_contract_digest": "",
            "expected_output_count": 0,
        }
    contract = plan_file.get("plan_contract") if isinstance(plan_file.get("plan_contract"), dict) else {}
    persisted_plan_ref = str(
        plan_file.get("file_id") or plan_file.get("conversation_file_id") or ""
    ).strip()
    plan_ref = persisted_plan_ref or (
        "plan_"
        + _recovery_digest(plan_file.get("name") or plan_file.get("file_name") or "")
    )
    revision = str(contract.get("revision") or "").strip()
    if not revision:
        revision = _recovery_digest(
            {
                "plan_ref": plan_ref,
                "goal": contract.get("goal"),
                "confirmed_scope": contract.get("confirmed_scope"),
                "expected_outputs": contract.get("expected_outputs"),
                "content": plan_file.get("content_text") or plan_file.get("text") or "",
            }
        )
    expected_outputs = contract.get("expected_outputs")
    if not isinstance(expected_outputs, list):
        expected_outputs = []
    acceptance_contract = {
        "goal": contract.get("goal"),
        "confirmed_scope": contract.get("confirmed_scope"),
        "expected_outputs": expected_outputs,
    }
    return {
        "plan_ref": _safe_recovery_identifier(plan_ref),
        "plan_revision": _safe_recovery_identifier(revision),
        "acceptance_contract_digest": _recovery_digest(acceptance_contract),
        "expected_output_count": len(expected_outputs),
    }


def _recovery_task_tree_projection(
    snapshot: Any,
    *,
    fetch_status: str,
) -> dict[str, Any]:
    normalized_fetch_status = str(fetch_status or "not_applicable").strip().lower()
    if normalized_fetch_status not in {"ok", "missing", "unavailable", "not_applicable"}:
        normalized_fetch_status = "unavailable"
    brief = build_task_tree_snapshot_brief(
        snapshot if normalized_fetch_status == "ok" and isinstance(snapshot, dict) else None
    )
    return {
        "authority": "server_task_tree",
        "role": "read_only_observation",
        "fetch_status": normalized_fetch_status,
        "run_id": _safe_recovery_identifier(brief.get("run_id")),
        "root_node_id": _safe_recovery_identifier(brief.get("root_node_id")),
        "snapshot_digest": (
            _recovery_digest(brief) if normalized_fetch_status == "ok" else ""
        ),
    }


def _recovery_result_refs(parsed: Any) -> tuple[str, list[str], str, str]:
    candidates: list[dict[str, Any]] = []
    if isinstance(parsed, dict):
        candidates.append(parsed)
        for key in ("result", "data", "payload", "job"):
            nested = parsed.get(key)
            if isinstance(nested, dict):
                candidates.append(nested)
    result_ref = ""
    job_id = ""
    status = ""
    artifact_ids: list[str] = []
    for candidate in candidates:
        if not result_ref:
            result_ref = str(
                candidate.get("result_ref")
                or candidate.get("raw_ref")
                or candidate.get("conversation_file_id")
                or candidate.get("file_id")
                or ""
            ).strip()
        if not job_id:
            job_id = str(candidate.get("job_id") or candidate.get("sandbox_job_id") or "").strip()
        if not status:
            status = str(candidate.get("status") or candidate.get("sandbox_status") or "").strip()
        for key in ("artifact_id", "file_id", "conversation_file_id", "report_file_id"):
            value = str(candidate.get(key) or "").strip()
            if value and value not in artifact_ids:
                artifact_ids.append(value)
        raw_ids = candidate.get("artifact_ids")
        if isinstance(raw_ids, list):
            for value in raw_ids:
                normalized = str(value or "").strip()
                if normalized and normalized not in artifact_ids:
                    artifact_ids.append(normalized)
    return (
        _safe_recovery_identifier(result_ref),
        [_safe_recovery_identifier(item) for item in artifact_ids],
        _safe_recovery_identifier(job_id),
        _safe_recovery_enum(status, _RECOVERY_RESULT_STATUSES, default="unknown"),
    )


def _recovery_collection(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "items": list(items),
        "total_count": len(items),
        "truncated": False,
        "digest": _recovery_digest(items),
    }


def _recovery_completion_projection(value: Any) -> dict[str, Any]:
    signal = normalize_tool_completion_signal(value)
    if not signal:
        return {}
    acceptance = signal.get("acceptance") if isinstance(signal.get("acceptance"), dict) else {}
    deliverables = signal.get("deliverable_files") if isinstance(signal.get("deliverable_files"), list) else []
    projected = {
        "status": _safe_recovery_enum(
            signal.get("status"),
            _RECOVERY_COMPLETION_STATUSES,
            default="unknown",
        ),
        "should_stop_tool_loop": bool(signal.get("should_stop_tool_loop")),
        "acceptance_digest": _recovery_digest(acceptance) if acceptance else "",
        "deliverable_count": len(deliverables),
    }
    final_answer = str(acceptance.get("final_answer") or "").strip()
    if final_answer:
        projected["final_answer"] = final_answer
    return projected


def _recovery_outcome_identity(
    *,
    tool: Any,
    ok: Any,
    completion: dict[str, Any],
) -> tuple[str, bool, str, str]:
    return (
        _safe_recovery_identifier(tool),
        bool(ok),
        str(completion.get("status") or ""),
        str(completion.get("acceptance_digest") or ""),
    )


def _recovery_tool_result_observations(
    continuation_messages: Any,
    tool_outcomes: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    messages = _normalize_continuation_messages(continuation_messages)
    calls: dict[str, dict[str, Any]] = {}
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for position, call in enumerate(list(getattr(message, "tool_calls", None) or [])):
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or f"call_{position}").strip()
            calls[call_id] = call

    observed: list[dict[str, Any]] = []
    protocol_gaps: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        tool_call_id = str(getattr(message, "tool_call_id", "") or "").strip()
        call = calls.get(tool_call_id) or {}
        tool_name = str(getattr(message, "name", "") or call.get("name") or "tool").strip()
        arguments = call.get("args") if isinstance(call, dict) else None
        if arguments is None and isinstance(call, dict):
            arguments = call.get("arguments")
        parsed = _parse_tool_message_content(message)
        error_kind = str(parsed.get("error_kind") or "").strip() if isinstance(parsed, dict) else ""
        if error_kind in {"runtime_tool_result_missing", "runtime_tool_call_invalid"}:
            protocol_gaps.append(
                {
                    "tool_call_id": _safe_recovery_identifier(tool_call_id),
                    "tool": _safe_recovery_identifier(tool_name),
                    "protocol_state": (
                        "result_unknown"
                        if error_kind == "runtime_tool_result_missing"
                        else "not_executed_invalid_args"
                    ),
                    "error_kind": error_kind,
                }
            )
            continue
        result_ref, artifact_ids, job_id, status = _recovery_result_refs(parsed)
        outcome = _tool_outcome_from_output(tool_name, message)
        completion = _recovery_completion_projection(outcome.get("completion_signal"))
        capability_outcome = _compact_recovery_capability_outcome(
            outcome.get("capability_outcome")
        )
        observed.append(
            {
                "tool_call_id": _safe_recovery_identifier(tool_call_id),
                "tool": _safe_recovery_identifier(tool_name),
                "arguments_digest": _recovery_digest(arguments or {}),
                "protocol_state": "tool_end_observed",
                "result_status": status,
                "ok": bool(outcome.get("ok")),
                "identity_complete": bool(tool_call_id and call),
                "source": "continuation_protocol",
                "job_id": job_id,
                "result_ref": result_ref,
                "artifact_count": len(artifact_ids),
                "artifact_ids_digest": _recovery_digest(artifact_ids) if artifact_ids else "",
                **({"completion": completion} if completion else {}),
                **({"capability_outcome": capability_outcome} if capability_outcome else {}),
            }
        )

    observed_identity_counts: dict[tuple[str, bool, str, str], int] = {}
    observed_call_ids = {
        str(item.get("tool_call_id") or "").strip()
        for item in observed
        if str(item.get("tool_call_id") or "").strip()
    }
    for item in observed:
        completion = item.get("completion") if isinstance(item.get("completion"), dict) else {}
        identity = _recovery_outcome_identity(
            tool=item.get("tool"),
            ok=item.get("ok"),
            completion=completion,
        )
        observed_identity_counts[identity] = observed_identity_counts.get(identity, 0) + 1

    projected_prior_results: list[dict[str, Any]] = []
    for outcome in _normalize_tool_outcomes(tool_outcomes):
        outcome_call_id = _safe_recovery_identifier(outcome.get("tool_call_id"))
        arguments_digest = _safe_recovery_identifier(outcome.get("arguments_digest"))
        completion = _recovery_completion_projection(outcome.get("completion_signal"))
        capability_outcome = _compact_recovery_capability_outcome(
            outcome.get("capability_outcome")
        )
        identity = _recovery_outcome_identity(
            tool=outcome.get("tool"),
            ok=outcome.get("ok"),
            completion=completion,
        )
        if outcome_call_id:
            if outcome_call_id in observed_call_ids:
                continue
        else:
            exact_count = observed_identity_counts.get(identity, 0)
            if exact_count > 0:
                observed_identity_counts[identity] = exact_count - 1
                continue
        result_status = _safe_recovery_enum(
            outcome.get("result_status"),
            _RECOVERY_RESULT_STATUSES,
            default="succeeded" if outcome.get("ok") else "failed",
        )
        projected_prior_results.append(
            {
                "tool": _safe_recovery_identifier(outcome.get("tool")),
                "tool_call_id": outcome_call_id,
                "arguments_digest": arguments_digest,
                "result_status": result_status,
                "ok": bool(outcome.get("ok")),
                "identity_complete": bool(outcome_call_id and arguments_digest),
                "source": "timeline_projection",
                **({"completion": completion} if completion else {}),
                **({"capability_outcome": capability_outcome} if capability_outcome else {}),
            }
        )
    return (
        _recovery_collection(observed),
        _recovery_collection(protocol_gaps),
        _recovery_collection(projected_prior_results),
    )


def _build_recovery_state_envelope(
    *,
    state: NativeAgentState | dict[str, Any],
    interruption: dict[str, Any],
    continuation_messages: Any,
    task_tree_snapshot: Any,
    task_tree_fetch_status: str,
    tool_schema_snapshot: Any,
    tool_outcomes: Any,
    pending_sandbox_jobs: Any,
) -> dict[str, Any]:
    normalized_messages = _normalize_continuation_messages(continuation_messages)
    boundary_rows = [
        {
            "type": message.__class__.__name__,
            "id": str(getattr(message, "id", "") or ""),
            "tool_call_id": str(getattr(message, "tool_call_id", "") or ""),
        }
        for message in normalized_messages
    ]
    activated_skills = _normalize_activated_skills(state.get("activated_skills"))
    capability_ids = sorted(
        {
            str(capability_id)
            for skill in activated_skills
            for capability_id in (skill.get("capability_ids") or [])
            if str(capability_id or "").strip()
        }
    )
    snapshot = tool_schema_snapshot if isinstance(tool_schema_snapshot, dict) else {}
    observed_tool_results, protocol_gaps, projected_prior_results = (
        _recovery_tool_result_observations(
            continuation_messages,
            tool_outcomes,
        )
    )
    all_pending_external = [
        {
            "app_id": _safe_recovery_identifier(job.get("app_id")),
            "job_id": _safe_recovery_identifier(job.get("job_id")),
            "status": _safe_recovery_enum(
                job.get("status"),
                _RECOVERY_RESULT_STATUSES,
                default="unknown",
            ),
            "execution_mode": _safe_recovery_enum(
                job.get("agent_execution_mode"),
                _RECOVERY_EXECUTION_MODES,
                default="",
            ),
        }
        for job in list(pending_sandbox_jobs or [])
        if isinstance(job, dict)
    ]
    loaded_tool_ids = _normalize_loaded_tool_ids(state.get("loaded_tool_ids"))
    activated_skill_ids = sorted(_activated_skill_ids_from_state(state))
    completed_side_effects = _normalize_side_effect_ledger(
        state.get("side_effect_ledger")
    )
    recovery_reason = _safe_recovery_identifier(
        interruption.get("reason") or "model_stream_stalled"
    )
    finalization_only = recovery_reason == "execution_limit_reached"
    return {
        "schema_version": "recovery_state_envelope_v1",
        "request_id": _safe_recovery_identifier(state.get("request_id")),
        "recovery": {
            "reason": recovery_reason,
            "mode": (
                "finalize_from_existing_evidence"
                if finalization_only
                else "resume_execution"
            ),
            "new_tool_calls_allowed": not finalization_only,
            "phase": _safe_recovery_identifier(interruption.get("phase") or "unknown"),
            "call_index": int(interruption.get("call_index") or 0),
            "consecutive_attempt": int(interruption.get("consecutive_attempt") or 0),
            "cumulative_attempt": int(interruption.get("cumulative_attempt") or 0),
            "max_attempts": int(interruption.get("max_attempts") or 0),
            "max_total_attempts": int(interruption.get("max_total_attempts") or 0),
            "discarded_partial_model_output": bool(
                interruption.get("discarded_partial_model_output")
            ),
            "discarded_partial_output_chars": int(
                interruption.get("discarded_partial_output_chars") or 0
            ),
        },
        "continuation_boundary": {
            "policy": "last_complete_protocol_message",
            "committed_message_count": len(normalized_messages),
            "last_message_type": (
                normalized_messages[-1].__class__.__name__ if normalized_messages else ""
            ),
            "boundary_digest": _recovery_digest(boundary_rows),
        },
        "goal": _recovery_goal_projection(state.get("conversation_files")),
        "task_tree_observation": _recovery_task_tree_projection(
            task_tree_snapshot,
            fetch_status=task_tree_fetch_status,
        ),
        "capabilities": {
            "loaded_tool_ids": [_safe_recovery_identifier(item) for item in loaded_tool_ids],
            "loaded_tool_count": len(loaded_tool_ids),
            "loaded_tool_ids_digest": _recovery_digest(loaded_tool_ids),
            "activated_skill_ids": [
                _safe_recovery_identifier(item) for item in activated_skill_ids
            ],
            "activated_skill_count": len(activated_skill_ids),
            "activated_skill_ids_digest": _recovery_digest(activated_skill_ids),
            "selected_capability_ids": [
                _safe_recovery_identifier(item) for item in capability_ids
            ],
            "selected_capability_count": len(capability_ids),
            "selected_capability_ids_digest": _recovery_digest(capability_ids),
            "schema_snapshot_id": _safe_recovery_identifier(snapshot.get("snapshot_id")),
            "schema_tokens": int(snapshot.get("schema_tokens") or 0),
        },
        "actions": {
            "completed_side_effects": completed_side_effects,
            "observed_tool_results": observed_tool_results,
            "protocol_gaps": protocol_gaps,
            "projected_prior_results": projected_prior_results,
            "pending_external": _recovery_collection(all_pending_external),
        },
    }


def _render_recovery_state_envelope(envelope: Any) -> str:
    if not isinstance(envelope, dict) or not envelope:
        return ""
    payload = json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload = payload.replace(_RECOVERY_ENVELOPE_OPEN, "RECOVERY_STATE_ENVELOPE_ESCAPED_OPEN")
    payload = payload.replace(_RECOVERY_ENVELOPE_CLOSE, "RECOVERY_STATE_ENVELOPE_ESCAPED_CLOSE")
    rendered = (
        _RECOVERY_ENVELOPE_OPEN
        + "\n"
        + payload
        + "\n"
        + _RECOVERY_ENVELOPE_CLOSE
    )
    return rendered


def _build_sandbox_resume_state_prompt(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict) or not payload:
        return ""
    jobs = [
        {
            "job_id": _safe_recovery_identifier(item.get("job_id")),
            "app_id": _safe_recovery_identifier(item.get("app_id")),
            "status": _safe_recovery_enum(
                item.get("status"),
                _RECOVERY_RESULT_STATUSES,
                default="unknown",
            ),
        }
        for item in (payload.get("jobs") or [payload])
        if isinstance(item, dict) and str(item.get("job_id") or "").strip()
    ]
    if not jobs:
        return ""
    return (
        "[SANDBOX_RESUME_STATE]\n"
        + json.dumps(
            {
                "schema_version": "sandbox_resume_state_v2",
                "authority": "sandbox_scheduler",
                "role": "read_only_observation",
                "jobs": jobs,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n[/SANDBOX_RESUME_STATE]"
    )


async def _promote_authoritative_sandbox_resume_jobs(
    *,
    state: NativeAgentState,
    context: dict[str, Any],
    project_id: str,
    conversation_id: str,
    user_id: int,
) -> list[dict[str, Any]]:
    """Promote scheduler wakeups through the original submit executor path.

    Scheduler state is a wake signal only.  Before the model is called, each
    terminal job is resolved to the exact TaskTree pending receipt, fetched,
    output-validated, registered with original lineage, and committed back to
    that action.  A mismatched wakeup remains an explicit failure and cannot
    become an unbound ``sandbox_get_result`` query.
    """

    payload = state.get("sandbox_resume_payload")
    if not isinstance(payload, dict):
        return []
    jobs = [
        item
        for item in (payload.get("jobs") or [payload])
        if isinstance(item, dict)
        and str(item.get("job_id") or "").strip()
        and str(item.get("status") or "").strip().lower()
        in _SANDBOX_TERMINAL_STATUSES
    ]
    if not jobs:
        return []

    from src.capabilities.runtime import adapt_sandbox_tools_with_executor
    from src.capabilities.state_store import TaskTreeCapabilityStateStore
    from src.tools.sandbox_tools import (
        build_sandbox_tools,
        compile_sandbox_typed_schema_build,
        fetch_sandbox_exposure_context,
    )

    # Resolve job bindings first so this wakeup builds only the original submit
    # capabilities it actually needs.  An unrelated accepted app with a broken
    # typed schema must not block promotion of an otherwise valid completed job.
    promotion_store = TaskTreeCapabilityStateStore()
    promotable_jobs: list[dict[str, Any]] = []
    promoted: list[dict[str, Any]] = []
    required_submit_tool_ids: set[str] = set()
    for item in jobs:
        job_id = str(item.get("job_id") or "").strip()
        try:
            promotion = await promotion_store.resolve_sandbox_promotion(
                job_id=job_id,
                request_id=str(state.get("request_id") or "").strip(),
                project_id=project_id,
                conversation_id=conversation_id,
                user_id=user_id,
            )
            promotable_jobs.append(item)
            required_submit_tool_ids.add(promotion.tool_name)
        except Exception as exc:  # noqa: BLE001 - explicit fail-closed result
            promoted.append(
                {
                    "job_id": job_id,
                    "ok": False,
                    "status": "promotion_rejected",
                    "error_type": exc.__class__.__name__,
                    "summary": str(exc)[:1000],
                }
            )
    if not promotable_jobs:
        return promoted

    exposure_context = context.get("sandbox_exposure_context")
    if exposure_context is None:
        exposure_context = fetch_sandbox_exposure_context()
        context["sandbox_exposure_context"] = exposure_context
    typed_schema_build = compile_sandbox_typed_schema_build(exposure_context)
    loaded_tool_ids = sorted(
        {*required_submit_tool_ids, "sandbox_get_result"}
    )
    sandbox_tools = build_sandbox_tools(
        project_id=project_id,
        conversation_id=conversation_id,
        user_id=user_id,
        request_id=str(state.get("request_id") or "").strip(),
        exposure_context=exposure_context,
        loaded_tool_ids=loaded_tool_ids,
        typed_schema_build=typed_schema_build,
    )
    runtime = adapt_sandbox_tools_with_executor(
        sandbox_tools,
        exposure_context,
        request_id=str(
            state.get("task_authority_request_id")
            or state.get("request_id")
            or ""
        ).strip(),
        task_phase=TaskPhase.EXECUTE,
        project_id=project_id,
        conversation_id=conversation_id,
        user_id=user_id,
        loaded_tool_ids=loaded_tool_ids,
        task_tree_binding_required=True,
    )
    if runtime.executor is None:
        raise RuntimeError("SANDBOX_PROMOTION_EXECUTOR_UNAVAILABLE")
    runtime.executor.bind_transport_request_id(
        str(state.get("request_id") or "").strip()
    )

    for item in promotable_jobs:
        job_id = str(item.get("job_id") or "").strip()
        try:
            result, promotion = await runtime.executor.promote_sandbox_job(
                job_id=job_id,
                request_id=str(state.get("request_id") or "").strip(),
                project_id=project_id,
                conversation_id=conversation_id,
                user_id=user_id,
                task_phase=TaskPhase.EXECUTE,
            )
            promoted.append(
                {
                    "job_id": job_id,
                    "ok": bool(result.ok),
                    "status": result.status.value,
                    "task_node_id": promotion.call.task_node_id,
                    "begin_call_id": promotion.call.call_id,
                    "capability_id": promotion.call.capability_id,
                    "capability_version": promotion.call.capability_version,
                    "artifact_count": len(result.artifacts),
                    "summary": str(result.summary or "")[:1000],
                }
            )
        except Exception as exc:  # noqa: BLE001 - surface, never downgrade to query
            promoted.append(
                {
                    "job_id": job_id,
                    "ok": False,
                    "status": "promotion_rejected",
                    "error_type": exc.__class__.__name__,
                    "summary": str(exc)[:1000],
                }
            )
    return promoted


def _object_size_chars(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:  # noqa: BLE001
        return len(str(value))


def _tool_call_names(value: Any, *, limit: int = 8) -> str:
    if not isinstance(value, (list, tuple)):
        return ""
    names: list[str] = []
    for item in value[:limit]:
        name = ""
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("function", {}).get("name") or "")
        else:
            name = str(getattr(item, "name", "") or "")
        if name:
            names.append(name[:120])
    return ",".join(names)


def _invalid_tool_call_summary(value: Any, *, limit: int = 3) -> str:
    if not isinstance(value, (list, tuple)):
        return ""
    parts: list[str] = []
    for item in value[:limit]:
        if isinstance(item, dict):
            name = str(item.get("name") or "")
            error = str(item.get("error") or "")
            args = item.get("args")
        else:
            name = str(getattr(item, "name", "") or "")
            error = str(getattr(item, "error", "") or "")
            args = getattr(item, "args", None)
        args_chars = _object_size_chars(args)
        summary = f"name={name or '?'} error={error[:160] or '?'} args_chars={args_chars}"
        parts.append(summary)
    return " | ".join(parts)


def _response_finish_reason(output_obj: Any) -> str:
    metadata = getattr(output_obj, "response_metadata", None)
    if isinstance(metadata, dict):
        value = metadata.get("finish_reason") or metadata.get("stop_reason")
        if value:
            return str(value)
    additional = getattr(output_obj, "additional_kwargs", None)
    if isinstance(additional, dict):
        value = additional.get("finish_reason") or additional.get("stop_reason")
        if value:
            return str(value)
    return ""


def _messages_content_chars(messages: list[Any]) -> int:
    total = 0
    for message in messages:
        total += _object_size_chars(getattr(message, "content", ""))
    return total


def _continuation_messages_ledger_content(messages: Any) -> str:
    rows: list[dict[str, Any]] = []
    for message in _normalize_continuation_messages(messages):
        row: dict[str, Any] = {
            "type": message.__class__.__name__,
            "content": getattr(message, "content", ""),
        }
        if isinstance(message, AIMessage):
            row["tool_calls"] = list(getattr(message, "tool_calls", None) or [])
            row["invalid_tool_calls"] = list(
                getattr(message, "invalid_tool_calls", None) or []
            )
        if isinstance(message, ToolMessage):
            row["tool_call_id"] = str(getattr(message, "tool_call_id", "") or "")
            row["name"] = str(getattr(message, "name", "") or "")
        rows.append(row)
    return json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def _continuation_messages_ledger_part(messages: Any) -> dict[str, Any]:
    content = _continuation_messages_ledger_content(messages)
    return {
        "category": "continuation_replay",
        "label": "Committed continuation messages",
        "content": content,
        "tokens_estimated": estimate_tokens(content),
        "source_type": "runtime_protocol_replay",
        "priority": 100,
        "meta": {"reasoning_content_replayed": False},
    }


def _parse_tool_message_content(output_obj: Any) -> Any:
    raw = _extract_full_output_text(output_obj)
    if not raw and isinstance(output_obj, (dict, list)):
        return output_obj
    if not raw and isinstance(output_obj, str):
        raw = output_obj
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        pass
    try:
        return ast.literal_eval(text)
    except Exception:  # noqa: BLE001
        return text


def _tool_message_error_kind(output_obj: Any) -> str:
    parsed = _parse_tool_message_content(output_obj)
    if not isinstance(parsed, dict):
        return ""
    return str(parsed.get("error_kind") or "").strip()


def _capability_result_payload(parsed: Any) -> Any:
    """Return the machine result carried by a unified CapabilityResult.

    Native runtime control tools predate the unified envelope and can still
    return a direct dictionary in compatibility paths. Runtime code accepts
    both shapes without mistaking envelope metadata for the machine result.
    """

    if isinstance(parsed, dict) and isinstance(parsed.get("result"), dict):
        return parsed["result"]
    return parsed


def _tool_message_projection_sizes(output_obj: Any) -> tuple[int, int]:
    """Separate model-visible ToolMessage content from runtime-only artifact data."""

    if not isinstance(output_obj, ToolMessage):
        return _object_size_chars(output_obj), 0
    return (
        _object_size_chars(getattr(output_obj, "content", None)),
        _object_size_chars(getattr(output_obj, "artifact", None)),
    )


def _is_durable_resource_identifier(value: Any) -> bool:
    normalized = str(value or "").strip()
    return normalized.startswith(("resource:conversation-file:", "conversation-file://"))


def _payload_has_durable_resource(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if _is_durable_resource_identifier(payload.get("raw_ref")):
        return True
    for key in ("resource", "source_sidecar_resource"):
        resource = payload.get(key)
        if isinstance(resource, dict) and any(
            _is_durable_resource_identifier(resource.get(field))
            for field in ("resource_id", "uri")
        ):
            return True
    resources = payload.get("resources")
    if isinstance(resources, list) and any(
        isinstance(resource, dict)
        and any(
            _is_durable_resource_identifier(resource.get(field))
            for field in ("resource_id", "uri")
        )
        for resource in resources
    ):
        return True
    machine_projection = payload.get("machine_projection")
    return (
        _payload_has_durable_resource(machine_projection)
        if isinstance(machine_projection, dict)
        else False
    )


def _tool_output_has_durable_resource(output_obj: Any) -> bool:
    if _payload_has_durable_resource(_parse_tool_message_content(output_obj)):
        return True
    artifact = getattr(output_obj, "artifact", None)
    return _payload_has_durable_resource(artifact)


def _tool_runtime_control_projection(output_obj: Any) -> dict[str, Any]:
    """Return the trusted, runtime-only control projection for one tool call."""

    artifact = getattr(output_obj, "artifact", None)
    if not isinstance(artifact, dict):
        return {}
    if str(artifact.get("schema_version") or "") != "evoengine.tool-runtime-artifact/v1":
        return {}
    control = artifact.get("control_projection")
    return dict(control) if isinstance(control, dict) else {}


def _tool_runtime_machine_control(output_obj: Any) -> dict[str, Any]:
    control = _tool_runtime_control_projection(output_obj)
    machine = control.get("machine")
    return dict(machine) if isinstance(machine, dict) else {}


def _tool_completion_signal_from_output(tool_name: str, output_obj: Any) -> dict[str, Any]:
    runtime_control = _tool_runtime_control_projection(output_obj)
    trusted_signal = normalize_tool_completion_signal(
        runtime_control.get("completion_signal")
    )
    if trusted_signal:
        return trusted_signal
    parsed = _parse_tool_message_content(output_obj)
    if isinstance(parsed, dict):
        signal = normalize_tool_completion_signal(parsed.get("completion_signal") or parsed.get("completion"))
        if signal:
            return signal
    return infer_tool_completion_signal(str(tool_name or ""), parsed, ok=None)


def _capability_outcome_from_output(output_obj: Any) -> dict[str, Any]:
    """Read canonical per-call facts from the envelope without inferring legacy data."""
    parsed = _parse_tool_message_content(output_obj)
    if not isinstance(parsed, dict):
        return {}
    candidates: list[Any] = [parsed.get("capability_outcome")]
    for key in ("result", "data", "payload"):
        nested = parsed.get(key)
        if isinstance(nested, dict):
            candidates.append(nested.get("capability_outcome"))
    for candidate in candidates:
        outcome = normalize_capability_outcome(candidate)
        if outcome:
            return outcome
    return {}


def _tool_result_payload(output_obj: Any) -> dict[str, Any]:
    parsed = _parse_tool_message_content(output_obj)
    if not isinstance(parsed, dict):
        return {}
    result = parsed.get("result")
    return result if isinstance(result, dict) else parsed


def _saved_plan_context_item(output_obj: Any) -> dict[str, Any] | None:
    parsed = _parse_tool_message_content(output_obj)
    payload = _tool_result_payload(output_obj)
    source_payload = payload
    if not source_payload or not bool(source_payload.get("ok", True)):
        return None
    if source_payload.get("noop") or source_payload.get("rejected_as_final_report"):
        return None
    plan_text = str(source_payload.get("plan_text") or "").strip()
    raw_contract = source_payload.get("plan_contract")
    plan_contract = dict(raw_contract) if isinstance(raw_contract, dict) else {}
    file_id = source_payload.get("conversation_file_id") or source_payload.get("file_id") or payload.get("conversation_file_id") or payload.get("file_id")
    file_name = str(source_payload.get("file_name") or payload.get("file_name") or "执行计划.md").strip() or "执行计划.md"
    return {
        "file_id": file_id,
        "conversation_file_id": file_id,
        "name": file_name,
        "file_name": file_name,
        "ext": "md",
        "kind": "text",
        "source": "agent_generated",
        "source_type": "agent_generated",
        "mime_type": str(payload.get("mime_type") or "text/markdown; charset=utf-8"),
        "size_bytes": int(source_payload.get("size_bytes") or len(plan_text.encode("utf-8"))),
        "parse_status": "completed",
        "drawer_section": "plan_file",
        "archive_status": "pending",
        "tool_name": "save_execution_plan",
        "retention_policy": "keep",
        "text": "" if plan_contract else plan_text,
        "content_text": "" if plan_contract else plan_text,
        "content_path": str(source_payload.get("content_path") or ""),
        "status_path": str(source_payload.get("status_path") or ""),
        "plan_contract": plan_contract,
    }


def _normalize_loaded_tool_ids(raw: Any) -> list[str]:
    if not isinstance(raw, (list, tuple, set)):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for item in raw:
        tool_id = str(item or "").strip()
        if not tool_id or tool_id in seen:
            continue
        seen.add(tool_id)
        result.append(tool_id)
    return result


def _normalize_activated_skills(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        skill_id = str(item.get("skill_id") or "").strip()
        if not skill_id or skill_id in seen:
            continue
        seen.add(skill_id)
        raw_capability_ids = item.get("capability_ids")
        capability_ids = (
            [
                str(capability_id).strip()
                for capability_id in raw_capability_ids
                if str(capability_id).strip()
            ]
            if isinstance(raw_capability_ids, list)
            else []
        )
        normalized.append(
            {
                "skill_id": skill_id,
                "name": str(item.get("name") or skill_id).strip(),
                "activation_mode": str(item.get("activation_mode") or "").strip(),
                "capability_ids": capability_ids,
            }
        )
    return normalized


def _activated_skill_ids_from_state(state: NativeAgentState | dict[str, Any]) -> set[str]:
    return {item["skill_id"] for item in _normalize_activated_skills(state.get("activated_skills"))}


def _build_activated_skill_state_prompt(raw: Any) -> str:
    activated = _normalize_activated_skills(raw)
    if not activated:
        return ""
    allowed_modes = {
        "unknown",
        "guide_only",
        "already_activated",
        "executed",
        "activated",
        "runtime",
    }
    safe_skills: list[dict[str, Any]] = []
    for item in activated:
        capability_ids = [
            _safe_recovery_identifier(capability_id)
            for capability_id in (item.get("capability_ids") or [])
        ]
        projection = {
            "skill_id": _safe_recovery_identifier(item.get("skill_id")),
            "activation_mode": _safe_recovery_enum(
                item.get("activation_mode"),
                allowed_modes,
                default="unknown",
            ),
            "capability_ids": capability_ids,
        }
        projection["state_digest"] = _recovery_digest(projection)
        safe_skills.append(projection)
    payload = {
        "schema_version": "activated_skills_state_v2",
        "authority": "activate_skill_result_projection",
        "role": "read_only_observation",
        "skills": safe_skills,
        "total_count": len(activated),
        "truncated": False,
        "digest": _recovery_digest(activated),
    }
    return (
        "[ACTIVATED_SKILLS]\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n[/ACTIVATED_SKILLS]"
    )


def _extract_activate_skill_state(output: Any) -> dict[str, Any] | None:
    parsed = _parse_tool_message_content(output)
    candidates: list[Any] = []
    if isinstance(parsed, dict):
        candidates.extend([parsed.get("result"), parsed.get("model_summary"), parsed.get("summary")])
    else:
        candidates.append(parsed)
    if isinstance(output, ToolMessage):
        candidates.append(output.content)
    text = "\n".join(str(item) for item in candidates if item not in (None, ""))
    if "activation_mode:" not in text:
        return None
    skill_match = re.search(r"(?m)^skill_id:\s*`?([^`\s]+)`?\s*$", text)
    if not skill_match:
        return None
    mode_match = re.search(r"(?m)^activation_mode:\s*([^\n]+?)\s*$", text)
    name_match = re.search(r"(?m)^✅ 已(?:激活|匹配)技能:\s*([^\n]+?)\s*$", text)
    capability_ids_match = re.search(
        r"(?m)^\s*capability_ids:\s*([^\n]+?)\s*$",
        text,
    )
    capability_ids: list[str] = []
    if capability_ids_match:
        raw_capability_ids = capability_ids_match.group(1).strip()
        if raw_capability_ids and raw_capability_ids not in {"无", "none", "None"}:
            capability_ids = [
                part.strip(" `")
                for part in raw_capability_ids.split(",")
                if part.strip(" `")
            ]
    return {
        "skill_id": skill_match.group(1).strip(),
        "name": name_match.group(1).strip() if name_match else skill_match.group(1).strip(),
        "activation_mode": mode_match.group(1).strip() if mode_match else "activated",
        "capability_ids": capability_ids,
    }


def _merge_activated_skill_state(
    existing: Any,
    activated_skill: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    normalized = _normalize_activated_skills(existing)
    if not activated_skill:
        return normalized
    incoming = _normalize_activated_skills([activated_skill])
    if not incoming:
        return normalized
    replacement = incoming[0]
    merged: list[dict[str, Any]] = []
    replaced = False
    for item in normalized:
        if item.get("skill_id") == replacement["skill_id"]:
            merged.append(replacement)
            replaced = True
        else:
            merged.append(item)
    if not replaced:
        merged.append(replacement)
    return merged


def _extract_load_tools_effective_tool_ids(tool_name: str, output_obj: Any) -> list[str]:
    if str(tool_name or "") != "load_tools":
        return []
    parsed = _parse_tool_message_content(output_obj)
    if not isinstance(parsed, dict):
        return []
    if str(parsed.get("evo_control") or "") != "load_tools":
        return []
    if not bool(parsed.get("reload_required")):
        return []
    effective = _normalize_loaded_tool_ids(parsed.get("effective_tool_ids"))
    if effective:
        return effective
    return _normalize_loaded_tool_ids(parsed.get("loaded_tool_ids"))


def _extract_load_tools_requested_tool_ids(tool_name: str, input_obj: Any) -> list[str]:
    if str(tool_name or "") != "load_tools":
        return []
    payload = input_obj
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return []
    if not isinstance(payload, dict):
        return []
    return _normalize_loaded_tool_ids(payload.get("tool_ids"))


def _extract_load_tools_metrics_payload(tool_name: str, output_obj: Any) -> dict[str, Any]:
    if str(tool_name or "") != "load_tools":
        return {}
    parsed = _parse_tool_message_content(output_obj)
    if not isinstance(parsed, dict):
        return {}
    denied_raw = parsed.get("denied_tool_ids") or parsed.get("denied") or []
    denied_ids = [
        str(item.get("tool_id") or "").strip()
        if isinstance(item, dict)
        else str(item or "").strip()
        for item in denied_raw
        if (isinstance(item, dict) and str(item.get("tool_id") or "").strip())
        or (not isinstance(item, dict) and str(item or "").strip())
    ]
    return {
        "loaded": _normalize_loaded_tool_ids(parsed.get("loaded_tool_ids")),
        "already_loaded": _normalize_loaded_tool_ids(parsed.get("already_loaded_tool_ids")),
        "denied": _normalize_loaded_tool_ids(denied_ids),
        "unknown": _normalize_loaded_tool_ids(parsed.get("unknown_tool_ids")),
        "no_op": bool(parsed.get("noop")),
    }


def _tool_outcome_from_output(tool_name: str, output_obj: Any) -> dict[str, Any]:
    parsed = _parse_tool_message_content(output_obj)
    ok = True
    summary = ""
    error_kind = ""
    severity = ""
    if isinstance(parsed, dict):
        if "ok" in parsed:
            ok = bool(parsed.get("ok"))
        elif "success" in parsed:
            ok = bool(parsed.get("success"))
        elif parsed.get("error") or parsed.get("error_kind"):
            ok = False
        summary = str(parsed.get("summary") or parsed.get("message") or parsed.get("error") or "").strip()
        error_kind = str(parsed.get("error_kind") or "").strip()
        severity = str(parsed.get("severity") or "").strip().lower()
    elif isinstance(parsed, str):
        lowered = parsed.strip().lower()
        ok = not lowered.startswith(("error ", "error:", "timeouterror:", "exception:"))
        summary = "工具返回文本结果；完整内容保留在 ToolMessage。"
    if not summary:
        summary = "工具返回成功" if ok else "工具返回失败"
    summary = re.sub(r"https?://[^\s,，)）]+", "[url redacted]", summary)
    summary = re.sub(r"(?i)(endpoint|full_url|url)\s*=\s*[^\s,，)）]+", r"\1=[redacted]", summary)
    outcome: dict[str, Any] = {
        "tool": str(tool_name or "tool"),
        "ok": bool(ok),
        "summary": summary,
        "error_kind": error_kind,
        "severity": severity,
    }
    result_ref, artifact_ids, job_id, result_status = _recovery_result_refs(parsed)
    outcome["result_status"] = (
        result_status
        if result_status != "unknown"
        else ("succeeded" if ok else "failed")
    )
    if result_ref:
        outcome["result_ref"] = result_ref
    if artifact_ids:
        outcome["artifact_ids"] = artifact_ids
    if job_id:
        outcome["job_id"] = job_id
    if isinstance(parsed, dict):
        for key in ("complete", "has_more"):
            if isinstance(parsed.get(key), bool):
                outcome[key] = parsed[key]
        if parsed.get("cursor") not in (None, ""):
            outcome["cursor"] = str(parsed["cursor"])
        if isinstance(parsed.get("resources"), list):
            outcome["resources"] = list(parsed["resources"])
        result = parsed.get("result") if isinstance(parsed.get("result"), dict) else {}
        execution_result = (
            result.get("execution_result")
            if isinstance(result.get("execution_result"), dict)
            else {}
        )
        for key in ("skill_id", "script_name"):
            value = next(
                (
                    str(candidate.get(key) or "").strip()
                    for candidate in (parsed, result, execution_result)
                    if str(candidate.get(key) or "").strip()
                ),
                "",
            )
            if value:
                outcome[key] = value
    completion_signal = _tool_completion_signal_from_output(tool_name, output_obj)
    if completion_signal:
        outcome["completion_signal"] = completion_signal
    capability_outcome = _capability_outcome_from_output(output_obj)
    if capability_outcome:
        outcome["capability_outcome"] = capability_outcome
    return outcome


def _canonical_protocol_projections(
    protocol_ledger: CanonicalProtocolLedger,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Derive bounded recovery views solely from the canonical call ledger."""

    outcomes: list[dict[str, Any]] = []
    side_effects = _normalize_side_effect_ledger({})
    for completed in protocol_ledger.completed_calls():
        call_id = str(completed.get("tool_call_id") or "").strip()
        tool_name = str(completed.get("tool_name") or "").strip()
        if not call_id or not tool_name:
            continue
        outcome = _tool_outcome_from_output(
            tool_name,
            completed.get("result"),
        )
        outcome["tool_call_id"] = call_id
        outcome["arguments_digest"] = _recovery_digest(
            completed.get("tool_input") or {}
        )
        outcomes = _append_tool_outcome(outcomes, outcome)
        if _is_side_effect_tool(tool_name):
            side_effects = _append_side_effect_identity(side_effects, outcome)
    return outcomes, side_effects


def _build_tool_fallback_reply(*, error_type: str, tool_outcomes: list[dict[str, Any]]) -> str:
    if error_type == "tool_failure_isolated":
        intro = "检测到至少一个工具失败。按本轮要求，我不重试、不更换工具，直接基于已返回的工具结果给出保底总结。"
        detail = "这不是工具继续卡住；失败工具已经被隔离，当前回合已安全收敛。"
    else:
        intro = "工具调用已经返回，但最后自然语言汇总阶段超时。我先基于已返回的工具结果给出保底总结，避免整轮对话失败。"
        detail = f"收尾异常类型：`{error_type}`。这发生在工具返回之后，不是工具本身继续卡住。"

    lines = [
        intro,
        "",
        "| 工具 | 状态 | 摘要 |",
        "| --- | --- | --- |",
    ]
    for outcome in tool_outcomes:
        status = "成功" if outcome.get("ok") else "失败（已隔离）"
        summary = str(outcome.get("summary") or "").replace("\n", " ").strip()
        if outcome.get("error_kind"):
            summary = f"{summary}；error_kind={outcome.get('error_kind')}"
        lines.append(f"| `{outcome.get('tool') or 'tool'}` | {status} | {summary} |")
    any_failed = any(not bool(item.get("ok")) for item in tool_outcomes)
    lines.extend(
        [
            "",
            detail,
            "结论：本轮工具结果已被保留；"
            + ("失败工具已隔离，" if any_failed else "")
            + "你可以基于上表继续追问，或让我重新生成更完整的自然语言总结。",
        ]
    )
    return "\n".join(lines)


def _reasoning_block_id(state: NativeAgentState, segment_index: int) -> str:
    request_id = str(state.get("request_id") or "req")
    resume_count = int(state.get("resume_count") or 0)
    return f"reasoning:{request_id}:{resume_count}:{segment_index}"


_SANDBOX_SUBMIT_TOOL_PREFIX = "sandbox_submit_"


def _extract_sandbox_submission(tool_name: str, output: Any) -> dict[str, Any] | None:
    name = str(tool_name or "")
    if name != "sandbox_submit" and not name.startswith(_SANDBOX_SUBMIT_TOOL_PREFIX):
        return None
    # Scheduler state is control-plane data.  It must come from the
    # runtime-only artifact produced after CapabilityExecutor validation, not
    # from the model-visible envelope or display prose.
    machine = _tool_runtime_machine_control(output)
    candidates: list[dict[str, Any]] = [machine] if machine else []
    for candidate in candidates:
        job = candidate.get("job") if isinstance(candidate.get("job"), dict) else {}
        job_id = str(
            candidate.get("job_id")
            or candidate.get("sandbox_job_id")
            or job.get("job_id")
            or ""
        ).strip()
        if not job_id:
            continue
        app_id = str(
            candidate.get("app_id")
            or candidate.get("sandbox_app_id")
            or job.get("app_id")
            or ""
        ).strip()
        if not app_id and name.startswith(_SANDBOX_SUBMIT_TOOL_PREFIX):
            app_id = str(name[len(_SANDBOX_SUBMIT_TOOL_PREFIX):]).strip()
        status = str(
            candidate.get("status")
            or candidate.get("sandbox_status")
            or job.get("status")
            or "queued"
        ).strip() or "queued"
        execution_mode = str(
            candidate.get("agent_execution_mode")
            or job.get("agent_execution_mode")
            or "nonblocking_long"
        ).strip().lower()
        return {
            "app_id": app_id or "sandbox",
            "job_id": job_id,
            "status": status,
            "agent_execution_mode": execution_mode,
            "message": json.dumps(candidate, ensure_ascii=False, default=str),
        }
    return None


def _extract_sandbox_result_status(tool_name: str, output: Any) -> dict[str, Any] | None:
    if str(tool_name or "") != "sandbox_get_result":
        return None
    machine = _tool_runtime_machine_control(output)
    candidates: list[dict[str, Any]] = [machine] if machine else []
    for candidate in candidates:
        job_id = str(candidate.get("job_id") or candidate.get("sandbox_job_id") or "").strip()
        status = str(candidate.get("status") or candidate.get("sandbox_status") or "").strip().lower()
        if not job_id or not status:
            continue
        terminal = status in _SANDBOX_TERMINAL_STATUSES
        return {"job_id": job_id, "status": status, "terminal": terminal}
    return None


def _extract_trusted_sandbox_suspend_binding(
    tool_name: str,
    output: Any,
    *,
    task_authority_request_id: str,
    model_tool_call_id: str,
    runtime_event_id: str,
    tool_input: Any,
) -> dict[str, Any] | None:
    """Validate the runtime-only receipt that authorizes sandbox suspension.

    TaskTree action status and sandbox job lists are receipt-owned Server
    projections.  A model-authored ``task_tree_update_node`` payload can never
    establish this authority.  The adapter instead carries the exact verified
    Capability call/begin/node identity outside model content; this function
    joins it to the canonical model call and the observed runtime event.  The
    model-facing tool schema intentionally has no ``task_node_id``: runtime
    projection chooses that identity before begin/receipt.
    """

    normalized_tool_name = str(tool_name or "").strip()
    if (
        normalized_tool_name != "sandbox_submit"
        and not normalized_tool_name.startswith(_SANDBOX_SUBMIT_TOOL_PREFIX)
    ):
        return None
    normalized_authority_request_id = str(task_authority_request_id or "").strip()
    normalized_model_call_id = str(model_tool_call_id or "").strip()
    normalized_runtime_event_id = str(runtime_event_id or "").strip()
    if not (
        normalized_authority_request_id
        and normalized_model_call_id
        and normalized_runtime_event_id
    ):
        return None

    control = _tool_runtime_control_projection(output)
    receipt = control.get("capability_receipt")
    machine = control.get("machine")
    if not isinstance(receipt, dict) or not isinstance(machine, dict):
        return None
    if str(receipt.get("schema_version") or "") != (
        "evoengine.capability-runtime-receipt-control/v1"
    ):
        return None
    receipt_fields = {
        key: str(receipt.get(key) or "").strip()
        for key in (
            "request_id",
            "capability_call_id",
            "begin_call_id",
            "capability_id",
            "capability_version",
            "tool_name",
            "task_node_id",
            "result_status",
        )
    }
    if (
        not all(receipt_fields.values())
        or receipt_fields["request_id"] != normalized_authority_request_id
        or receipt_fields["tool_name"] != normalized_tool_name
        or receipt_fields["result_status"] != "pending"
        or not bool(receipt.get("task_tree_binding_required"))
    ):
        return None
    _ = tool_input
    job_id = str(machine.get("job_id") or machine.get("sandbox_job_id") or "").strip()
    status = str(machine.get("status") or machine.get("sandbox_status") or "").strip().lower()
    execution_mode = str(machine.get("agent_execution_mode") or "").strip().lower()
    if (
        not job_id
        or not status
        or status in _SANDBOX_TERMINAL_STATUSES
        or execution_mode != "nonblocking_long"
    ):
        return None
    return {
        "schema_version": "evoengine.sandbox-suspend-binding/v1",
        "job_id": job_id,
        "task_node_id": receipt_fields["task_node_id"],
        "capability_call_id": receipt_fields["capability_call_id"],
        "begin_call_id": receipt_fields["begin_call_id"],
        "capability_id": receipt_fields["capability_id"],
        "capability_version": receipt_fields["capability_version"],
        "task_authority_request_id": receipt_fields["request_id"],
        "model_tool_call_id": normalized_model_call_id,
        "runtime_event_id": normalized_runtime_event_id,
    }


def _apply_sandbox_result_status(
    pending_sandbox_jobs: list[dict[str, Any]],
    result_status: dict[str, Any],
) -> list[dict[str, Any]]:
    job_id = str(result_status.get("job_id") or "").strip()
    if not job_id:
        return pending_sandbox_jobs
    terminal = bool(result_status.get("terminal"))
    next_pending: list[dict[str, Any]] = []
    for item in pending_sandbox_jobs:
        if str(item.get("job_id") or "").strip() != job_id:
            next_pending.append(item)
            continue
        if terminal:
            continue
        updated = dict(item)
        updated["status"] = str(result_status.get("status") or updated.get("status") or "running")
        next_pending.append(updated)
    return next_pending


def _apply_sandbox_resume_statuses(
    pending_sandbox_jobs: list[dict[str, Any]],
    resume_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Reconcile checkpoint jobs with authoritative resume statuses."""

    if not isinstance(resume_payload, dict):
        return list(pending_sandbox_jobs)
    raw_jobs = resume_payload.get("jobs")
    candidates = (
        [item for item in raw_jobs if isinstance(item, dict)]
        if isinstance(raw_jobs, list)
        else [resume_payload]
    )
    reconciled = list(pending_sandbox_jobs)
    for candidate in candidates:
        # sandbox_resume_payload is scheduler-authenticated request control,
        # not model/tool output, so it is read directly at this boundary.
        job_id = str(
            candidate.get("job_id") or candidate.get("sandbox_job_id") or ""
        ).strip()
        status_value = str(
            candidate.get("status") or candidate.get("sandbox_status") or ""
        ).strip().lower()
        if job_id and status_value:
            reconciled = _apply_sandbox_result_status(
                reconciled,
                {
                    "job_id": job_id,
                    "status": status_value,
                    "terminal": status_value in _SANDBOX_TERMINAL_STATUSES,
                },
            )
    return reconciled


def _restore_hitl_resume_state_from_timeline(
    blocks: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Restore the interrupted turn from its persisted HITL timeline.

    This is deliberately scoped to one HITL continuation and does not restore
    unrelated turns.
    """

    outcomes: list[dict[str, Any]] = []
    activated_skills: list[dict[str, Any]] = []
    loaded_tool_ids: list[str] = []
    pending_sandbox_jobs: list[dict[str, Any]] = []
    side_effect_ledger = _normalize_side_effect_ledger({})
    seen_call_ids: set[str] = set()
    replay_messages: list[BaseMessage] = []

    ordered_blocks = sorted(
        [block for block in (blocks or []) if isinstance(block, dict)],
        key=lambda block: (
            int(block.get("order") or 0),
            int(block.get("timestamp") or 0),
            str(block.get("block_id") or ""),
        ),
    )
    for block in ordered_blocks:
        if str(block.get("kind") or "") != "thinking_tool":
            continue
        payload = block.get("payload") if isinstance(block.get("payload"), dict) else {}
        if str(payload.get("phase") or "") != "tool_done":
            continue
        tool_name = str(payload.get("tool_name") or "").strip()
        if not tool_name or tool_name == "request_human_input":
            continue
        call_id = str(payload.get("tool_call_id") or "").strip()
        runtime_run_id = str(payload.get("runtime_run_id") or "").strip()
        if call_id and call_id in seen_call_ids:
            continue
        if call_id:
            seen_call_ids.add(call_id)

        output_preview = payload.get("output_preview")
        severity = str(payload.get("severity") or "").strip().lower()
        outcome = _tool_outcome_from_output(tool_name, output_preview)
        if call_id:
            outcome["tool_call_id"] = call_id
        if runtime_run_id:
            outcome["runtime_run_id"] = runtime_run_id
        input_preview = payload.get("input_preview")
        outcome["arguments_digest"] = _recovery_digest(input_preview or {})
        outcome["summary"] = str(payload.get("summary") or outcome.get("summary") or "").strip()
        if severity:
            outcome["severity"] = severity
        completion_signal = normalize_tool_completion_signal(payload.get("completion_signal"))
        if completion_signal:
            outcome["completion_signal"] = completion_signal
        if output_preview not in (None, "", [], {}):
            outcome["resume_preview"] = output_preview
        outcomes = _append_tool_outcome(outcomes, outcome)

        # Legacy conversations may predate durable TurnCheckpoint storage and
        # therefore have only a UI timeline preview. Keep protocol shape for
        # compatibility, but never present that preview as the complete machine
        # result. New conversations restore the exact ToolMessages from their
        # checkpoint and do not enter this fallback.
        if output_preview not in (None, "", [], {}):
            replay_call_id = "hitl_" + _recovery_digest(
                {
                    "tool": tool_name,
                    "call_id": call_id,
                    "arguments": input_preview,
                    "result": output_preview,
                }
            )[:20]
            replay_args = input_preview if isinstance(input_preview, dict) else {}
            replay_messages.extend(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": replay_call_id,
                                "name": tool_name,
                                "args": replay_args,
                            }
                        ],
                    ),
                    ToolMessage(
                        content=json.dumps(
                            {
                                "schema_version": "evoengine.legacy-timeline-preview/v1",
                                "projection_kind": "legacy_timeline_preview",
                                "preview_authority": "ui_timeline_only",
                                "content_complete": False,
                                "full_result_recoverable": False,
                                "tool": tool_name,
                                "preview": output_preview,
                                "limitation": (
                                    "旧会话没有持久化完整 ToolMessage；此内容仅用于恢复线索，"
                                    "不得视为完整机器结果。"
                                ),
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                            default=str,
                        ),
                        tool_call_id=replay_call_id,
                        name=tool_name,
                    ),
                ]
            )
        if _is_side_effect_tool(tool_name):
            side_effect_ledger = _append_side_effect_identity(
                side_effect_ledger,
                outcome,
            )

        if tool_name == "activate_skill":
            activated_skills = _merge_activated_skill_state(
                activated_skills,
                _extract_activate_skill_state(output_preview),
            )
        loaded_tool_ids.extend(_extract_load_tools_effective_tool_ids(tool_name, output_preview))

        submission = _extract_sandbox_submission(tool_name, output_preview)
        if submission is not None:
            status = str(submission.get("status") or "").strip().lower()
            terminal = status in _SANDBOX_TERMINAL_STATUSES
            job_id = str(submission.get("job_id") or "").strip()
            if not terminal and job_id and all(
                str(item.get("job_id") or "").strip() != job_id
                for item in pending_sandbox_jobs
            ):
                pending_sandbox_jobs.append(submission)
        result_status = _extract_sandbox_result_status(tool_name, output_preview)
        if result_status is not None:
            pending_sandbox_jobs = _apply_sandbox_result_status(
                pending_sandbox_jobs,
                result_status,
            )

    return {
        "tool_outcomes": outcomes,
        "side_effect_ledger": side_effect_ledger,
        "activated_skills": activated_skills,
        "loaded_tool_ids": _normalize_loaded_tool_ids(loaded_tool_ids),
        "pending_sandbox_jobs": pending_sandbox_jobs,
        # Legacy sessions created before TurnCheckpoint still replay their
        # recorded protocol once.  Do not apply a second, Agent-local context
        # policy here; current sessions restore the checkpoint directly and
        # Server's Context Composer owns history bounds.
        "hitl_resume_messages": replay_messages,
    }


def _checkpoint_message_row(message: BaseMessage) -> dict[str, Any] | None:
    if isinstance(message, HumanMessage):
        message_type = "human"
    elif isinstance(message, SystemMessage):
        message_type = "system"
    elif isinstance(message, AIMessage):
        message_type = "ai"
    elif isinstance(message, ToolMessage):
        message_type = "tool"
    else:
        return None
    row: dict[str, Any] = {
        "type": message_type,
        "content": getattr(message, "content", ""),
    }
    message_id = str(getattr(message, "id", "") or "").strip()
    if message_id:
        row["id"] = message_id
    additional_kwargs = dict(getattr(message, "additional_kwargs", None) or {})
    if additional_kwargs:
        row["additional_kwargs"] = additional_kwargs
    if isinstance(message, AIMessage):
        row["tool_calls"] = list(getattr(message, "tool_calls", None) or [])
        row["invalid_tool_calls"] = list(
            getattr(message, "invalid_tool_calls", None) or []
        )
    if isinstance(message, ToolMessage):
        row["tool_call_id"] = str(getattr(message, "tool_call_id", "") or "")
        row["name"] = str(getattr(message, "name", "") or "")
        row["status"] = str(getattr(message, "status", "") or "")
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
                row["artifact"] = bounded_artifact
    return row


def _checkpoint_messages(rows: Any) -> list[BaseMessage]:
    if not isinstance(rows, list):
        return []
    messages: list[BaseMessage] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        message_type = str(row.get("type") or "").strip().lower()
        content = row.get("content", "")
        additional_kwargs = (
            dict(row.get("additional_kwargs") or {})
            if isinstance(row.get("additional_kwargs"), dict)
            else {}
        )
        message_id = str(row.get("id") or "").strip() or None
        if message_type == "human":
            messages.append(
                HumanMessage(
                    content=content,
                    additional_kwargs=additional_kwargs,
                    id=message_id,
                )
            )
        elif message_type == "system":
            messages.append(
                SystemMessage(
                    content=content,
                    additional_kwargs=additional_kwargs,
                    id=message_id,
                )
            )
        elif message_type == "ai":
            messages.append(
                AIMessage(
                    content=content,
                    tool_calls=list(row.get("tool_calls") or []),
                    invalid_tool_calls=list(row.get("invalid_tool_calls") or []),
                    additional_kwargs=additional_kwargs,
                    id=message_id,
                )
            )
        elif message_type == "tool":
            tool_call_id = str(row.get("tool_call_id") or "").strip()
            if tool_call_id:
                status = str(row.get("status") or "").strip().lower()
                messages.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=tool_call_id,
                        name=str(row.get("name") or "").strip() or None,
                        additional_kwargs=additional_kwargs,
                        artifact=(
                            dict(row.get("artifact") or {})
                            if isinstance(row.get("artifact"), dict)
                            else None
                        ),
                        status=(
                            status
                            if status in {"success", "error"}
                            else "success"
                        ),
                        id=message_id,
                    )
                )
    return messages


def _checkpoint_side_effect_replays(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in list(getattr(message, "tool_calls", None) or []):
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or "").strip()
            tool_name = str(call.get("name") or "").strip()
            arguments = call.get("args") if isinstance(call.get("args"), dict) else {}
            if call_id and tool_name and _is_checkpoint_replay_tool(tool_name):
                calls[call_id] = (tool_name, dict(arguments))
    replays: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        call_id = str(getattr(message, "tool_call_id", "") or "").strip()
        call = calls.get(call_id)
        if call is None:
            continue
        tool_name, arguments = call
        parsed = _parse_tool_message_content(message)
        outcome = _tool_outcome_from_output(tool_name, message)
        if not bool(outcome.get("ok")):
            continue
        replays.append(
            {
                "tool": tool_name,
                "arguments_digest": _recovery_digest(arguments),
                "content": getattr(message, "content", ""),
                "ok": True,
            }
        )
    return replays


def _same_request_usage_ledger(value: Any, *, request_id: str) -> dict[str, Any]:
    """Return one bounded usage ledger only when its request identity matches."""

    normalized_request_id = str(request_id or "").strip()
    raw_request_id = (
        str(value.get("request_id") or "").strip()
        if isinstance(value, dict)
        else ""
    )
    if raw_request_id and raw_request_id != normalized_request_id:
        return normalize_usage_ledger(None, request_id=normalized_request_id)
    normalized = normalize_usage_ledger(value, request_id=normalized_request_id)
    if str(normalized.get("request_id") or "").strip() != normalized_request_id:
        return normalize_usage_ledger(None, request_id=normalized_request_id)
    return normalized


def _build_turn_checkpoint(
    *,
    state: NativeAgentState,
    protocol_ledger: CanonicalProtocolLedger,
    pending_bundle: dict[str, Any],
    pending_sandbox_jobs: list[dict[str, Any]],
    pending_capability_authorization: dict[str, Any] | None = None,
    status: str = "waiting_human",
) -> dict[str, Any]:
    if not protocol_ledger.is_closed():
        raise RuntimeError("CANONICAL_PROTOCOL_CHECKPOINT_REQUIRES_CLOSED_LEDGER")
    continuation_messages = protocol_ledger.snapshot_messages()
    message_rows = [
        row
        for message in continuation_messages
        if (row := _checkpoint_message_row(message)) is not None
    ]
    payload: dict[str, Any] = {
        "schema_version": "evoengine.turn-checkpoint/v1",
        "request_id": str(state.get("request_id") or ""),
        "status": status,
        "user_goal": str(state.get("user_text") or ""),
        "task_state": dict(state.get("task_state") or {}),
        "loaded_tool_ids": _normalize_loaded_tool_ids(state.get("loaded_tool_ids")),
        "loaded_tool_entries": _normalize_loaded_tool_ids(
            state.get("loaded_tool_entries")
        ),
        "activated_skills": _normalize_activated_skills(
            state.get("activated_skills")
        ),
        "resource_access_ledger": normalize_resource_access_ledger(
            state.get("resource_access_ledger")
        ),
        "reference_context": _normalize_reference_context_payload(
            state.get("reference_context")
        ),
        "session_compact_checkpoint": dict(
            state.get("session_compact_checkpoint") or {}
        ),
        "pending_sandbox_jobs": list(pending_sandbox_jobs),
        "protocol_messages": message_rows,
        "pending_question_bundle": dict(pending_bundle),
        "max_turns": (
            max(1, int(state["max_turns"]))
            if state.get("max_turns") not in (None, "")
            else None
        ),
        "model_turns_used": max(0, int(state.get("model_turns_used") or 0)),
        "tool_schema_reload_count": max(
            0,
            int(state.get("tool_schema_reload_count") or 0),
        ),
        # model_usage_v2 is already bounded to MODEL_USAGE_MAX_CALLS and stores
        # numeric accounting only.  Re-scope here so a caller cannot smuggle a
        # different transport request's ledger through a continuation.
        "usage": _same_request_usage_ledger(
            state.get("usage"),
            request_id=str(state.get("request_id") or ""),
        ),
    }
    if isinstance(pending_capability_authorization, dict):
        payload["runtime_control"] = {
            "pending_capability_authorization": dict(
                pending_capability_authorization
            )
        }
    payload["checkpoint_id"] = "checkpoint_" + hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode(
            "utf-8"
        )
    ).hexdigest()[:24]
    runtime_control = payload.get("runtime_control")
    if isinstance(runtime_control, dict):
        pending = runtime_control.get("pending_capability_authorization")
        if isinstance(pending, dict):
            pending["checkpoint_id"] = payload["checkpoint_id"]
    return payload


def _restore_runtime_from_turn_checkpoint(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or value.get("schema_version") != "evoengine.turn-checkpoint/v1":
        return None
    protocol_messages = _checkpoint_messages(value.get("protocol_messages"))
    checkpoint_request_id = str(value.get("request_id") or "").strip()
    return {
        "resource_access_ledger": normalize_resource_access_ledger(
            value.get("resource_access_ledger")
        ),
        "reference_context": _normalize_reference_context_payload(
            value.get("reference_context")
        ),
        "session_compact_checkpoint": dict(
            value.get("session_compact_checkpoint") or {}
        ),
        "activated_skills": _normalize_activated_skills(
            value.get("activated_skills")
        ),
        "loaded_tool_ids": _normalize_loaded_tool_ids(value.get("loaded_tool_ids")),
        "loaded_tool_entries": _normalize_loaded_tool_ids(
            value.get("loaded_tool_entries")
        ),
        "pending_sandbox_jobs": list(value.get("pending_sandbox_jobs") or []),
        # Checkpoint protocol is transport-neutral. The request adapter below
        # decides whether a resume carries new human input (HITL) or merely a
        # scheduler-authoritative sandbox continuation.
        "checkpoint_protocol_messages": protocol_messages,
        "task_state": dict(value.get("task_state") or {}),
        "runtime_control": dict(value.get("runtime_control") or {}),
        "max_turns": (
            max(1, int(value["max_turns"]))
            if value.get("max_turns") not in (None, "")
            else None
        ),
        "model_turns_used": max(0, int(value.get("model_turns_used") or 0)),
        "tool_schema_reload_count": max(
            0,
            int(value.get("tool_schema_reload_count") or 0),
        ),
        "checkpoint_request_id": checkpoint_request_id,
        "usage": _same_request_usage_ledger(
            value.get("usage"),
            request_id=checkpoint_request_id,
        ),
    }


def _typed_capability_hitl_from_tool_output(
    output: Any,
    *,
    tool_name: str,
) -> dict[str, Any] | None:
    """Read trusted NEEDS_INPUT control only from ToolMessage.artifact."""

    if not isinstance(output, ToolMessage):
        return None
    artifact = getattr(output, "artifact", None)
    if not isinstance(artifact, dict):
        return None
    runtime_control = artifact.get("runtime_control")
    if not isinstance(runtime_control, dict):
        return None
    if str(runtime_control.get("kind") or "") != "capability_needs_input":
        return None
    if str(runtime_control.get("tool_name") or "") != str(tool_name or ""):
        raise ValueError("CAPABILITY_NEEDS_INPUT_CONTROL_INVALID")
    call = CapabilityCall.model_validate(runtime_control.get("capability_call"))
    control = CapabilityNeedsInput.model_validate(
        runtime_control.get("needs_input_control")
    )
    snapshot_payload = runtime_control.get("registry_snapshot")
    snapshot = (
        CapabilityRegistrySnapshot.model_validate(snapshot_payload)
        if isinstance(snapshot_payload, dict)
        else None
    )
    tool_call_id = str(getattr(output, "tool_call_id", "") or "").strip()
    if (
        not tool_call_id
        or call.call_id != control.pending_call_id
        or call.capability_id != control.capability_id
        or call.capability_version != control.capability_version
        or call.registry_snapshot_id != control.registry_snapshot_id
        or (
            bool(call.registry_snapshot_id)
            and (snapshot is None or snapshot.snapshot_id != call.registry_snapshot_id)
        )
    ):
        raise ValueError("CAPABILITY_AUTHORIZATION_PENDING_CALL_MISMATCH")
    return {
        "schema_version": "evoengine.pending-capability-authorization/v1",
        "tool_name": str(tool_name or ""),
        "tool_call_id": tool_call_id,
        "capability_call": call.model_dump(mode="json", exclude_none=True),
        "needs_input_control": control.model_dump(mode="json", exclude_none=True),
        **(
            {"registry_snapshot": snapshot.model_dump(mode="json")}
            if snapshot is not None
            else {}
        ),
    }


def _replace_tool_message_for_call(
    messages: list[BaseMessage],
    *,
    tool_call_id: str,
    replacement: ToolMessage,
) -> list[BaseMessage]:
    replaced = False
    result: list[BaseMessage] = []
    for message in messages:
        if (
            isinstance(message, ToolMessage)
            and str(getattr(message, "tool_call_id", "") or "") == tool_call_id
        ):
            if replaced:
                continue
            result.append(replacement)
            replaced = True
        else:
            result.append(message)
    if not replaced:
        raise RuntimeError("CAPABILITY_AUTHORIZATION_PENDING_CALL_MISMATCH")
    return result


async def _resume_pending_capability_call(
    *,
    state: NativeAgentState,
    dynamic_agent: Any,
    messages: list[BaseMessage],
) -> tuple[list[BaseMessage], dict[str, Any] | None]:
    resume_control = state.get("capability_resume_control")
    pending = state.get("pending_capability_authorization")
    if not isinstance(resume_control, dict):
        return messages, None
    if not isinstance(pending, dict):
        raise RuntimeError("CAPABILITY_AUTHORIZATION_RESUME_CONTROL_MISSING")
    checkpoint = state.get("turn_checkpoint")
    checkpoint_id = str(
        checkpoint.get("checkpoint_id") if isinstance(checkpoint, dict) else ""
    )
    call = CapabilityCall.model_validate(pending.get("capability_call"))
    tool_call_id = str(pending.get("tool_call_id") or "").strip()
    if (
        not checkpoint_id
        or checkpoint_id != str(resume_control.get("checkpoint_id") or "")
        or call.call_id != str(resume_control.get("pending_call_id") or "")
        or call.call_id
        != str(
            CapabilityNeedsInput.model_validate(
                pending.get("needs_input_control")
            ).pending_call_id
        )
        or not tool_call_id
    ):
        raise RuntimeError("CAPABILITY_AUTHORIZATION_PENDING_CALL_MISMATCH")

    tool_name = str(pending.get("tool_name") or "").strip()
    decision = str(resume_control.get("decision") or "").strip()
    if decision == "rejected":
        envelope = {
            "ok": False,
            "status": "cancelled",
            "error_kind": "authorization_denied",
            "summary": "用户未授权执行该操作",
            "capability_call_id": call.call_id,
        }
        replacement = ToolMessage(
            content=json.dumps(envelope, ensure_ascii=False, separators=(",", ":")),
            tool_call_id=tool_call_id,
            name=tool_name or None,
        )
        return (
            _replace_tool_message_for_call(
                messages,
                tool_call_id=tool_call_id,
                replacement=replacement,
            ),
            {
                "tool": tool_name,
                "tool_call_id": tool_call_id,
                "ok": False,
                "status": "authorization_denied",
            },
        )
    if decision != "approved" or not str(resume_control.get("grant_id") or ""):
        raise RuntimeError("CAPABILITY_AUTHORIZATION_RESUME_CONTROL_MISSING")

    executor = getattr(dynamic_agent, "_evo_capability_executor", None)
    if executor is None:
        raise RuntimeError("CAPABILITY_AUTHORIZATION_RESUME_CONTROL_MISSING")
    snapshot_payload = pending.get("registry_snapshot")
    if isinstance(snapshot_payload, dict):
        snapshot = CapabilityRegistrySnapshot.model_validate(snapshot_payload)
        executor = executor.scoped(snapshot)
    elif call.registry_snapshot_id:
        raise RuntimeError("CAPABILITY_AUTHORIZATION_RESUME_CONTROL_MISSING")

    result = await executor.invoke(
        call,
        task_phase=str(state.get("task_phase") or TaskPhase.EXECUTE.value),
        provider_metadata={
            "capability_authorization": {
                "grant_id": str(resume_control.get("grant_id") or ""),
                "parent_request_id": str(
                    resume_control.get("parent_request_id") or ""
                ),
                "continuation_request_id": str(
                    resume_control.get("continuation_request_id") or ""
                ),
                "pending_call_id": call.call_id,
                "request_digest": str(
                    pending.get("needs_input_control", {}).get("request_digest")
                    if isinstance(pending.get("needs_input_control"), dict)
                    else ""
                ),
                "effect_capability_id": str(
                    resume_control.get("effect_capability_id") or ""
                ),
                "effect_request_digest": str(
                    resume_control.get("effect_request_digest") or ""
                ),
            }
        },
    )
    if result.status.value == "needs_input":
        raise RuntimeError("CAPABILITY_NEEDS_INPUT_CONTROL_INVALID")
    from src.capabilities.tool_adapter import (
        capability_result_to_legacy_envelope,
        capability_result_to_runtime_artifact,
    )

    envelope = capability_result_to_legacy_envelope(
        result,
        tool_name=tool_name,
        category="deliverable",
    )
    artifact = capability_result_to_runtime_artifact(
        result,
        envelope=envelope,
        call=call,
        tool_name=tool_name,
    )
    replacement = ToolMessage(
        content=json.dumps(envelope, ensure_ascii=False, separators=(",", ":")),
        artifact=artifact,
        tool_call_id=tool_call_id,
        name=tool_name or None,
    )
    return (
        _replace_tool_message_for_call(
            messages,
            tool_call_id=tool_call_id,
            replacement=replacement,
        ),
        {
            "tool": tool_name,
            "tool_call_id": tool_call_id,
            "ok": bool(result.ok),
            "status": result.status.value,
            "capability_call_id": call.call_id,
        },
    )


def _reference_context_for_current_prompt(
    state: NativeAgentState,
    *,
    hitl_resume_messages: list[BaseMessage],
) -> dict[str, Any] | None:
    """Inject the selected-reference catalog only on its original user prompt.

    A HITL checkpoint already contains that original HumanMessage.  The answer
    bundle is a new protocol message, while the retained context remains
    available to resource tools through ``NativeAgentState``.
    """

    if hitl_resume_messages:
        return None
    return _normalize_reference_context_payload(state.get("reference_context"))


def _build_sandbox_waiting_reply(submissions: list[dict[str, Any]]) -> str:
    if not submissions:
        return "沙盒任务已提交，正在等待计算完成。"
    if len(submissions) == 1:
        item = submissions[0]
        return (
            "已提交沙盒任务，正在等待计算完成。\n"
            f"- app={item.get('app_id') or '?'}\n"
            f"- job_id={item.get('job_id') or '?'}\n"
            f"- status={item.get('status') or 'queued'}\n"
            "如果本次目标还要求结果、报告或结果文件，下一步应继续查询 sandbox_get_result；如果任务耗时较长，再向用户说明当前等待状态。"
        )
    lines = ["已提交多个沙盒任务，正在等待计算完成："]
    for item in submissions:
        lines.append(
            f"- app={item.get('app_id') or '?'} | job_id={item.get('job_id') or '?'} | status={item.get('status') or 'queued'}"
        )
    lines.append("如果本次目标还要求结果、报告或结果文件，下一步应继续查询 sandbox_get_result；如果任务耗时较长，再向用户说明当前等待状态。")
    return "\n".join(lines)


async def _run_native_agent_turn_once(
    state: NativeAgentState,
    context: NativeAgentContext,
    *,
    event_writer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    # The native runner owns its event sink directly.  It no longer obtains
    # hidden execution state or a writer from LangGraph configuration.
    writer = event_writer or (lambda _event: None)
    runtime_context = getattr(context, "context", None)
    if isinstance(runtime_context, dict):
        context = runtime_context
    elif not isinstance(context, dict):
        context = {}
    settings = context.get("settings")
    llm_base = context.get("llm_base")
    if not isinstance(settings, Settings):
        raise RuntimeError("missing native agent settings in runtime.context")
    if llm_base is None:
        raise RuntimeError("missing llm_base in runtime.context")

    protocol_ledger = state.get("_canonical_protocol_ledger")
    if not isinstance(protocol_ledger, CanonicalProtocolLedger):
        protocol_ledger = CanonicalProtocolLedger()
        state["_canonical_protocol_ledger"] = protocol_ledger
    compact_checkpoint = state.get("session_compact_checkpoint")
    compact_completion_authority = completion_authority_for_request(
        compact_checkpoint,
        request_id=str(state.get("request_id") or ""),
        task_authority=dict(state.get("task_authority") or {}),
    )
    if compact_completion_authority is not None:
        protocol_ledger.restore_terminal_seal(compact_completion_authority)

    adapter = _adapter_from_state(state)
    if compact_completion_authority is not None:
        receipt = compact_completion_authority["completion_receipt"]
        restored_summary = str(compact_completion_authority.get("summary") or "")
        completion_text = (
            restored_summary if restored_summary.strip() else "任务已完成"
        )
        state["completion_receipt"] = dict(receipt)
        if completion_text and completion_text not in adapter.final_text:
            _emit_events(
                writer,
                adapter.append_answer_delta(
                    completion_text,
                    timestamp=int(time.time() * 1000),
                    fake_stream=True,
                ),
            )
        if "[[cite:" in completion_text:
            _hydrate_citation_registry_from_reference_store(
                adapter,
                request_id=str(state.get("request_id") or ""),
                project_id=str(state.get("project_id") or "").strip() or None,
                conversation_id=str(state.get("conversation_id") or "").strip()
                or None,
            )
        if not bool(state.get("_completion_receipt_seal_emitted")):
            state["_completion_receipt_seal_emitted"] = True
            writer(
                {
                    "type": "completion_receipt_sealed",
                    "request_id": str(state.get("request_id") or ""),
                    "payload": {
                        "completion_receipt": dict(receipt),
                        "summary": completion_text,
                    },
                }
            )
        if not bool(state.get("_terminal_completion_emitted")):
            state["_terminal_completion_emitted"] = True
            writer(
                adapter.build_turn_completed(
                    cancelled=False,
                    status=_task_terminal_status_from_receipt(receipt),
                    final_reply_text=completion_text,
                    completion_receipt=dict(receipt),
                )
            )
        _transition_native_task_state(
            state,
            TaskPhase.COMPLETED,
            resume_reason="session_compact_completion_authority_restored",
            pending_job_ids=[],
        )
        return {
            "any_tool_called": True,
            **_state_from_adapter(state, adapter),
            "completion_receipt": dict(receipt),
            "pending_question_bundle": None,
            "pending_sandbox_jobs": [],
            "status": _task_terminal_status_from_receipt(receipt),
        }
    if not isinstance(state.get("task_state"), dict):
        _transition_native_task_state(state, TaskPhase.INTAKE)
    native_reasoning = is_native_reasoning_model(settings.llm_model)
    project_id = _context_project_id(context, state)
    conversation_id = _context_conversation_id(context, state)
    user_id = _context_user_id(context, state)
    task_tree_summary = ""
    task_tree_snapshot: dict[str, Any] | None = None
    task_tree_fetch_status = "not_applicable"
    if project_id and conversation_id and user_id:
        resume_payload = state.get("sandbox_resume_payload")
        resume_jobs = (
            list(resume_payload.get("jobs") or [])
            if isinstance(resume_payload, dict)
            else []
        )
        if resume_jobs:
            try:
                sandbox_promotions = await _promote_authoritative_sandbox_resume_jobs(
                    state=state,
                    context=context,
                    project_id=project_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
                state["sandbox_promotion_results"] = sandbox_promotions
                _worker_checkpoint(
                    "native_sandbox_receipts_promoted",
                    request_id=str(state.get("request_id") or ""),
                    promotion_count=len(sandbox_promotions),
                    succeeded_count=sum(
                        1
                        for item in sandbox_promotions
                        if str(item.get("status") or "") == "succeeded"
                    ),
                    rejected_count=sum(
                        1
                        for item in sandbox_promotions
                        if str(item.get("status") or "") == "promotion_rejected"
                    ),
                )
            except Exception as exc:
                state["sandbox_promotion_results"] = [
                    {
                        "ok": False,
                        "status": "promotion_runtime_unavailable",
                        "error_type": exc.__class__.__name__,
                        "summary": str(exc)[:1000],
                    }
                ]
                _worker_checkpoint(
                    "native_sandbox_receipt_promotion_error",
                    request_id=str(state.get("request_id") or ""),
                    error_type=exc.__class__.__name__,
                )
            try:
                reconciliation = reconcile_sandbox_nodes_from_authoritative_jobs(
                    project_id=project_id,
                    session_id=conversation_id,
                    user_id=user_id,
                    jobs=resume_jobs,
                )
                _worker_checkpoint(
                    "native_task_tree_sandbox_reconciled",
                    request_id=str(state.get("request_id") or ""),
                    updated_node_count=len(reconciliation.get("updated_node_ids") or []),
                    skipped_node_count=len(reconciliation.get("skipped_node_ids") or []),
                    error_count=len(reconciliation.get("errors") or []),
                )
            except Exception as exc:
                _worker_checkpoint(
                    "native_task_tree_sandbox_reconcile_error",
                    request_id=str(state.get("request_id") or ""),
                    error_type=exc.__class__.__name__,
                )
        task_tree_started_ms = int(time.time() * 1000)
        _worker_checkpoint(
            "native_task_tree_summary_start",
            request_id=str(state.get("request_id") or ""),
            project_id=project_id,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        try:
            task_tree_snapshot = get_latest_snapshot(
                project_id=project_id,
                user_id=user_id,
                session_id=conversation_id,
            )
            task_tree_fetch_status = "ok" if isinstance(task_tree_snapshot, dict) else "missing"
            task_tree_summary = build_task_tree_summary(task_tree_snapshot)
            _worker_checkpoint(
                "native_task_tree_summary_done",
                request_id=str(state.get("request_id") or ""),
                elapsed_ms=int(time.time() * 1000) - task_tree_started_ms,
                summary_chars=len(task_tree_summary or ""),
            )
        except Exception as exc:
            task_tree_summary = ""
            task_tree_snapshot = None
            task_tree_fetch_status = "unavailable"
            _worker_checkpoint(
                "native_task_tree_summary_error",
                request_id=str(state.get("request_id") or ""),
                elapsed_ms=int(time.time() * 1000) - task_tree_started_ms,
                error_type=exc.__class__.__name__,
                error=str(exc)[:300],
            )
    activated_skill_prompt = _build_activated_skill_state_prompt(state.get("activated_skills"))
    plan_execution_contract_prompt = _build_plan_execution_contract_prompt(
        state.get("conversation_files"),
    )
    continuation_messages, transcript_repairs = _normalize_tool_message_protocol(
        state.get("continuation_messages")
    )
    hitl_resume_messages, hitl_replay_repairs = _normalize_tool_message_protocol(
        state.get("hitl_resume_messages")
    )
    resource_access_ledger = normalize_resource_access_ledger(
        state.get("resource_access_ledger")
    )
    session_checkpoint_state = (
        state.get("session_compact_checkpoint")
        if isinstance(state.get("session_compact_checkpoint"), dict)
        else {}
    )
    state["session_compact_checkpoint"] = session_checkpoint_state
    update_resource_access_ledger_from_messages(
        resource_access_ledger,
        [*hitl_resume_messages, *continuation_messages],
    )
    state["resource_access_ledger"] = resource_access_ledger
    transcript_repairs.extend(hitl_replay_repairs)
    if transcript_repairs:
        _worker_checkpoint(
            "native_continuation_transcript_repaired",
            request_id=str(state.get("request_id") or ""),
            repair_count=len(transcript_repairs),
            repairs=",".join(transcript_repairs[:12]),
        )
    missing_protocol_results = [
        repair
        for repair in transcript_repairs
        if repair.startswith("missing_tool_result:")
    ]
    if missing_protocol_results:
        raise RuntimeError(
            "CANONICAL_PROTOCOL_INCOMPLETE:"
            + ",".join(missing_protocol_results[:12])
        )
    capability_discovery_requirements = (
        _build_staged_capability_discovery_requirements(
            current_user_message=str(state.get("user_text") or ""),
            conversation_files=state.get("conversation_files"),
            task_tree_snapshot=task_tree_snapshot,
            loaded_tool_ids=_normalize_loaded_tool_ids(state.get("loaded_tool_ids")),
        )
    )
    current_tool_state_prompt = (
        ""
        if continuation_messages or hitl_resume_messages
        else _build_current_turn_tool_state_prompt(
            tool_outcomes=_normalize_tool_outcomes(state.get("tool_outcomes")),
            pending_sandbox_jobs=list(state.get("pending_sandbox_jobs") or []),
        )
    )
    extra_system_prompt_parts = [str(state.get("system_prompt") or "").strip(), _task_state_prompt(state)]
    if plan_execution_contract_prompt:
        extra_system_prompt_parts.append(plan_execution_contract_prompt)
    if activated_skill_prompt:
        extra_system_prompt_parts.append(activated_skill_prompt)
    if current_tool_state_prompt:
        extra_system_prompt_parts.append(current_tool_state_prompt)
    execution_budget_prompt = _execution_budget_prompt(state)
    if execution_budget_prompt:
        extra_system_prompt_parts.append(execution_budget_prompt)
    sandbox_resume_state_prompt = _build_sandbox_resume_state_prompt(
        state.get("sandbox_resume_payload")
    )
    if sandbox_resume_state_prompt:
        extra_system_prompt_parts.append(sandbox_resume_state_prompt)
    sandbox_promotion_results = [
        dict(item)
        for item in (state.get("sandbox_promotion_results") or [])[:16]
        if isinstance(item, dict)
    ]
    if sandbox_promotion_results:
        extra_system_prompt_parts.append(
            "[SANDBOX_PROMOTION_RESULTS]\n"
            + json.dumps(
                {
                    "schema_version": "sandbox_action_promotion_v1",
                    "authority": "capability_executor_task_tree_receipt",
                    "results": sandbox_promotion_results,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n[/SANDBOX_PROMOTION_RESULTS]"
        )
    runtime_extra_system_prompt = "\n\n".join(part for part in extra_system_prompt_parts if part)

    build_started_ms = int(time.time() * 1000)
    _worker_checkpoint(
        "native_agent_build_start",
        request_id=str(state.get("request_id") or ""),
        model=str(settings.llm_model or ""),
        native_reasoning=native_reasoning,
        thinking_mode=os.environ.get("EVOENGINE_THINKING_MODE", "auto").strip().lower() or "auto",
        task_tree_summary_chars=len(task_tree_summary or ""),
        has_project_context=bool(project_id and conversation_id and user_id),
    )
    try:
        from src.skills.input_context import infer_available_inputs

        dynamic_agent = build_dynamic_agent(
            settings,
            llm_base,
            project_id,
            conversation_id,
            user_id,
            str(state.get("request_id") or "").strip() or None,
            task_authority_request_id=str(
                state.get("task_authority_request_id")
                or state.get("request_id")
                or ""
            ).strip()
            or None,
            native_reasoning=native_reasoning,
            task_tree_summary=task_tree_summary,
            task_tree_fetch_status=task_tree_fetch_status,
            extra_system_prompt=runtime_extra_system_prompt,
            task_phase=str(state.get("task_phase") or TaskPhase.INTAKE.value),
            tool_selection_query=str(state.get("user_text") or ""),
            loaded_tool_ids=_normalize_loaded_tool_ids(state.get("loaded_tool_ids")),
            current_turn_tool_names=[
                str(item.get("tool") or "")
                for item in _normalize_tool_outcomes(state.get("tool_outcomes"))
                if str(item.get("tool") or "").strip()
            ],
            sandbox_exposure_context=context.get("sandbox_exposure_context"),
            available_inputs=infer_available_inputs(
                attachments=state.get("attachments"),
                conversation_files=state.get("conversation_files"),
            ),
            activated_skill_ids=_activated_skill_ids_from_state(state),
            replayed_side_effects=_checkpoint_side_effect_replays(
                [*hitl_resume_messages, *continuation_messages]
            ),
            resource_access_ledger=resource_access_ledger,
            reference_context=_normalize_reference_context_payload(
                state.get("reference_context")
            ),
            session_checkpoint_state=session_checkpoint_state,
            capability_discovery_requirements=capability_discovery_requirements,
            protocol_ledger=protocol_ledger,
        )
        transport_request_id = str(state.get("request_id") or "").strip()
        for executor_attribute in (
            "_evo_capability_executor",
            "_evo_sandbox_capability_executor",
        ):
            executor = getattr(dynamic_agent, executor_attribute, None)
            binder = getattr(executor, "bind_transport_request_id", None)
            if callable(binder):
                binder(transport_request_id)
        existing_sandbox_context = context.get("sandbox_exposure_context")
        built_sandbox_context = getattr(
            dynamic_agent,
            "_evo_sandbox_exposure_context",
            None,
        )
        if existing_sandbox_context is None and built_sandbox_context is not None:
            context["sandbox_exposure_context"] = built_sandbox_context
        elif (
            existing_sandbox_context is not None
            and built_sandbox_context is not existing_sandbox_context
        ):
            raise RuntimeError(
                "sandbox exposure context changed within one native agent turn"
            )
        rejected_sandbox_tool_ids = set(
            _normalize_loaded_tool_ids(
                getattr(dynamic_agent, "_evo_rejected_sandbox_tool_ids", None)
            )
        )
        if rejected_sandbox_tool_ids:
            previous_entries = _normalize_loaded_tool_ids(
                state.get("loaded_tool_entries")
            )
            remaining_entries = [
                entry
                for entry in previous_entries
                if entry not in rejected_sandbox_tool_ids
                and not _dynamic_cache_entry_expanded_tools(entry).intersection(
                    rejected_sandbox_tool_ids
                )
            ]
            state["loaded_tool_entries"] = remaining_entries
            state["loaded_tool_ids"] = [
                tool_id
                for tool_id in _normalize_loaded_tool_ids(
                    state.get("loaded_tool_ids")
                )
                if tool_id not in rejected_sandbox_tool_ids
            ]
            evicted_entries = _evict_dynamic_tool_cache_ids(
                str(state.get("tool_cache_scope") or ""),
                rejected_sandbox_tool_ids,
            )
            metrics = normalize_tool_load_metrics(state.get("tool_load_metrics"))
            if previous_entries and not remaining_entries and metrics["cache_hit_count"]:
                metrics["cache_hit_count"] -= 1
                metrics["cache_miss_count"] += 1
            state["tool_load_metrics"] = metrics
            _worker_checkpoint(
                "native_stale_sandbox_schema_pruned",
                request_id=str(state.get("request_id") or ""),
                rejected_tool_ids=",".join(sorted(rejected_sandbox_tool_ids)),
                evicted_entries=",".join(sorted(evicted_entries)),
            )
        _worker_checkpoint(
            "native_agent_build_done",
            request_id=str(state.get("request_id") or ""),
            elapsed_ms=int(time.time() * 1000) - build_started_ms,
        )
    except Exception as exc:
        _worker_checkpoint(
            "native_agent_build_error",
            request_id=str(state.get("request_id") or ""),
            elapsed_ms=int(time.time() * 1000) - build_started_ms,
            error_type=exc.__class__.__name__,
            error=str(exc)[:500],
        )
        raise

    tool_schema_snapshot = getattr(dynamic_agent, "_evo_tool_schema_snapshot", None)
    if not isinstance(tool_schema_snapshot, dict):
        tool_schema_snapshot = {}
    tool_load_metrics = record_tool_schema_snapshot(
        state.get("tool_load_metrics"),
        tool_schema_snapshot,
    )
    state["tool_schema_snapshot"] = tool_schema_snapshot
    state["tool_load_metrics"] = tool_load_metrics

    # ── Skill summary thinking event ──────────────────────────
    try:
        skill_reg = get_skill_registry("skills")
        load_events = skill_reg.flush_load_events()
        if skill_reg.list_skills():
            # Count ALL enabled skills, separate those with tools vs knowledge-only
            tool_skill_ids: set[str] = set()
            tool_counts: dict[str, int] = {}
            mcp_servers: list[str] = []

            for e in (load_events or []):
                if e.phase == "tools_registered" and e.skill_id:
                    tool_skill_ids.add(e.skill_id)
                    tool_counts[e.skill_id] = tool_counts.get(e.skill_id, 0) + int(e.detail.get("tool_count", 0))
                if e.phase == "mcp_connected" and e.detail.get("status") == "connected":
                    mcp_servers.append(str(e.detail.get("server", "")))

            all_skills = skill_reg.list_skills()
            tool_count_total = sum(tool_counts.values())
            knowledge_count = len(all_skills) - len(tool_skill_ids)

            # Summary: "加载 8 个技能（2 含工具共 2 工具 + 6 知识）"
            parts: list[str] = [f"{len(all_skills)} 个技能"]
            if tool_count_total:
                parts.append(f"{len(tool_skill_ids)} 含工具（{tool_count_total} 工具）")
            if knowledge_count:
                parts.append(f"{knowledge_count} 知识参考")
            skill_summary_text = "加载 " + "、".join(parts)

            items: list[dict[str, Any]] = [
                {"label": "总技能", "value": str(len(all_skills))},
                {"label": "含工具", "value": f"{len(tool_skill_ids)} 个技能 / {tool_count_total} 工具"},
                {"label": "知识参考", "value": f"{knowledge_count} 个技能（提供代码模式和最佳实践指导）"},
            ]
            if mcp_servers:
                items.append({"label": "外部数据源", "value": ", ".join(mcp_servers)})

            writer(
                {
                    "type": "agent_thinking",
                    "phase": "skill_summary",
                    "tool_call_id": None,
                    "tool_name": None,
                    "summary": skill_summary_text,
                    "icon": "package",
                    "items": items,
                }
            )
        else:
            writer(
                {
                    "type": "agent_thinking",
                    "phase": "skill_summary",
                    "tool_call_id": None,
                    "tool_name": None,
                    "summary": "技能系统未加载到任何 Skill",
                    "icon": "package",
                    "items": [{"label": "诊断", "value": "load_events 为空，检查 EVOENGINE_SKILL_INCLUDE_TEST 和 skills/ 目录"}],
                }
            )
    except Exception as _skill_exc:
        import logging
        _log = logging.getLogger("evoengine.skill")
        _log.error("skill_summary_emit_failed: %s", _skill_exc)
        writer(
            {
                "type": "agent_thinking",
                "phase": "skill_summary",
                "tool_call_id": None,
                "tool_name": None,
                "summary": f"技能加载异常: {_skill_exc}",
                "icon": "package",
                "items": [],
            }
        )

    hitl_resume_messages, resumed_capability_outcome = (
        await _resume_pending_capability_call(
            state=state,
            dynamic_agent=dynamic_agent,
            messages=hitl_resume_messages,
        )
    )
    if resumed_capability_outcome is not None:
        _worker_checkpoint(
            "native_capability_authorization_resumed",
            request_id=str(state.get("request_id") or ""),
            tool=str(resumed_capability_outcome.get("tool") or ""),
            status=str(resumed_capability_outcome.get("status") or ""),
            ok=bool(resumed_capability_outcome.get("ok")),
        )

    current_user_text = str(state.get("user_text") or "")
    if isinstance(state.get("human_answer_bundle"), dict):
        current_user_text = (
            current_user_text
            + "\n\n[HUMAN_ANSWER_BUNDLE]\n"
            + json.dumps(
                state["human_answer_bundle"],
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
            + "\n[/HUMAN_ANSWER_BUNDLE]"
        )
    prompt_reference_context = _reference_context_for_current_prompt(
        state,
        hitl_resume_messages=hitl_resume_messages,
    )
    user_prompt = compose_user_prompt(
        current_user_text,
        history=None,
        project_summary=str(state.get("project_summary") or ""),
        attachments=state.get("attachments"),
        conversation_files=state.get("conversation_files"),
        reference_context=prompt_reference_context,
        native_reasoning=native_reasoning,
    )
    user_prompt_parts = compose_user_prompt_ledger_parts(
        current_user_text,
        history=None,
        project_summary=str(state.get("project_summary") or ""),
        attachments=state.get("attachments"),
        conversation_files=state.get("conversation_files"),
        reference_context=prompt_reference_context,
        native_reasoning=native_reasoning,
    )
    resumed_protocol_messages: list[BaseMessage] = []
    if continuation_messages:
        resumed_protocol_messages = list(continuation_messages)
    elif hitl_resume_messages:
        resumed_protocol_messages = [
            *hitl_resume_messages,
            HumanMessage(user_prompt),
        ]
    recovery_state_prompt = ""
    existing_recovery_envelope = state.get("recovery_state_envelope")
    if isinstance(existing_recovery_envelope, dict) and existing_recovery_envelope:
        interruption = (
            dict(existing_recovery_envelope.get("recovery") or {})
            if isinstance(existing_recovery_envelope.get("recovery"), dict)
            else {}
        )
        refreshed_recovery_envelope = _build_recovery_state_envelope(
            state=state,
            interruption=interruption,
            continuation_messages=continuation_messages,
            task_tree_snapshot=task_tree_snapshot,
            task_tree_fetch_status=task_tree_fetch_status,
            tool_schema_snapshot=tool_schema_snapshot,
            tool_outcomes=state.get("tool_outcomes"),
            pending_sandbox_jobs=state.get("pending_sandbox_jobs"),
        )
        state["recovery_state_envelope"] = refreshed_recovery_envelope
        recovery_state_prompt = _render_recovery_state_envelope(
            refreshed_recovery_envelope
        )
        _worker_checkpoint(
            "native_recovery_envelope_prepared",
            request_id=str(state.get("request_id") or ""),
            envelope_chars=len(recovery_state_prompt),
            envelope_digest=_recovery_digest(refreshed_recovery_envelope),
            committed_message_count=len(continuation_messages),
            observed_tool_result_count=int(
                ((refreshed_recovery_envelope.get("actions") or {}).get("observed_tool_results") or {}).get(
                    "total_count"
                )
                or 0
            ),
        )

    # Keep the durable protocol transcript untouched.  Only the list sent to
    # the model is composed; later tool/model events continue to append to the
    # complete transcript used by checkpoint and recovery state.
    protocol_messages: list[BaseMessage] = list(resumed_protocol_messages) or [
        *history_protocol_messages(state.get("history")),
        HumanMessage(user_prompt),
    ]
    observed_discovery_requirement_ids = (
        _observed_capability_discovery_requirement_ids(protocol_messages)
    )
    prefetch_messages = _capability_prefetch_messages(
        getattr(dynamic_agent, "_evo_capability_prefetch", ""),
        current_user_message=str(state.get("user_text") or ""),
        seen_requirement_ids=observed_discovery_requirement_ids,
    )
    if prefetch_messages:
        protocol_messages.extend(prefetch_messages)
        prefetch_payloads = [
            _parse_tool_message_content(message.content)
            for message in prefetch_messages
            if isinstance(message, ToolMessage)
        ]
        _worker_checkpoint(
            "native_capability_prefetch_attached",
            request_id=str(state.get("request_id") or ""),
            requirement_count=len(prefetch_payloads),
            candidate_count=sum(
                len(payload.get("candidates") or [])
                for payload in prefetch_payloads
            ),
            payload_chars=sum(
                len(str(message.content or ""))
                for message in prefetch_messages
                if isinstance(message, ToolMessage)
            ),
        )
    protocol_ledger.seed_if_empty(protocol_messages)
    continuation_messages = protocol_ledger.snapshot_messages()
    state["continuation_messages"] = list(continuation_messages)
    # Context composition has exactly one runtime authority: the middleware
    # immediately before each model call.  Do not pre-compose here and then
    # compose the same messages a second time inside the Agent.
    composed_protocol_messages = list(continuation_messages)
    messages: list[BaseMessage] = list(composed_protocol_messages)
    if recovery_state_prompt:
        messages.append(SystemMessage(content=recovery_state_prompt))

    agent_context_parts = getattr(dynamic_agent, "_evo_context_ledger_parts", None)
    context_usage = build_context_usage(
        request_id=str(state.get("request_id") or ""),
        model=str(settings.llm_model or ""),
        task_phase=str(state.get("task_phase") or TaskPhase.INTAKE.value),
        task_state=state.get("task_state") if isinstance(state.get("task_state"), dict) else {},
        context_window=DEFAULT_CONTEXT_WINDOW_TOKENS,
        parts=[
            *(agent_context_parts if isinstance(agent_context_parts, list) else []),
            *(
                [
                    _continuation_messages_ledger_part(composed_protocol_messages)
                ]
                if resumed_protocol_messages or len(protocol_messages) > 1
                else user_prompt_parts
            ),
            *(
                [
                    {
                        "category": "recovery_state",
                        "label": "Recovery state envelope",
                        "content": recovery_state_prompt,
                        "source_type": "runtime_recovery_state",
                        "priority": 100,
                    }
                ]
                if recovery_state_prompt
                else []
            ),
        ],
    )
    context_usage["context_composition"] = {
        "schema_version": "evoengine.context-composition/v2",
        "authority": "agent_middleware",
        "precomposition_applied": False,
        "original_message_count": len(protocol_messages),
        "composed_message_count": len(protocol_messages),
    }
    context_usage["tool_schema_ledger"] = tool_load_metrics
    context_snapshot_id = build_context_snapshot_id(context_usage)
    context_usage["snapshot_id"] = context_snapshot_id
    selected_tools: list[str] = []
    selected_tool_packs = getattr(dynamic_agent, "_evo_selected_tool_packs", None)
    if isinstance(agent_context_parts, list):
        for part in agent_context_parts:
            if not isinstance(part, dict) or str(part.get("category") or "") != "tool_definitions":
                continue
            meta = part.get("meta")
            if isinstance(meta, dict) and isinstance(meta.get("tool_names"), list):
                selected_tools = [str(name) for name in meta.get("tool_names") or [] if str(name or "").strip()]
                break
    context_plan = build_context_plan(
        request_id=str(state.get("request_id") or ""),
        task_phase=str(state.get("task_phase") or TaskPhase.INTAKE.value),
        task_state=state.get("task_state") if isinstance(state.get("task_state"), dict) else {},
        submode="",
        modules=[
            {
                "key": str(item.get("category") or item.get("id") or ""),
                "label": str(item.get("label") or item.get("category") or ""),
                "tokens_estimated": int(item.get("tokens_estimated") or 0),
                "included": item.get("included", True) is not False,
                "truncated": bool(item.get("truncated")),
            }
            for item in context_usage.get("items", [])
            if isinstance(item, dict)
        ],
        selected_tools=selected_tools,
        selected_tool_groups=(
            [str(name) for name in selected_tool_packs]
            if isinstance(selected_tool_packs, list)
            else sorted({name.split("_", 1)[0] for name in selected_tools if "_" in name})
        ),
        budget={
            "context_window": DEFAULT_CONTEXT_WINDOW_TOKENS,
            "task_state": state.get("task_state") if isinstance(state.get("task_state"), dict) else {},
            "loaded_tool_ids": _normalize_loaded_tool_ids(state.get("loaded_tool_ids")),
            "tool_schema_reload_count": int(state.get("tool_schema_reload_count") or 0),
            "tool_schema_snapshot_id": str(tool_schema_snapshot.get("snapshot_id") or ""),
            "tool_schema_tokens": int(tool_schema_snapshot.get("schema_tokens") or 0),
            "context_composition": dict(context_usage["context_composition"]),
        },
    )
    writer({"type": "context_plan", **context_plan})
    writer(
        {
            "type": "context_usage",
            **context_usage,
            "meta": {
                "native_reasoning": native_reasoning,
                "history_count": len(state.get("history") or []),
                "attachment_count": len(state.get("attachments") or []),
                "conversation_file_count": len(state.get("conversation_files") or []),
                "seed_timeline_blocks": len(state.get("timeline_blocks") or []),
                "loaded_tool_ids": _normalize_loaded_tool_ids(state.get("loaded_tool_ids")),
                "tool_schema_reload_count": int(state.get("tool_schema_reload_count") or 0),
                "tool_schema_snapshot_id": str(tool_schema_snapshot.get("snapshot_id") or ""),
                "tool_schema_tokens": int(tool_schema_snapshot.get("schema_tokens") or 0),
                "tool_load_metrics": normalize_tool_load_metrics(state.get("tool_load_metrics")),
            },
        }
    )
    thinking_mode = os.environ.get("EVOENGINE_THINKING_MODE", "auto").strip().lower() or "auto"
    stream_recovery_mode = bool(state.get("stream_recovery_mode"))
    stream_recovery_attempts = int(state.get("stream_recovery_attempts") or 0)
    stream_recovery_total_attempts = int(
        state.get("stream_recovery_total_attempts") or 0
    )
    stream_first_chunk_timeout_s = (
        _native_stream_recovery_timeout_seconds()
        if stream_recovery_mode
        else _native_stream_first_chunk_timeout_seconds()
    )
    stream_stall_timeout_s = (
        _native_stream_stall_timeout_seconds()
    )
    recovery_prefix_suppressor = (
        _RecoveryPrefixSuppressor(state.get("stream_recovery_replay_prefix") or "")
        if stream_recovery_mode
        else None
    )
    _worker_checkpoint(
        "native_model_context_prepared",
        request_id=str(state.get("request_id") or ""),
        model=str(settings.llm_model or ""),
        thinking_mode=thinking_mode,
        native_reasoning=native_reasoning,
        main_read_timeout_s=os.environ.get("EVO_MAIN_LLM_READ_TIMEOUT_SEC", ""),
        main_max_tokens=os.environ.get("EVO_MAIN_LLM_MAX_TOKENS", "") or "provider_default",
        main_disable_streaming=str(
            getattr(llm_base, "disable_streaming", os.environ.get("EVO_MAIN_LLM_DISABLE_STREAMING", "false"))
        ),
        stream_recovery_mode=stream_recovery_mode,
        stream_recovery_attempts=stream_recovery_attempts,
        stream_recovery_total_attempts=stream_recovery_total_attempts,
        llm_transport_generation=int(
            context.get("llm_transport_generation") or 0
        ),
        native_stream_first_chunk_timeout_s=stream_first_chunk_timeout_s,
        native_stream_stall_timeout_s=stream_stall_timeout_s,
        recursion_limit=EVO_AGENT_RECURSION_LIMIT,
        message_count=len(messages),
        message_content_chars=_messages_content_chars(messages),
        history_count=len(state.get("history") or []),
        seed_timeline_blocks=len(state.get("timeline_blocks") or []),
        conversation_file_count=len(state.get("conversation_files") or []),
        attachment_count=len(state.get("attachments") or []),
    )

    marker_stripper = _StreamMarkerStripper()

    def _safe_emit_token(text: str) -> str:
        cleaned = marker_stripper.feed(text)
        return _EVO_MARKER_RE.sub("", cleaned)

    reasoning_segment_index = 0
    reasoning_id = _reasoning_block_id(state, reasoning_segment_index)
    reasoning_text = ""
    reasoning_open = False
    reasoning_needs_new_segment = False
    pending_bundle: dict[str, Any] | None = None
    pending_sandbox_jobs: list[dict[str, Any]] = list(state.get("pending_sandbox_jobs") or [])
    pending_tool_inputs: dict[str, Any] = {}
    current_batch_tool_calls: dict[str, str] = {}
    deferred_hitl: tuple[dict[str, Any], int] | None = None
    deferred_capability_hitl: tuple[dict[str, Any], int] | None = None
    active_tool_runs: dict[str, tuple[str, int]] = {}
    anonymous_tools_active = 0
    tools_active = 0
    request_id = str(state.get("request_id") or "")
    emitted_session_checkpoint_ids: set[str] = set()
    usage_ledger = normalize_usage_ledger(state.get("usage"), request_id=request_id)
    # The native graph receives nested LangChain model events as well as main
    # agent events.  Mark the ledger complete for this graph once those nested
    # events are recorded below; server-side model calls remain separate cost
    # events and do not change the existing user-credit aggregate.
    usage_ledger["coverage"] = "all_model_calls"
    adapter.record_usage_snapshot(usage_ledger)
    total_input_tokens = int(usage_ledger.get("cumulative_input_tokens") or 0)
    total_output_tokens = int(usage_ledger.get("cumulative_output_tokens") or 0)
    last_model_name = str(usage_ledger.get("model") or "")
    answer_emitted_in_call = False
    model_call_answer_start_chars = len(_adapter_answer_text(adapter))
    model_call_start_block_ids = {
        str(block.get("block_id") or "")
        for block in adapter.snapshot_blocks()
        if str(block.get("block_id") or "")
    }
    last_model_answer_text = ""
    last_model_finish_reason = ""
    last_model_had_tool_calls = False
    last_model_had_invalid_tool_calls = False
    final_answer_committed = False
    task_complete_accepted = False
    task_complete_summary = ""
    task_complete_deferred_at_ms: int | None = None
    any_tool_called = bool(state.get("_canonical_any_tool_called"))
    held_answer_text = ""
    tool_rounds_completed = 0
    tool_outcomes, side_effect_ledger = _canonical_protocol_projections(
        protocol_ledger
    )
    state["tool_outcomes"] = list(tool_outcomes)
    state["side_effect_ledger"] = side_effect_ledger
    tool_result_context: list[dict[str, Any]] = []
    activated_skills = _normalize_activated_skills(state.get("activated_skills"))
    trusted_sandbox_suspend_bindings: dict[str, dict[str, Any]] = {}
    pending_sandbox_bookkeeping_observed = False
    pending_sandbox_result_still_pending = False
    tool_schema_reload_count = int(state.get("tool_schema_reload_count") or 0)
    stream_event_count = 0
    model_call_index = next_model_call_index(usage_ledger, request_id=request_id) - 1
    model_call_started_ms: int | None = None
    model_call_last_chunk_ms: int | None = None
    model_call_last_progress_ms: int | None = None
    model_call_last_progress_log_ms: int | None = None
    model_call_inflight = False
    model_call_chunk_count = 0
    model_call_empty_chunk_count = 0
    model_call_reasoning_chars = 0
    model_call_reasoning_chunk_count = 0
    model_call_text_chars = 0
    model_call_text_chunk_count = 0
    model_call_tool_call_chunks = 0
    model_call_last_chunk_kind = ""
    model_call_last_progress_kind = ""
    model_call_last_chunk_text_chars = 0
    model_call_last_chunk_reasoning_chars = 0
    model_call_last_chunk_tool_call_chunks = 0
    model_call_input_chars = 0
    nested_model_calls: dict[str, tuple[int, int]] = {}
    compaction_model_calls: dict[str, tuple[int, int]] = {}

    def _sync_canonical_runtime_projection() -> dict[str, Any] | None:
        """Project canonical protocol facts into resumable runtime state.

        Event callbacks may enrich UI blocks and metrics, but this projection
        is sufficient even when LangChain omits an ``on_tool_end`` callback
        for a middleware short-circuit.
        """

        nonlocal continuation_messages
        nonlocal current_batch_tool_calls
        nonlocal tool_outcomes
        nonlocal side_effect_ledger
        nonlocal task_complete_accepted
        nonlocal task_complete_summary
        nonlocal pending_sandbox_jobs
        nonlocal pending_sandbox_bookkeeping_observed
        nonlocal pending_sandbox_result_still_pending

        runtime_event_ids = {
            str(item.get("tool_call_id") or "").strip(): str(
                item.get("runtime_run_id") or ""
            ).strip()
            for item in tool_outcomes
            if str(item.get("tool_call_id") or "").strip()
            and str(item.get("runtime_run_id") or "").strip()
        }
        continuation_messages = protocol_ledger.snapshot_messages()
        state["continuation_messages"] = list(continuation_messages)
        current_batch_tool_calls = protocol_ledger.pending_calls()
        tool_outcomes, side_effect_ledger = _canonical_protocol_projections(
            protocol_ledger
        )
        for item in tool_outcomes:
            call_id = str(item.get("tool_call_id") or "").strip()
            runtime_event_id = runtime_event_ids.get(call_id)
            if runtime_event_id:
                item["runtime_run_id"] = runtime_event_id
        side_effect_ledger = _normalize_side_effect_ledger({})
        for item in tool_outcomes:
            if _is_side_effect_tool(str(item.get("tool") or "")):
                side_effect_ledger = _append_side_effect_identity(
                    side_effect_ledger,
                    item,
                )
        state["tool_outcomes"] = list(tool_outcomes)
        state["side_effect_ledger"] = side_effect_ledger

        seal = protocol_ledger.terminal_seal()
        if seal is None:
            return None
        receipt = seal.get("completion_receipt")
        if not isinstance(receipt, dict):
            return None
        public_receipt = _public_task_completion_receipt(receipt)
        task_complete_accepted = True
        sealed_summary = str(seal.get("summary") or "")
        task_complete_summary = (
            sealed_summary if sealed_summary.strip() else "任务已完成"
        )
        state["completion_receipt"] = dict(receipt)
        if public_receipt is not None:
            compact_state = state.get("session_compact_checkpoint")
            attached_checkpoint = attach_completion_authority_to_checkpoint(
                compact_state,
                completion_authority=seal,
                request_id=request_id,
                task_authority=dict(state.get("task_authority") or {}),
            )
            if attached_checkpoint is not None:
                if isinstance(compact_state, dict):
                    compact_state.clear()
                    compact_state.update(attached_checkpoint)
                    attached_checkpoint = compact_state
                else:
                    state["session_compact_checkpoint"] = attached_checkpoint
                checkpoint_id = str(
                    attached_checkpoint.get("checkpoint_id") or ""
                )
                if (
                    checkpoint_id
                    and checkpoint_id not in emitted_session_checkpoint_ids
                ):
                    emitted_session_checkpoint_ids.add(checkpoint_id)
                    writer(
                        {
                            "type": "session_compact_checkpoint",
                            "request_id": request_id,
                            "payload": {
                                "checkpoint": dict(attached_checkpoint)
                            },
                        }
                    )
        # A trusted TaskComplete seal is terminal. Late local sandbox
        # bookkeeping cannot downgrade it.
        pending_sandbox_jobs = []
        pending_sandbox_bookkeeping_observed = False
        pending_sandbox_result_still_pending = False
        if public_receipt is not None and not bool(
            state.get("_completion_receipt_seal_emitted")
        ):
            state["_completion_receipt_seal_emitted"] = True
            writer(
                {
                    "type": "completion_receipt_sealed",
                    "request_id": request_id,
                    "payload": {
                        "completion_receipt": public_receipt,
                        "summary": task_complete_summary,
                    },
                }
            )
        return seal

    async def _await_canonical_tool_commit(
        *,
        observed_call_id: str,
        runtime_run_id: str,
    ) -> dict[str, Any] | None:
        """Wait for the outer middleware commit behind an observational event.

        LangChain emits ``on_tool_end`` from inside the wrapped handler.  The
        callback can therefore reach this consumer a few event-loop turns
        before the outermost ``CanonicalProtocolMiddleware`` regains control
        and commits the returned ToolMessage.  The event is never allowed to
        author protocol state; we only wait briefly for the real ledger write.
        """

        deadline = asyncio.get_running_loop().time() + 1.0
        exact_ids = {
            value
            for value in (
                str(observed_call_id or "").strip(),
                str(runtime_run_id or "").strip(),
            )
            if value
        }
        while True:
            _sync_canonical_runtime_projection()
            matched = next(
                (
                    row
                    for row in protocol_ledger.completed_calls()
                    if str(row.get("tool_call_id") or "") in exact_ids
                ),
                None,
            )
            if matched is not None:
                return matched
            if asyncio.get_running_loop().time() >= deadline:
                return None
            await asyncio.sleep(0.001)

    def _record_model_call_usage(
        output_obj: Any,
        *,
        call_index: int,
        status: str = "completed",
        started_at_ms: int | None = None,
        completed_at_ms: int | None = None,
        recovery_call: bool | None = None,
        scope: str = "agent_main",
    ) -> dict[str, Any]:
        nonlocal usage_ledger, total_input_tokens, total_output_tokens, last_model_name
        extracted = extract_model_usage(output_obj)
        call = build_model_call_record(
            request_id=request_id,
            call_index=call_index,
            usage=extracted,
            scope=scope,
            provider="deepseek",
            status=status,
            started_at_ms=started_at_ms,
            completed_at_ms=completed_at_ms,
            context_snapshot_id=context_snapshot_id,
            tool_schema_reload_count=tool_schema_reload_count,
            tool_schema_snapshot_id=str(tool_schema_snapshot.get("snapshot_id") or ""),
            tool_schema_tokens=int(tool_schema_snapshot.get("schema_tokens") or 0),
            recovery_mode=stream_recovery_mode if recovery_call is None else recovery_call,
        )
        usage_ledger = append_model_call(usage_ledger, call, request_id=request_id)
        # Checkpoints are built from state so keep the exact current-request
        # ledger synchronized after every provider-backed model call.
        state["usage"] = usage_ledger
        total_input_tokens = int(usage_ledger.get("cumulative_input_tokens") or 0)
        total_output_tokens = int(usage_ledger.get("cumulative_output_tokens") or 0)
        if str(extracted.get("model") or ""):
            last_model_name = str(extracted.get("model") or "")
        adapter.record_usage_snapshot(usage_ledger)
        writer({"type": "agent_usage", **usage_ledger})
        return extracted

    def _start_next_reasoning_segment() -> None:
        nonlocal reasoning_segment_index, reasoning_id, reasoning_text, reasoning_needs_new_segment
        reasoning_segment_index += 1
        reasoning_id = _reasoning_block_id(state, reasoning_segment_index)
        reasoning_text = ""
        reasoning_needs_new_segment = False

    def _emit_reasoning_delta(delta: str, *, timestamp_ms: int) -> None:
        nonlocal reasoning_text, reasoning_open, reasoning_needs_new_segment
        text = str(delta or "")
        if not text:
            return
        if reasoning_needs_new_segment or (not reasoning_open and reasoning_text):
            _start_next_reasoning_segment()
        reasoning_text = f"{reasoning_text}{text}"
        payload = {
            "tool_call_id": reasoning_id,
            "phase": "reasoning",
            "tool_name": "",
            "summary": reasoning_text,
            "icon": "bolt",
            "items": [],
            "input_preview": None,
            "output_preview": None,
            "timestamp": timestamp_ms,
        }
        _emit_events(
            writer,
            adapter.upsert_thinking_block(
                block_id=reasoning_id,
                kind="thinking_reasoning",
                payload=payload,
                timestamp=timestamp_ms,
                completed=False,
            ),
        )
        reasoning_open = True

    def _complete_reasoning(*, timestamp_ms: int) -> None:
        nonlocal reasoning_open, reasoning_needs_new_segment
        if not reasoning_open:
            return
        payload = {
            "tool_call_id": reasoning_id,
            "phase": "reasoning_done",
            "tool_name": "",
            "summary": reasoning_text,
            "icon": "bolt",
            "items": [],
            "input_preview": None,
            "output_preview": None,
            "timestamp": timestamp_ms,
        }
        _emit_events(
            writer,
            adapter.upsert_thinking_block(
                block_id=reasoning_id,
                kind="thinking_reasoning",
                payload=payload,
                timestamp=timestamp_ms,
                completed=True,
            ),
        )
        reasoning_open = False
        reasoning_needs_new_segment = True

    def _open_hitl_interrupt(
        tool_input: Any,
        *,
        timestamp_ms: int,
        pending_capability_authorization: dict[str, Any] | None = None,
    ) -> None:
        nonlocal pending_bundle
        normalized_input = tool_input if isinstance(tool_input, dict) else {}
        typed_control = (
            pending_capability_authorization.get("needs_input_control")
            if isinstance(pending_capability_authorization, dict)
            else None
        )
        if isinstance(typed_control, dict):
            bundle = CapabilityNeedsInput.model_validate(
                typed_control
            ).question_bundle
        else:
            bundle = HumanQuestionBundle(
                bundle_id=f"{state.get('request_id') or 'req'}_bundle_{int(timestamp_ms)}",
                bundle_title=str(normalized_input.get("bundle_title") or "").strip()
                or "需要你补充信息",
                bundle_summary=str(normalized_input.get("bundle_summary") or ""),
                questions=normalized_input.get("questions") or [],
            )
        pending_bundle = bundle.model_dump(mode="json")
        missing_fields = [
            str(question.get("question_id") or question.get("id") or "").strip()
            for question in pending_bundle.get("questions") or []
            if isinstance(question, dict) and bool(question.get("required", True))
        ]
        task_fields = _transition_native_task_state(
            state,
            TaskPhase.SUSPENDED,
            resume_reason="awaiting_human_input",
            missing_fields=[field for field in missing_fields if field],
        )
        pending_bundle["task_context"] = task_fields["task_state"]
        _complete_reasoning(timestamp_ms=timestamp_ms)
        turn_checkpoint = _build_turn_checkpoint(
            state=state,
            protocol_ledger=protocol_ledger,
            pending_bundle=pending_bundle,
            pending_sandbox_jobs=pending_sandbox_jobs,
            pending_capability_authorization=pending_capability_authorization,
        )
        state["turn_checkpoint"] = turn_checkpoint
        writer(
            {
                "type": "turn_checkpoint",
                "request_id": str(state.get("request_id") or ""),
                "payload": {"checkpoint": turn_checkpoint},
            }
        )
        _emit_events(
            writer,
            adapter.open_interrupt(
                bundle=pending_bundle,
                thread_id=str(state.get("request_id") or ""),
                resume_token=str(state.get("request_id") or ""),
                timestamp=timestamp_ms,
            ),
        )

    def _prepare_answer_delta(raw_text: str) -> str:
        answer_delta = _safe_emit_token(raw_text)
        if not answer_delta:
            return ""
        if stream_recovery_mode and recovery_prefix_suppressor is not None:
            answer_delta = recovery_prefix_suppressor.feed(answer_delta)
        return answer_delta

    def _emit_answer_delta(answer_delta: str, *, timestamp_ms: int, fake_stream: bool = False) -> None:
        nonlocal answer_emitted_in_call
        if not answer_delta:
            return
        if reasoning_open:
            _complete_reasoning(timestamp_ms=timestamp_ms)
        answer_emitted_in_call = True
        _emit_events(
            writer,
            adapter.append_answer_delta(answer_delta, timestamp=timestamp_ms, fake_stream=fake_stream),
        )

    def _emit_answer_text(raw_text: str, *, timestamp_ms: int, fake_stream: bool = False) -> None:
        _emit_answer_delta(
            _prepare_answer_delta(raw_text),
            timestamp_ms=timestamp_ms,
            fake_stream=fake_stream,
        )

    async def _emit_answer_text_paced(raw_text: str, *, timestamp_ms: int) -> None:
        answer_delta = _prepare_answer_delta(raw_text)
        if not answer_delta:
            return
        chunk_chars = _native_fake_stream_chunk_chars(len(answer_delta))
        delay_s = _native_fake_stream_delay_seconds()
        offset = 0
        while offset < len(answer_delta):
            piece = answer_delta[offset: offset + chunk_chars]
            offset += chunk_chars
            _emit_answer_delta(piece, timestamp_ms=int(time.time() * 1000), fake_stream=True)
            if delay_s > 0 and offset < len(answer_delta):
                await asyncio.sleep(delay_s)

    def _sealed_completion_update() -> dict[str, Any] | None:
        nonlocal last_model_answer_text
        nonlocal last_model_had_tool_calls
        nonlocal final_answer_committed

        seal = _sync_canonical_runtime_projection()
        if seal is None:
            return None
        receipt = seal.get("completion_receipt")
        if not isinstance(receipt, dict):
            return None
        public_receipt = _public_task_completion_receipt(receipt)
        sealed_summary = str(seal.get("summary") or "")
        completion_text = (
            sealed_summary if sealed_summary.strip() else "任务已完成"
        )
        if "[[cite:" in completion_text:
            _hydrate_citation_registry_from_reference_store(
                adapter,
                request_id=str(state.get("request_id") or ""),
                project_id=str(state.get("project_id") or "").strip() or None,
                conversation_id=str(state.get("conversation_id") or "").strip()
                or None,
            )
        if completion_text not in last_model_answer_text:
            _emit_answer_text(
                ("\n\n" if last_model_answer_text else "") + completion_text,
                timestamp_ms=int(time.time() * 1000),
            )
        last_model_answer_text = completion_text
        last_model_had_tool_calls = False
        final_answer_committed = True
        _complete_reasoning(timestamp_ms=int(time.time() * 1000))
        if not bool(state.get("_terminal_completion_emitted")):
            state["_terminal_completion_emitted"] = True
            writer(
                adapter.build_turn_completed(
                    cancelled=False,
                    status=_task_terminal_status_from_receipt(receipt),
                    final_reply_text=completion_text,
                    completion_receipt=public_receipt,
                )
            )
        _transition_native_task_state(
            state,
            TaskPhase.COMPLETED,
            resume_reason="task_complete_receipt_sealed",
            pending_job_ids=[],
        )
        return {
            "any_tool_called": True,
            **_state_from_adapter(state, adapter),
            "completion_receipt": dict(receipt),
            "pending_question_bundle": None,
            "pending_sandbox_jobs": [],
            "status": _task_terminal_status_from_receipt(receipt),
        }

    def _flush_recovery_prefix(*, timestamp_ms: int) -> None:
        if not stream_recovery_mode or recovery_prefix_suppressor is None:
            return
        _emit_answer_delta(
            recovery_prefix_suppressor.flush(),
            timestamp_ms=timestamp_ms,
            fake_stream=True,
        )

    def _flush_held_answer_text(*, timestamp_ms: int) -> None:
        nonlocal held_answer_text
        if not held_answer_text:
            return
        buffered = held_answer_text
        held_answer_text = ""
        _emit_answer_text(buffered, timestamp_ms=timestamp_ms)
        trailing_text = _EVO_MARKER_RE.sub("", marker_stripper.flush())
        if trailing_text:
            _emit_answer_delta(trailing_text, timestamp_ms=timestamp_ms)

    def _model_stream_wait_state(now_ms: int) -> tuple[float, str, float, int | None]:
        if model_call_last_progress_ms is None:
            base_ms = model_call_started_ms or now_ms
            since_ms = now_ms - base_ms
            timeout_s = stream_first_chunk_timeout_s
            phase = "before_first_progress"
        else:
            since_ms = now_ms - model_call_last_progress_ms
            timeout_s = stream_stall_timeout_s
            phase = "after_progress"
        remaining_s = max(0.0, timeout_s - (since_ms / 1000.0))
        return remaining_s, phase, timeout_s, since_ms

    event_stream: Any = None
    deferred_tool_end_events: list[dict[str, Any]] = []
    event_backlog: list[dict[str, Any]] = []
    try:
        restored_completion = _sealed_completion_update()
        if restored_completion is not None:
            return restored_completion
        stream_config = {"recursion_limit": EVO_AGENT_RECURSION_LIMIT}
        try:
            event_stream = dynamic_agent.astream_events(
                {"messages": messages},
                version="v2",
                config=stream_config,
            )
        except TypeError:
            event_stream = dynamic_agent.astream_events({"messages": messages}, version="v2")
        event_iter = event_stream.__aiter__()
        while True:
            watchdog_phase = ""
            watchdog_timeout_s = 0.0
            watchdog_since_progress_ms: int | None = None
            try:
                if event_backlog:
                    event = event_backlog.pop(0)
                elif model_call_inflight:
                    wait_now_ms = int(time.time() * 1000)
                    remaining_s, watchdog_phase, watchdog_timeout_s, watchdog_since_progress_ms = _model_stream_wait_state(wait_now_ms)
                    if watchdog_timeout_s > 0:
                        if remaining_s <= 0:
                            raise asyncio.TimeoutError()
                        event = await asyncio.wait_for(event_iter.__anext__(), timeout=remaining_s)
                    else:
                        event = await event_iter.__anext__()
                else:
                    event = await event_iter.__anext__()
            except StopAsyncIteration:
                if deferred_tool_end_events:
                    _sync_canonical_runtime_projection()
                    completed_ids = {
                        str(row.get("tool_call_id") or "").strip()
                        for row in protocol_ledger.completed_calls()
                    }
                    ready_events = [
                        {**item, "_evo_deferred_canonical_event": True}
                        for item in deferred_tool_end_events
                        if str(item.get("_evo_observed_call_id") or "").strip()
                        in completed_ids
                    ]
                    if len(ready_events) != len(deferred_tool_end_events):
                        missing_ids = [
                            str(item.get("_evo_observed_call_id") or "").strip()
                            or str(item.get("run_id") or "").strip()
                            or str(item.get("name") or "").strip()
                            or "unknown"
                            for item in deferred_tool_end_events
                            if str(item.get("_evo_observed_call_id") or "").strip()
                            not in completed_ids
                        ]
                        raise RuntimeError(
                            "CANONICAL_PROTOCOL_UNCOMMITTED_TOOL_EVENT:"
                            + ",".join(missing_ids)
                        )
                    deferred_tool_end_events = []
                    event_backlog.extend(ready_events)
                    continue
                break
            except asyncio.TimeoutError as exc:
                now_ms = int(time.time() * 1000)
                elapsed_ms = (now_ms - model_call_started_ms) if model_call_started_ms else None
                since_last_chunk_ms = (now_ms - model_call_last_chunk_ms) if model_call_last_chunk_ms else None
                since_last_progress_ms = (
                    (now_ms - model_call_last_progress_ms)
                    if model_call_last_progress_ms is not None
                    else ((now_ms - model_call_started_ms) if model_call_started_ms else watchdog_since_progress_ms)
                )
                _worker_checkpoint(
                    "native_model_stream_stall",
                    request_id=str(state.get("request_id") or ""),
                    call_index=model_call_index,
                    phase=watchdog_phase,
                    timeout_s=watchdog_timeout_s,
                    elapsed_ms=elapsed_ms,
                    since_last_chunk_ms=since_last_chunk_ms,
                    since_last_progress_ms=since_last_progress_ms,
                    stream_event_count=stream_event_count,
                    chunk_count=model_call_chunk_count,
                    empty_chunk_count=model_call_empty_chunk_count,
                    reasoning_chars=model_call_reasoning_chars,
                    reasoning_chunk_count=model_call_reasoning_chunk_count,
                    text_chars=model_call_text_chars,
                    text_chunk_count=model_call_text_chunk_count,
                    tool_call_chunks=model_call_tool_call_chunks,
                    last_chunk_kind=model_call_last_chunk_kind,
                    last_progress_kind=model_call_last_progress_kind,
                    last_chunk_text_chars=model_call_last_chunk_text_chars,
                    last_chunk_reasoning_chars=model_call_last_chunk_reasoning_chars,
                    last_chunk_tool_call_chunks=model_call_last_chunk_tool_call_chunks,
                    input_chars=model_call_input_chars,
                    tool_outcome_count=len(tool_outcomes),
                )
                raise NativeModelStreamStall(
                    call_index=model_call_index,
                    phase=watchdog_phase,
                    timeout_s=watchdog_timeout_s,
                    elapsed_ms=elapsed_ms,
                    since_last_progress_ms=since_last_progress_ms,
                ) from exc
            if (
                deferred_tool_end_events
                and not bool(event.get("_evo_deferred_canonical_event"))
            ):
                _sync_canonical_runtime_projection()
                completed_ids = {
                    str(row.get("tool_call_id") or "").strip()
                    for row in protocol_ledger.completed_calls()
                }
                ready_events = [
                    {**item, "_evo_deferred_canonical_event": True}
                    for item in deferred_tool_end_events
                    if str(item.get("_evo_observed_call_id") or "").strip()
                    in completed_ids
                ]
                if ready_events:
                    ready_ids = {
                        str(item.get("_evo_observed_call_id") or "").strip()
                        for item in ready_events
                    }
                    deferred_tool_end_events = [
                        item
                        for item in deferred_tool_end_events
                        if str(item.get("_evo_observed_call_id") or "").strip()
                        not in ready_ids
                    ]
                    event_backlog = [*ready_events[1:], event, *event_backlog]
                    event = ready_events[0]
            stream_event_count += 1
            name = str(event.get("event") or "")
            now_ms = int(time.time() * 1000)

            # Tool implementations may invoke their own LLMs (for example the
            # retrieval subagent planning queries and synthesizing an Evidence
            # Pack). LangChain surfaces those nested model events on the same
            # stream. They are tool internals, not user-visible assistant
            # output, so they must not update the main answer/reasoning blocks,
            # token usage, or native stream watchdog state.
            parent_run_ids = {
                str(parent_id or "").strip()
                for parent_id in (event.get("parent_ids") or [])
                if str(parent_id or "").strip()
            }
            event_tags = {
                str(tag or "").strip()
                for tag in (event.get("tags") or [])
                if str(tag or "").strip()
            }
            session_compaction_model = bool(
                name.startswith("on_chat_model_")
                and SESSION_COMPACTION_MODEL_TAG in event_tags
            )
            if session_compaction_model:
                # Session checkpoint generation is intentionally a strict,
                # non-streaming model call.  It can legitimately take longer
                # than the main stream's first-progress deadline because no
                # chunk exists before the complete forced tool call arrives.
                # Keep its usage in the common ledger, but leave liveness to
                # the compactor's own HTTP timeout and reconnect budget.
                nested_run_id = str(event.get("run_id") or "")
                if name == "on_chat_model_start":
                    model_call_index += 1
                    compaction_model_calls[nested_run_id] = (
                        model_call_index,
                        now_ms,
                    )
                    _worker_checkpoint(
                        "native_session_compaction_model_start",
                        request_id=request_id,
                        call_index=model_call_index,
                        input_chars=_object_size_chars(
                            event.get("data", {}).get("input")
                        ),
                    )
                elif name == "on_chat_model_end":
                    nested_index, nested_started_ms = compaction_model_calls.pop(
                        nested_run_id,
                        (model_call_index + 1, now_ms),
                    )
                    if nested_index > model_call_index:
                        model_call_index = nested_index
                    _record_model_call_usage(
                        event.get("data", {}).get("output"),
                        call_index=nested_index,
                        scope="agent_session_compaction",
                        started_at_ms=nested_started_ms,
                        completed_at_ms=now_ms,
                    )
                    _worker_checkpoint(
                        "native_session_compaction_model_end",
                        request_id=request_id,
                        call_index=nested_index,
                        elapsed_ms=max(0, now_ms - nested_started_ms),
                    )
                elif name == "on_chat_model_error":
                    nested_index, nested_started_ms = compaction_model_calls.pop(
                        nested_run_id,
                        (model_call_index + 1, now_ms),
                    )
                    if nested_index > model_call_index:
                        model_call_index = nested_index
                    _record_model_call_usage(
                        None,
                        call_index=nested_index,
                        scope="agent_session_compaction",
                        status="failed",
                        started_at_ms=nested_started_ms,
                        completed_at_ms=now_ms,
                    )
                continue
            nested_tool_model = bool(
                name.startswith("on_chat_model_")
                and tools_active > 0
                and (
                    bool(parent_run_ids & set(active_tool_runs))
                    if parent_run_ids
                    else True
                )
            )
            if nested_tool_model:
                nested_run_id = str(event.get("run_id") or "")
                if name == "on_chat_model_start":
                    model_call_index += 1
                    nested_model_calls[nested_run_id] = (model_call_index, now_ms)
                elif name == "on_chat_model_end":
                    nested_index, nested_started_ms = nested_model_calls.pop(
                        nested_run_id,
                        (model_call_index + 1, now_ms),
                    )
                    if nested_index > model_call_index:
                        model_call_index = nested_index
                    _record_model_call_usage(
                        event.get("data", {}).get("output"),
                        call_index=nested_index,
                        scope="agent_tool_internal",
                        started_at_ms=nested_started_ms,
                        completed_at_ms=now_ms,
                    )
                elif name == "on_chat_model_error":
                    nested_index, nested_started_ms = nested_model_calls.pop(
                        nested_run_id,
                        (model_call_index + 1, now_ms),
                    )
                    if nested_index > model_call_index:
                        model_call_index = nested_index
                    _record_model_call_usage(
                        None,
                        call_index=nested_index,
                        scope="agent_tool_internal",
                        status="failed",
                        started_at_ms=nested_started_ms,
                        completed_at_ms=now_ms,
                    )
                continue

            if (
                name == "on_chat_model_start"
                and tools_active > 0
                and parent_run_ids
            ):
                # LangGraph cannot advance to the next main-model node while a
                # tool is still executing.  A main-model start whose ancestry
                # contains none of the active tool run IDs therefore proves
                # that one or more on_tool_end observation events were lost.
                # Reconcile only the event bookkeeping: never invent a tool
                # result or mark a task/source successful.
                stale_runs = list(active_tool_runs.items())
                stale_ages_ms = [
                    max(0, now_ms - int(started_ms or now_ms))
                    for _run_id, (_tool_name, started_ms) in stale_runs
                ]
                logger.warning(
                    "native_stale_tool_runs_reclaimed request_id=%s count=%d run_ids=%s",
                    state.get("request_id") or "",
                    len(stale_runs) + anonymous_tools_active,
                    ",".join(run_id for run_id, _details in stale_runs),
                )
                _worker_checkpoint(
                    "native_stale_tool_runs_reclaimed",
                    request_id=str(state.get("request_id") or ""),
                    stale_count=len(stale_runs) + anonymous_tools_active,
                    stale_run_ids=",".join(run_id for run_id, _details in stale_runs),
                    stale_tool_names=",".join(
                        details[0] for _run_id, details in stale_runs
                    ),
                    max_age_ms=max(stale_ages_ms, default=0),
                )
                for stale_run_id in active_tool_runs:
                    pending_tool_inputs.pop(stale_run_id, None)
                active_tool_runs.clear()
                anonymous_tools_active = 0
                tools_active = 0

            if name == "on_chat_model_start":
                _record_model_turn(state)
                session_checkpoint = state.get("session_compact_checkpoint")
                checkpoint_id = (
                    str(session_checkpoint.get("checkpoint_id") or "")
                    if isinstance(session_checkpoint, dict)
                    else ""
                )
                if (
                    checkpoint_id
                    and checkpoint_id not in emitted_session_checkpoint_ids
                    and str(session_checkpoint.get("created_by_request_id") or "")
                    == request_id
                ):
                    emitted_session_checkpoint_ids.add(checkpoint_id)
                    writer(
                        {
                            "type": "session_compact_checkpoint",
                            "request_id": request_id,
                            "payload": {"checkpoint": dict(session_checkpoint)},
                        }
                    )
                model_call_index += 1
                model_call_started_ms = now_ms
                model_call_last_chunk_ms = None
                model_call_last_progress_ms = None
                model_call_last_progress_log_ms = None
                model_call_inflight = True
                model_call_chunk_count = 0
                model_call_empty_chunk_count = 0
                model_call_reasoning_chars = 0
                model_call_reasoning_chunk_count = 0
                model_call_text_chars = 0
                model_call_text_chunk_count = 0
                model_call_tool_call_chunks = 0
                model_call_last_chunk_kind = ""
                model_call_last_progress_kind = ""
                model_call_last_chunk_text_chars = 0
                model_call_last_chunk_reasoning_chars = 0
                model_call_last_chunk_tool_call_chunks = 0
                model_call_input_chars = _object_size_chars(event.get("data", {}).get("input"))
                _worker_checkpoint(
                    "native_model_start",
                    request_id=str(state.get("request_id") or ""),
                    call_index=model_call_index,
                    stream_event_count=stream_event_count,
                    model=str(settings.llm_model or ""),
                    thinking_mode=thinking_mode,
                    native_reasoning=native_reasoning,
                    input_chars=model_call_input_chars,
                    tool_outcome_count=len(tool_outcomes),
                    tools_active=tools_active,
                    tool_rounds_completed=tool_rounds_completed,
                    answer_emitted=answer_emitted_in_call,
                )
                _flush_held_answer_text(timestamp_ms=now_ms)
                _emit_events(writer, adapter.close_active_answer_block(timestamp=now_ms))
                answer_emitted_in_call = False
                model_call_answer_start_chars = len(_adapter_answer_text(adapter))
                model_call_start_block_ids = {
                    str(block.get("block_id") or "")
                    for block in adapter.snapshot_blocks()
                    if str(block.get("block_id") or "")
                }
                final_answer_committed = False
                held_answer_text = ""
                continue

            if name == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if not isinstance(chunk, AIMessageChunk):
                    continue
                model_call_chunk_count += 1
                if model_call_started_ms is None:
                    model_call_started_ms = now_ms
                model_call_last_chunk_ms = now_ms
                blocks = getattr(chunk, "content_blocks", None) or []
                raw_content = getattr(chunk, "content", None)
                if not blocks and isinstance(raw_content, str) and raw_content:
                    blocks = [{"type": "text", "text": raw_content}]
                extra = getattr(chunk, "additional_kwargs", None) or {}
                reasoning_from_kwargs = str(extra.get("reasoning_content") or "")
                tool_call_chunks = getattr(chunk, "tool_call_chunks", None) or []
                chunk_kind_parts: list[str] = []
                chunk_reasoning_chars = len(reasoning_from_kwargs)
                chunk_text_chars = 0
                chunk_tool_call_chunks = len(tool_call_chunks or [])
                if reasoning_from_kwargs:
                    chunk_kind_parts.append("reasoning_kwargs")
                if tool_call_chunks:
                    chunk_kind_parts.append("tool_call")
                meaningful_progress = bool(reasoning_from_kwargs or tool_call_chunks)
                model_call_tool_call_chunks += len(tool_call_chunks or [])
                if reasoning_from_kwargs:
                    model_call_reasoning_chars += len(reasoning_from_kwargs)
                    _emit_reasoning_delta(reasoning_from_kwargs, timestamp_ms=now_ms)

                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    block_type = str(block.get("type") or "")
                    if block_type == "reasoning":
                        if reasoning_from_kwargs:
                            continue
                        raw = block.get("reasoning")
                        if raw is None:
                            raw = block.get("text", "")
                        reasoning_delta = str(raw or "")
                        if reasoning_delta:
                            meaningful_progress = True
                            chunk_reasoning_chars += len(reasoning_delta)
                            chunk_kind_parts.append("reasoning_block")
                        model_call_reasoning_chars += len(reasoning_delta)
                        _emit_reasoning_delta(reasoning_delta, timestamp_ms=now_ms)
                        continue
                    if block_type != "text" or tools_active > 0:
                        continue
                    text = str(block.get("text") or "")
                    if not text:
                        continue
                    meaningful_progress = True
                    chunk_text_chars += len(text)
                    chunk_kind_parts.append("text")
                    model_call_text_chars += len(text)
                    if held_answer_text:
                        held_answer_text = f"{held_answer_text}{text}"
                        continue
                    visible_text, json_candidate = _split_trailing_json_candidate(text)
                    if visible_text:
                        _emit_answer_text(visible_text, timestamp_ms=now_ms)
                    if json_candidate:
                        held_answer_text = json_candidate
                if not chunk_kind_parts:
                    chunk_kind_parts.append("empty")
                model_call_last_chunk_kind = "+".join(dict.fromkeys(chunk_kind_parts))
                model_call_last_chunk_text_chars = chunk_text_chars
                model_call_last_chunk_reasoning_chars = chunk_reasoning_chars
                model_call_last_chunk_tool_call_chunks = chunk_tool_call_chunks
                if chunk_reasoning_chars > 0:
                    model_call_reasoning_chunk_count += 1
                if chunk_text_chars > 0:
                    model_call_text_chunk_count += 1
                if not meaningful_progress:
                    model_call_empty_chunk_count += 1
                    continue
                first_progress = model_call_last_progress_ms is None
                model_call_last_progress_ms = now_ms
                progress_kind_parts: list[str] = []
                if chunk_tool_call_chunks > 0:
                    progress_kind_parts.append("tool_call")
                if chunk_reasoning_chars > 0:
                    progress_kind_parts.append("reasoning")
                if chunk_text_chars > 0:
                    progress_kind_parts.append("text")
                model_call_last_progress_kind = "+".join(progress_kind_parts) or model_call_last_chunk_kind
                if first_progress:
                    _worker_checkpoint(
                        "native_model_first_chunk",
                        request_id=str(state.get("request_id") or ""),
                        call_index=model_call_index,
                        elapsed_ms=(now_ms - model_call_started_ms) if model_call_started_ms else 0,
                        stream_event_count=stream_event_count,
                        reasoning_chars=model_call_reasoning_chars,
                        reasoning_chunk_count=model_call_reasoning_chunk_count,
                        text_chars=model_call_text_chars,
                        text_chunk_count=model_call_text_chunk_count,
                        tool_call_chunks=model_call_tool_call_chunks,
                        last_chunk_kind=model_call_last_chunk_kind,
                        last_progress_kind=model_call_last_progress_kind,
                        empty_chunk_count=model_call_empty_chunk_count,
                    )
                    model_call_last_progress_log_ms = now_ms
                elif model_call_last_progress_log_ms is None or now_ms - model_call_last_progress_log_ms >= 30_000:
                    _worker_checkpoint(
                        "native_model_stream_progress",
                        request_id=str(state.get("request_id") or ""),
                        call_index=model_call_index,
                        elapsed_ms=(now_ms - model_call_started_ms) if model_call_started_ms else 0,
                        stream_event_count=stream_event_count,
                        chunk_count=model_call_chunk_count,
                        empty_chunk_count=model_call_empty_chunk_count,
                        reasoning_chars=model_call_reasoning_chars,
                        reasoning_chunk_count=model_call_reasoning_chunk_count,
                        text_chars=model_call_text_chars,
                        text_chunk_count=model_call_text_chunk_count,
                        tool_call_chunks=model_call_tool_call_chunks,
                        last_chunk_kind=model_call_last_chunk_kind,
                        last_progress_kind=model_call_last_progress_kind,
                        last_chunk_text_chars=model_call_last_chunk_text_chars,
                        last_chunk_reasoning_chars=model_call_last_chunk_reasoning_chars,
                        last_chunk_tool_call_chunks=model_call_last_chunk_tool_call_chunks,
                    )
                    model_call_last_progress_log_ms = now_ms
                continue

            if name == "on_chat_model_end":
                had_model_start = model_call_inflight
                model_call_inflight = False
                if not had_model_start:
                    model_call_index += 1
                trailing_text = _safe_emit_token(marker_stripper.flush())
                if trailing_text:
                    if stream_recovery_mode and recovery_prefix_suppressor is not None:
                        trailing_text = recovery_prefix_suppressor.feed(trailing_text)
                    if reasoning_open:
                        _complete_reasoning(timestamp_ms=now_ms)
                    if trailing_text:
                        answer_emitted_in_call = True
                        _emit_events(
                            writer,
                            adapter.append_answer_delta(
                                trailing_text,
                                timestamp=now_ms,
                                fake_stream=stream_recovery_mode,
                            ),
                        )
                output_obj = event.get("data", {}).get("output")
                _sync_canonical_runtime_projection()
                # This clears the shadow state used by any later outer rebuild.
                # The already-sent envelope remains an immutable recovery-entry
                # observation inside this ReAct invocation; newer AI/tool
                # messages are appended after it and therefore supersede it.
                state["recovery_state_envelope"] = None
                extracted_usage = _record_model_call_usage(
                    output_obj,
                    call_index=model_call_index,
                    started_at_ms=model_call_started_ms if had_model_start else now_ms,
                    completed_at_ms=now_ms,
                )
                input_tokens = int(extracted_usage.get("input_tokens") or 0)
                output_tokens = int(extracted_usage.get("output_tokens") or 0)
                model_name = str(extracted_usage.get("model") or "")
                final_tool_calls = getattr(output_obj, "tool_calls", None) or []
                current_batch_tool_calls = protocol_ledger.pending_calls()
                invalid_tool_calls = getattr(output_obj, "invalid_tool_calls", None) or []
                final_tool_call_count = len(final_tool_calls) if isinstance(final_tool_calls, (list, tuple)) else 0
                invalid_tool_call_count = len(invalid_tool_calls) if isinstance(invalid_tool_calls, (list, tuple)) else 0
                finish_reason = _response_finish_reason(output_obj)
                _worker_checkpoint(
                    "native_model_end",
                    request_id=str(state.get("request_id") or ""),
                    call_index=model_call_index,
                    elapsed_ms=(now_ms - model_call_started_ms) if model_call_started_ms else 0,
                    since_last_chunk_ms=(now_ms - model_call_last_chunk_ms) if model_call_last_chunk_ms else None,
                    since_last_progress_ms=(now_ms - model_call_last_progress_ms) if model_call_last_progress_ms else None,
                    stream_event_count=stream_event_count,
                    chunk_count=model_call_chunk_count,
                    empty_chunk_count=model_call_empty_chunk_count,
                    reasoning_chars=model_call_reasoning_chars,
                    reasoning_chunk_count=model_call_reasoning_chunk_count,
                    text_chars=model_call_text_chars,
                    text_chunk_count=model_call_text_chunk_count,
                    tool_call_chunks=model_call_tool_call_chunks,
                    last_chunk_kind=model_call_last_chunk_kind,
                    last_progress_kind=model_call_last_progress_kind,
                    last_chunk_text_chars=model_call_last_chunk_text_chars,
                    last_chunk_reasoning_chars=model_call_last_chunk_reasoning_chars,
                    last_chunk_tool_call_chunks=model_call_last_chunk_tool_call_chunks,
                    input_chars=model_call_input_chars,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model_name=model_name,
                    answer_emitted=answer_emitted_in_call,
                    final_tool_calls=final_tool_call_count,
                    final_tool_call_names=_tool_call_names(final_tool_calls),
                    invalid_tool_calls=invalid_tool_call_count,
                    invalid_tool_call_summary=_invalid_tool_call_summary(invalid_tool_calls),
                    finish_reason=finish_reason,
                )
                if model_call_tool_call_chunks > 0 and final_tool_call_count == 0:
                    _worker_checkpoint(
                        "native_tool_call_parse_failed",
                        request_id=str(state.get("request_id") or ""),
                        call_index=model_call_index,
                        tool_call_chunks=model_call_tool_call_chunks,
                        invalid_tool_calls=invalid_tool_call_count,
                        invalid_tool_call_summary=_invalid_tool_call_summary(invalid_tool_calls),
                        finish_reason=finish_reason,
                        text_chars=model_call_text_chars,
                        reasoning_chars=model_call_reasoning_chars,
                    )
                # Some providers do not stream text deltas but still return the
                # complete AIMessage at model end. Preserve that native text.
                if not answer_emitted_in_call and not (
                    stream_recovery_mode
                    and recovery_prefix_suppressor is not None
                    and recovery_prefix_suppressor.received_text
                ):
                    if stream_recovery_mode:
                        await _emit_answer_text_paced(_extract_full_output_text(output_obj), timestamp_ms=now_ms)
                    else:
                        _emit_answer_text(_extract_full_output_text(output_obj), timestamp_ms=now_ms)
                _flush_recovery_prefix(timestamp_ms=now_ms)
                current_call_answer_text = _answer_text_since(
                    adapter,
                    model_call_answer_start_chars,
                )
                recovered_existing_prefix = (
                    recovery_prefix_suppressor.recovered_existing_prefix
                    if stream_recovery_mode
                    and recovery_prefix_suppressor is not None
                    else ""
                )
                last_model_answer_text = (
                    f"{recovered_existing_prefix}{current_call_answer_text}"
                ).strip()
                last_model_finish_reason = finish_reason
                last_model_had_invalid_tool_calls = invalid_tool_call_count > 0
                last_model_had_tool_calls = bool(
                    final_tool_call_count > 0
                    or invalid_tool_call_count > 0
                    or model_call_tool_call_chunks > 0
                )
                final_answer_committed = bool(
                    last_model_answer_text
                    and not last_model_had_tool_calls
                    and not _finish_reason_is_truncated(last_model_finish_reason)
                )
                if stream_recovery_mode:
                    stream_recovery_mode = False
                    stream_recovery_attempts = 0
                    state["stream_recovery_mode"] = False
                    state["stream_recovery_attempts"] = 0
                    state["stream_recovery_replay_prefix"] = ""
                    stream_first_chunk_timeout_s = _native_stream_first_chunk_timeout_seconds()
                    stream_stall_timeout_s = _native_stream_stall_timeout_seconds()
                continue

            if name == "on_chat_model_error":
                had_model_start = model_call_inflight
                model_call_inflight = False
                if not had_model_start:
                    model_call_index += 1
                _record_model_call_usage(
                    None,
                    call_index=model_call_index,
                    status="failed",
                    started_at_ms=model_call_started_ms if had_model_start else now_ms,
                    completed_at_ms=now_ms,
                )
                continue

            if name == "on_tool_start":
                model_call_inflight = False
                tool_name = str(event.get("name") or "")
                run_id = str(event.get("run_id") or "")
                tool_input = event.get("data", {}).get("input")
                if isinstance(tool_input, dict) and isinstance(tool_input.get("input"), (dict, str)):
                    tool_input = tool_input["input"]
                if held_answer_text:
                    if _held_json_matches_tool_input(held_answer_text, tool_input):
                        held_answer_text = ""
                    else:
                        _flush_held_answer_text(timestamp_ms=now_ms)
                if run_id:
                    pending_tool_inputs[run_id] = tool_input
                if tool_name == "request_human_input":
                    any_tool_called = True
                    logger.info(
                        "native_tool_start request_id=%s run_id=%s tool=%s input_type=%s input_chars=%d hitl=true",
                        state.get("request_id") or "",
                        run_id,
                        tool_name,
                        type(tool_input).__name__,
                        _object_size_chars(tool_input),
                    )
                    _worker_checkpoint(
                        "native_tool_start",
                        request_id=str(state.get("request_id") or ""),
                        run_id=run_id,
                        tool=tool_name,
                        input_type=type(tool_input).__name__,
                        input_chars=_object_size_chars(tool_input),
                        hitl=True,
                    )
                    continue
                any_tool_called = True
                if run_id:
                    active_tool_runs.setdefault(run_id, (tool_name, now_ms))
                else:
                    anonymous_tools_active += 1
                tools_active = len(active_tool_runs) + anonymous_tools_active
                logger.info(
                    "native_tool_start request_id=%s run_id=%s tool=%s input_type=%s input_chars=%d tools_active=%d",
                    state.get("request_id") or "",
                    run_id,
                    tool_name,
                    type(tool_input).__name__,
                    _object_size_chars(tool_input),
                    tools_active,
                )
                _worker_checkpoint(
                    "native_tool_start",
                    request_id=str(state.get("request_id") or ""),
                    run_id=run_id,
                    tool=tool_name,
                    input_type=type(tool_input).__name__,
                    input_chars=_object_size_chars(tool_input),
                    tools_active=tools_active,
                )
                _complete_reasoning(timestamp_ms=now_ms)
                _emit_events(writer, adapter.close_active_answer_block(timestamp=now_ms))
                labels = {"icon": "bolt"}
                try:
                    from src.agents.lead_agent import _get_tool_labels  # local import keeps public surface small

                    labels = _get_tool_labels(tool_name)
                except Exception:  # noqa: BLE001
                    pass
                payload = {
                    "tool_call_id": "",
                    "runtime_run_id": run_id,
                    "phase": "tool_start",
                    "tool_name": tool_name,
                    "summary": summarize_tool_start(tool_name, tool_input),
                    "icon": labels.get("icon") or "bolt",
                    "items": [],
                    "input_preview": build_tool_input_preview(tool_input),
                    "output_preview": None,
                    "timestamp": now_ms,
                }
                _emit_events(
                    writer,
                    adapter.upsert_thinking_block(
                        block_id=f"tool:{run_id or int(now_ms)}",
                        kind="thinking_tool",
                        payload=payload,
                        timestamp=now_ms,
                        completed=False,
                    ),
                )
                continue

            if name == "on_tool_error":
                tool_name = str(event.get("name") or "")
                run_id = str(event.get("run_id") or "")
                _sync_canonical_runtime_projection()
                completed_rows = protocol_ledger.completed_calls()
                matched = next(
                    (
                        row
                        for row in completed_rows
                        if str(row.get("tool_call_id") or "") == run_id
                    ),
                    None,
                )
                if matched is None:
                    same_name_rows = [
                        row
                        for row in completed_rows
                        if str(row.get("tool_name") or "") == tool_name
                    ]
                    matched = same_name_rows[0] if len(same_name_rows) == 1 else None
                if matched is None:
                    _worker_checkpoint(
                        "native_tool_error_waiting_for_canonical_commit",
                        request_id=str(state.get("request_id") or ""),
                        run_id=run_id,
                        tool=tool_name,
                    )
                    continue
                matched_call_id = str(matched.get("tool_call_id") or "")
                output = matched.get("result")
                _worker_checkpoint(
                    "native_tool_error_observed",
                    request_id=str(state.get("request_id") or ""),
                    run_id=run_id,
                    tool=tool_name,
                    tool_call_id=matched_call_id,
                    error_kind=_tool_message_error_kind(output),
                )
                event = {
                    **event,
                    "event": "on_tool_end",
                    "_evo_tool_error": True,
                    "data": {**dict(event.get("data") or {}), "output": output},
                }
                name = "on_tool_end"

            if name == "on_tool_end":
                tool_name = str(event.get("name") or "")
                run_id = str(event.get("run_id") or "")
                observed_tool_error = bool(event.get("_evo_tool_error"))
                if run_id:
                    pending_tool_inputs.pop(run_id, None)
                observed_output = event.get("data", {}).get("output")
                observed_call_id = _tool_message_call_id(observed_output)
                canonical_row = await _await_canonical_tool_commit(
                    observed_call_id=observed_call_id,
                    runtime_run_id=run_id,
                )
                if canonical_row is None:
                    if not bool(event.get("_evo_deferred_canonical_event")):
                        deferred_tool_end_events.append(
                            {
                                **event,
                                "_evo_observed_call_id": observed_call_id,
                            }
                        )
                        _worker_checkpoint(
                            "native_tool_end_deferred_for_canonical_commit",
                            request_id=str(state.get("request_id") or ""),
                            run_id=run_id,
                            tool=tool_name,
                            tool_call_id=observed_call_id,
                            deferred_count=len(deferred_tool_end_events),
                        )
                        continue
                    raise RuntimeError(
                        "CANONICAL_PROTOCOL_UNCOMMITTED_TOOL_EVENT:"
                        + (observed_call_id or run_id or tool_name or "unknown")
                    )
                model_tool_call_id = str(
                    canonical_row.get("tool_call_id") or ""
                )
                tool_name = str(canonical_row.get("tool_name") or tool_name)
                tool_input = canonical_row.get("tool_input")
                output = canonical_row.get("result")
                model_output_chars, runtime_artifact_chars = _tool_message_projection_sizes(output)
                stored_as_resource = _tool_output_has_durable_resource(output)
                # The outer middleware has already committed the exact result
                # identity before this observational callback is emitted.
                # Never reconstruct or name-match protocol state from events.
                try:
                    typed_capability_hitl = _typed_capability_hitl_from_tool_output(
                        output,
                        tool_name=tool_name,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        str(exc) or "CAPABILITY_NEEDS_INPUT_CONTROL_INVALID"
                    ) from exc
                if typed_capability_hitl is not None:
                    if deferred_capability_hitl is not None:
                        raise RuntimeError(
                            "CAPABILITY_AUTHORIZATION_MULTIPLE_PENDING"
                        )
                    if run_id:
                        active_tool_runs.pop(run_id, None)
                    elif anonymous_tools_active > 0:
                        anonymous_tools_active -= 1
                    tools_active = len(active_tool_runs) + anonymous_tools_active
                    if current_batch_tool_calls:
                        deferred_capability_hitl = (
                            typed_capability_hitl,
                            now_ms,
                        )
                        _worker_checkpoint(
                            "native_capability_hitl_deferred_for_sibling_tools",
                            request_id=str(state.get("request_id") or ""),
                            pending_call_count=len(current_batch_tool_calls),
                        )
                        continue
                    _open_hitl_interrupt(
                        {},
                        timestamp_ms=now_ms,
                        pending_capability_authorization=typed_capability_hitl,
                    )
                    break
                if tool_name == "request_human_input" and not observed_tool_error:
                    logger.info(
                        "native_tool_end request_id=%s run_id=%s tool=%s output_type=%s output_chars=%d hitl=true",
                        state.get("request_id") or "",
                        run_id,
                        tool_name,
                        type(output).__name__,
                        _object_size_chars(output),
                    )
                    _worker_checkpoint(
                        "native_tool_end",
                        request_id=str(state.get("request_id") or ""),
                        run_id=run_id,
                        tool=tool_name,
                        output_type=type(output).__name__,
                        output_chars=_object_size_chars(output),
                        model_output_chars=model_output_chars,
                        runtime_artifact_chars=runtime_artifact_chars,
                        stored_as_resource=stored_as_resource,
                        hitl=True,
                    )
                    normalized_hitl_input = tool_input if isinstance(tool_input, dict) else {}
                    if current_batch_tool_calls:
                        deferred_hitl = (dict(normalized_hitl_input), now_ms)
                        _worker_checkpoint(
                            "native_hitl_deferred_for_sibling_tools",
                            request_id=str(state.get("request_id") or ""),
                            pending_call_count=len(current_batch_tool_calls),
                            pending_tools=",".join(current_batch_tool_calls.values()),
                        )
                        continue
                    _open_hitl_interrupt(normalized_hitl_input, timestamp_ms=now_ms)
                    break

                if run_id:
                    active_tool_runs.pop(run_id, None)
                elif anonymous_tools_active > 0:
                    anonymous_tools_active -= 1
                tools_active = len(active_tool_runs) + anonymous_tools_active
                requested_load_tool_ids = _extract_load_tools_requested_tool_ids(tool_name, tool_input)
                schema_reload_tool_ids = _extract_load_tools_effective_tool_ids(tool_name, output)
                load_metrics_payload = _extract_load_tools_metrics_payload(tool_name, output)
                if schema_reload_tool_ids:
                    existing_loaded_ids = set(_normalize_loaded_tool_ids(state.get("loaded_tool_ids")))
                    new_schema_reload_tool_ids = [
                        tool_id for tool_id in schema_reload_tool_ids if tool_id not in existing_loaded_ids
                    ]
                    if not new_schema_reload_tool_ids:
                        _worker_checkpoint(
                            "native_tool_schema_reload_skipped_duplicate",
                            request_id=str(state.get("request_id") or ""),
                            tool=tool_name,
                            requested_tool_ids=",".join(requested_load_tool_ids),
                            effective_tool_ids=",".join(schema_reload_tool_ids),
                            loaded_tool_ids=",".join(sorted(existing_loaded_ids)),
                        )
                        schema_reload_tool_ids = []
                    else:
                        schema_reload_tool_ids = new_schema_reload_tool_ids
                if str(tool_name or "") == "load_tools":
                    tool_load_metrics = record_tool_load_request(
                        state.get("tool_load_metrics"),
                        requested=requested_load_tool_ids,
                        added=schema_reload_tool_ids,
                        already_loaded=load_metrics_payload.get("already_loaded"),
                        denied=load_metrics_payload.get("denied"),
                        unknown=load_metrics_payload.get("unknown"),
                        no_op=bool(load_metrics_payload.get("no_op")),
                    )
                    state["tool_load_metrics"] = tool_load_metrics
                    context_usage["tool_schema_ledger"] = tool_load_metrics
                tool_outcome = _tool_outcome_from_output(tool_name, output)
                if model_tool_call_id:
                    tool_outcome["tool_call_id"] = model_tool_call_id
                if run_id:
                    tool_outcome["runtime_run_id"] = run_id
                tool_outcome["arguments_digest"] = _recovery_digest(tool_input or {})
                existing_outcome_index = next(
                    (
                        index
                        for index, item in enumerate(tool_outcomes)
                        if model_tool_call_id
                        and str(item.get("tool_call_id") or "").strip()
                        == model_tool_call_id
                    ),
                    None,
                )
                if existing_outcome_index is None:
                    tool_outcomes = _append_tool_outcome(
                        tool_outcomes,
                        tool_outcome,
                    )
                else:
                    merged_outcomes = list(tool_outcomes)
                    merged_outcomes[existing_outcome_index] = {
                        **merged_outcomes[existing_outcome_index],
                        **tool_outcome,
                    }
                    tool_outcomes = _normalize_tool_outcomes(merged_outcomes)
                if _is_side_effect_tool(tool_name):
                    side_effect_ledger = _append_side_effect_identity(
                        side_effect_ledger,
                        tool_outcome,
                    )
                    state["side_effect_ledger"] = side_effect_ledger
                if tool_name == "save_execution_plan":
                    plan_item = _saved_plan_context_item(output)
                    if plan_item is not None:
                        conversation_files = list(state.get("conversation_files") or [])
                        plan_file_id = str(
                            plan_item.get("conversation_file_id") or plan_item.get("file_id") or ""
                        ).strip()
                        conversation_files = [
                            item
                            for item in conversation_files
                            if not (
                                isinstance(item, dict)
                                and str(item.get("drawer_section") or "").strip().lower() == "plan_file"
                            )
                        ]
                        conversation_files.append(plan_item)
                        state["conversation_files"] = conversation_files
                        _transition_native_task_state(
                            state,
                            TaskPhase.EXECUTE,
                            plan_ref=plan_file_id or str(plan_item.get("file_name") or ""),
                            resume_reason="plan_saved",
                            missing_fields=[],
                        )
                if str(tool_name or "") == "activate_skill":
                    activated_skills = _merge_activated_skill_state(
                        activated_skills,
                        _extract_activate_skill_state(output),
                    )
                logger.info(
                    "native_tool_end request_id=%s run_id=%s tool=%s output_type=%s output_chars=%d tools_active=%d",
                    state.get("request_id") or "",
                    run_id,
                    tool_name,
                    type(output).__name__,
                    _object_size_chars(output),
                    tools_active,
                )
                _worker_checkpoint(
                    "native_tool_end",
                    request_id=str(state.get("request_id") or ""),
                    run_id=run_id,
                    tool=tool_name,
                    output_type=type(output).__name__,
                    output_chars=_object_size_chars(output),
                    model_output_chars=model_output_chars,
                    runtime_artifact_chars=runtime_artifact_chars,
                    stored_as_resource=stored_as_resource,
                    tools_active=tools_active,
                    requested_tool_ids=",".join(requested_load_tool_ids),
                    effective_tool_ids=",".join(schema_reload_tool_ids),
                    ok=bool(tool_outcome.get("ok")),
                    error_kind=str(tool_outcome.get("error_kind") or ""),
                    severity=str(tool_outcome.get("severity") or ""),
                )
                if str(tool_name or "") != "load_tools":
                    _mark_dynamic_tool_cache_used(
                        scope=str(state.get("tool_cache_scope") or ""),
                        turn=int(state.get("tool_cache_turn") or 0),
                        loaded_entries=_normalize_loaded_tool_ids(state.get("loaded_tool_entries")),
                        tool_name=str(tool_name or ""),
                    )
                sandbox_submission = _extract_sandbox_submission(tool_name, output)
                trusted_sandbox_binding = _extract_trusted_sandbox_suspend_binding(
                    tool_name,
                    output,
                    task_authority_request_id=str(
                        state.get("task_authority_request_id")
                        or state.get("request_id")
                        or ""
                    ),
                    model_tool_call_id=model_tool_call_id,
                    runtime_event_id=run_id,
                    tool_input=tool_input,
                )
                if trusted_sandbox_binding is not None:
                    bound_job_id = str(
                        trusted_sandbox_binding.get("job_id") or ""
                    ).strip()
                    if bound_job_id:
                        trusted_sandbox_suspend_bindings[bound_job_id] = dict(
                            trusted_sandbox_binding
                        )
                    if (
                        sandbox_submission is not None
                        and str(sandbox_submission.get("job_id") or "").strip()
                        == bound_job_id
                    ):
                        sandbox_submission["runtime_binding"] = dict(
                            trusted_sandbox_binding
                        )
                if sandbox_submission is not None:
                    submission_status = str(sandbox_submission.get("status") or "").strip().lower()
                    submission_terminal = submission_status in _SANDBOX_TERMINAL_STATUSES
                    submitted_job_id = str(sandbox_submission.get("job_id") or "").strip()
                    if not submission_terminal and submitted_job_id and all(
                        str(item.get("job_id") or "").strip() != submitted_job_id
                        for item in pending_sandbox_jobs
                    ):
                        pending_sandbox_jobs.append(sandbox_submission)
                sandbox_result_status = _extract_sandbox_result_status(tool_name, output)
                if sandbox_result_status is not None:
                    pending_sandbox_jobs = _apply_sandbox_result_status(
                        pending_sandbox_jobs,
                        sandbox_result_status,
                    )
                    if not bool(sandbox_result_status.get("terminal")):
                        pending_sandbox_result_still_pending = True
                        result_job_id = str(sandbox_result_status.get("job_id") or "").strip()
                        if result_job_id and all(
                            str(item.get("job_id") or "").strip() != result_job_id
                            for item in pending_sandbox_jobs
                        ):
                            pending_sandbox_jobs.append(
                                {
                                    "app_id": "sandbox",
                                    "job_id": result_job_id,
                                    "status": str(sandbox_result_status.get("status") or "running"),
                                    "agent_execution_mode": "nonblocking_long",
                                }
                            )
                current_pending_job_ids = {
                    str(item.get("job_id") or "").strip()
                    for item in pending_sandbox_jobs
                    if str(item.get("job_id") or "").strip()
                }
                pending_sandbox_bookkeeping_observed = bool(
                    current_pending_job_ids
                    and current_pending_job_ids.issubset(
                        trusted_sandbox_suspend_bindings
                    )
                )
                completion_signal = _tool_completion_signal_from_output(tool_name, output)
                capability_outcome = _capability_outcome_from_output(output)
                items = extract_tool_items(tool_name, output, tool_input=tool_input)
                summary = summarize_tool_done(tool_name, output, items)
                context_attribution = build_tool_result_context(
                    tool_call_id=model_tool_call_id,
                    tool_name=tool_name,
                    raw_output=output,
                    summary=summary,
                    stored_as_artifact=bool(
                        stored_as_resource
                        or sandbox_submission is not None
                        or sandbox_result_status is not None
                    ),
                    included_in_main_context=True,
                    included_in_history=True,
                    included_in_runtime_delta=True,
                    reason="native_tool_message",
                )
                context_attribution["model_output_chars"] = model_output_chars
                context_attribution["runtime_artifact_chars"] = runtime_artifact_chars
                context_attribution["stored_as_resource"] = stored_as_resource
                if completion_signal:
                    context_attribution["completion_signal"] = completion_signal
                if capability_outcome:
                    context_attribution["capability_outcome"] = capability_outcome
                tool_result_context.append(context_attribution)
                context_usage["tool_result_context"] = tool_result_context
                context_snapshot_id = build_context_snapshot_id(context_usage)
                context_usage["snapshot_id"] = context_snapshot_id
                writer({"type": "context_usage", **context_usage})
                incoming_citations: list[dict[str, Any]] = []
                verified_sources = verified_sources_from_tool_output(output)
                if verified_sources:
                    try:
                        from src.subagents.reference_store import append_verified_reference_candidates

                        append_verified_reference_candidates(
                            verified_sources,
                            request_id=str(
                                state.get("task_authority_request_id")
                                or state.get("request_id")
                                or ""
                            )
                            or None,
                            project_id=str(state.get("project_id") or "") or None,
                            conversation_id=str(state.get("conversation_id") or "") or None,
                            source_tool=tool_name,
                        )
                    except Exception:  # reference persistence must not hide a valid tool result
                        pass
                for candidate in build_citations_from_tool_output(tool_name, output):
                    if isinstance(candidate, dict):
                        citation_id = adapter.registry.add(candidate)
                        if citation_id is not None:
                            incoming_citations.append(adapter.registry.citations[citation_id - 1])
                try:
                    from src.agents.lead_agent import _get_tool_labels  # local import keeps public surface small

                    labels = _get_tool_labels(tool_name)
                except Exception:  # noqa: BLE001
                    labels = {"icon": "bolt"}
                payload = {
                    "tool_call_id": model_tool_call_id,
                    "runtime_run_id": run_id,
                    "phase": "tool_done",
                    "tool_name": tool_name,
                    "summary": summary,
                    "icon": labels.get("icon") or "bolt",
                    "items": items,
                    "input_preview": build_tool_input_preview(tool_input),
                    "output_preview": build_tool_output_preview(output),
                    "context_attribution": context_attribution,
                    "timestamp": now_ms,
                }
                if completion_signal:
                    payload["completion_signal"] = completion_signal
                if capability_outcome:
                    payload["capability_outcome"] = capability_outcome
                if incoming_citations:
                    payload["citations"] = incoming_citations
                _emit_events(
                    writer,
                    adapter.upsert_thinking_block(
                        block_id=f"tool:{run_id or int(now_ms)}",
                        kind="thinking_tool",
                        payload=payload,
                        timestamp=now_ms,
                        completed=True,
                    ),
                )
                if deferred_hitl is not None and not current_batch_tool_calls:
                    deferred_hitl_input, deferred_hitl_started_ms = deferred_hitl
                    deferred_hitl = None
                    _worker_checkpoint(
                        "native_hitl_sibling_tools_drained",
                        request_id=str(state.get("request_id") or ""),
                        deferred_ms=max(0, now_ms - deferred_hitl_started_ms),
                    )
                    _open_hitl_interrupt(deferred_hitl_input, timestamp_ms=now_ms)
                    break
                if (
                    deferred_capability_hitl is not None
                    and not current_batch_tool_calls
                ):
                    pending_authorization, deferred_started_ms = (
                        deferred_capability_hitl
                    )
                    deferred_capability_hitl = None
                    _worker_checkpoint(
                        "native_capability_hitl_sibling_tools_drained",
                        request_id=str(state.get("request_id") or ""),
                        deferred_ms=max(0, now_ms - deferred_started_ms),
                    )
                    _open_hitl_interrupt(
                        {},
                        timestamp_ms=now_ms,
                        pending_capability_authorization=pending_authorization,
                    )
                    break
                if task_complete_accepted:
                    if current_batch_tool_calls:
                        if task_complete_deferred_at_ms is None:
                            task_complete_deferred_at_ms = now_ms
                            _worker_checkpoint(
                                "native_task_complete_deferred_for_sibling_tools",
                                request_id=str(state.get("request_id") or ""),
                                pending_call_count=len(current_batch_tool_calls),
                                pending_tools=",".join(current_batch_tool_calls.values()),
                            )
                        continue
                    if task_complete_deferred_at_ms is not None:
                        _worker_checkpoint(
                            "native_task_complete_sibling_tools_drained",
                            request_id=str(state.get("request_id") or ""),
                            deferred_ms=max(0, now_ms - task_complete_deferred_at_ms),
                        )
                        task_complete_deferred_at_ms = None
                    # A sibling sandbox submission may finish after the
                    # authoritative task_complete result and repopulate the
                    # local pending list.  Completion acceptance wins over
                    # that late bookkeeping just as it does when the submit
                    # result arrives first.
                    pending_sandbox_jobs = []
                    pending_sandbox_bookkeeping_observed = False
                    pending_sandbox_result_still_pending = False
                    # task_complete has already validated the task tree and
                    # deliverables.  Do not let a missing/late on_tool_end event
                    # outside the model-declared batch keep an accepted turn alive.
                    completion_text = task_complete_summary or "任务已完成"
                    if completion_text not in last_model_answer_text:
                        _emit_answer_text(
                            ("\n\n" if last_model_answer_text else "") + completion_text,
                            timestamp_ms=now_ms,
                        )
                    # ``task_complete.summary`` is the validated user-facing
                    # terminal payload. Model prose emitted before the control
                    # call is progress/reasoning, not part of that payload.
                    last_model_answer_text = completion_text
                    last_model_had_tool_calls = False
                    final_answer_committed = bool(last_model_answer_text)
                    await _close_event_stream_bounded(event_stream)
                    break
                if tools_active == 0:
                    tool_rounds_completed += 1
                    if pending_sandbox_jobs and (
                        pending_sandbox_bookkeeping_observed or pending_sandbox_result_still_pending
                    ):
                        _worker_checkpoint(
                            "native_pending_sandbox_suspend",
                            request_id=str(state.get("request_id") or ""),
                            pending_job_ids=",".join(
                                str(item.get("job_id") or "") for item in pending_sandbox_jobs
                            ),
                            bookkeeping_observed=pending_sandbox_bookkeeping_observed,
                            result_still_pending=pending_sandbox_result_still_pending,
                        )
                        await _close_event_stream_bounded(event_stream)
                        break
                continue

        ready_transition = protocol_ledger.pop_ready_transition()
        if ready_transition is not None:
            raise CanonicalRuntimeTransition(*ready_transition)

        sealed_completion = _sealed_completion_update()
        if sealed_completion is not None:
            return sealed_completion
        unresolved_protocol_calls = protocol_ledger.pending_calls()
        if unresolved_protocol_calls:
            raise RuntimeError(
                "CANONICAL_PROTOCOL_INCOMPLETE:"
                + ",".join(sorted(unresolved_protocol_calls))
            )

        if held_answer_text:
            _flush_held_answer_text(timestamp_ms=int(time.time() * 1000))
            last_model_answer_text = _answer_text_since(
                adapter,
                model_call_answer_start_chars,
            )
            final_answer_committed = bool(
                last_model_answer_text
                and not last_model_had_tool_calls
                and not _finish_reason_is_truncated(last_model_finish_reason)
            )

        _complete_reasoning(timestamp_ms=int(time.time() * 1000))

        if pending_bundle is not None or adapter.waiting_human:
            question_bundle = pending_bundle or state.get("pending_question_bundle")
            writer(
                adapter.build_turn_suspended(
                    status="waiting_human",
                    reason="awaiting_human_input",
                    final_reply_text=_question_bundle_reply_text(question_bundle),
                )
            )
            return {
                "any_tool_called": any_tool_called,
                **_state_from_adapter(state, adapter),
                "pending_question_bundle": question_bundle,
                "pending_sandbox_jobs": pending_sandbox_jobs,
                "status": "waiting_human",
            }

        if pending_sandbox_jobs and not (
            pending_sandbox_bookkeeping_observed
            or pending_sandbox_result_still_pending
        ):
            raise RuntimeError("SANDBOX_SUSPEND_AUTHORITY_MISSING")

        if pending_sandbox_jobs:
            turn_checkpoint = _build_turn_checkpoint(
                state=state,
                protocol_ledger=protocol_ledger,
                pending_bundle={},
                pending_sandbox_jobs=pending_sandbox_jobs,
                status="waiting_sandbox",
            )
            state["turn_checkpoint"] = turn_checkpoint
            writer(
                {
                    "type": "turn_checkpoint",
                    "request_id": str(state.get("request_id") or ""),
                    "payload": {"checkpoint": turn_checkpoint},
                }
            )
            writer(
                adapter.build_turn_suspended(
                    status="waiting_sandbox",
                    reason="awaiting_sandbox_result",
                    pending_sandbox_jobs=pending_sandbox_jobs,
                )
            )
            _transition_native_task_state(
                state,
                TaskPhase.SUSPENDED,
                resume_reason="awaiting_sandbox_result",
                pending_job_ids=[
                    str(item.get("job_id") or "").strip()
                    for item in pending_sandbox_jobs
                    if str(item.get("job_id") or "").strip()
                ],
            )
            return {
                "any_tool_called": any_tool_called,
                **_state_from_adapter(state, adapter),
                "pending_question_bundle": None,
                "pending_sandbox_jobs": pending_sandbox_jobs,
                "status": "waiting_sandbox",
            }

        if final_answer_committed:
            # Rehydration only resolves explicit source bindings; an uncited
            # answer must not carry unused request references into final state.
            if "[[cite:" in last_model_answer_text:
                _hydrate_citation_registry_from_reference_store(
                    adapter,
                    request_id=str(state.get("request_id") or ""),
                    project_id=str(state.get("project_id") or "").strip() or None,
                    conversation_id=str(state.get("conversation_id") or "").strip() or None,
                )
            completed_event = adapter.build_turn_completed(
                cancelled=False,
                status=_task_terminal_status_from_receipt(
                    state.get("completion_receipt")
                ),
                final_reply_text=last_model_answer_text,
                completion_receipt=(
                    _public_task_completion_receipt(
                        state.get("completion_receipt")
                    )
                ),
            )
            writer(completed_event)
            _transition_native_task_state(
                state,
                TaskPhase.COMPLETED,
                resume_reason="answer_delivered",
                pending_job_ids=[],
            )
            return {
                "any_tool_called": any_tool_called,
                **_state_from_adapter(state, adapter),
                "pending_question_bundle": None,
                "pending_sandbox_jobs": pending_sandbox_jobs,
                "status": _task_terminal_status_from_receipt(
                    state.get("completion_receipt")
                ),
            }

        truncated_answer_continuation_attempts = int(
            state.get("truncated_answer_continuation_attempts") or 0
        )
        if (
            truncated_answer_continuation_attempts < 1
            and _finish_reason_is_truncated(last_model_finish_reason)
        ):
            _worker_checkpoint(
                "native_truncated_answer_continuation_start",
                request_id=str(state.get("request_id") or ""),
                finish_reason=last_model_finish_reason,
                stale_answer_chars=len(_adapter_answer_text(adapter)),
                last_call_answer_chars=len(last_model_answer_text),
                last_call_had_tool_calls=last_model_had_tool_calls,
                tool_outcome_count=len(tool_outcomes),
            )
            recovery_state: NativeAgentState = dict(state)
            recovery_state.update(_state_from_adapter(state, adapter))
            recovery_state["timeline_blocks"] = adapter.snapshot_blocks()
            recovery_state["usage"] = usage_ledger
            recovery_state["pending_question_bundle"] = None
            recovery_state["pending_sandbox_jobs"] = pending_sandbox_jobs
            recovery_state["activated_skills"] = activated_skills
            recovery_state["tool_outcomes"] = tool_outcomes
            recovery_state["continuation_messages"] = continuation_messages
            recovery_state["resume_count"] = int(state.get("resume_count") or 0) + 1
            recovery_state["stream_recovery_mode"] = False
            recovery_state["stream_recovery_attempts"] = stream_recovery_attempts
            recovery_state["stream_recovery_total_attempts"] = (
                stream_recovery_total_attempts
            )
            recovery_state["stream_recovery_replay_prefix"] = ""
            recovery_state["truncated_answer_continuation_attempts"] = (
                truncated_answer_continuation_attempts + 1
            )
            recovery_state["system_prompt"] = _strip_legacy_recovery_state(
                state.get("system_prompt")
            )
            recovery_state["recovery_state_envelope"] = _build_recovery_state_envelope(
                state=recovery_state,
                interruption={
                    "reason": "answer_truncated",
                    "phase": "after_complete_truncated_message",
                    "call_index": model_call_index,
                    "consecutive_attempt": truncated_answer_continuation_attempts + 1,
                    "max_attempts": 1,
                    "discarded_partial_model_output": False,
                    "discarded_partial_output_chars": 0,
                },
                continuation_messages=continuation_messages,
                task_tree_snapshot=task_tree_snapshot,
                task_tree_fetch_status=task_tree_fetch_status,
                tool_schema_snapshot=tool_schema_snapshot,
                tool_outcomes=tool_outcomes,
                pending_sandbox_jobs=pending_sandbox_jobs,
            )
            recovery_state["_canonical_any_tool_called"] = bool(any_tool_called)
            transition = CanonicalRuntimeTransition(
                "retry_turn",
                {"reason": "answer_truncated"},
            )
            transition.next_state = recovery_state
            raise transition

        final_answer_recovery_attempts = int(
            state.get("final_answer_recovery_attempts") or 0
        )
        final_answer_recovery_limit = (
            2 if last_model_had_invalid_tool_calls else 1
        )
        if final_answer_recovery_attempts < final_answer_recovery_limit:
            _worker_checkpoint(
                "native_final_answer_recovery_start",
                request_id=str(state.get("request_id") or ""),
                attempt=final_answer_recovery_attempts + 1,
                tool_outcome_count=len(tool_outcomes),
                stale_answer_chars=len(_adapter_answer_text(adapter)),
                last_call_had_tool_calls=last_model_had_tool_calls,
            )
            recovery_state: NativeAgentState = dict(state)
            recovery_state.update(_state_from_adapter(state, adapter))
            recovery_state["timeline_blocks"] = adapter.snapshot_blocks()
            recovery_state["usage"] = usage_ledger
            recovery_state["pending_question_bundle"] = None
            recovery_state["pending_sandbox_jobs"] = pending_sandbox_jobs
            recovery_state["activated_skills"] = activated_skills
            recovery_state["tool_outcomes"] = tool_outcomes
            recovery_state["continuation_messages"] = continuation_messages
            recovery_state["resume_count"] = int(state.get("resume_count") or 0) + 1
            recovery_state["stream_recovery_mode"] = False
            recovery_state["stream_recovery_attempts"] = stream_recovery_attempts
            recovery_state["stream_recovery_total_attempts"] = (
                stream_recovery_total_attempts
            )
            recovery_state["stream_recovery_replay_prefix"] = ""
            recovery_state["final_answer_recovery_attempts"] = (
                final_answer_recovery_attempts + 1
            )
            recovery_state["system_prompt"] = _strip_legacy_recovery_state(
                state.get("system_prompt")
            )
            recovery_state["recovery_state_envelope"] = _build_recovery_state_envelope(
                state=recovery_state,
                interruption={
                    "reason": "final_answer_incomplete",
                    "phase": (
                        "after_complete_tool_protocol"
                        if tool_outcomes
                        else "after_complete_without_final_answer"
                    ),
                    "call_index": model_call_index,
                    "consecutive_attempt": final_answer_recovery_attempts + 1,
                    "cumulative_attempt": final_answer_recovery_attempts + 1,
                    "max_attempts": final_answer_recovery_limit,
                    "discarded_partial_model_output": False,
                    "discarded_partial_output_chars": 0,
                },
                continuation_messages=continuation_messages,
                task_tree_snapshot=task_tree_snapshot,
                task_tree_fetch_status=task_tree_fetch_status,
                tool_schema_snapshot=tool_schema_snapshot,
                tool_outcomes=tool_outcomes,
                pending_sandbox_jobs=pending_sandbox_jobs,
            )
            recovery_state["_canonical_any_tool_called"] = bool(any_tool_called)
            transition = CanonicalRuntimeTransition(
                "retry_turn",
                {"reason": "final_answer_incomplete"},
            )
            transition.next_state = recovery_state
            raise transition

        failed_event = adapter.build_turn_failed(
            "模型未能提交完整最终答案，可继续本轮任务重试。",
            error_kind="final_answer_incomplete",
        )
        writer(failed_event)
        _transition_native_task_state(
            state,
            TaskPhase.FAILED,
            resume_reason="final_answer_incomplete",
        )
        return {
            "any_tool_called": any_tool_called,
            **_state_from_adapter(state, adapter),
            "pending_question_bundle": None,
            "pending_sandbox_jobs": [],
            "status": "failed",
            "error_message": "model did not commit a complete final answer",
        }
    except CanonicalRuntimeTransition as transition:
        _sync_canonical_runtime_projection()
        if transition.kind == "retry_turn":
            await _close_event_stream_bounded(event_stream)
            if not isinstance(transition.next_state, dict):
                raise RuntimeError(
                    "CANONICAL_RUNTIME_TRANSITION_STATE_MISSING:retry_turn"
                ) from transition
            raise
        if transition.kind == "terminal_complete":
            sealed_update = _sealed_completion_update()
            if sealed_update is None:
                raise RuntimeError("CANONICAL_TERMINAL_SEAL_INVALID") from transition
            return sealed_update

        if transition.kind == "suspend_human":
            await _close_event_stream_bounded(event_stream)
            if pending_bundle is None and not adapter.waiting_human:
                _open_hitl_interrupt(
                    transition.payload.get("tool_input"),
                    timestamp_ms=int(time.time() * 1000),
                )
            question_bundle = pending_bundle or state.get("pending_question_bundle")
            writer(
                adapter.build_turn_suspended(
                    status="waiting_human",
                    reason="awaiting_human_input",
                    final_reply_text=_question_bundle_reply_text(question_bundle),
                )
            )
            return {
                "any_tool_called": True,
                **_state_from_adapter(state, adapter),
                "pending_question_bundle": question_bundle,
                "pending_sandbox_jobs": pending_sandbox_jobs,
                "continuation_messages": protocol_ledger.snapshot_messages(),
                "tool_outcomes": tool_outcomes,
                "side_effect_ledger": side_effect_ledger,
                "status": "waiting_human",
            }

        await _close_event_stream_bounded(event_stream)
        next_state: NativeAgentState = dict(state)
        next_state.update(_state_from_adapter(state, adapter))
        next_state["timeline_blocks"] = adapter.snapshot_blocks()
        next_state["usage"] = usage_ledger
        next_state["pending_question_bundle"] = None
        next_state["pending_sandbox_jobs"] = pending_sandbox_jobs
        next_state["activated_skills"] = activated_skills
        next_state["tool_outcomes"] = tool_outcomes
        next_state["side_effect_ledger"] = side_effect_ledger
        next_state["continuation_messages"] = protocol_ledger.snapshot_messages()
        next_state["resume_count"] = int(state.get("resume_count") or 0) + 1
        next_state["stream_recovery_mode"] = stream_recovery_mode
        next_state["stream_recovery_attempts"] = stream_recovery_attempts
        next_state["stream_recovery_total_attempts"] = (
            stream_recovery_total_attempts
        )
        next_state["stream_recovery_replay_prefix"] = ""
        next_state["_canonical_protocol_ledger"] = protocol_ledger
        next_state["_canonical_any_tool_called"] = bool(any_tool_called)

        if transition.kind == "reload_tools":
            requested_ids = _normalize_loaded_tool_ids(
                list(transition.payload.get("tool_ids") or [])
            )
            existing_loaded = _normalize_loaded_tool_ids(
                state.get("loaded_tool_ids")
            )
            merged_loaded = _normalize_loaded_tool_ids(
                [*existing_loaded, *requested_ids]
            )
            existing_entries = _normalize_loaded_tool_ids(
                state.get("loaded_tool_entries")
            )
            requested_entries = get_runtime_dynamic_entry_ids(requested_ids)
            merged_entries = _normalize_loaded_tool_ids(
                [
                    *existing_entries,
                    *requested_entries,
                ]
            )
            _store_dynamic_tool_cache_entries(
                scope=str(state.get("tool_cache_scope") or ""),
                turn=int(state.get("tool_cache_turn") or 0),
                entries=merged_entries,
            )
            next_state["loaded_tool_ids"] = merged_loaded
            next_state["loaded_tool_entries"] = merged_entries
            next_state["tool_schema_reload_count"] = tool_schema_reload_count + 1
            _worker_checkpoint(
                "native_tool_schema_reload_transition",
                request_id=request_id,
                reload_count=tool_schema_reload_count + 1,
                loaded_tool_ids=",".join(merged_loaded),
                loaded_tool_entries=",".join(merged_entries),
            )
        elif transition.kind == "refresh_tree":
            _worker_checkpoint(
                "native_capability_discovery_refresh_transition",
                request_id=request_id,
                reason="trusted_task_frontier_changed",
            )
        else:
            raise RuntimeError(
                f"CANONICAL_RUNTIME_TRANSITION_UNSUPPORTED:{transition.kind}"
            ) from transition

        transition.next_state = next_state
        raise
    except asyncio.CancelledError:
        sealed_update = _sealed_completion_update()
        if sealed_update is not None:
            return sealed_update
        for nested_index, nested_started_ms in list(nested_model_calls.values()):
            _record_model_call_usage(
                None,
                call_index=nested_index,
                scope="agent_tool_internal",
                status="cancelled",
                started_at_ms=nested_started_ms,
                completed_at_ms=int(time.time() * 1000),
            )
        nested_model_calls.clear()
        if model_call_inflight and model_call_index > 0:
            _record_model_call_usage(
                None,
                call_index=model_call_index,
                status="cancelled",
                started_at_ms=model_call_started_ms,
                completed_at_ms=int(time.time() * 1000),
            )
            model_call_inflight = False
        _emit_events(
            writer,
            adapter.supersede_model_attempt(
                existing_block_ids=model_call_start_block_ids,
                reason="model_call_cancelled",
            ),
        )
        completed_event = adapter.build_turn_completed(cancelled=True)
        writer(completed_event)
        _transition_native_task_state(state, TaskPhase.SUSPENDED, resume_reason="cancelled")
        return {
            "any_tool_called": any_tool_called,
            **_state_from_adapter(state, adapter),
            "pending_sandbox_jobs": pending_sandbox_jobs,
            "status": "cancelled",
        }
    except Exception as exc:  # noqa: BLE001
        sealed_update = _sealed_completion_update()
        if sealed_update is not None:
            return sealed_update
        error_type = exc.__class__.__name__
        error_message = str(exc).strip() or repr(exc)
        model_call_was_inflight = bool(model_call_inflight)
        now_ms = int(time.time() * 1000)
        for nested_index, nested_started_ms in list(nested_model_calls.values()):
            _record_model_call_usage(
                None,
                call_index=nested_index,
                scope="agent_tool_internal",
                status="failed",
                started_at_ms=nested_started_ms,
                completed_at_ms=now_ms,
            )
        nested_model_calls.clear()
        if model_call_inflight and model_call_index > 0:
            _record_model_call_usage(
                None,
                call_index=model_call_index,
                status="failed",
                started_at_ms=model_call_started_ms,
                completed_at_ms=now_ms,
            )
            model_call_inflight = False
        _worker_checkpoint(
            "native_model_exception",
            request_id=str(state.get("request_id") or ""),
            call_index=model_call_index,
            error_type=error_type,
            error_message=error_message[:300],
            elapsed_ms=(now_ms - model_call_started_ms) if model_call_started_ms else None,
            since_last_chunk_ms=(now_ms - model_call_last_chunk_ms) if model_call_last_chunk_ms else None,
            since_last_progress_ms=(now_ms - model_call_last_progress_ms) if model_call_last_progress_ms else None,
            stream_event_count=stream_event_count,
            chunk_count=model_call_chunk_count,
            empty_chunk_count=model_call_empty_chunk_count,
            reasoning_chars=model_call_reasoning_chars,
            reasoning_chunk_count=model_call_reasoning_chunk_count,
            text_chars=model_call_text_chars,
            text_chunk_count=model_call_text_chunk_count,
            tool_call_chunks=model_call_tool_call_chunks,
            last_chunk_kind=model_call_last_chunk_kind,
            last_progress_kind=model_call_last_progress_kind,
            last_chunk_text_chars=model_call_last_chunk_text_chars,
            last_chunk_reasoning_chars=model_call_last_chunk_reasoning_chars,
            last_chunk_tool_call_chunks=model_call_last_chunk_tool_call_chunks,
            input_chars=model_call_input_chars,
            tool_outcome_count=len(tool_outcomes),
        )
        traceback_summary = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        logger.exception(
            "native_agent_exception request_id=%s error_type=%s message=%s",
            state.get("request_id") or "",
            error_type,
            error_message[:500],
        )
        stream_stall = isinstance(exc, NativeModelStreamStall)
        recoverable_transport_error = bool(
            stream_stall
            or (
                model_call_was_inflight
                and _retryable_model_transport_error(exc)
            )
        )
        timeout_like = "timeout" in error_type.lower() or "timeout" in error_message.lower()
        execution_limit_reached = error_type == "GraphRecursionError"
        fail_closed_runtime_error = error_message.startswith(
            (
                "CANONICAL_PROTOCOL_",
                "CANONICAL_TERMINAL_",
                "SANDBOX_SUSPEND_AUTHORITY_",
                "SANDBOX_RESUME_",
                "SESSION_COMPACT_",
                "CAPABILITY_AUTHORIZATION_",
                "CAPABILITY_NEEDS_INPUT_",
                "TASK_AUTHORITY_",
            )
        )
        if fail_closed_runtime_error:
            _complete_reasoning(timestamp_ms=now_ms)
            adapter.record_usage_snapshot(usage_ledger, timeout=False)
            writer(
                adapter.build_turn_failed(
                    error_message,
                    error_kind="runtime_protocol_violation",
                    error_type=error_type,
                    traceback_summary=traceback_summary,
                )
            )
            _transition_native_task_state(
                state,
                TaskPhase.FAILED,
                resume_reason="runtime_protocol_violation",
            )
            return {
                "any_tool_called": any_tool_called,
                **_state_from_adapter(state, adapter),
                "pending_sandbox_jobs": pending_sandbox_jobs,
                "status": "failed",
                "error_message": error_message,
            }
        transport_recreate_failed = False
        if recoverable_transport_error:
            max_recovery_attempts = _native_stream_recovery_max_attempts()
            max_total_recovery_attempts = _native_stream_recovery_total_max_attempts()
            current_consecutive_attempts = int(stream_recovery_attempts or 0)
            current_total_attempts = int(stream_recovery_total_attempts or 0)
            if (
                current_consecutive_attempts < max_recovery_attempts
                and current_total_attempts < max_total_recovery_attempts
            ):
                consecutive_attempt = current_consecutive_attempts + 1
                cumulative_attempt = current_total_attempts + 1
                emitted_call_delta = _adapter_answer_text(adapter)[
                    max(0, int(model_call_answer_start_chars or 0)) :
                ]
                _emit_events(
                    writer,
                    adapter.supersede_model_attempt(
                        existing_block_ids=model_call_start_block_ids,
                        reason=(
                            "model_stream_stalled"
                            if stream_stall
                            else "model_transport_error"
                        ),
                        timestamp=now_ms,
                    ),
                )
                _worker_checkpoint(
                    "native_model_stream_recovery_start",
                    request_id=str(state.get("request_id") or ""),
                    call_index=model_call_index,
                    attempt=consecutive_attempt,
                    consecutive_attempt=consecutive_attempt,
                    cumulative_attempt=cumulative_attempt,
                    max_attempts=max_recovery_attempts,
                    max_total_attempts=max_total_recovery_attempts,
                    phase=(
                        getattr(exc, "phase", "")
                        if stream_stall
                        else "transport_error"
                    ),
                    error_type=exc.__class__.__name__,
                    answer_chars=len(emitted_call_delta),
                    tool_outcome_count=len(tool_outcomes),
                    pending_sandbox_jobs=len(pending_sandbox_jobs),
                    llm_transport_generation=int(
                        context.get("llm_transport_generation") or 0
                    ),
                    next_llm_transport_generation=(
                        int(context.get("llm_transport_generation") or 0) + 1
                    ),
                )
                stream_close_result = await _close_event_stream_bounded(event_stream)
                stream_closed = bool(stream_close_result.get("closed"))
                current_close_result = await _close_owned_llm_transport(context)
                if current_close_result.get("owned"):
                    _worker_checkpoint(
                        "native_model_stream_transport_release_result",
                        request_id=str(state.get("request_id") or ""),
                        call_index=model_call_index,
                        llm_transport_generation=int(
                            context.get("llm_transport_generation") or 0
                        ),
                        stream_closed=stream_closed,
                        stream_close_attempted=bool(
                            stream_close_result.get("attempted")
                        ),
                        stream_close_error_type=str(
                            stream_close_result.get("error_type") or ""
                        ),
                        close_attempted=int(
                            current_close_result.get("attempted") or 0
                        ),
                        close_succeeded=int(current_close_result.get("closed") or 0),
                        close_error_types=list(
                            current_close_result.get("errors") or []
                        )[:4],
                    )
                recovery_state: NativeAgentState = dict(state)
                recovery_state.update(_state_from_adapter(state, adapter))
                recovery_state["timeline_blocks"] = adapter.snapshot_blocks()
                recovery_state["pending_question_bundle"] = pending_bundle or state.get("pending_question_bundle")
                recovery_state["pending_sandbox_jobs"] = pending_sandbox_jobs
                recovery_state["activated_skills"] = activated_skills
                recovery_state["tool_outcomes"] = tool_outcomes
                recovery_state["continuation_messages"] = continuation_messages
                recovery_state["resume_count"] = int(state.get("resume_count") or 0) + 1
                recovery_state["stream_recovery_attempts"] = consecutive_attempt
                recovery_state["stream_recovery_total_attempts"] = cumulative_attempt
                recovery_state["stream_recovery_mode"] = True
                # Superseded prose is hidden from both the UI and model state, so
                # the fresh provider call must be allowed to replay it verbatim.
                recovery_state["stream_recovery_replay_prefix"] = ""
                recovery_state["loaded_tool_ids"] = _normalize_loaded_tool_ids(state.get("loaded_tool_ids"))
                recovery_state["tool_schema_reload_count"] = int(state.get("tool_schema_reload_count") or 0)
                recovery_state["system_prompt"] = _strip_legacy_recovery_state(
                    state.get("system_prompt")
                )
                recovery_state["recovery_state_envelope"] = _build_recovery_state_envelope(
                    state=recovery_state,
                    interruption={
                        "reason": (
                            "model_stream_stalled"
                            if stream_stall
                            else "model_transport_error"
                        ),
                        "phase": (
                            getattr(exc, "phase", "")
                            if stream_stall
                            else "transport_error"
                        )
                        or "unknown",
                        "call_index": int(getattr(exc, "call_index", 0) or 0),
                        "consecutive_attempt": consecutive_attempt,
                        "cumulative_attempt": cumulative_attempt,
                        "max_attempts": max_recovery_attempts,
                        "max_total_attempts": max_total_recovery_attempts,
                        "discarded_partial_model_output": bool(
                            (
                                stream_stall
                                and getattr(exc, "phase", "")
                                == "after_progress"
                            )
                            or emitted_call_delta
                        ),
                        "discarded_partial_output_chars": len(emitted_call_delta),
                    },
                    continuation_messages=continuation_messages,
                    task_tree_snapshot=task_tree_snapshot,
                    task_tree_fetch_status=task_tree_fetch_status,
                    tool_schema_snapshot=tool_schema_snapshot,
                    tool_outcomes=tool_outcomes,
                    pending_sandbox_jobs=pending_sandbox_jobs,
                )
                try:
                    recovery_context, recovery_owner = _fresh_recovery_context(
                        context
                    )
                except Exception as transport_exc:  # noqa: BLE001
                    transport_recreate_failed = True
                    error_type = transport_exc.__class__.__name__
                    error_message = (
                        "model recovery transport recreation failed: "
                        f"{error_type}"
                    )
                    _worker_checkpoint(
                        "native_model_stream_transport_recreate_failed",
                        request_id=str(state.get("request_id") or ""),
                        call_index=model_call_index,
                        llm_transport_generation=int(
                            context.get("llm_transport_generation") or 0
                        ),
                        error_type=error_type,
                    )
                else:
                    next_generation = int(
                        recovery_context.get("llm_transport_generation") or 0
                    )
                    _worker_checkpoint(
                        "native_model_stream_transport_recreated",
                        request_id=str(state.get("request_id") or ""),
                        call_index=model_call_index,
                        previous_llm_transport_generation=int(
                            context.get("llm_transport_generation") or 0
                        ),
                        llm_transport_generation=next_generation,
                        fresh_client=bool(recovery_owner is not None),
                        stream_closed=stream_closed,
                        stream_close_attempted=bool(
                            stream_close_result.get("attempted")
                        ),
                        stream_close_error_type=str(
                            stream_close_result.get("error_type") or ""
                        ),
                    )
                    recovery_state["_canonical_any_tool_called"] = bool(
                        any_tool_called
                    )
                    transition = CanonicalRuntimeTransition(
                        "retry_turn",
                        {"reason": "model_transport_recovery"},
                    )
                    transition.next_state = recovery_state
                    transition.next_context = recovery_context
                    raise transition
            if not transport_recreate_failed:
                _worker_checkpoint(
                    "native_model_stream_recovery_exhausted",
                    request_id=str(state.get("request_id") or ""),
                    attempts=current_total_attempts,
                    consecutive_attempts=current_consecutive_attempts,
                    cumulative_attempts=current_total_attempts,
                    max_attempts=max_recovery_attempts,
                    max_total_attempts=max_total_recovery_attempts,
                    call_index=model_call_index,
                    error_type=exc.__class__.__name__,
                )
        if model_call_was_inflight:
            _emit_events(
                writer,
                adapter.supersede_model_attempt(
                    existing_block_ids=model_call_start_block_ids,
                    reason=(
                        "model_stream_recovery_exhausted"
                        if recoverable_transport_error
                        else "model_call_failed"
                    ),
                    timestamp=now_ms,
                ),
            )
        if execution_limit_reached and any_tool_called and tool_outcomes:
            final_answer_recovery_attempts = int(
                state.get("final_answer_recovery_attempts") or 0
            )
            if final_answer_recovery_attempts < 1:
                _worker_checkpoint(
                    "native_execution_limit_finalization_start",
                    request_id=str(state.get("request_id") or ""),
                    call_index=model_call_index,
                    attempt=final_answer_recovery_attempts + 1,
                    tool_outcome_count=len(tool_outcomes),
                )
                await _close_event_stream_bounded(event_stream)
                recovery_state: NativeAgentState = dict(state)
                recovery_state.update(_state_from_adapter(state, adapter))
                recovery_state["timeline_blocks"] = adapter.snapshot_blocks()
                recovery_state["usage"] = usage_ledger
                recovery_state["pending_question_bundle"] = None
                recovery_state["pending_sandbox_jobs"] = pending_sandbox_jobs
                recovery_state["activated_skills"] = activated_skills
                recovery_state["tool_outcomes"] = tool_outcomes
                recovery_state["continuation_messages"] = continuation_messages
                recovery_state["resume_count"] = int(state.get("resume_count") or 0) + 1
                recovery_state["stream_recovery_mode"] = False
                recovery_state["stream_recovery_attempts"] = stream_recovery_attempts
                recovery_state["stream_recovery_total_attempts"] = (
                    stream_recovery_total_attempts
                )
                recovery_state["stream_recovery_replay_prefix"] = ""
                recovery_state["final_answer_recovery_attempts"] = (
                    final_answer_recovery_attempts + 1
                )
                recovery_state["system_prompt"] = _strip_legacy_recovery_state(
                    state.get("system_prompt")
                )
                recovery_state["recovery_state_envelope"] = _build_recovery_state_envelope(
                    state=recovery_state,
                    interruption={
                        "reason": "execution_limit_reached",
                        "phase": "after_tool_loop",
                        "call_index": model_call_index,
                        "consecutive_attempt": final_answer_recovery_attempts + 1,
                        "cumulative_attempt": final_answer_recovery_attempts + 1,
                        "max_attempts": 1,
                        "discarded_partial_model_output": bool(
                            _adapter_answer_text(adapter)[
                                max(0, int(model_call_answer_start_chars or 0)) :
                            ]
                        ),
                        "discarded_partial_output_chars": len(
                            _adapter_answer_text(adapter)[
                                max(0, int(model_call_answer_start_chars or 0)) :
                            ]
                        ),
                    },
                    continuation_messages=continuation_messages,
                    task_tree_snapshot=task_tree_snapshot,
                    task_tree_fetch_status=task_tree_fetch_status,
                    tool_schema_snapshot=tool_schema_snapshot,
                    tool_outcomes=tool_outcomes,
                    pending_sandbox_jobs=pending_sandbox_jobs,
                )
                recovery_state["_canonical_any_tool_called"] = bool(any_tool_called)
                transition = CanonicalRuntimeTransition(
                    "retry_turn",
                    {"reason": "execution_limit_finalization"},
                )
                transition.next_state = recovery_state
                raise transition
        if timeout_like and any_tool_called and tool_outcomes and not stream_stall:
            now_ms = int(time.time() * 1000)
            _complete_reasoning(timestamp_ms=now_ms)
            adapter.record_usage_snapshot(usage_ledger, timeout=True)
            writer(
                adapter.build_turn_failed(
                    "模型在工具结果返回后超时，尚未提交最终答案。",
                    error_kind="final_answer_incomplete",
                    error_type=error_type,
                    traceback_summary=traceback_summary,
                )
            )
            _transition_native_task_state(state, TaskPhase.FAILED, resume_reason="model_timeout_after_tools")
            return {
                "any_tool_called": any_tool_called,
                **_state_from_adapter(state, adapter),
                "pending_sandbox_jobs": pending_sandbox_jobs,
                "status": "failed",
                "error_message": f"{error_type}: {error_message}",
            }
        if recoverable_transport_error and any_tool_called and tool_outcomes:
            _worker_checkpoint(
                "native_model_stream_stall_tool_fallback",
                request_id=str(state.get("request_id") or ""),
                call_index=model_call_index,
                tool_outcome_count=len(tool_outcomes),
                has_answer_content=_has_answer_content(adapter),
            )
            _complete_reasoning(timestamp_ms=now_ms)
            adapter.record_usage_snapshot(usage_ledger, timeout=True)
            writer(
                adapter.build_turn_failed(
                    "模型在工具结果返回后流式中断，尚未提交最终答案。",
                    error_kind="final_answer_incomplete",
                    error_type=error_type,
                    traceback_summary=traceback_summary,
                )
            )
            _transition_native_task_state(state, TaskPhase.FAILED, resume_reason="model_stream_stalled_after_tools")
            return {
                "any_tool_called": any_tool_called,
                **_state_from_adapter(state, adapter),
                "pending_sandbox_jobs": pending_sandbox_jobs,
                "status": "failed",
                "error_message": f"{error_type}: {error_message}",
            }
        if any_tool_called and tool_outcomes:
            _worker_checkpoint(
                "native_model_exception_tool_fallback",
                request_id=str(state.get("request_id") or ""),
                call_index=model_call_index,
                error_type=error_type,
                tool_outcome_count=len(tool_outcomes),
                pending_sandbox_jobs=len(pending_sandbox_jobs),
                has_answer_content=_has_answer_content(adapter),
            )
            _complete_reasoning(timestamp_ms=now_ms)
            adapter.record_usage_snapshot(usage_ledger, timeout=False)
            writer(
                adapter.build_turn_failed(
                    "模型在工具结果返回后异常退出，尚未提交最终答案。",
                    error_kind="final_answer_incomplete",
                    error_type=error_type,
                    traceback_summary=traceback_summary,
                )
            )
            _transition_native_task_state(state, TaskPhase.FAILED, resume_reason="model_error_after_tools")
            return {
                "any_tool_called": any_tool_called,
                **_state_from_adapter(state, adapter),
                "pending_sandbox_jobs": pending_sandbox_jobs,
                "status": "failed",
                "error_message": f"{error_type}: {error_message}",
            }
        if stream_stall:
            _complete_reasoning(timestamp_ms=now_ms)
            adapter.record_usage_snapshot(usage_ledger, timeout=True)
            failed_event = adapter.build_turn_failed(
                "模型流式输出超过进展等待阈值，已中止本轮以保留任务树和工具状态。请继续时先读取任务树、沙盒任务和文件抽屉后从断点推进。",
                error_kind="model_stream_stalled",
                error_type=error_type,
                traceback_summary=traceback_summary,
            )
            writer(failed_event)
            _transition_native_task_state(state, TaskPhase.FAILED, resume_reason="model_stream_stalled")
            return {
                "any_tool_called": any_tool_called,
                **_state_from_adapter(state, adapter),
                "pending_sandbox_jobs": pending_sandbox_jobs,
                "status": "failed",
                "error_message": f"{error_type}: {error_message}",
            }
        failed_event = adapter.build_turn_failed(
            f"{error_type}: {error_message}",
            error_kind="agent_failed",
            error_type=error_type,
            traceback_summary=traceback_summary,
        )
        writer(failed_event)
        _transition_native_task_state(state, TaskPhase.FAILED, resume_reason="agent_failed")
        return {
            "any_tool_called": any_tool_called,
            **_state_from_adapter(state, adapter),
            "pending_sandbox_jobs": pending_sandbox_jobs,
            "status": "failed",
            "error_message": f"{error_type}: {error_message}",
        }


async def run_native_agent_turn(
    state: NativeAgentState,
    context: NativeAgentContext,
    *,
    event_writer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run typed runtime rebuilds iteratively on one canonical ledger."""

    current_state = state
    current_context = context
    try:
        while True:
            try:
                return await _run_native_agent_turn_once(
                    current_state,
                    current_context,
                    event_writer=event_writer,
                )
            except CanonicalRuntimeTransition as transition:
                if not isinstance(transition.next_state, dict):
                    raise RuntimeError(
                        "CANONICAL_RUNTIME_TRANSITION_STATE_MISSING:"
                        f"{transition.kind}"
                    ) from transition
                current_state = transition.next_state
                if transition.next_context is not None:
                    current_context = transition.next_context
    finally:
        if current_context is not context:
            close_result = await _close_owned_llm_transport(current_context)
            _worker_checkpoint(
                "native_model_stream_transport_release",
                request_id=str(current_state.get("request_id") or ""),
                llm_transport_generation=int(
                    current_context.get("llm_transport_generation") or 0
                ),
                already_closed=not bool(close_result.get("owned")),
                close_attempted=int(close_result.get("attempted") or 0),
                close_succeeded=int(close_result.get("closed") or 0),
                close_error_types=list(close_result.get("errors") or [])[:4],
            )


def _request_to_native_state(request: AgentInvokeRequest) -> NativeAgentState:
    request_id = str(request.request_id or "").strip() or f"req_agent_{uuid.uuid4().hex}"
    from src.context.task_state import build_initial_task_state

    seed_timeline_blocks = list(request.timeline_blocks or [])
    hitl_parent_request_id = str(request.hitl_parent_request_id or "").strip()
    is_hitl_resume = bool(hitl_parent_request_id)
    task_authority = (
        request.task_authority.model_dump(mode="json")
        if request.task_authority is not None
        else None
    )
    capability_resume_control = (
        request.capability_resume_control.model_dump(mode="json")
        if request.capability_resume_control is not None
        else None
    )
    if capability_resume_control is not None and (
        not is_hitl_resume
        or capability_resume_control["parent_request_id"] != hitl_parent_request_id
        or capability_resume_control["continuation_request_id"] != request_id
        or not isinstance(request.turn_checkpoint, dict)
        or capability_resume_control["checkpoint_id"]
        != str(request.turn_checkpoint.get("checkpoint_id") or "")
    ):
        raise ValueError("CAPABILITY_AUTHORIZATION_RESUME_CONTROL_MISSING")
    sandbox_resume_payload = (
        dict(request.sandbox_resume_payload)
        if isinstance(request.sandbox_resume_payload, dict)
        else None
    )
    is_sandbox_resume = bool(sandbox_resume_payload)
    if task_authority is None:
        if is_hitl_resume or is_sandbox_resume or capability_resume_control is not None:
            raise ValueError("TASK_AUTHORITY_CONTROL_MISSING")
        task_authority = {
            "schema_version": "evoengine.task-authority-control/v1",
            "transport_request_id": request_id,
            "task_authority_request_id": request_id,
            "task_run_id": None,
            "continuation_parent_request_id": None,
        }
    if str(task_authority.get("transport_request_id") or "") != request_id:
        raise ValueError("TASK_AUTHORITY_TRANSPORT_REQUEST_MISMATCH")
    control_parent_request_id = str(
        task_authority.get("continuation_parent_request_id") or ""
    ).strip()
    if is_hitl_resume and control_parent_request_id != hitl_parent_request_id:
        raise ValueError("TASK_AUTHORITY_CONTINUATION_PARENT_MISMATCH")
    task_authority_request_id = str(
        task_authority.get("task_authority_request_id") or ""
    ).strip()
    task_authority_run_id = str(
        task_authority.get("task_run_id") or ""
    ).strip() or None
    if (
        not task_authority_request_id
        or (
            task_authority_request_id != request_id
            and task_authority_run_id is None
        )
    ):
        raise ValueError("TASK_AUTHORITY_CONTROL_INVALID")
    turn_checkpoint = (
        dict(request.turn_checkpoint)
        if isinstance(request.turn_checkpoint, dict)
        else None
    )
    if is_sandbox_resume:
        if (
            turn_checkpoint is None
            or str(turn_checkpoint.get("request_id") or "").strip() != request_id
        ):
            raise ValueError("SANDBOX_RESUME_CHECKPOINT_REQUEST_MISMATCH")
        if str(turn_checkpoint.get("status") or "").strip() != "waiting_sandbox":
            raise ValueError("SANDBOX_RESUME_CHECKPOINT_STATUS_MISMATCH")
        payload_request_id = str(sandbox_resume_payload.get("request_id") or "").strip()
        if payload_request_id and payload_request_id != request_id:
            raise ValueError("SANDBOX_RESUME_PAYLOAD_REQUEST_MISMATCH")
        payload_checkpoint_id = str(
            sandbox_resume_payload.get("checkpoint_id") or ""
        ).strip()
        if payload_checkpoint_id and payload_checkpoint_id != str(
            turn_checkpoint.get("checkpoint_id") or ""
        ).strip():
            raise ValueError("SANDBOX_RESUME_PAYLOAD_CHECKPOINT_MISMATCH")
    restored_runtime = (
        _restore_runtime_from_turn_checkpoint(turn_checkpoint)
        if (is_hitl_resume or is_sandbox_resume)
        else None
    )
    if restored_runtime is None:
        restored_runtime = _restore_hitl_resume_state_from_timeline(
            seed_timeline_blocks if (is_hitl_resume or is_sandbox_resume) else []
        )
    if restored_runtime is None:
        restored_runtime = {}
    if is_sandbox_resume and not restored_runtime:
        raise ValueError("SANDBOX_RESUME_CHECKPOINT_INVALID")
    # Older checkpoints and callers may still carry max_turns.  It is ignored
    # deliberately: root-model call count is not a completion or safety gate.
    effective_max_turns = None
    restored_model_turns_used = max(
        0,
        int(restored_runtime.get("model_turns_used") or 0),
    )
    request_usage = _same_request_usage_ledger(
        request.usage_ledger,
        request_id=request_id,
    )
    restored_usage = _same_request_usage_ledger(
        restored_runtime.get("usage"),
        request_id=request_id,
    )
    restored_usage_is_same_request = (
        str(restored_runtime.get("checkpoint_request_id") or "").strip()
        == request_id
    )
    effective_usage = (
        restored_usage
        if restored_usage_is_same_request
        and int(restored_usage.get("all_model_call_count") or 0) > 0
        else request_usage
    )
    restored_tool_schema_reload_count = max(
        0,
        int(restored_runtime.get("tool_schema_reload_count") or 0),
    )
    restored_pending_sandbox_jobs = list(
        restored_runtime.get("pending_sandbox_jobs") or []
    )
    if is_sandbox_resume:
        restored_pending_sandbox_jobs = _apply_sandbox_resume_statuses(
            restored_pending_sandbox_jobs,
            sandbox_resume_payload,
        )
    task_state = build_initial_task_state(
        request_id=request_id,
        task_id=str(request.conversation_id or request_id),
        conversation_files=list(request.conversation_files or []),
    )
    if is_hitl_resume or is_sandbox_resume:
        restored_task_state = restored_runtime.get("task_state")
        if isinstance(restored_task_state, dict) and restored_task_state:
            try:
                task_state = TaskState.model_validate(restored_task_state)
            except Exception:
                pass
        task_state = task_state.model_copy(
            update={
                "resume_reason": (
                    "sandbox_result_ready" if is_sandbox_resume else "human_input_received"
                ),
                "missing_fields": [],
            }
        )
    initial_tool_cache = _build_initial_tool_cache_state(request)
    resident_loaded_tool_ids = _expand_loaded_tool_ids(get_runtime_resident_tool_ids())
    restored_loaded_tool_ids = _normalize_loaded_tool_ids(restored_runtime.get("loaded_tool_ids"))
    resume_required_tool_ids = ["sandbox_get_result"] if is_sandbox_resume else []
    initial_tool_cache["loaded_tool_ids"] = _normalize_loaded_tool_ids(
        [
            *resident_loaded_tool_ids,
            *list(initial_tool_cache.get("loaded_tool_ids") or []),
            *restored_loaded_tool_ids,
            *resume_required_tool_ids,
        ]
    )
    initial_tool_cache["loaded_tool_entries"] = _normalize_loaded_tool_ids(
        [
            *list(initial_tool_cache.get("loaded_tool_entries") or []),
            *list(restored_runtime.get("loaded_tool_entries") or []),
        ]
    )
    return {
        "request_id": request_id,
        "task_authority": task_authority,
        "task_authority_request_id": task_authority_request_id,
        "task_authority_run_id": task_authority_run_id,
        "user_text": request.text.strip(),
        "task_phase": task_state.phase.value,
        "task_state": task_state.model_dump(mode="json"),
        "project_summary": str(request.project_summary or ""),
        "system_prompt": str(request.system_prompt or ""),
        "history": list(request.history or []),
        "knowledge": dict(request.knowledge or {}),
        "database": dict(request.database or {}),
        "attachments": list(request.attachments or []),
        "conversation_files": list(request.conversation_files or []),
        "reference_context": (
            request.reference_context.model_dump(mode="json")
            if request.reference_context is not None
            else _normalize_reference_context_payload(
                restored_runtime.get("reference_context")
            )
        ),
        "project_id": request.project_id,
        "conversation_id": request.conversation_id,
        "user_id": request.user_id,
        "pending_question_bundle": None,
        "pending_sandbox_jobs": restored_pending_sandbox_jobs,
        "sandbox_resume_payload": sandbox_resume_payload,
        "human_answer_bundle": (
            dict(request.human_answer_bundle)
            if isinstance(request.human_answer_bundle, dict)
            else None
        ),
        "capability_resume_control": (
            capability_resume_control
        ),
        "pending_capability_authorization": (
            dict(
                (restored_runtime.get("runtime_control") or {}).get(
                    "pending_capability_authorization"
                )
            )
            if isinstance(restored_runtime.get("runtime_control"), dict)
            and isinstance(
                (restored_runtime.get("runtime_control") or {}).get(
                    "pending_capability_authorization"
                ),
                dict,
            )
            else None
        ),
        "turn_checkpoint": (
            turn_checkpoint
        ),
        "session_compact_checkpoint": (
            dict(request.session_compact_checkpoint)
            if isinstance(request.session_compact_checkpoint, dict)
            else dict(restored_runtime.get("session_compact_checkpoint") or {})
        ),
        "resource_access_ledger": normalize_resource_access_ledger(
            restored_runtime.get("resource_access_ledger")
        ),
        "timeline_blocks": seed_timeline_blocks,
        "usage": effective_usage,
        "citations": [],
        "final_reply_text": "",
        "reply_incomplete": False,
        "error_message": "",
        "resume_count": 1 if (is_hitl_resume or is_sandbox_resume) else 0,
        "stream_recovery_attempts": 0,
        "stream_recovery_total_attempts": 0,
        "stream_recovery_mode": False,
        "stream_recovery_replay_prefix": "",
        "recovery_state_envelope": None,
        "truncated_answer_continuation_attempts": 0,
        "final_answer_recovery_attempts": 0,
        "tool_outcomes": [],
        "side_effect_ledger": _normalize_side_effect_ledger({}),
        "continuation_messages": (
            list(
                restored_runtime.get("checkpoint_protocol_messages")
                or restored_runtime.get("hitl_resume_messages")
                or []
            )
            if is_sandbox_resume
            else []
        ),
        "hitl_resume_messages": (
            list(
                restored_runtime.get("checkpoint_protocol_messages")
                or restored_runtime.get("hitl_resume_messages")
                or []
            )
            if is_hitl_resume
            else []
        ),
        **initial_tool_cache,
        "activated_skills": list(restored_runtime.get("activated_skills") or []),
        "max_turns": effective_max_turns,
        "model_turns_used": restored_model_turns_used,
        "tool_schema_reload_count": restored_tool_schema_reload_count,
        "status": "running",
    }


async def stream_native_agent_events(
    settings: Settings,
    llm_base: Any,
    request: AgentInvokeRequest,
    *,
    llm_factory: Callable[[], Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """直接运行内层 ReAct；不保存或恢复外层 LangGraph 程序状态。"""
    state = _request_to_native_state(request)
    llm_for_context = _clone_llm_disable_streaming(llm_base) if bool(getattr(request, "disable_streaming", False)) else llm_base
    context = build_runtime_context(
        settings,
        llm_for_context,
        llm_factory=(
            (lambda: _clone_llm_disable_streaming(llm_factory()))
            if llm_factory is not None
            and bool(getattr(request, "disable_streaming", False))
            else llm_factory
        ),
        project_id=request.project_id,
        conversation_id=request.conversation_id,
        user_id=request.user_id,
    )
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    def _write(event: dict[str, Any]) -> None:
        queue.put_nowait(event)

    async def _run() -> None:
        nonlocal state
        try:
            update = await run_native_agent_turn(state, context, event_writer=_write)
            state.update(update)
        finally:
            queue.put_nowait(None)

    task = asyncio.create_task(_run())
    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event
        await task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
