# Reading excerpt; functions/classes retain their original bodies.
# Origin: server/backend/server_backend.py @ 8ee96f3171f20eb56c08e4a8aba6a43ab02f0f69
# Module setup, unselected helpers and service wiring are omitted.
# See source_manifest.json and docs/DEPENDENCIES.md. Do not execute this slice.

# ORIGINAL L1-L1
from __future__ import annotations

# ORIGINAL L3-L3
import asyncio

# ORIGINAL L4-L4
import json

# ORIGINAL L5-L5
import logging

# ORIGINAL L6-L6
import os

# ORIGINAL L7-L7
import re

# ORIGINAL L8-L8
import uuid

# ORIGINAL L9-L9
from datetime import UTC, datetime

# ORIGINAL L10-L10
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

# ORIGINAL L11-L11
from dataclasses import dataclass, field

# ORIGINAL L12-L12
from typing import Any, Literal

# ORIGINAL L13-L13
import time

# ORIGINAL L15-L15
import httpx

# ORIGINAL L16-L16
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect

# ORIGINAL L17-L17
from fastapi.middleware.cors import CORSMiddleware

# ORIGINAL L18-L18
from fastapi.responses import JSONResponse

# ORIGINAL L19-L19
from routers import auth

# ORIGINAL L20-L20
from routers import activity

# ORIGINAL L21-L21
from routers import assets

# ORIGINAL L22-L22
from routers import literature

# ORIGINAL L23-L23
from routers import chat_attachments

# ORIGINAL L24-L24
from routers import conversation_files

# ORIGINAL L25-L25
from routers import support

# ORIGINAL L26-L26
from routers import agent_feedback

# ORIGINAL L27-L27
from routers import announcements

# ORIGINAL L28-L28
from routers import homepage_capabilities

# ORIGINAL L29-L29
from routers import admin_auth

# ORIGINAL L30-L30
from routers import admin_dashboard

# ORIGINAL L31-L31
from routers import admin_exports

# ORIGINAL L32-L32
from routers import admin_file_parsing

# ORIGINAL L33-L33
from routers import admin_model_costs

# ORIGINAL L34-L34
from routers import admin_sandbox

# ORIGINAL L35-L35
from routers import admin_smoke_tests

# ORIGINAL L36-L36
from routers import export

# ORIGINAL L37-L37
from routers import payments

# ORIGINAL L38-L38
from routers import admin_payments

# ORIGINAL L39-L39
from routers import sandbox

# ORIGINAL L40-L40
from routers import datalake

# ORIGINAL L41-L41
from routers import task_tree

# ORIGINAL L42-L42
from routers import version_reward

# ORIGINAL L43-L43
from routers import admin_chat

# ORIGINAL L44-L44
from routers import rd_agents

# ORIGINAL L45-L45
from routers import rd_cases

# ORIGINAL L46-L46
from routers import rd_evals

# ORIGINAL L47-L47
from routers import rd_models

# ORIGINAL L48-L48
from routers import rd_experiments

# ORIGINAL L49-L49
from routers import rd_observability

# ORIGINAL L50-L50
from routers import rd_workflows

# ORIGINAL L51-L51
from routers import reference_resources

# ORIGINAL L52-L52
from routers import scientific_figures

# ORIGINAL L53-L53
from routers import deliverable_capabilities

# ORIGINAL L54-L54
from utils.posthog_client import capture as ph_capture

# ORIGINAL L55-L55
from admin_bootstrap import ensure_admin_user

# ORIGINAL L56-L56
from activity_service import ensure_default_activity

# ORIGINAL L57-L57
from abstractdataclasses.docstree import DocsTree, TreeOperation

# ORIGINAL L58-L58
from pydantic import BaseModel

# ORIGINAL L60-L60
from security import get_current_user, get_user_from_token

# ORIGINAL L61-L76
from database import (
    AgentTurnFeedback,
    AssetStatus,
    ChatAttachment,
    CreditTransaction,
    ParseStatus,
    ProjectAsset,
    ProjectConversationSession,
    ProjectConversationMessage,
    ProjectConversationTurn,
    ProjectSession,
    ProjectTaskTreeRun,
    ProjectTreeVersion,
    User,
    engine,
)

# ORIGINAL L77-L77
from sqlmodel import Session, select

# ORIGINAL L78-L78
from sqlalchemy import func

# ORIGINAL L79-L79
from sqlalchemy.exc import IntegrityError

# ORIGINAL L80-L87
from services import (
    payment_scheduler_runtime,
    project_bootstrap_service,
    sandbox_service,
    smart_point_service,
    task_completion_receipt_service,
    task_tree_service,
)

# ORIGINAL L88-L88
from services.file_parsing_admin_service import bootstrap_defaults as bootstrap_file_parsing_defaults

# ORIGINAL L89-L89
from services.chat_attachment_service import build_chat_attachment_status_payload

# ORIGINAL L90-L94
from services.citation_policy import (
    project_final_answer,
    sanitize_citation_container,
    sanitize_citations,
)

# ORIGINAL L95-L101
from services.agent_turn_runtime import (
    AgentTurnTransportDriver,
    ProgressLivenessWatchdog,
    SandboxContinuationContractError,
    build_next_sandbox_continuation_transition,
    is_sandbox_terminal_status,
)

# ORIGINAL L102-L106
from services.conversation_file_service import (
    build_agent_conversation_file_attachment_ref,
    build_agent_conversation_file_context,
    register_conversation_file_from_chat_attachment,
)

# ORIGINAL L107-L107
from schemas.reference_refs import ResolvedReference

# ORIGINAL L108-L113
from services.reference_resolution_service import (
    ReferenceServiceError,
    build_reference_context,
    normalize_reference_inputs,
    resolve_reference_inputs,
)

# ORIGINAL L114-L117
from services.turn_authorization_service import (
    TurnAuthorizationError,
    issue_or_reject_authorization_in_session,
)

# ORIGINAL L118-L121
from services.model_usage_ledger import (
    merge_usage_ledgers,
    normalize_usage_ledger,
)

# ORIGINAL L122-L126
from services.model_cost_service import (
    extract_provider_usage,
    persist_usage_ledger_events,
    record_usage_event,
)

# ORIGINAL L127-L127
from services import rd_workflow_service

# ORIGINAL L128-L128
from services.sandbox_manifest_service import sync_sandbox_manifests

# ORIGINAL L129-L129
from run_migrations import apply_pending_migrations

# ORIGINAL L130-L130
from shared.tree_cleanup import cleanup_legacy_plan_tree_snapshot

# ORIGINAL L131-L131
from events import emit_event_nowait

# ORIGINAL L132-L132
from utils.observability import check_tools_health, log_tools_health, setup_logging

# ORIGINAL L133-L133
from app_version import get_app_version

# ORIGINAL L134-L138
from context_budget import (
    ContextBudgetManager,
    ContextBudget,
    count_tokens,
)

# ORIGINAL L176-L176
from token_service import token_service, BILLING_VERSION

# ORIGINAL L177-L177
from version_reward_service import ensure_default_version_reward

# ORIGINAL L817-L817
import json as _json

# ORIGINAL L818-L818
import re as _re

# ORIGINAL L819-L819
from pathlib import Path as _Path

# ORIGINAL L820-L820
from fastapi.responses import JSONResponse as _JSONResponse

# ORIGINAL L821-L821
from database import get_session, SkillConfig

# ORIGINAL L1571-L1571
import collections  # noqa: E402  — keep import close to the structures that use it.

# ORIGINAL L3312-L3341
def _persist_turn_checkpoint_payload(
    project_session: ProjectSession,
    request_id: str,
    checkpoint: dict[str, Any],
) -> None:
    """Persist the canonical resume state before a HITL interrupt is exposed."""
    if checkpoint.get("schema_version") != "evoengine.turn-checkpoint/v1":
        raise ValueError("unsupported turn checkpoint schema")
    checkpoint_request_id = str(checkpoint.get("request_id") or "").strip()
    if checkpoint_request_id and checkpoint_request_id != str(request_id or "").strip():
        raise ValueError("turn checkpoint request_id mismatch")
    with Session(engine) as db:
        turn = db.exec(
            select(ProjectConversationTurn).where(
                ProjectConversationTurn.project_session_id == project_session.id,
                ProjectConversationTurn.request_id == request_id,
            )
        ).first()
        if not turn:
            raise ValueError("turn checkpoint target not found")
        trace_payload = (
            dict(turn.thinking_trace_json)
            if isinstance(turn.thinking_trace_json, dict)
            else {"schema_version": "v1"}
        )
        trace_payload["turn_checkpoint"] = dict(checkpoint)
        turn.thinking_trace_json = trace_payload
        turn.updated_at = datetime.now(UTC).replace(tzinfo=None)
        db.add(turn)
        db.commit()

# ORIGINAL L3344-L3361
def _load_turn_checkpoint_payload(
    project_session_id: int,
    request_id: str,
) -> dict[str, Any] | None:
    with Session(engine) as db:
        turn = db.exec(
            select(ProjectConversationTurn).where(
                ProjectConversationTurn.project_session_id == int(project_session_id),
                ProjectConversationTurn.request_id == request_id,
            )
        ).first()
        trace = turn.thinking_trace_json if turn and isinstance(turn.thinking_trace_json, dict) else {}
        checkpoint = trace.get("turn_checkpoint")
        if not isinstance(checkpoint, dict):
            return None
        if checkpoint.get("schema_version") != "evoengine.turn-checkpoint/v1":
            return None
        return dict(checkpoint)

# ORIGINAL L5496-L5691
def _finish_turn_success(
    project_session: ProjectSession,
    user_id: int,
    request_id: str,
    final_reply_text: str,
    billing_meta: dict[str, Any] | None = None,
    thinking_events: list[dict[str, Any]] | None = None,
    timeline_blocks: list[dict[str, Any]] | None = None,
    citations: list[dict[str, Any]] | None = None,
    turn_phase: str = "success",
    error_kind: str | None = None,
    context_usage: dict[str, Any] | None = None,
    usage_ledger: dict[str, Any] | None = None,
    completion_receipt: dict[str, Any] | None = None,
) -> None:
    now = datetime.now(UTC).replace(tzinfo=None)
    citations = _filter_agent_citations(citations)
    with Session(engine) as db:
        turn = db.exec(
            select(ProjectConversationTurn).where(
                ProjectConversationTurn.project_session_id == project_session.id,
                ProjectConversationTurn.request_id == request_id,
            )
        ).first()
        if not turn:
            conversation = _ensure_project_default_conversation(db, project_session=project_session)
            turn = ProjectConversationTurn(
                project_session_id=project_session.id,
                conversation_session_id=int(conversation.id or 0),
                conversation_id=conversation.conversation_id,
                user_id=user_id,
                request_id=request_id,
                input_text="",
                status="running",
                created_at=now,
                updated_at=now,
            )
            db.add(turn)
            db.commit()
            db.refresh(turn)

        turn.final_reply_text = final_reply_text
        turn.status = "success"
        turn.error_code = None
        turn.error_message = None
        bm = billing_meta or {}
        turn.consumed_credits = _to_credit_decimal(bm.get("consumed_credits", 0.0))
        turn.output_char_count = int(bm.get("output_char_count", 0))
        turn.token_ratio_snapshot = _to_credit_decimal(bm.get("token_ratio", 0.0))
        turn.billing_version = str(bm.get("billing_version") or BILLING_VERSION)
        # token_weighted_v1 fields
        turn.input_tokens = int(bm.get("input_tokens", 0))
        turn.output_tokens = int(bm.get("output_tokens", 0))
        turn.model_name = str(bm.get("model_name") or "") or None
        turn.billing_w_in_snapshot = _to_credit_decimal(bm.get("billing_w_in_snapshot", 0.0))
        turn.billing_w_out_snapshot = _to_credit_decimal(bm.get("billing_w_out_snapshot", 0.0))
        turn.model_weight_snapshot = _to_credit_decimal(bm.get("model_weight_snapshot", 1.0))
        finalized_thinking = _finalize_thinking_events_for_storage(thinking_events)
        existing_trace = turn.thinking_trace_json if isinstance(turn.thinking_trace_json, dict) else {}
        existing_meta = existing_trace.get("meta") if isinstance(existing_trace.get("meta"), dict) else {}
        trace_meta: dict[str, Any] = dict(existing_meta)
        trace_meta["phase"] = turn_phase or "success"
        trace_meta.pop("pending_sandbox_jobs", None)
        if final_reply_text and not str(trace_meta.get("answer_started_at") or "").strip():
            trace_meta["answer_started_at"] = _to_utc_iso(now)
        trace_meta["answer_completed_at"] = _to_utc_iso(now)
        trace_payload: dict[str, Any] = {
            "schema_version": "v1",
            "meta": trace_meta,
        }
        trace_payload = _preserve_trace_runtime_state(existing_trace, trace_payload)
        if error_kind:
            trace_payload["meta"]["error_kind"] = error_kind
        if finalized_thinking:
            trace_payload["events"] = finalized_thinking
        if citations:
            trace_payload["citations"] = citations
        if isinstance(usage_ledger, dict) and usage_ledger:
            trace_payload["usage_ledger"] = normalize_usage_ledger(
                usage_ledger,
                request_id=request_id,
            )
        if isinstance(context_usage, dict) and context_usage:
            trace_payload["context_usage"] = context_usage
        if isinstance(completion_receipt, dict) and completion_receipt:
            trace_payload["completion_receipt"] = dict(completion_receipt)
        trace_payload = _sync_trace_block_timeline(turn, trace_payload)
        trace_payload = _sync_trace_timeline_v3(trace_payload, timeline_blocks=timeline_blocks)
        if finalized_thinking or citations or turn_phase or context_usage:
            turn.thinking_trace_json = trace_payload
        turn.updated_at = now
        db.add(turn)
        _safe_persist_turn_model_usage(
            db,
            turn=turn,
            project_session=project_session,
            user_id=user_id,
            request_id=request_id,
            usage_ledger=trace_payload.get("usage_ledger"),
        )
        _safe_ensure_workflow_run(
            db,
            workflow_run_id=request_id,
            title=turn.input_text[:120] if turn.input_text else turn.conversation_id or request_id,
            project_id=project_session.project_id,
            conversation_id=turn.conversation_id,
            request_id=request_id,
            user_id=user_id,
            traffic_label="prod_user",
            status="succeeded",
            context_json={"turn_phase": turn_phase},
        )
        _safe_upsert_workflow_step(
            db,
            workflow_run_id=request_id,
            step_key="user_turn",
            title="用户输入",
            step_type="conversation_turn",
            status="succeeded",
            request_id=request_id,
        )
        _safe_upsert_workflow_step(
            db,
            workflow_run_id=request_id,
            step_key="agent_reply",
            title="智能体完成回复",
            step_type="agent_reply",
            status="succeeded",
            request_id=request_id,
            payload_json={
                "final_reply_text": final_reply_text[:2000],
                "billing_meta": billing_meta or {},
                "citation_count": len(citations or []),
            },
        )

        seq_no = _next_message_seq(db, turn.id)
        msg = ProjectConversationMessage(
            turn_id=turn.id,
            project_session_id=project_session.id,
            conversation_session_id=turn.conversation_session_id,
            conversation_id=turn.conversation_id,
            user_id=user_id,
            request_id=request_id,
            role="agent",
            message_type="final",
            seq_no=seq_no,
            content_text=final_reply_text,
            content_json={
                "schema_version": "v1",
                "source": "agent_bridge",
                "project_id": project_session.project_id,
                "request_id": request_id,
                "consumed_credits": float(turn.consumed_credits),
                "output_char_count": int(turn.output_char_count),
                "token_ratio": float(turn.token_ratio_snapshot),
                "billing_version": turn.billing_version,
                "thinking_events": finalized_thinking,
                "block_timeline": _normalize_timeline_v3_blocks(timeline_blocks),
                "block_timeline_version": "v3" if timeline_blocks else str(trace_payload.get("meta", {}).get("block_timeline_version") or "v2"),
                "citations": [c for c in (citations or []) if isinstance(c, dict)],
            },
            created_at=now,
        )
        db.add(msg)
        conversation = (
            db.get(ProjectConversationSession, int(turn.conversation_session_id or 0))
            if turn.conversation_session_id
            else _ensure_project_default_conversation(db, project_session=project_session)
        )
        if conversation:
            _update_conversation_auto_fields(
                conversation,
                first_input_text=turn.input_text,
                event_time=now,
            )
            db.add(conversation)
        db.commit()
        _safe_emit_server_event(
            "backend.turn_succeeded",
            source_service="backend.server_backend",
            request_id=request_id,
            conversation_id=turn.conversation_id if turn else None,
            project_id=project_session.project_id,
            user_id=user_id,
            workflow_run_id=request_id,
            payload_json={
                "turn_phase": turn_phase,
                "output_char_count": int(turn.output_char_count or 0) if turn else 0,
                "input_tokens": int(turn.input_tokens or 0) if turn else 0,
                "output_tokens": int(turn.output_tokens or 0) if turn else 0,
                "model_name": turn.model_name if turn else None,
            },
        )

    _persist_seq_counters.pop(request_id, None)

# ORIGINAL L6141-L6319
async def _finalize_turn_terminal(
    *,
    frontend_ws: WebSocket | None,
    project_session: ProjectSession,
    user_id: int,
    request_id: str,
    kind: Literal["success", "cancelled"],
    final_reply_text: str,
    usage_payload: dict[str, Any] | None,
    thinking_events: list[dict[str, Any]] | None,
    timeline_blocks: list[dict[str, Any]] | None,
    citations: list[dict[str, Any]] | None,
    turn_phase: str,
    ws_alive: bool,
    error_kind: str | None = None,
    billing_meta_override: dict[str, Any] | None = None,
    extra_done_payload: dict[str, Any] | None = None,
    context_usage: dict[str, Any] | None = None,
) -> FinalizeResult:
    """First terminal wins — guarded by per-request lock and running status."""
    started = time.monotonic()
    runtime = _get_active_turn(request_id)
    if runtime is not None:
        frontend_ws = runtime.get("frontend_ws")
        ws_alive = frontend_ws is not None
    raw_citations = citations
    final_reply_text, citations = _project_agent_final_answer(
        final_reply_text,
        raw_citations,
    )
    timeline_blocks = _normalize_timeline_v3_blocks(
        timeline_blocks,
        fallback_citations=(raw_citations if isinstance(raw_citations, list) else None),
    )
    lock = _turn_finalize_locks.setdefault(request_id, asyncio.Lock())
    async with lock:
        current_status = _get_turn_status_for_session(project_session.id, request_id)
        allowed_current_statuses = {"running"} if kind == "success" else {"running", "waiting_human"}
        if current_status not in allowed_current_statuses:
            _log_turn_checkpoint(
                "turn_finalize_skipped",
                request_id=request_id,
                project_id=project_session.project_id,
                project_session_id=int(project_session.id or 0),
                user_id=user_id,
                kind=kind,
                turn_phase=turn_phase,
                current_status=current_status,
                allowed_statuses=sorted(allowed_current_statuses),
                ws_alive=ws_alive,
                error_kind=error_kind or "",
            )
            return FinalizeResult(applied=False, skipped_status=current_status)

        _log_turn_checkpoint(
            "turn_finalize_started",
            request_id=request_id,
            project_id=project_session.project_id,
            project_session_id=int(project_session.id or 0),
            user_id=user_id,
            kind=kind,
            turn_phase=turn_phase,
            current_status=current_status,
            ws_alive=ws_alive,
            error_kind=error_kind or "",
            final_chars=len(final_reply_text or ""),
            thinking_count=len(thinking_events or []),
            timeline_block_count=len(timeline_blocks or []),
            citation_count=len(citations or []),
        )

        if billing_meta_override is not None:
            billing_meta = billing_meta_override
        else:
            billing_meta = _compute_billing_meta(
                usage_payload,
                user_id=user_id,
                project_id=project_session.project_id,
                request_id=request_id,
                output_char_count=len(final_reply_text),
            )

        usage = normalize_usage_ledger(
            usage_payload if isinstance(usage_payload, dict) else {},
            request_id=request_id,
        )
        reconciled_context_usage = (
            _reconcile_context_usage_with_provider(
                context_usage,
                usage=usage,
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
            )
            if isinstance(context_usage, dict) and context_usage
            else {}
        )

        if kind == "cancelled":
            _finish_turn_cancelled(
                project_session=project_session,
                user_id=user_id,
                request_id=request_id,
                final_reply_text=final_reply_text,
                billing_meta=billing_meta,
                thinking_events=thinking_events,
                timeline_blocks=timeline_blocks,
                citations=citations,
                context_usage=reconciled_context_usage,
                usage_ledger=usage,
            )
        else:
            _finish_turn_success(
                project_session=project_session,
                user_id=user_id,
                request_id=request_id,
                final_reply_text=final_reply_text,
                billing_meta=billing_meta,
                thinking_events=thinking_events,
                timeline_blocks=timeline_blocks,
                citations=citations,
                turn_phase=turn_phase,
                error_kind=error_kind,
                context_usage=reconciled_context_usage,
                usage_ledger=usage,
                completion_receipt=(
                    dict(extra_done_payload.get("completion_receipt"))
                    if isinstance(extra_done_payload, dict)
                    and isinstance(extra_done_payload.get("completion_receipt"), dict)
                    else None
                ),
            )

        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        done_payload: dict[str, Any] = {
            "text": final_reply_text,
            "usage": usage,
            "consumed_credits": billing_meta.get("consumed_credits", 0.0),
            "smart_point_balance": billing_meta.get("smart_point_balance", billing_meta.get("ai_credits", 0.0)),
            "ai_credits": billing_meta.get("ai_credits", billing_meta.get("smart_point_balance", 0.0)),
            "output_char_count": billing_meta.get("output_char_count", len(final_reply_text)),
            "billing_version": billing_meta.get("billing_version", BILLING_VERSION),
            "citations": citations or [],
            "block_timeline": _normalize_timeline_v3_blocks(timeline_blocks),
            "block_timeline_version": "v3" if timeline_blocks else "v2",
            "turn_phase": turn_phase,
            "answer_completed_at": _to_utc_iso(datetime.now(UTC).replace(tzinfo=None)),
        }
        if reconciled_context_usage:
            done_payload["context_usage"] = reconciled_context_usage
        if kind == "cancelled":
            done_payload["cancelled"] = True
            done_payload["reply_incomplete"] = True
        if error_kind:
            done_payload["error_kind"] = error_kind
        if extra_done_payload:
            done_payload.update(extra_done_payload)

        await _push_ws_event(
            frontend_ws,
            _build_message("agent_done", done_payload, request_id=request_id),
        )

        _log_turn_checkpoint(
            "turn_finalize_applied",
            request_id=request_id,
            project_id=project_session.project_id,
            project_session_id=int(project_session.id or 0),
            user_id=user_id,
            kind=kind,
            turn_phase=turn_phase,
            ws_alive=ws_alive,
            error_kind=error_kind or "",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            consumed_credits=billing_meta.get("consumed_credits", 0.0),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return FinalizeResult(applied=True, kind=kind)

# ORIGINAL L6792-L6845
def _sandbox_turn_resume_snapshot(
    *,
    user_id: int,
    project_id: str,
    request_id: str,
) -> dict[str, Any] | None:
    project_session = _get_project_session(
        user_id=user_id,
        project_id=project_id,
        include_inactive=False,
    )
    if project_session is None:
        return None
    with Session(engine) as db:
        turn = db.exec(
            select(ProjectConversationTurn).where(
                ProjectConversationTurn.project_session_id == int(project_session.id or 0),
                ProjectConversationTurn.user_id == int(user_id),
                ProjectConversationTurn.request_id == request_id,
            )
        ).first()
        if turn is None:
            return None
        trace = turn.thinking_trace_json if isinstance(turn.thinking_trace_json, dict) else {}
        meta = trace.get("meta") if isinstance(trace.get("meta"), dict) else {}
        turn_checkpoint = (
            dict(trace["turn_checkpoint"])
            if isinstance(trace.get("turn_checkpoint"), dict)
            else None
        )
        return {
            "project_session": project_session,
            "conversation_id": str(turn.conversation_id or "").strip(),
            "input_text": str(turn.input_text or ""),
            "status": str(turn.status or "").strip().lower(),
            "phase": str(meta.get("phase") or "").strip().lower(),
            "pending_sandbox_jobs": _normalize_pending_sandbox_jobs(
                meta.get("pending_sandbox_jobs")
            ),
            "timeline_blocks": _normalize_timeline_v3_blocks(
                meta.get("timeline_v3") if isinstance(meta.get("timeline_v3"), list) else []
            ),
            "usage_ledger": (
                dict(trace.get("usage_ledger"))
                if isinstance(trace.get("usage_ledger"), dict)
                else {}
            ),
            "context_usage": (
                dict(trace.get("context_usage"))
                if isinstance(trace.get("context_usage"), dict)
                else {}
            ),
            "turn_checkpoint": turn_checkpoint,
        }

# ORIGINAL L6848-L6894
async def _wait_for_sandbox_turn_pause(
    *,
    user_id: int,
    project_id: str,
    request_id: str,
    timeout_seconds: float = 30.0,
) -> dict[str, Any] | None:
    watchdog = ProgressLivenessWatchdog(max(0.01, float(timeout_seconds)))
    last_signature: tuple[str, str, str, tuple[tuple[str, str], ...]] | None = None
    while True:
        snapshot = _sandbox_turn_resume_snapshot(
            user_id=user_id,
            project_id=project_id,
            request_id=request_id,
        )
        if snapshot is None or snapshot.get("status") != "running":
            return None
        checkpoint = snapshot.get("turn_checkpoint")
        checkpoint_id = (
            str(checkpoint.get("checkpoint_id") or "")
            if isinstance(checkpoint, dict)
            else ""
        )
        job_signature = tuple(
            sorted(
                (
                    str(item.get("job_id") or "").strip(),
                    str(item.get("status") or "").strip().lower(),
                )
                for item in snapshot.get("pending_sandbox_jobs") or []
                if isinstance(item, dict)
            )
        )
        signature = (
            str(snapshot.get("status") or ""),
            str(snapshot.get("phase") or ""),
            checkpoint_id,
            job_signature,
        )
        if signature != last_signature:
            watchdog.touch("sandbox_turn_state_change")
            last_signature = signature
        if snapshot.get("phase") in {"waiting_sandbox", "resuming"}:
            return snapshot
        if watchdog.expired:
            return None
        await asyncio.sleep(min(0.25, max(0.01, watchdog.remaining_seconds())))

# ORIGINAL L6897-L7103
async def _continue_sandbox_turn(job: dict[str, Any]) -> None:
    job_id = str(job.get("job_id") or "").strip()
    request_id = str(job.get("request_id") or "").strip()
    project_id = str(job.get("project_id") or "").strip()
    conversation_id = str(job.get("session_id") or "").strip()
    status = str(job.get("status") or "").strip().lower()
    try:
        user_id = int(job.get("user_id") or 0)
    except (TypeError, ValueError):
        user_id = 0
    if not all((job_id, request_id, project_id, conversation_id, user_id)):
        return

    snapshot = await _wait_for_sandbox_turn_pause(
        user_id=user_id,
        project_id=project_id,
        request_id=request_id,
    )
    if snapshot is None:
        return
    project_session = snapshot["project_session"]
    if conversation_id != str(snapshot.get("conversation_id") or ""):
        return
    pending_jobs = _normalize_pending_sandbox_jobs(
        snapshot.get("pending_sandbox_jobs")
    )
    if pending_jobs and job_id not in {
        str(item.get("job_id") or "").strip() for item in pending_jobs
    }:
        _log_turn_checkpoint(
            "sandbox_turn_continuation_rejected_stale_job",
            request_id=request_id,
            project_id=project_id,
            conversation_id=conversation_id,
            project_session_id=int(project_session.id or 0),
            user_id=user_id,
            job_id=job_id,
            pending_job_ids=[
                str(item.get("job_id") or "").strip() for item in pending_jobs
            ],
        )
        return
    try:
        terminal_jobs = await asyncio.to_thread(
            _terminal_pending_sandbox_payloads,
            pending_jobs,
        )
        if not terminal_jobs:
            terminal_jobs = [job]
        transition = build_next_sandbox_continuation_transition(
            request_id=request_id,
            turn_checkpoint=(
                dict(snapshot.get("turn_checkpoint"))
                if isinstance(snapshot.get("turn_checkpoint"), dict)
                else None
            ),
            usage_ledger=(
                dict(snapshot.get("usage_ledger"))
                if isinstance(snapshot.get("usage_ledger"), dict)
                else None
            ),
            jobs=terminal_jobs,
        )
        resume_payload = dict(transition.sandbox_resume_payload)
    except SandboxContinuationContractError as exc:
        _log_turn_checkpoint(
            "sandbox_turn_continuation_contract_rejected",
            request_id=request_id,
            project_id=project_id,
            conversation_id=conversation_id,
            project_session_id=int(project_session.id or 0),
            user_id=user_id,
            job_id=job_id,
            error_kind=str(exc),
            level=logging.WARNING,
        )
        return

    # Legacy free-text summaries are no longer model context authority. They
    # remain in existing rows for compatibility/audit, but only the validated
    # Session Compact Checkpoint may cover old transcript turns.
    project_summary = ""
    session_compact_checkpoint = _load_latest_session_compact_checkpoint(
        int(project_session.id or 0),
        conversation_id,
        request_id,
    )
    effective_after_turn_id = _session_checkpoint_history_boundary(
        session_compact_checkpoint
    )
    history_payload = _build_agent_history(
        project_session,
        after_turn_id=effective_after_turn_id,
        conversation_id=conversation_id,
    )
    conversation_files = build_agent_conversation_file_context(
        user_id=user_id,
        project_id=project_id,
        conversation_id=conversation_id,
    )
    resume_text = str(snapshot.get("input_text") or "").strip()
    if not resume_text:
        resume_text = "继续完成上一条用户请求。"
    current_task = asyncio.current_task()
    runtime = _get_active_turn(request_id)
    if runtime is None:
        _register_active_turn(
            request_id,
            task=current_task,
            project_session_id=int(project_session.id or 0),
            cancel_event=asyncio.Event(),
            snapshot={
                "partial_text": _answer_text_from_timeline(
                    snapshot.get("timeline_blocks") or []
                ),
                "usage": snapshot.get("usage_ledger") or {},
                "context_usage": snapshot.get("context_usage") or {},
                "timeline_blocks": snapshot.get("timeline_blocks") or [],
                "pending_sandbox_jobs": pending_jobs,
                "turn_phase": "resuming",
                "ws_alive": False,
            },
            frontend_ws=None,
        )
        runtime = _get_active_turn(request_id)
    runtime = runtime or {}
    if runtime:
        runtime["task"] = current_task
        runtime["phase"] = "resuming"
    _persist_turn_progress(
        project_session,
        request_id,
        timeline_blocks=snapshot.get("timeline_blocks") or [],
        partial_reply=_answer_text_from_timeline(snapshot.get("timeline_blocks") or []),
        turn_phase="resuming",
        usage=snapshot.get("usage_ledger") or {},
        context_usage=snapshot.get("context_usage") or {},
    )
    frontend_ws = runtime.get("frontend_ws")
    ws_alive_flag = {"alive": frontend_ws is not None}
    outcome = await _stream_from_agent_http(
        frontend_ws,
        project_session,
        resume_text,
        [],
        conversation_files,
        request_id,
        history_payload,
        user_id=user_id,
        project_id=project_id,
        project_summary=project_summary,
        system_prompt=TASK_TREE_SYSTEM_PROMPT,
        cancel_event=runtime.get("cancel_event"),
        ws_alive_flag=ws_alive_flag,
        agent_payload={
            "system_prompt": TASK_TREE_SYSTEM_PROMPT,
            "text": resume_text,
            "attachments": [],
            "conversation_files": conversation_files,
            "request_id": request_id,
            "session_id": conversation_id,
            "project_id": project_id,
            "conversation_id": conversation_id,
            "project_summary": project_summary,
            "history": history_payload,
            **(
                {"session_compact_checkpoint": session_compact_checkpoint}
                if session_compact_checkpoint
                else {}
            ),
            "timeline_blocks": snapshot.get("timeline_blocks") or [],
            "usage_ledger": snapshot.get("usage_ledger") or {},
            "turn_checkpoint": snapshot.get("turn_checkpoint"),
            "sandbox_resume_payload": resume_payload,
        },
    )

    if outcome.waiting_sandbox:
        next_snapshot = {
            "partial_text": outcome.partial_text or "",
            "usage": outcome.usage or {},
            "context_usage": outcome.context_usage or {},
            "thinking_events": outcome.thinking_events or [],
            "timeline_blocks": outcome.timeline_blocks or [],
            "citations": outcome.citations or [],
            "pending_sandbox_jobs": outcome.pending_sandbox_jobs or [],
            "turn_phase": "waiting_sandbox",
        }
        if runtime:
            runtime["task"] = None
            runtime["phase"] = "waiting_sandbox"
            runtime["snapshot"] = next_snapshot
        await _recover_sandbox_turn_continuations(
            outcome.pending_sandbox_jobs
        )
        return

    await _finalize_agent_stream_outcome(
        frontend_ws=frontend_ws,
        project_session=project_session,
        user_id=user_id,
        request_id=request_id,
        outcome=outcome,
        cancel_requested=bool(
            runtime.get("cancel_event") and runtime["cancel_event"].is_set()
        ),
    )

# ORIGINAL L7106-L7127
async def _drain_sandbox_turn_continuations(request_id: str) -> None:
    while True:
        queued = _sandbox_continuation_queue.get(request_id)
        if not queued:
            _sandbox_continuation_queue.pop(request_id, None)
            return
        job_id = next(iter(queued))
        job = queued.pop(job_id)
        if not queued:
            _sandbox_continuation_queue.pop(request_id, None)
        inflight_key = f"{request_id}:{job_id}"
        _sandbox_continuation_inflight.add(inflight_key)
        try:
            await _continue_sandbox_turn(job)
        except Exception:  # noqa: BLE001
            logger.exception(
                "sandbox_turn_continuation_job_failed job_id=%s request_id=%s",
                job_id,
                request_id,
            )
        finally:
            _sandbox_continuation_inflight.discard(inflight_key)

# ORIGINAL L7130-L7165
def _schedule_sandbox_turn_continuation(job: dict[str, Any]) -> None:
    job_id = str(job.get("job_id") or "").strip()
    request_id = str(job.get("request_id") or "").strip()
    status = str(job.get("status") or "").strip().lower()
    if not job_id or not request_id or not is_sandbox_terminal_status(status):
        return
    inflight_key = f"{request_id}:{job_id}"
    queued = _sandbox_continuation_queue.setdefault(request_id, {})
    if inflight_key in _sandbox_continuation_inflight or job_id in queued:
        return
    queued[job_id] = dict(job)
    existing = _sandbox_continuation_tasks.get(request_id)
    if isinstance(existing, asyncio.Task) and not existing.done():
        return
    task = asyncio.create_task(
        _drain_sandbox_turn_continuations(request_id),
        name=f"sandbox_continuation_{request_id}",
    )
    _sandbox_continuation_tasks[request_id] = task

    def _done(done_task: asyncio.Task, *, rid: str = request_id) -> None:
        _sandbox_continuation_tasks.pop(rid, None)
        runtime = _active_turn_registry.get(rid)
        if runtime is not None and runtime.get("task") is done_task:
            _unregister_active_turn(rid, done_task)
        try:
            done_task.result()
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001
            logger.exception(
                "sandbox_turn_continuation_failed request_id=%s",
                rid,
            )

    task.add_done_callback(_done)

# ORIGINAL L7168-L7180
def _terminal_pending_sandbox_payloads(
    pending_jobs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    with Session(engine) as db:
        for item in _normalize_pending_sandbox_jobs(pending_jobs):
            payload = sandbox_service.build_job_terminal_callback_payload(
                db,
                job_id=item["job_id"],
            )
            if is_sandbox_terminal_status(payload.get("status")):
                payloads.append(payload)
    return payloads

# ORIGINAL L7209-L7220
async def _recover_sandbox_turn_continuations(
    pending_jobs: list[dict[str, Any]] | None = None,
) -> int:
    loader = (
        lambda: _terminal_pending_sandbox_payloads(pending_jobs or [])
        if pending_jobs is not None
        else _recoverable_sandbox_turn_payloads()
    )
    payloads = await asyncio.to_thread(loader)
    for payload in payloads:
        _schedule_sandbox_turn_continuation(payload)
    return len(payloads)
