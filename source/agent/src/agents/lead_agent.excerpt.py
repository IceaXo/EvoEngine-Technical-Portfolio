# Reading excerpt; functions/classes retain their original bodies.
# Origin: agent/src/agents/lead_agent.py @ 057cdcb9c61e59e8e63f73e844d5361e1ea67358
# Module setup, unselected helpers and service wiring are omitted.
# See source_manifest.json and docs/DEPENDENCIES.md. Do not execute this slice.

# ORIGINAL L1-L1
from __future__ import annotations

# ORIGINAL L3-L3
import asyncio

# ORIGINAL L4-L4
import hashlib

# ORIGINAL L5-L5
import httpx

# ORIGINAL L6-L6
import json

# ORIGINAL L7-L7
import logging

# ORIGINAL L8-L8
import os

# ORIGINAL L9-L9
import re

# ORIGINAL L10-L10
import time

# ORIGINAL L11-L11
import uuid

# ORIGINAL L12-L12
from typing import Any, AsyncIterator, Literal

# ORIGINAL L14-L14
from langchain.agents import create_agent

# ORIGINAL L15-L15
from langchain.messages import AIMessageChunk, HumanMessage, SystemMessage

# ORIGINAL L16-L16
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ORIGINAL L20-L20
from src.config.settings import Settings, get_settings, is_native_reasoning_model

# ORIGINAL L21-L21
from src.capabilities.sandbox_contracts import SandboxExposureContext

# ORIGINAL L22-L22
from src.schemas.hitl import HumanQuestionBundle, HumanQuestionItem, HumanQuestionOption

# ORIGINAL L23-L23
from src.services.orchestrator import compose_user_prompt, history_protocol_messages

# ORIGINAL L24-L24
from src.services.dual_channel_stream import BodyOnlySplitter, DualChannelSplitter

# ORIGINAL L25-L28
from src.services.context_ledger import (
    serialize_tools_for_ledger,
    split_system_prompt_parts,
)

# ORIGINAL L29-L29
from src.services.tool_schema_ledger import build_tool_schema_snapshot

# ORIGINAL L30-L30
from src.agents.citation_settings import citations_enabled

# ORIGINAL L31-L31
from src.agents.context_composer_middleware import ContextComposerMiddleware

# ORIGINAL L32-L32
from src.agents.task_tree_dependency_middleware import TaskTreeDependencyOrderMiddleware

# ORIGINAL L33-L33
from src.agents.invalid_tool_call_middleware import InvalidToolCallRepairMiddleware

# ORIGINAL L34-L34
from src.skills.registry import get_skill_registry

# ORIGINAL L35-L35
from src.support.chat_deepseek_safe import ChatDeepSeekThinkingSafe

# ORIGINAL L340-L366
class TaskCompleteArgs(BaseModel):
    summary: str = Field(
        ...,
        description=(
            "面向用户的最终完成答复；成功接受后将直接作为本轮最终回复。"
            "应包含关键结论、必要限制，以及任务明确要求时的主要交付文件。"
            "本字段会替换工具调用前流式输出的正文；已经使用来源形成结论时，"
            "把对应的 [[cite:KEY]] 一并写入本字段。"
        ),
    )
    deliverable_file_ids: list[int] = Field(
        default_factory=list,
        max_length=64,
        description=(
            "可选。填写成功 ToolResult 已经登记为 result_file 的 exact conversation_file_id。"
            "Server 只验收 ID、producer、SHA 和 lineage，不会在完成时替文件改变 drawer 状态；"
            "不要填写输入文件、temporary_output 或仅供后续步骤使用的中间件。"
        ),
    )
    terminal_status: Literal["completed", "blocked"] = Field(
        default="completed",
        description=(
            "completed 表示所有必要义务均已有正常或科学阴性/证据不足/不适用结果；"
            "blocked 仅用于 required 工具调用已有权威 failed/cancelled receipt、"
            "安全重试或 replacement 已耗尽，但仍需交付已有结果和阻塞说明。"
        ),
    )

# ORIGINAL L384-L386
def _task_complete_impl(summary: str) -> str:
    message = str(summary or "").strip()
    return message or "任务已完成"

# ORIGINAL L389-L749
def build_task_complete_tool(
    *,
    project_id: str | None = None,
    conversation_id: str | None = None,
    user_id: int | None = None,
    request_id: str | None = None,
    transport_request_id: str | None = None,
):
    from langchain_core.tools import StructuredTool

    def _citation_binding_error(summary: str) -> dict[str, Any] | None:
        """Validate explicit Citation markers without owning task completion."""

        active_request = str(transport_request_id or request_id or "").strip()
        if (
            not citations_enabled()
            or not active_request
            or "[[cite:" not in str(summary or "")
        ):
            return None

        from src.subagents.citation_policy import (
            available_citation_claim_bindings,
            citation_stable_key,
            count_verified_markdown_citation_markers,
            diagnose_verified_markdown_citation_markers,
            render_verified_markdown_footnotes,
            sanitize_citation_candidate,
        )
        from src.subagents.reference_store import load_reference_records
        from src.subagents.source_ledger import normalize_source_record

        candidates: list[dict[str, Any]] = []
        for index, record in enumerate(
            load_reference_records(
                request_id=active_request,
                project_id=str(project_id or "").strip() or None,
                conversation_id=str(conversation_id or "").strip() or None,
            ),
            1,
        ):
            normalized = normalize_source_record(record, fallback_index=index)
            if normalized is None:
                continue
            citation = sanitize_citation_candidate(normalized)
            if citation is not None:
                candidates.append(citation)
        marker_count, bound_marker_count = count_verified_markdown_citation_markers(
            summary,
            candidates,
            require_claim_id=False,
        )
        marker_diagnostics = diagnose_verified_markdown_citation_markers(
            summary,
            candidates,
            require_claim_id=False,
        )
        _rendered, bound = render_verified_markdown_footnotes(summary, candidates)
        invalid_marker_count = marker_count - bound_marker_count
        if marker_count > 0 and invalid_marker_count == 0:
            return None

        stable_keys: list[str] = []
        seen_keys: set[str] = set()
        for citation in candidates:
            key = citation_stable_key(citation)
            normalized_key = key.lower()
            if not key or normalized_key in seen_keys:
                continue
            seen_keys.add(normalized_key)
            stable_keys.append(key)

        if not candidates:
            error_code = "citation_reference_unavailable"
            error_prefix = (
                "最终摘要包含引用标记，但当前请求没有可用于发布的已核验来源；"
            )
        elif marker_count == 0:
            error_code = "citation_binding_invalid"
            error_prefix = "最终摘要包含无法解析的 Citation 标记；"
        else:
            error_code = "citation_binding_invalid"
            error_prefix = (
                "最终摘要中的一个或多个引用标记没有绑定到当前请求的已核验来源；"
                "请根据 marker_diagnostics 定位具体标记；"
            )
        return {
            "ok": False,
            "accepted": False,
            "error_code": error_code,
            "error": error_prefix
            + "请使用当前请求返回的 [[cite:引用键|claim=ClaimID]] 后重新调用 task_complete。",
            "next_tool": "task_complete",
            "next_action": "rewrite_summary_with_verified_citations",
            "completion_signal": {
                "schema_version": "tool_completion_signal_v1",
                "status": error_code,
                "should_stop_tool_loop": False,
                "next_action": "rewrite_summary_with_verified_citations_then_retry_task_complete",
                "reason": "最终摘要包含无法解析的 Citation 合同。",
                "acceptance": {
                    "accepted": False,
                    "source_tool": "task_complete",
                    "evidence_permits_external_claims": True,
                },
            },
            "citation_binding": {
                "schema_version": "evoengine.citation-binding/v1",
                "available_reference_count": len(candidates),
                "bound_reference_count": len(bound),
                "marker_count": marker_count,
                "bound_marker_count": bound_marker_count,
                "invalid_marker_count": invalid_marker_count,
                "marker_diagnostics": marker_diagnostics,
                "available_reference_keys": stable_keys,
                "available_claim_bindings": available_citation_claim_bindings(
                    candidates,
                ),
                "required_marker_format": "[[cite:KEY|claim=CLAIM_ID]]",
            },
        }

    def _validated_task_complete(
        summary: str,
        deliverable_file_ids: list[int] | None = None,
        terminal_status: Literal["completed", "blocked"] = "completed",
    ) -> dict[str, Any]:
        raw_message = str(summary or "")
        message = raw_message if raw_message.strip() else "任务已完成"
        requested_terminal = str(terminal_status or "completed").strip().lower()
        project = str(project_id or "").strip()
        conversation = str(conversation_id or "").strip()
        if not project or not conversation or user_id is None:
            return {"ok": True, "accepted": True, "summary": message}

        from src.services import task_tree_client

        snapshot = task_tree_client.get_latest_snapshot(
            project_id=project,
            user_id=int(user_id),
            session_id=conversation,
            request_id=str(request_id or "").strip() or None,
        )
        if task_tree_client.is_terminal_snapshot(snapshot):
            # Task-tree lookup is conversation-scoped.  Only an exact request
            # replay may consume a sealed CompletionReceipt; a later request
            # must not inherit the historical run's completion or files.
            snapshot_request_id = str((snapshot or {}).get("request_id") or "").strip()
            active_request_id = str(request_id or "").strip()
            same_request = bool(
                active_request_id and snapshot_request_id == active_request_id
            )
            existing_terminal_status = str(
                (snapshot or {}).get("mission_status")
                or (snapshot or {}).get("status")
                or ""
            ).strip().lower()
            if same_request and existing_terminal_status in {"completed", "blocked"}:
                root = (
                    (snapshot or {}).get("root")
                    if isinstance((snapshot or {}).get("root"), dict)
                    else {}
                )
                authoritative_summary = str(
                    (snapshot or {}).get("status_reason")
                    or root.get("result_summary")
                    or ""
                )
                if not authoritative_summary.strip():
                    return {
                        "ok": False,
                        "accepted": False,
                        "code": "completion_replay_summary_missing",
                        "error": (
                            "当前请求已完成，但 Server 未返回权威完成摘要。"
                        ),
                        "retryable": False,
                    }
                # A sealed task replay must never depend on the model
                # reproducing a potentially long summary byte-for-byte.  Use
                # only the Server-persisted terminal state and deliverables;
                # the Server revalidates the immutable receipt below.
                message = authoritative_summary
                requested_terminal = existing_terminal_status
                authoritative_receipt = (
                    (snapshot or {}).get("completion_receipt")
                    if isinstance((snapshot or {}).get("completion_receipt"), dict)
                    else {}
                )
                deliverable_file_ids = [
                    int(item.get("conversation_file_id"))
                    for item in authoritative_receipt.get("deliverables") or []
                    if isinstance(item, dict)
                    and item.get("conversation_file_id") is not None
                ]
            if same_request:
                if existing_terminal_status not in {"completed", "blocked"}:
                    return {
                        "ok": False,
                        "accepted": False,
                        "code": "task_tree_run_terminal",
                        "error": "当前请求的任务树已终止，不能重新标记为完成。",
                        "retryable": False,
                    }
            else:
                snapshot = None
        root = snapshot.get("root") if isinstance(snapshot, dict) and isinstance(snapshot.get("root"), dict) else None
        if not root:
            # TaskTree is a projection, not completion permission.  A normal
            # turn (including one that registered files directly from a
            # ToolResult) can finish without manufacturing a tree solely for
            # TaskComplete.  Citation publication remains independently
            # validated.
            if requested_terminal == "blocked":
                return {
                    "ok": False,
                    "accepted": False,
                    "code": "blocked_terminal_requires_task_authority",
                    "error": "阻塞交付必须有权威任务树和失败 ToolResult，不能仅由模型声明。",
                    "retryable": False,
                }
            citation_binding_error = _citation_binding_error(message)
            if citation_binding_error is not None:
                return citation_binding_error
            return {"ok": True, "accepted": True, "summary": message}

        citation_binding_error = _citation_binding_error(message)
        if citation_binding_error is not None:
            return citation_binding_error
        run_id = str(snapshot.get("run_id") or "").strip()
        root_node_id = str(root.get("node_id") or "").strip()
        if not run_id or not root_node_id:
            return {
                "ok": False,
                "accepted": False,
                "code": "task_tree_root_identity_missing",
                "error": "任务树缺少明确的 run_id 或 root node_id，不能正式结束任务。",
                "next_action": "refresh_task_tree_snapshot",
            }
        try:
            completion_result = task_tree_client.complete_task_root(
                run_id=run_id,
                root_node_id=root_node_id,
                result_summary=message,
                deliverable_file_ids=list(deliverable_file_ids or []),
                terminal_status=requested_terminal,
            )
        except task_tree_client.TaskTreeRequestError as exc:
            logger.info(
                "task_complete_contract_rejected run_id=%s root_node_id=%s code=%s",
                run_id,
                root_node_id,
                exc.code,
            )
            next_action = "refresh_task_tree_snapshot"
            if (
                exc.code == "required_capability_receipts_missing"
                and requested_terminal == "completed"
            ):
                next_action = (
                    "refresh_task_tree_then_recover_failed_actions_or_use_"
                    "terminal_status_blocked_only_for_nonretryable_failures"
                )
            return {
                "ok": False,
                "accepted": False,
                "code": exc.code,
                "error": exc.message,
                "details": exc.details,
                "retryable": exc.status_code >= 500,
                "next_action": next_action,
            }
        except Exception as exc:
            logger.warning(
                "task_complete_root_update_failed run_id=%s root_node_id=%s error_type=%s",
                run_id,
                root_node_id,
                exc.__class__.__name__,
            )
            return {
                "ok": False,
                "accepted": False,
                "code": "task_tree_root_completion_failed",
                "error": "任务树根目标完成状态写入失败，请稍后重试。",
                "retryable": True,
                "next_action": "retry_task_complete",
            }
        completed_snapshot = (
            completion_result.get("snapshot")
            if isinstance(completion_result, dict)
            and isinstance(completion_result.get("snapshot"), dict)
            else {}
        )
        completion_receipt = (
            completion_result.get("completion_receipt")
            if isinstance(completion_result, dict)
            and isinstance(completion_result.get("completion_receipt"), dict)
            else {}
        )
        completed_root = (
            completed_snapshot.get("root")
            if isinstance(completed_snapshot, dict)
            and isinstance(completed_snapshot.get("root"), dict)
            else {}
        )
        if (
            str(completed_root.get("node_id") or "").strip() != root_node_id
            or str(completed_root.get("status") or "").strip().lower()
            != ("completed" if requested_terminal == "completed" else "partial")
            or str(completed_snapshot.get("mission_status") or "").strip().lower()
            != requested_terminal
            or str(completion_receipt.get("schema_version") or "")
            != "evoengine.task-completion-receipt/v1"
            or str(completion_receipt.get("run_id") or "") != run_id
        ):
            return {
                "ok": False,
                "accepted": False,
                "code": "task_tree_root_completion_unconfirmed",
                "error": "任务树服务未确认根目标与任务 run 已完成，不能返回完成成功。",
                "retryable": True,
                "next_action": "refresh_task_tree_snapshot",
            }
        return {
            "ok": True,
            "accepted": True,
            "summary": message,
            "task_tree_run_id": run_id,
            "root_node_id": root_node_id,
            "completion_receipt": completion_receipt,
            "terminal_status": str(
                completion_receipt.get("terminal_status") or "completed"
            ),
            "goal_outcome": str(
                completion_receipt.get("goal_outcome") or "achieved"
            ),
            "result_file_count": len(completion_receipt.get("deliverables") or []),
            "deliverable_files": list(completion_receipt.get("deliverables") or []),
        }

    return StructuredTool.from_function(
        func=_validated_task_complete,
        name="task_complete",
        description=(
            "当全部必要义务都已有权威解决结果、最终交付已经生成，可以结束本次长链路时调用。"
            "解决结果不等于全为阳性：能力正常执行后得到阴性、证据不足或不适用结论，"
            "以及因此由 Server 机械跳过的下游动作，都可以构成完整交付；不得为凑成功伪造结果。"
            "Server 会在同一事务内权威校验 required obligations、CapabilityOutcome、ToolResult receipts 和 exact result_file IDs；"
            "若用户明确要求交付某些已由成功 ToolResult 登记的 result_file，必须把这些文件在"
            "task_tree_get_snapshot/capability receipt 中的 exact conversation_file_id 填入"
            "deliverable_file_ids；Server 只验收而不提升文件状态，不能只在 summary 中声称已交付；"
            "deliverable_file_ids 只能来自当前请求的权威成功结果；历史请求文件必须先在当前请求中"
            "经合法的字节不变复制/打包动作重新登记，不能直接沿用旧 ID；"
            "document 只是展示容器，不要求模型复制文件状态。"
            "summary 是最终交付正文而不是内部进度摘要，应完整保留结论、限制、文件链接和已使用的 [[cite:KEY]]；"
            "不要把本工具当作探测验收条件的尝试。调用此工具表示执行流程已完整收束；"
            "若 required 工具已有权威失败/取消结果且安全恢复已耗尽，可用 terminal_status=blocked 交付已有结果；"
            "blocked 不是成功，不能把 pending/running、缺文件或只是发生可重试错误伪装成阻塞。"
        ),
        args_schema=TaskCompleteArgs,
    )

# ORIGINAL L2103-L2103
from src.tools.research_toolkit import load_structured_research_tools

# ORIGINAL L2104-L2104
from src.tools.web_search import build_web_page_read_tool, build_web_search_tool

# ORIGINAL L2105-L2105
from src.tools.enterprise_knowledge import build_enterprise_tools

# ORIGINAL L2405-L2405
from src.capabilities.catalog import project_tool_catalog_entries, register_tool_catalog_entries

# ORIGINAL L2406-L2406
from src.capabilities.registry import CapabilityRegistry

# ORIGINAL L3293-L4119
def build_dynamic_agent(
    settings: Settings,
    llm: Any,
    project_id: str | None = None,
    conversation_id: str | None = None,
    user_id: int | None = None,
    request_id: str | None = None,
    *,
    task_authority_request_id: str | None = None,
    native_reasoning: bool = False,
    task_tree_summary: str = "",
    task_tree_fetch_status: str = "not_applicable",
    extra_system_prompt: str = "",
    task_phase: str = "intake",
    tool_selection_query: str = "",
    loaded_tool_ids: list[str] | None = None,
    current_turn_tool_names: list[str] | None = None,
    sandbox_exposure_context: SandboxExposureContext | None = None,
    available_inputs: list[str] | set[str] | tuple[str, ...] | None = None,
    activated_skill_ids: list[str] | set[str] | tuple[str, ...] | None = None,
    replayed_side_effects: list[dict[str, Any]] | None = None,
    resource_access_ledger: dict[str, Any] | None = None,
    reference_context: dict[str, Any] | None = None,
    session_checkpoint_state: dict[str, Any] | None = None,
    capability_discovery_requirements: list[dict[str, Any]] | None = None,
    protocol_ledger: Any | None = None,
):
    """【学术科研动态工具装配】：无路由直给版，统一 RAG 优先

    ``native_reasoning``: when True (deepseek-v4-flash / r1 etc.), the system
    prompt drops the EVO_BODY monologue contract — the text channel becomes
    pure Markdown body, and thinking is captured automatically via the native
    reasoning_content channel.
    """
    authority_request_id = (
        str(task_authority_request_id or "").strip()
        or str(request_id or "").strip()
        or None
    )
    tools: list[Any] = []
    resolved_available_inputs = tuple(
        sorted(
            {
                str(item or "").strip()
                for item in (available_inputs or ())
                if str(item or "").strip()
            }
        )
    )
    sandbox_capability_runtime = None
    runtime_task_phase = str(task_phase or "intake").strip().lower() or "intake"
    # A confirmed missing tree is the only tree lookup result that disables
    # binding.  An unavailable lookup must fail closed for business actions;
    # otherwise a transient Server error silently turns the same turn into an
    # unbound execution path.
    task_tree_binding_required = _task_tree_binding_required(
        summary=task_tree_summary,
        fetch_status=task_tree_fetch_status,
    )
    from src.capabilities.runtime import enabled_executor_provider_types

    enabled_capability_provider_types = enabled_executor_provider_types()
    plan_revision = ""
    if enabled_capability_provider_types and project_id and conversation_id and user_id is not None:
        try:
            from src.services.conversation_file_registry import get_plan_revision_fingerprint

            plan_revision = get_plan_revision_fingerprint(
                user_id=user_id,
                project_id=project_id,
                conversation_id=conversation_id,
            )
        except Exception:
            plan_revision = ""
    current_turn_tools = set(_normalize_tool_ids(list(current_turn_tool_names or [])))
    if sandbox_exposure_context is None:
        from src.tools.sandbox_tools import fetch_sandbox_exposure_context

        sandbox_exposure_context = fetch_sandbox_exposure_context()
    from src.tools.sandbox_tools import compile_sandbox_typed_schema_build

    sandbox_typed_schema_build = compile_sandbox_typed_schema_build(
        sandbox_exposure_context
    )
    for failed_tool_id, failure_reason in sandbox_typed_schema_build.failures:
        logger.warning(
            "sandbox_typed_schema_build_failed request_id=%s tool_id=%s error=%s",
            request_id,
            failed_tool_id,
            failure_reason,
        )
    from src.capabilities.skill_contracts import compile_skill_typed_schema_build

    # Freeze model-visible Skill contracts only after applying the current
    # admin enablement state.  Internal/disabled contracts must never enter
    # discovery, load_tools availability, loader, or executor schemas.
    skill_registry = get_skill_registry("skills")
    skill_registry.reset_load_events()
    skill_registry._apply_enabled_config()
    public_enabled_skill_ids = _public_enabled_skill_ids(skill_registry)

    skill_typed_schema_build = compile_skill_typed_schema_build(
        "skills",
        reserved_tool_ids={
            *TOOL_CATALOG_BY_ID,
            *sandbox_exposure_context.snapshot.typed_tool_ids,
        },
        allowed_skill_ids=public_enabled_skill_ids,
    )
    for failed_tool_id, failure_reason in skill_typed_schema_build.failures:
        logger.warning(
            "skill_typed_schema_build_failed request_id=%s tool_id=%s error=%s",
            request_id,
            failed_tool_id,
            failure_reason,
        )
    from src.capabilities.skill_runtime_availability import (
        build_skill_runtime_availability_snapshot,
        get_skill_runtime_availability_snapshot,
    )

    normalized_request_id = str(request_id or "").strip()
    skill_runtime_availability = (
        get_skill_runtime_availability_snapshot(
            skill_typed_schema_build,
            request_id=normalized_request_id,
        )
        if normalized_request_id
        else build_skill_runtime_availability_snapshot(
            skill_typed_schema_build,
        )
    )
    if skill_runtime_availability.unavailable_tool_ids:
        logger.info(
            "skill_runtime_dependencies_unavailable request_id=%s tool_count=%d tool_ids=%s",
            normalized_request_id,
            len(skill_runtime_availability.unavailable_tool_ids),
            ",".join(sorted(skill_runtime_availability.unavailable_tool_ids)),
        )
    constructible_tool_ids = _available_tool_ids_for_build(
        settings,
        project_id=project_id,
        conversation_id=conversation_id,
        user_id=user_id,
        sandbox_exposure_context=sandbox_exposure_context,
        sandbox_typed_schema_build=sandbox_typed_schema_build,
        skill_typed_schema_build=skill_typed_schema_build,
        unavailable_skill_tool_ids=(
            skill_runtime_availability.unavailable_tool_ids
        ),
    )
    resident_tool_ids = (
        set(_expand_loaded_tool_ids(get_runtime_resident_tool_ids()))
        & set(constructible_tool_ids)
    )
    # A parameter/configuration file is itself an instruction-bearing input.
    # Make its bounded reader available in this turn instead of relying on the
    # model to guess a tool ID before it can map file values into a typed Skill
    # or sandbox contract. Ordinary data files remain lazy-loaded.
    if (
        {"parameter_file", "configuration_file"}
        & {str(item or "").strip().lower() for item in resolved_available_inputs}
        and "conversation_file_read" in constructible_tool_ids
    ):
        resident_tool_ids.add("conversation_file_read")
    expanded_requested_tool_ids = _expand_loaded_tool_ids(
        _normalize_tool_ids(loaded_tool_ids)
    )
    accepted_typed_tool_ids = set(
        sandbox_exposure_context.snapshot.typed_tool_ids
        if sandbox_exposure_context is not None
        else ()
    )
    rejected_sandbox_tool_ids = {
        tool_id
        for tool_id in expanded_requested_tool_ids
        if tool_id.startswith("sandbox_submit_")
        and tool_id not in accepted_typed_tool_ids
    }
    requested_loaded_tool_ids = {
        tool_id
        for tool_id in expanded_requested_tool_ids
        if tool_id in constructible_tool_ids
    }
    explicit_loaded_tool_ids = set([*resident_tool_ids, *requested_loaded_tool_ids])
    allow_incremental_load_tools = _bool_env_local("EVO_ALLOW_INCREMENTAL_LOAD_TOOLS", True)
    expose_load_tools = (not explicit_loaded_tool_ids) or allow_incremental_load_tools
    if not expose_load_tools:
        # With incremental loading disabled, discovery may only advertise
        # schemas mounted in this exact build.  Keeping load_tools in the
        # resident manifest must not make it appear mounted after the factory
        # intentionally omitted it.
        explicit_loaded_tool_ids.discard("load_tools")
        available_tool_ids = frozenset(explicit_loaded_tool_ids)
    else:
        available_tool_ids = constructible_tool_ids
    selected_pack_set = _pack_keys_for_tool_ids(explicit_loaded_tool_ids) if explicit_loaded_tool_ids else {"core"}
    if explicit_loaded_tool_ids.intersection(accepted_typed_tool_ids):
        selected_pack_set.add("sandbox_submit")
    selected_pack_keys = [key for key in TOOL_PACK_CATALOG if key in selected_pack_set]
    selection_reasons = {
        pack_key: ("resident_or_explicit_load_tools" if explicit_loaded_tool_ids else "core_only_initial_schema")
        for pack_key in selected_pack_keys
    }
    tool_pack_selection = {
        "selected": selected_pack_keys,
        "reasons": selection_reasons,
        "query_preview": str(tool_selection_query or "")[:200],
        "strategy": "core_only_plus_explicit_load_tools",
    }

    if (
        sandbox_exposure_context is None
        and any(pack.startswith("sandbox_") for pack in selected_pack_set)
    ):
        from src.tools.sandbox_tools import fetch_sandbox_exposure_context

        sandbox_exposure_context = fetch_sandbox_exposure_context()

    if expose_load_tools:
        tools.append(build_load_tools_tool(
            already_loaded_tool_ids=explicit_loaded_tool_ids,
            current_turn_tool_names=current_turn_tools,
            sandbox_exposure_context=sandbox_exposure_context,
            available_tool_ids=available_tool_ids,
            sandbox_typed_schema_build=sandbox_typed_schema_build,
            skill_typed_schema_build=skill_typed_schema_build,
        ))

    # One resident cross-provider discovery entry replaces catalog guessing.
    # It is bound to the exact same immutable sandbox exposure context used by
    # load_tools, schema construction and executor adaptation in this build.
    match_capability_tool = build_match_capability_tool(
        current_user_message=tool_selection_query,
        already_loaded_tool_ids=explicit_loaded_tool_ids,
        activated_skill_ids=activated_skill_ids,
        sandbox_exposure_context=sandbox_exposure_context,
        available_inputs=resolved_available_inputs,
        available_tool_ids=available_tool_ids,
        sandbox_typed_schema_build=sandbox_typed_schema_build,
        skill_typed_schema_build=skill_typed_schema_build,
    )
    tools.append(match_capability_tool)
    from src.tools.resource_tools import build_resource_tools

    tools.extend(
        build_resource_tools(
            project_id=project_id,
            conversation_id=conversation_id,
            user_id=user_id,
            resource_access_ledger=resource_access_ledger,
            reference_context=reference_context,
        )
    )
    # Capability awareness must survive transport recovery and HITL resume.
    # ``current_turn_tools`` may contain only control-plane calls from an
    # earlier attempt (for example request_human_input); using it as a gate
    # made the read-only candidate set disappear before execution resumed.
    # Once an executable typed capability has actually been loaded, the
    # ordinary continuation transcript and its exact schema are authoritative
    # and prefetch is no longer needed.
    executable_typed_tool_ids = {
        *accepted_typed_tool_ids,
        *set(getattr(skill_typed_schema_build, "tool_ids", ()) or ()),
    }
    has_loaded_typed_capability = bool(
        explicit_loaded_tool_ids.intersection(executable_typed_tool_ids)
    )
    capability_prefetch_prompt_block = (
        render_capability_prefetch_prompt(
            match_capability_tool,
            current_user_message=tool_selection_query,
            max_results=3,
            discovery_requirements=capability_discovery_requirements,
        )
        if capability_discovery_requirements is not None
        else (
            ""
            if has_loaded_typed_capability
            else render_capability_prefetch_prompt(
                match_capability_tool,
                current_user_message=tool_selection_query,
                max_results=3,
            )
        )
    )

    if "knowledge_assets" in selected_pack_set:
        enterprise_tools = build_enterprise_tools(
            project_id=project_id,
            user_id=user_id,
            conversation_id=conversation_id,
            request_id=authority_request_id,
            usage_caller="lead",
        )
        tools.extend(enterprise_tools)

    if "web_search" in selected_pack_set:
        tools.extend(
            [
                build_web_search_tool(settings),
                build_web_page_read_tool(),
            ]
        )

    if "retrieval_subagent" in selected_pack_set:
        from src.subagents.retrieval_subagent import build_retrieval_subagent_tool
        tools.append(
            build_retrieval_subagent_tool(
                settings=settings,
                skill_typed_schema_build=skill_typed_schema_build,
                available_tool_ids=available_tool_ids,
                project_id=project_id,
                conversation_id=conversation_id,
                user_id=user_id,
                request_id=authority_request_id,
            )
        )

    if settings.enable_research_tools and "bio_database" in selected_pack_set:
        research_tools, _ = load_structured_research_tools(
            project_id=project_id,
            conversation_id=conversation_id,
            user_id=user_id,
            request_id=authority_request_id,
            allowlist=None,
        )
        tools.extend(_filter_research_tools_for_loaded_ids(research_tools, explicit_loaded_tool_ids))

    # ── 沙盒计算工具：通用生产入口 + 逐应用 typed 迁移候选 ──
    if any(pack.startswith("sandbox_") for pack in selected_pack_set):
        from src.tools.sandbox_tools import build_sandbox_tools
        from src.capabilities.runtime import adapt_sandbox_tools_with_executor

        sandbox_tools = build_sandbox_tools(
            project_id=project_id,
            conversation_id=conversation_id,
            user_id=user_id,
            request_id=request_id,
            exposure_context=sandbox_exposure_context,
            loaded_tool_ids=explicit_loaded_tool_ids,
            typed_schema_build=sandbox_typed_schema_build,
        )
        sandbox_capability_runtime = adapt_sandbox_tools_with_executor(
            sandbox_tools,
            sandbox_exposure_context,
            task_phase="execute",
            request_id=str(authority_request_id or "request_unknown"),
            project_id=project_id,
            conversation_id=conversation_id,
            user_id=user_id,
            plan_revision=plan_revision,
            loaded_tool_ids=explicit_loaded_tool_ids,
            task_tree_binding_required=task_tree_binding_required,
        )
        sandbox_tools = sandbox_capability_runtime.tools
        tools.extend(
            _filter_sandbox_tools_for_packs(
                sandbox_tools,
                selected_pack_set,
                loaded_tool_ids=explicit_loaded_tool_ids,
                sandbox_exposure_context=sandbox_exposure_context,
            )
        )

    if "conversation_files" in selected_pack_set:
        from src.tools.conversation_file_tools import build_conversation_file_tools
        conversation_file_tools = build_conversation_file_tools(
            project_id=project_id,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        tools.extend(conversation_file_tools)

    if "deliverable" in selected_pack_set:
        from src.capabilities.providers.deliverable import build_deliverable_subagent_tool
        from src.tools.generate_doc_tools import build_generate_doc_tools
        generate_doc_tools = build_generate_doc_tools(
            project_id=project_id,
            conversation_id=conversation_id,
            user_id=user_id,
            request_id=authority_request_id,
        )
        tools.extend(generate_doc_tools)
        tools.append(build_deliverable_subagent_tool())

    if "artifact_execution" in selected_pack_set and _artifact_tools_enabled():
        from src.tools.artifact_tools import build_artifact_tools
        from src.tools.report_generation_tools import build_report_generation_tools
        from src.services.tool_result_envelope import wrap_structured_tools

        artifact_tools = build_artifact_tools(
            project_id=project_id,
            conversation_id=conversation_id,
            user_id=user_id,
            request_id=authority_request_id,
        )
        artifact_finalize = next(
            (tool for tool in artifact_tools if str(getattr(tool, "name", "")) == "artifact_finalize"),
            None,
        )
        artifact_tools = _filter_tools_by_loaded_ids(
            artifact_tools,
            explicit_loaded_tool_ids,
        )
        tools.extend(wrap_structured_tools(
            artifact_tools,
            category="artifact_execution",
        ))
        report_tools = _filter_tools_by_loaded_ids(
            build_report_generation_tools(
                project_id=project_id,
                conversation_id=conversation_id,
                user_id=user_id,
                request_id=authority_request_id,
                finalize_delivery=getattr(artifact_finalize, "func", None),
            ),
            explicit_loaded_tool_ids,
        )
        tools.extend(wrap_structured_tools(
            report_tools,
            category="deliverable",
        ))

    tools.append(build_request_human_input_tool())

    # ── Skill 系统工具（从 skills/ 目录动态装配）──
    # Executable Skills have one model-visible route:
    # match_capability -> load_tools(exact typed tool) -> typed tool call.
    # activate_skill remains only for guide-only Skills.  The generic loader
    # and executor stay constructible for internal compatibility tests but are
    # hidden from production model builds.
    logger.info(
        "skill_capability_dispatch request_id=%s total_skills=%d contracted_capabilities=%d",
        request_id,
        len(skill_registry.list_skills()),
        len(skill_typed_schema_build.items),
    )

    if "progress" in selected_pack_set:
        tools.append(build_report_progress_tool())
        tools.append(
            build_task_complete_tool(
                project_id=project_id,
                conversation_id=conversation_id,
                user_id=user_id,
                request_id=authority_request_id,
                transport_request_id=request_id,
            )
        )

    # ── Skill activation tool (permanent builtin) ──
    from src.skills.session import _activate_skill_impl
    from src.services.tool_result_envelope import wrap_structured_tool
    from langchain_core.tools import StructuredTool as _StructuredTool
    from pydantic import BaseModel as _BaseModel, Field as _Field

    class _ActivateSkillInput(_BaseModel):
        skill_id: str = _Field(
            description=(
                "仅填写 match_capability 返回 kind=guide 的 guide-only Skill ID；"
                "可执行/typed 候选必须按 load_tool_ids 装载精确工具，不能在此激活。"
            )
        )
        available_inputs: list[str] = _Field(
            default_factory=list,
            description="当前已识别的输入类型，例如 count_matrix、sample_metadata、fasta；没有则留空。",
        )

    def _activate_skill_for_current_mode(
        skill_id: str,
        available_inputs: list[str] | None = None,
    ) -> str:
        effective_available_inputs = sorted(
            {
                *resolved_available_inputs,
                *(
                    str(item or "").strip()
                    for item in (available_inputs or [])
                    if str(item or "").strip()
                ),
            }
        )
        return _activate_skill_impl(
            skill_id=skill_id,
            available_inputs=effective_available_inputs,
            detail_level="full",
            allow_non_public=False,
            request_id=authority_request_id,
            project_id=project_id,
            conversation_id=conversation_id,
            user_id=user_id,
            current_user_message=tool_selection_query,
            skill_typed_schema_build=skill_typed_schema_build,
            available_tool_ids=available_tool_ids,
        )

    if "skill_discovery" in selected_pack_set:
        tools.append(wrap_structured_tool(_StructuredTool.from_function(
            func=_activate_skill_for_current_mode,
            name="activate_skill",
            description=(
                "只激活 match_capability 明确返回 kind=guide 的 guide-only Skill，"
                "加载方法指南但不执行脚本。可执行/typed 候选不得调用本工具；"
                "必须直接按候选 load_tool_ids 装载精确执行工具。"
                "对 executable 候选调用本工具会被合同拒绝。"
            ),
            args_schema=_ActivateSkillInput,
        ), category="skill_discovery"))

    if (
        "skill_discovery" in selected_pack_set
        and "load_skill_capability" in explicit_loaded_tool_ids
        and "load_skill_capability" in available_tool_ids
    ):
        tools.append(
            wrap_structured_tool(
                build_load_skill_capability_tool(
                    skill_typed_schema_build,
                    skill_registry=skill_registry,
                ),
                category="skill_discovery",
            )
        )

    if (
        "skill_execution" in selected_pack_set
        and "execute_skill_capability" in explicit_loaded_tool_ids
        and "execute_skill_capability" in available_tool_ids
    ):
        skill_capability_tool = build_execute_skill_capability_tool(
            skill_typed_schema_build,
            skill_registry=skill_registry,
            request_id=authority_request_id,
            project_id=project_id,
            conversation_id=conversation_id,
            user_id=user_id,
            current_user_message=tool_selection_query,
        )
        # This tool already returns the canonical model/runtime projection
        # pair. Wrapping it as a legacy native tool would collapse the pair
        # into one envelope and expose the runtime artifact to the model.
        setattr(skill_capability_tool, "_evo_tool_category", "skill_execution")
        setattr(skill_capability_tool, "_evo_capability_dispatch", "skill_contract_generic")
        tools.append(skill_capability_tool)

    # Typed Skill capabilities are model-visible only after match_capability
    # returns their exact tool ID and load_tools mounts that ID.  Their schema
    # and implementation both come from the same frozen Skill build used by
    # discovery, so there is no second generic execution route.
    from src.capabilities.skill_runtime import build_skill_typed_tools

    tools.extend(
        build_skill_typed_tools(
            skill_typed_schema_build,
            skills_root="skills",
            available_tool_ids=available_tool_ids,
            loaded_tool_ids=explicit_loaded_tool_ids,
            request_id=request_id,
            project_id=project_id,
            conversation_id=conversation_id,
            user_id=user_id,
            current_user_message=tool_selection_query,
            task_tree_binding_required=task_tree_binding_required,
        )
    )

    tools = _filter_tools_by_loaded_ids(tools, explicit_loaded_tool_ids)

    from src.capabilities.runtime import (
        adapt_tools_with_executor,
        guard_checkpoint_replayed_side_effects,
    )

    capability_runtime = adapt_tools_with_executor(
        tools,
        task_phase="execute",
        request_id=str(authority_request_id or "request_unknown"),
        project_id=project_id,
        conversation_id=conversation_id,
        user_id=user_id,
        catalog_by_id=TOOL_CATALOG_BY_ID,
        plan_revision=plan_revision,
        task_tree_binding_required=task_tree_binding_required,
    )
    # Capability tools keep their exact frozen identities for Runtime action
    # projection. Model-facing TaskTree tools only create display containers
    # and never receive or author these identities.
    all_capability_tools = list(capability_runtime.tools)
    execution_capabilities = _execution_capability_identities(
        all_capability_tools
    )
    model_capability_tools = _constrain_task_bound_capability_tools(
        all_capability_tools,
        task_tree_binding_required=task_tree_binding_required,
        project_id=project_id,
        conversation_id=conversation_id,
        user_id=user_id,
    )

    assembled_tools = list(model_capability_tools)
    if "task_tree" in selected_pack_set:
        from src.tools.task_tree_tools import build_task_tree_tools

        if project_id and conversation_id and user_id is not None:
            task_tree_tools = build_task_tree_tools(
                project_id=project_id,
                conversation_id=conversation_id,
                user_id=user_id,
                request_id=authority_request_id,
                execution_capabilities=execution_capabilities,
            )
            assembled_tools.extend(
                tool
                for tool in task_tree_tools
                if str(getattr(tool, "name", "")) in explicit_loaded_tool_ids
            )
    tools = guard_checkpoint_replayed_side_effects(
        assembled_tools,
        replayed_side_effects,
    )

    logger.info(
        "dynamic_agent_tools_built request_id=%s task_phase=%s tool_count=%d tool_names=%s",
        request_id,
        runtime_task_phase,
        len(tools),
        ",".join(str(getattr(tool, "name", "")) for tool in tools[:20]),
    )

    # ── System Prompt ──────────────────────────────────────────────────
    catalog_reachability = _catalog_cutover_reachability(
        available_tool_ids,
        sandbox_exposure_context=sandbox_exposure_context,
        sandbox_typed_schema_build=sandbox_typed_schema_build,
        skill_typed_schema_build=skill_typed_schema_build,
        mounted_tool_ids=explicit_loaded_tool_ids,
    )
    catalog_cutover_blockers = list(catalog_reachability["unreachable"])
    # The prompt list is now only a fail-safe for a future unclassified tool.
    # The normal path has one discovery entry: match_capability.
    catalog_discovery_active = bool(
        expose_load_tools
        and "load_tools" not in current_turn_tools
        and catalog_cutover_blockers
    )
    tool_catalog_prompt_block = (
        render_tool_catalog_prompt(
            include_all=True,
            available_tool_ids=available_tool_ids,
        )
        if catalog_discovery_active
        else ""
    )
    loaded_tools_prompt_block = render_loaded_tools_prompt(explicit_loaded_tool_ids)
    # Compose exactly one execution protocol plus current state envelopes.
    system_prompt_text = (
        CORE_AGENT_CONTRACT_PROMPT_BLOCK
        + tool_catalog_prompt_block
        + loaded_tools_prompt_block
        + (f"[TASK_TREE_SNAPSHOT]\n{task_tree_summary}\n[/TASK_TREE_SNAPSHOT]\n" if str(task_tree_summary or "").strip() else "")
        + (f"[RUNTIME_ENVELOPE]\n{str(extra_system_prompt).strip()}\n[/RUNTIME_ENVELOPE]\n" if str(extra_system_prompt or "").strip() else "")
    )
    tool_schema_snapshot = build_tool_schema_snapshot(tools)
    # The composer is the only model-history budget authority.  Keeping it in
    # middleware makes the same policy apply to ordinary ReAct rounds, HITL
    # continuation, and transport recovery without mutating durable state.
    agent_middleware = []
    if protocol_ledger is not None:
        # This is the outermost tool wrapper. It commits the final result of
        # the complete middleware/handler chain, including a downstream
        # short-circuit, before callback events are projected for the UI.
        from src.agents.canonical_protocol_ledger import CanonicalProtocolMiddleware

        agent_middleware.append(CanonicalProtocolMiddleware(protocol_ledger))
    agent_middleware.append(
        ContextComposerMiddleware(
            request_id=str(request_id or ""),
            resource_access_ledger=resource_access_ledger,
            summary_model=llm,
            session_checkpoint_state=session_checkpoint_state,
        )
    )
    if protocol_ledger is None:
        agent_middleware.append(InvalidToolCallRepairMiddleware())
    if (
        project_id
        and conversation_id
        and user_id is not None
        and any(str(getattr(tool, "name", "") or "") == "task_tree_update_node" for tool in tools)
    ):
        agent_middleware.append(
            TaskTreeDependencyOrderMiddleware(
                project_id=project_id,
                conversation_id=conversation_id,
                user_id=int(user_id),
            )
        )
    agent = create_agent(
        model=llm,
        system_prompt=SystemMessage(system_prompt_text),
        tools=tools,
        middleware=agent_middleware,
    )
    # This attribute is part of the turn runtime contract, not optional ledger
    # metadata.  Attach it before best-effort observability work so recursive
    # rebuilds cannot accidentally fetch a different catalog when ledger
    # serialization fails.
    setattr(
        agent,
        "_evo_sandbox_exposure_context",
        sandbox_exposure_context,
    )
    setattr(
        agent,
        "_evo_rejected_sandbox_tool_ids",
        sorted(rejected_sandbox_tool_ids),
    )
    setattr(agent, "_evo_available_tool_ids", sorted(available_tool_ids))
    setattr(agent, "_evo_skill_typed_schema_build", skill_typed_schema_build)
    setattr(
        agent,
        "_evo_skill_runtime_availability",
        skill_runtime_availability,
    )
    setattr(
        agent,
        "_evo_skill_typed_schema_failures",
        dict(skill_typed_schema_build.failures),
    )
    setattr(
        agent,
        "_evo_sandbox_typed_schema_failures",
        dict(sandbox_typed_schema_build.failures),
    )
    try:
        tool_names = [str(getattr(tool, "name", "") or tool.__class__.__name__) for tool in tools]
        context_parts = split_system_prompt_parts(
            system_prompt_text=system_prompt_text,
            task_tree_summary=task_tree_summary,
            extra_system_prompt=extra_system_prompt,
            split_fragments=[
                capability_prefetch_prompt_block,
                tool_catalog_prompt_block,
                loaded_tools_prompt_block,
            ],
        )
        if tool_catalog_prompt_block or loaded_tools_prompt_block:
            context_parts.append({
                "category": "tool_catalog",
                "label": f"Tool catalog ({tool_catalog_prompt_block.count(chr(10) + '- `')} entries)",
                "content": tool_catalog_prompt_block + loaded_tools_prompt_block,
                "source_type": "tool_catalog",
                "meta": {
                    "catalog_entries": tool_catalog_prompt_block.count(chr(10) + "- `"),
                    "loaded_tool_ids": sorted(explicit_loaded_tool_ids),
                    "catalog_strategy": (
                        "unreachable_tool_fallback"
                        if tool_catalog_prompt_block
                        else "match_capability_only"
                    ),
                    "catalog_cutover_blockers": catalog_cutover_blockers,
                },
                "priority": 82,
            })
        context_parts.append(
            {
                "category": "tool_definitions",
                "label": f"Tool definitions ({len(tools)} tools)",
                "content": serialize_tools_for_ledger(tools),
                "source_type": "langchain_tools",
                "meta": {
                    "tool_count": len(tools),
                    "tool_names": tool_names,
                    "tool_names_truncated": False,
                },
                "priority": 90,
            }
        )
        setattr(agent, "_evo_context_ledger_parts", context_parts)
        setattr(agent, "_evo_tool_schema_snapshot", tool_schema_snapshot)
        setattr(agent, "_evo_selected_tool_packs", selected_pack_keys)
        setattr(agent, "_evo_tool_pack_selection", tool_pack_selection)
        setattr(agent, "_evo_loaded_tool_ids", sorted(explicit_loaded_tool_ids))
        setattr(agent, "_evo_tool_catalog_entries", TOOL_CATALOG_ENTRIES)
        setattr(
            agent,
            "_evo_capability_prefetch",
            capability_prefetch_prompt_block,
        )
        setattr(
            agent,
            "_evo_capability_discovery_requirements",
            [
                dict(item)
                for item in (capability_discovery_requirements or [])
                if isinstance(item, dict)
            ],
        )
        setattr(agent, "_evo_capability_registry", capability_runtime.registry)
        setattr(agent, "_evo_capability_executor", capability_runtime.executor)
        setattr(agent, "_evo_task_authority_request_id", authority_request_id)
        setattr(
            agent,
            "_evo_capability_adapted_tool_ids",
            list(capability_runtime.adapted_tool_ids),
        )
        setattr(
            agent,
            "_evo_sandbox_capability_registry",
            sandbox_capability_runtime.registry if sandbox_capability_runtime else None,
        )
        setattr(
            agent,
            "_evo_sandbox_capability_executor",
            sandbox_capability_runtime.executor if sandbox_capability_runtime else None,
        )
        setattr(
            agent,
            "_evo_sandbox_capability_adapted_tool_ids",
            list(sandbox_capability_runtime.adapted_tool_ids)
            if sandbox_capability_runtime
            else [],
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("context_ledger_agent_metadata_failed request_id=%s err=%s", request_id, exc)
    return agent

# ORIGINAL L4708-L5571
async def stream_agent_events(
    llm_base,
    user_text: str,
    history: list[dict] | None = None,
    project_summary: str = "",
    request_id: str | None = None,
    knowledge: dict | None = None,
    database: dict | None = None,
    attachments: list[dict[str, Any]] | None = None,
    project_id: str | None = None,
    conversation_id: str | None = None,
    user_id: int | None = None,
    conversation_files: list[dict[str, Any]] | None = None,
    reference_context: dict[str, Any] | None = None,
    system_prompt: str = "",
) -> AsyncIterator[dict]:
    settings = get_settings()
    native_reasoning = is_native_reasoning_model(settings.llm_model)
    from src.skills.input_context import infer_available_inputs

    dynamic_agent = build_dynamic_agent(
        settings,
        llm_base,
        project_id,
        conversation_id,
        user_id,
        request_id,
        native_reasoning=native_reasoning,
        extra_system_prompt=system_prompt,
        reference_context=reference_context,
        tool_selection_query=user_text,
        available_inputs=infer_available_inputs(
            attachments=attachments,
            conversation_files=conversation_files,
        ),
    )

    # ── Skill summary thinking event (user-visible timeline) ──────
    skill_reg = get_skill_registry("skills")
    load_events = skill_reg.flush_load_events()
    if load_events:
        # Collect per-skill data: any skill that registered tools is shown.
        # Skill = high-level capability (e.g. "蛋白质结构分析")
        # Tool  = concrete function     (e.g. "alphafold_query")
        skill_info: dict[str, dict[str, Any]] = {}  # skill_id → {name, tool_count}

        for e in load_events:
            if e.phase == "tools_registered" and e.skill_id:
                sid = e.skill_id
                if sid not in skill_info:
                    skill_def = skill_reg.get_skill(sid)
                    skill_info[sid] = {
                        "name": skill_def.name if skill_def else sid,
                        "tool_count": 0,
                    }
                skill_info[sid]["tool_count"] += int(e.detail.get("tool_count", 0))

        # Also note MCP connections
        mcp_servers: list[str] = []
        for e in load_events:
            if e.phase == "mcp_connected" and e.detail.get("status") == "connected":
                mcp_servers.append(str(e.detail.get("server", "")))

        # Build summary: "加载 1 个技能: Echo Tool（1 工具）"
        skill_parts: list[str] = []
        items: list[dict[str, Any]] = []
        for sid, info in skill_info.items():
            tc = info["tool_count"]
            skill_parts.append(
                f"{info['name']}（{tc} 工具）" if tc else info["name"]
            )
            items.append({
                "label": f"技能: {info['name']}",
                "value": f"{tc} 个工具",
            })

        skill_count = len(skill_info)
        skill_summary_text = ""
        if skill_count:
            base = "加载" if not any(e.phase == "l1_matched" and e.detail.get("loaded") for e in load_events) else "匹配"
            skill_summary_text = f"{base} {skill_count} 个技能: " + "、".join(skill_parts)

        if skill_summary_text:
            if mcp_servers:
                items.append({"label": "外部数据源", "value": ", ".join(mcp_servers)})

            yield {
                "type": "agent_thinking",
                "phase": "skill_summary",
                "tool_call_id": None,
                "tool_name": None,
                "summary": skill_summary_text,
                "icon": "package",
                "items": items,
            }

    messages = [
        *history_protocol_messages(history),
        HumanMessage(
            compose_user_prompt(
                user_text,
                history=None,
                project_summary=project_summary,
                attachments=attachments,
                conversation_files=conversation_files,
                reference_context=reference_context,
                native_reasoning=native_reasoning,
            )
        ),
    ]

    def _make_splitter() -> BodyOnlySplitter:
        # Both native and non-native models now treat the text channel as
        # answer-only. Native reasoning still arrives via reasoning_content /
        # reasoning blocks; non-native models default to having no thinking row.
        return BodyOnlySplitter()

    # Cross-chunk EVO marker stripper — guarantees no partial `<<<EVO_*>>>`
    # bytes ever reach the frontend, even when the marker is split across
    # streaming chunks.
    marker_stripper = _StreamMarkerStripper()

    def _safe_emit_token(text: str) -> str:
        # Apply cross-chunk stripping first, then a final per-chunk pass.
        cleaned = marker_stripper.feed(text)
        return _EVO_MARKER_RE.sub("", cleaned)

    tools_active = 0
    _event_count: int = 0       # Total astream_events events seen (P1 infinite-loop guard)
    _tool_round: int = 0        # Tool call rounds completed (P1 infinite-loop guard)
    _empty_streak: int = 0      # Consecutive non-productive events (P1 infinite-loop guard)
    splitter: "DualChannelSplitter | BodyOnlySplitter | None" = _make_splitter()
    _total_input_tokens: int = 0
    _total_output_tokens: int = 0
    _last_model_name: str = ""

    _raw_model_text: str = ""
    _answer_emitted: bool = False
    logger.debug(f"[PHASE] agent_built request_id={request_id} model={settings.llm_model}")
    _seen_tool_in_turn: bool = False
    _final_response_phase: bool = False
    _pending_tool_inputs: dict[str, Any] = {}
    _turn_tool_events: list[dict[str, Any]] = []
    _bridge_emitted: bool = False
    _model_reasoning_emitted: bool = False
    _answer_stream_started: bool = False
    _reasoning_done_emitted: bool = False
    _generating_emitted: bool = False
    # Tracks which content_blocks field the model last produced. DeepSeek V4
    # streams reasoning blocks first, then switches to text blocks; that field
    # transition is the precise moment to lock the reasoning timeline row.
    _prev_block_field: str = ""
    _human_input_requested: bool = False
    _hard_timeout_occurred: bool = False
    _deepseek_chunk_seq: int = 0
    _call_answer_text: list[str] = []   # buffered answer text per LLM call; flushed at model end
    _deepseek_trace_started_at = time.time()
    _deepseek_trace_last_at = _deepseek_trace_started_at
    _run_deadline = time.monotonic() + float(EVO_AGENT_HARD_TIMEOUT)
    _tool_started_in_current_call: bool = False

    def _remaining_run_seconds() -> float:
        return max(0.0, _run_deadline - time.monotonic())

    def _begin_answer_stream() -> list[dict[str, Any]]:
        nonlocal _answer_stream_started, _generating_emitted
        if _answer_stream_started:
            return []
        events: list[dict[str, Any]] = []
        if not _generating_emitted:
            _generating_emitted = True
            events.extend(_emit_generating_lifecycle())
        _answer_stream_started = True
        events.append(_emit_answer_stream_start())
        return events

    # Pick up cancel flag created by agent_server (or create if missing)
    _cancel_flag: asyncio.Event | None = None
    if request_id:
        from src.runtime.cancel_registry import cancel_flags
        _cancel_flag = cancel_flags.get(request_id)
        if _cancel_flag is None:
            _cancel_flag = asyncio.Event()
            cancel_flags[request_id] = _cancel_flag

    # Background task: if cancel flag is set, cancel the PARENT task
    _parent_task = asyncio.current_task()
    async def _cancel_watcher() -> None:
        if _cancel_flag is not None:
            await _cancel_flag.wait()
            if _parent_task is not None:
                _parent_task.cancel()
    if _cancel_flag is not None:
        _watcher_task = asyncio.create_task(_cancel_watcher())

    try:
        async with asyncio.timeout(EVO_AGENT_HARD_TIMEOUT):
            logger.debug(f"[PHASE] entering astream_events request_id={request_id}")
            _write_deepseek_chunk_trace(
                request_id,
                {
                    "event": "astream_start",
                    "model": settings.llm_model,
                    "native_reasoning": native_reasoning,
                    "attachment_count": len(attachments or []),
                },
            )
            stream_config = {"recursion_limit": EVO_AGENT_RECURSION_LIMIT}
            try:
                event_stream = dynamic_agent.astream_events(
                    {"messages": messages},
                    version="v2",
                    config=stream_config,
                )
            except TypeError:
                event_stream = dynamic_agent.astream_events({"messages": messages}, version="v2")
            async for ev in event_stream:
                _event_count += 1
                name = str(ev.get("event") or "")
                if _event_count > EVO_MAX_STREAM_EVENTS:
                    logger.warning(
                        "agent_event_overflow request_id=%s events=%d limit=%d",
                        request_id, _event_count, EVO_MAX_STREAM_EVENTS,
                    )
                    break
                # P1: consecutive no-progress events (empty stream chunks, internal noise)
                if name in ("on_tool_start", "on_tool_end", "on_chat_model_end"):
                    _empty_streak = 0
                elif name == "on_chat_model_stream":
                    chunk_probe = ev.get("data", {}).get("chunk")
                    productive = False
                    if isinstance(chunk_probe, AIMessageChunk):
                        probe_blocks = getattr(chunk_probe, "content_blocks", None) or []
                        probe_extra = getattr(chunk_probe, "additional_kwargs", None) or {}
                        if str(probe_extra.get("reasoning_content") or "").strip():
                            productive = True
                        if getattr(chunk_probe, "tool_call_chunks", None):
                            productive = True
                        for block in probe_blocks:
                            if isinstance(block, dict) and str(
                                block.get("text") or block.get("reasoning") or ""
                            ).strip():
                                productive = True
                                break
                    _empty_streak = 0 if productive else _empty_streak + 1
                else:
                    _empty_streak += 1
                if _empty_streak > EVO_MAX_EMPTY_STREAK:
                    logger.warning(
                        "agent_empty_streak_overflow request_id=%s streak=%d limit=%d last_event=%s",
                        request_id, _empty_streak, EVO_MAX_EMPTY_STREAK, name,
                    )
                    break
                if name == "on_chat_model_end":
                    # Accumulate real token counts from every LLM call in this turn.
                    output_obj = ev.get("data", {}).get("output")
                    _write_deepseek_chunk_trace(
                        request_id,
                        {
                            "event": "chat_model_end",
                            "event_count": _event_count,
                            "output": _json_safe(output_obj),
                        },
                    )
                    usage_meta = getattr(output_obj, "usage_metadata", None)
                    if usage_meta is None and hasattr(output_obj, "response_metadata"):
                        # Fallback: some providers put token info in response_metadata
                        rm = output_obj.response_metadata or {}
                        usage_meta = rm.get("token_usage") or rm.get("usage")
                    if isinstance(usage_meta, dict):
                        _total_input_tokens += int(usage_meta.get("input_tokens") or usage_meta.get("prompt_tokens") or 0)
                        _total_output_tokens += int(usage_meta.get("output_tokens") or usage_meta.get("completion_tokens") or 0)
                    elif usage_meta is not None:
                        # object-style (e.g. LangChain UsageMetadata)
                        _total_input_tokens += int(getattr(usage_meta, "input_tokens", 0))
                        _total_output_tokens += int(getattr(usage_meta, "output_tokens", 0))
                    # Capture model name from response metadata
                    if not _last_model_name and hasattr(output_obj, "response_metadata"):
                        rm = (output_obj.response_metadata or {}) if isinstance(output_obj.response_metadata, dict) else {}
                        _last_model_name = str(rm.get("model_name") or rm.get("model") or "")

                    # on_chat_model_end fallback: if the model's full content is available and
                    # no answer has been emitted yet, try one more recovery pass.
                    if not _answer_emitted and output_obj is not None:
                        full_content = ""
                        if hasattr(output_obj, "content"):
                            raw_c = output_obj.content
                            if isinstance(raw_c, str):
                                full_content = raw_c
                            elif isinstance(raw_c, list):
                                parts = [str(b.get("text", "")) for b in raw_c if isinstance(b, dict) and b.get("type") == "text"]
                                full_content = "".join(parts)
                        if full_content.strip() and full_content.strip() != _raw_model_text.strip():
                            _raw_model_text = _raw_model_text + full_content if not _raw_model_text else _raw_model_text

                    # Flush buffered answer text for this call. Final user-visible text
                    # now flows directly from the model stream instead of a terminal tool.
                    if (
                        _call_answer_text
                        and not _answer_emitted
                        and not _tool_started_in_current_call
                    ):
                        full = "".join(_call_answer_text)
                        _call_answer_text.clear()
                        if full.strip():
                            for start_ev in _begin_answer_stream():
                                yield start_ev
                            _answer_emitted = True
                            _raw_model_text += full
                            yield {"type": "agent_token", "text": full}
                    else:
                        _call_answer_text.clear()
                    continue

                if name == "on_chat_model_start":
                    _call_answer_text.clear()
                    _tool_started_in_current_call = False
                    _write_deepseek_chunk_trace(
                        request_id,
                        {
                            "event": "chat_model_start",
                            "event_count": _event_count,
                            "data": _json_safe(ev.get("data") or {}),
                        },
                    )
                    continue

                if name == "on_chat_model_stream":
                    chunk = ev.get("data", {}).get("chunk")
                    if not isinstance(chunk, AIMessageChunk):
                        continue
                    now_ts = time.time()
                    _deepseek_chunk_seq += 1
                    _write_deepseek_chunk_trace(
                        request_id,
                        {
                            "event": "chat_model_stream",
                            "event_count": _event_count,
                            "chunk_seq": _deepseek_chunk_seq,
                            "elapsed_ms": int((now_ts - _deepseek_trace_started_at) * 1000),
                            "since_prev_chunk_ms": int((now_ts - _deepseek_trace_last_at) * 1000),
                            **_summarize_ai_chunk(chunk),
                        },
                    )
                    _deepseek_trace_last_at = now_ts
                    blocks = getattr(chunk, "content_blocks", None) or []
                    extra = getattr(chunk, "additional_kwargs", None) or {}
                    reasoning_from_kwargs = str(extra.get("reasoning_content") or "")
                    tool_chunks = getattr(chunk, "tool_call_chunks", None) or []
                    # Prefer API-native reasoning_content; fall back to content_blocks.
                    if reasoning_from_kwargs:
                        reasoning_ev = _reasoning_thinking_event(
                            reasoning_from_kwargs, append=True
                        )
                        if reasoning_ev:
                            _model_reasoning_emitted = True
                            _prev_block_field = "reasoning"
                            yield reasoning_ev

                    if not blocks and not reasoning_from_kwargs:
                        continue

                    # Phase 1.2: process each block in arrival order so the precise
                    # reasoning_content → content field switch can be detected, even
                    # within a single chunk that interleaves the two.
                    for block in blocks:
                        block_type = str(block.get("type") or "")

                        # Native reasoning channel (DeepSeek V4 reasoning_content blocks).
                        # Streamed directly, bypassing body text parsing — no marker
                        # buffering, no preamble validation. Allowed even while tools
                        # are active so the model's between-tool planning is visible.
                        if block_type == "reasoning":
                            if reasoning_from_kwargs:
                                continue
                            raw = block.get("reasoning")
                            if raw is None:
                                raw = block.get("text", "")
                            delta = str(raw or "")
                            if not delta:
                                continue
                            reasoning_ev = _reasoning_thinking_event(delta, append=True)
                            if reasoning_ev:
                                _model_reasoning_emitted = True
                                _prev_block_field = "reasoning"
                                yield reasoning_ev
                            continue

                        if block_type != "text":
                            continue

                        # Text blocks: gated by tool activity + final-response phase.
                        if tools_active > 0:
                            continue
                        if not _final_response_phase:
                            # No tools were called this turn — first text block opens the gate.
                            _final_response_phase = True

                        text = str(block.get("text", ""))
                        if not text:
                            continue

                        # Bridge precise timing — fire at the moment we are about to
                        # consume the first text block. Two cases produce an event:
                        #   1. Tools ran this turn → synthesize a bridge summary.
                        #   2. Native reasoning streamed just before this text →
                        #      anchor row marks the field transition.
                        if not _bridge_emitted:
                            bridge = _emit_reasoning_bridge(_turn_tool_events)
                            if bridge is None and _prev_block_field == "reasoning":
                                bridge = _empty_reasoning_anchor()
                            if bridge:
                                _bridge_emitted = True
                                yield bridge

                        # Once we switch from reasoning to text, lock the reasoning row.
                        if (
                            _prev_block_field == "reasoning"
                            and not _reasoning_done_emitted
                        ):
                            _reasoning_done_emitted = True
                            yield {
                                "type": "agent_thinking",
                                "phase": "reasoning_done",
                                "tool_call_id": REASONING_TOOL_CALL_ID,
                                "tool_name": "",
                                "summary": "",
                                "icon": "bolt",
                                "items": [],
                            }

                        if splitter is None:
                            splitter = _make_splitter()

                        reasoning_delta, answer_delta, _reasoning_done = splitter.feed(text)
                        if answer_delta:
                            _raw_model_text += answer_delta
                            answer_delta = _safe_emit_token(
                                _dedupe_body_against_monologue(answer_delta, splitter)
                            )
                            if answer_delta:
                                _call_answer_text.append(answer_delta)
                                # Before the first real tool round completes, treat
                                # streamed text as provisional: some reasoning-native
                                # models emit a "draft tool query / JSON草稿" text
                                # block just before they actually call a tool. If we
                                # streamed it immediately, that draft would leak into
                                # the user-visible正文. We therefore buffer pre-tool
                                # text and only stream live after at least one tool
                                # round has actually completed, or flush at
                                # on_chat_model_end when this call truly used no
                                # tools.
                                if _seen_tool_in_turn and not _tool_started_in_current_call:
                                    for start_ev in _begin_answer_stream():
                                        yield start_ev
                                    _answer_emitted = True
                                    yield {"type": "agent_token", "text": answer_delta}
                        _prev_block_field = "text"
                    continue

                if name == "on_tool_start":
                    tool_name = str(ev.get("name") or "")
                    run_id = str(ev.get("run_id") or "")
                    tool_input_for_log = ev.get("data", {}).get("input")
                    logger.info(
                        "agent_tool_start request_id=%s run_id=%s tool=%s event_count=%d remaining_s=%.1f input_type=%s input_chars=%d",
                        request_id,
                        run_id,
                        tool_name,
                        _event_count,
                        _remaining_run_seconds(),
                        type(tool_input_for_log).__name__,
                        len(str(tool_input_for_log or "")),
                    )
                    if tool_name == "request_human_input":
                        tool_input = ev.get("data", {}).get("input")
                        if isinstance(tool_input, dict) and isinstance(tool_input.get("input"), (dict, str)):
                            tool_input = tool_input["input"]
                        _write_deepseek_chunk_trace(
                            request_id,
                            {
                                "event": f"{tool_name}_start",
                                "event_count": _event_count,
                                "run_id": run_id,
                                "input": _json_safe(tool_input),
                            },
                        )
                        if run_id:
                            _pending_tool_inputs[run_id] = tool_input
                        continue
                    tools_active += 1
                    _seen_tool_in_turn = True
                    _tool_started_in_current_call = True
                    _final_response_phase = False
                    _bridge_emitted = False
                    _model_reasoning_emitted = False
                    _reasoning_done_emitted = False
                    _generating_emitted = False
                    _prev_block_field = ""
                    splitter = None
                    _call_answer_text.clear()
                    if not _answer_emitted:
                        _raw_model_text = ""
                    if _model_reasoning_emitted and not _reasoning_done_emitted:
                        _reasoning_done_emitted = True
                        yield {
                            "type": "agent_thinking",
                            "phase": "reasoning_done",
                            "tool_call_id": REASONING_TOOL_CALL_ID,
                            "tool_name": "",
                            "summary": "",
                            "icon": "bolt",
                            "items": [],
                        }
                    yield {
                        "type": "agent_thinking",
                        "phase": "reasoning_reset",
                        "tool_call_id": REASONING_TOOL_CALL_ID,
                        "tool_name": "",
                        "summary": "",
                        "icon": "bolt",
                        "items": [],
                    }
                    run_id = str(ev.get("run_id") or "")
                    tool_input = ev.get("data", {}).get("input")
                    if isinstance(tool_input, dict) and isinstance(tool_input.get("input"), (dict, str)):
                        tool_input = tool_input["input"]
                    if run_id:
                        _pending_tool_inputs[run_id] = tool_input
                    labels = _get_tool_labels(tool_name)
                    input_preview = build_tool_input_preview(tool_input)
                    yield {
                        "type": "agent_thinking",
                        "phase": "tool_start",
                        "tool_name": tool_name,
                        "tool_call_id": run_id,
                        "summary": summarize_tool_start(tool_name, tool_input),
                        "icon": labels.get("icon") or "bolt",
                        "items": [],
                        "input_preview": input_preview,
                        "output_preview": None,
                    }
                    continue

                if name == "on_tool_end":
                    tool_name = str(ev.get("name") or "")
                    run_id = str(ev.get("run_id") or "")
                    output_for_log = ev.get("data", {}).get("output")
                    logger.info(
                        "agent_tool_end request_id=%s run_id=%s tool=%s event_count=%d remaining_s=%.1f output_type=%s output_chars=%d",
                        request_id,
                        run_id,
                        tool_name,
                        _event_count,
                        _remaining_run_seconds(),
                        type(output_for_log).__name__,
                        len(str(output_for_log or "")),
                    )
                    if tool_name == "request_human_input":
                        interrupt_input = _pending_tool_inputs.pop(run_id, None) if run_id else None
                        if not isinstance(interrupt_input, dict):
                            interrupt_input = {}
                        _write_deepseek_chunk_trace(
                            request_id,
                            {
                                "event": "request_human_input_end",
                                "event_count": _event_count,
                                "run_id": run_id,
                                "input": _json_safe(interrupt_input),
                            },
                        )
                        if _model_reasoning_emitted and not _reasoning_done_emitted:
                            _reasoning_done_emitted = True
                            yield {
                                "type": "agent_thinking",
                                "phase": "reasoning_done",
                                "tool_call_id": REASONING_TOOL_CALL_ID,
                                "tool_name": "",
                                "summary": "",
                                "icon": "bolt",
                                "items": [],
                            }
                        bundle = HumanQuestionBundle(
                            bundle_id=f"{request_id or 'req'}_bundle_{uuid.uuid4().hex[:8]}",
                            bundle_title=str(interrupt_input.get("bundle_title") or "").strip() or "需要你补充信息",
                            bundle_summary=str(interrupt_input.get("bundle_summary") or ""),
                            questions=interrupt_input.get("questions") or [],
                        )
                        _human_input_requested = True
                        yield {
                            "type": "human_input_required",
                            "request_id": request_id,
                            "thread_id": request_id,
                            "resume_token": request_id,
                            "bundle": bundle.model_dump(mode="json"),
                        }
                        break
                    tools_active = max(0, tools_active - 1)
                    output = ev.get("data", {}).get("output")
                    saved_input = _pending_tool_inputs.pop(run_id, None) if run_id else None
                    items = extract_tool_items(tool_name, output, tool_input=saved_input)
                    citations = build_citations_from_tool_output(tool_name, output)
                    verified_sources = verified_sources_from_tool_output(output)
                    if verified_sources:
                        try:
                            from src.subagents.reference_store import append_verified_reference_candidates

                            append_verified_reference_candidates(
                                verified_sources,
                                request_id=request_id,
                                project_id=project_id,
                                conversation_id=conversation_id,
                                source_tool=tool_name,
                            )
                        except Exception:
                            pass
                    labels = _get_tool_labels(tool_name)
                    summary = summarize_tool_done(tool_name, output, items)
                    input_preview = build_tool_input_preview(saved_input)
                    output_preview = build_tool_output_preview(output)
                    done_ev = {
                        "phase": "tool_done",
                        "tool_name": tool_name,
                        "tool_call_id": run_id,
                        "summary": summary,
                        "icon": labels.get("icon") or "bolt",
                        "items": items,
                        "input_preview": input_preview,
                        "output_preview": output_preview,
                    }
                    _turn_tool_events.append(done_ev)
                    yield {
                        "type": "agent_thinking",
                        **done_ev,
                        "citations": citations,
                    }
                    if tools_active == 0:
                        _tool_round += 1
                        _final_response_phase = True
                        splitter = _make_splitter()
                        if not _bridge_emitted:
                            bridge = _emit_reasoning_bridge(_turn_tool_events)
                            if bridge:
                                _bridge_emitted = True
                                yield bridge
                    continue

    except asyncio.CancelledError:
        _write_deepseek_chunk_trace(
            request_id,
            {
                "event": "astream_cancelled",
                "event_count": _event_count,
                "chunk_count": _deepseek_chunk_seq,
                "elapsed_ms": int((time.time() - _deepseek_trace_started_at) * 1000),
                "since_last_chunk_ms": int((time.time() - _deepseek_trace_last_at) * 1000),
            },
        )
        logger.warning(
            "stream_agent_cancelled request_id=%s events=%d tool_rounds=%d",
            request_id, _event_count, _tool_round,
        )
        yield {
            "type": "agent_usage",
            "input_tokens": _total_input_tokens,
            "output_tokens": _total_output_tokens,
            "total_tokens": _total_input_tokens + _total_output_tokens,
            "model": _last_model_name,
        }
        raise
    except asyncio.TimeoutError:
        _hard_timeout_occurred = True
        _write_deepseek_chunk_trace(
            request_id,
            {
                "event": "astream_hard_timeout",
                "event_count": _event_count,
                "chunk_count": _deepseek_chunk_seq,
                "elapsed_ms": int((time.time() - _deepseek_trace_started_at) * 1000),
                "since_last_chunk_ms": int((time.time() - _deepseek_trace_last_at) * 1000),
                "timeout_seconds": EVO_AGENT_HARD_TIMEOUT,
            },
        )
        logger.warning(
            "agent_hard_timeout request_id=%s timeout=%ds events=%d tool_rounds=%d",
            request_id, EVO_AGENT_HARD_TIMEOUT, _event_count, _tool_round,
        )
        # Fall through to post-stream + Phase 2 processing on timeout.
        # Don't raise — let the caller see whatever partial output we have.
        # The _hard_timeout_occurred flag will be picked up in the final
        # agent_usage event so the upstream knows this turn is degraded.
    except Exception as exc:
        _write_deepseek_chunk_trace(
            request_id,
            {
                "event": "astream_exception",
                "event_count": _event_count,
                "chunk_count": _deepseek_chunk_seq,
                "elapsed_ms": int((time.time() - _deepseek_trace_started_at) * 1000),
                "since_last_chunk_ms": int((time.time() - _deepseek_trace_last_at) * 1000),
                "exception_type": exc.__class__.__name__,
                "exception": str(exc),
            },
        )
        raise

    finally:
        # P1 safety net: if a tool start/end pair is broken by any abnormal
        # exit (CancelledError / TimeoutError / Exception), reset the counter
        # so post-stream text processing is never permanently skipped.
        tools_active = 0

    # --- Post-stream processing ---
    _post_stream_skipped = _human_input_requested or _remaining_run_seconds() <= 0
    if _post_stream_skipped:
        logger.warning(
            "agent_post_stream_skipped request_id=%s remaining=%.1fs timeout=%ds",
            request_id, _remaining_run_seconds(), EVO_AGENT_HARD_TIMEOUT,
        )

    _write_deepseek_chunk_trace(
        request_id,
        {
            "event": "astream_postprocess",
            "event_count": _event_count,
            "chunk_count": _deepseek_chunk_seq,
            "elapsed_ms": int((time.time() - _deepseek_trace_started_at) * 1000),
            "since_last_chunk_ms": int((time.time() - _deepseek_trace_last_at) * 1000),
            "answer_emitted": _answer_emitted,
            "raw_model_text_len": len(_raw_model_text),
        },
    )

    if not _post_stream_skipped and splitter is not None:
        r_tail, a_tail, flush_ops = splitter.flush()
        if r_tail:
            _model_reasoning_emitted = True
            reasoning_ev = _reasoning_thinking_event(r_tail, append=splitter.last_reasoning_append)
            if reasoning_ev:
                yield reasoning_ev
        # Emit reasoning_done exactly once if any reasoning was ever streamed and we
        # haven't already sent it via _transition_at_body_marker during the stream loop.
        if not _reasoning_done_emitted and (_model_reasoning_emitted or splitter.reasoning_streamed):
            _reasoning_done_emitted = True
            yield {
                "type": "agent_thinking",
                "phase": "reasoning_done",
                "tool_call_id": REASONING_TOOL_CALL_ID,
                "tool_name": "",
                "summary": "",
                "icon": "bolt",
                "items": [],
            }
        if a_tail:
            a_tail = _safe_emit_token(_dedupe_body_against_monologue(a_tail, splitter))
        # Flush any remaining hold the cross-chunk stripper may still be sitting on.
        residual = marker_stripper.flush()
        if residual:
            a_tail = (a_tail or "") + residual
        if a_tail:
            for start_ev in _begin_answer_stream():
                yield start_ev
            _answer_emitted = True
            yield {"type": "agent_token", "text": a_tail}
        # splitter ops suppressed — Phase 2 handles all operations
    elif not _post_stream_skipped and not _answer_emitted and _raw_model_text.strip():
        # Layer-2 fallback: splitter was cleared (e.g. tool-only edge) — single buffer.
        splitter_fb = BodyOnlySplitter() if not native_reasoning else DualChannelSplitter()
        splitter_fb.prepend_buffer(_raw_model_text)
        r_tail, a_tail, flush_ops = splitter_fb.flush()
        if native_reasoning and r_tail:
            _model_reasoning_emitted = True
            reasoning_ev = _reasoning_thinking_event(r_tail, append=splitter_fb.last_reasoning_append)
            if reasoning_ev:
                yield reasoning_ev
            yield {
                "type": "agent_thinking",
                "phase": "reasoning_done",
                "tool_call_id": REASONING_TOOL_CALL_ID,
                "tool_name": "",
                "summary": "",
                "icon": "bolt",
                "items": [],
            }
        elif native_reasoning and splitter_fb.reasoning_streamed:
            yield {
                "type": "agent_thinking",
                "phase": "reasoning_done",
                "tool_call_id": REASONING_TOOL_CALL_ID,
                "tool_name": "",
                "summary": "",
                "icon": "bolt",
                "items": [],
            }
        if a_tail:
            a_tail = _safe_emit_token(_dedupe_body_against_monologue(a_tail, splitter_fb))
        residual = marker_stripper.flush()
        if residual:
            a_tail = (a_tail or "") + residual
        if a_tail:
            for start_ev in _begin_answer_stream():
                yield start_ev
            _answer_emitted = True
            yield {"type": "agent_token", "text": a_tail}
        # splitter ops suppressed

    # Layer 3 fallback: raw text recovery.
    # Triggered only when no agent_token was emitted through normal or flush paths.
    if not _answer_emitted and _raw_model_text.strip():
        from src.services.dual_channel_stream import strip_dual_channel_envelope
        from src.services.response_parser import extract_agent_result

        mono = ""
        body = _raw_model_text
        recovered_ops: list[dict[str, Any]] = []
        if native_reasoning:
            mono, body, recovered_ops = strip_dual_channel_envelope(_raw_model_text)
            if not body.strip():
                body, recovered_ops = extract_agent_result(_raw_model_text)
        if native_reasoning and mono.strip():
            _model_reasoning_emitted = True
            reasoning_ev = _reasoning_thinking_event(mono, append=False)
            if reasoning_ev:
                yield reasoning_ev
            yield {
                "type": "agent_thinking",
                "phase": "reasoning_done",
                "tool_call_id": REASONING_TOOL_CALL_ID,
                "tool_name": "",
                "summary": "",
                "icon": "bolt",
                "items": [],
            }
        if body.strip():
            body = _safe_emit_token(body)
            residual = marker_stripper.flush()
            if residual:
                body = body + residual
            _answer_emitted = True
            for start_ev in _begin_answer_stream():
                yield start_ev
            yield {"type": "agent_token", "text": body}
            yield {"type": "agent_answer_recovered", "source": "raw_text"}
        # recovered_ops suppressed — Phase 2 handles all operations

    logger.debug(f"[PHASE] astream_events_done request_id={request_id} body_len={len(_raw_model_text)}")

    # Always clean up cancel watcher regardless of post-stream skip
    if _cancel_flag is not None:
        _watcher_task.cancel()
        try:
            await _watcher_task
        except asyncio.CancelledError:
            pass

    # Emit real token usage so agent_server can include it in agent_done
    yield {
        "type": "agent_usage",
        "input_tokens": _total_input_tokens,
        "output_tokens": _total_output_tokens,
        "total_tokens": _total_input_tokens + _total_output_tokens,
        "model": _last_model_name,
        "timeout": _hard_timeout_occurred,
    }
