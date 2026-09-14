from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Callable

from langchain_core.tools import StructuredTool

from src.capabilities.models import (
    CapabilityApplicability,
    CapabilityCoverageState,
    CapabilityEvidenceState,
    CapabilityOutcome,
    CapabilityRecoveryAction,
    ResourceRef,
    SourceOutcome,
    SourceProcessingStatus,
)
from src.services.resource_store import (
    ResourceScope,
    persist_source_sidecar_resource,
    persist_tool_result_resource,
)
from src.services.tool_model_projection import (
    model_result_exceeds_inline_limit,
    project_tool_result_for_model,
    redact_model_secrets,
)
from src.services.tool_ui_projection import build_tool_ui_projection


logger = logging.getLogger("evoengine-agent")

READY_TO_ANSWER_STATUSES: frozenset[str] = frozenset(
    {
        "ready_to_answer",
        "answer_ready",
        "completed",
        "complete",
        "task_complete",
    }
)
READY_TO_ANSWER_NEXT_ACTIONS: frozenset[str] = frozenset(
    {
        "final_answer",
        "answer_user",
        "answer_from_evidence_pack_without_more_retrieval",
        "stop",
    }
)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _apply_capability_result_contract(
    *,
    tool_name: str,
    category: str,
    raw_result: Any,
    envelope: dict[str, Any],
    elapsed_ms: int,
) -> None:
    mode = os.getenv("EVO_CAPABILITY_CONTRACT_MODE", "off").strip().lower()
    if mode not in {"shadow", "enforce"}:
        return
    try:
        from src.capabilities.shadow import observe_legacy_tool_envelope

        observe_legacy_tool_envelope(
            tool_name=tool_name,
            category=category,
            raw_result=raw_result,
            envelope=envelope,
            elapsed_ms=elapsed_ms,
            mode=mode,
        )
    except Exception as exc:  # noqa: BLE001 - shadow must never affect tool execution
        if mode == "enforce":
            raise
        logger.warning(
            "capability_shadow_observer_failed tool=%s error_type=%s",
            tool_name,
            exc.__class__.__name__,
        )


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _json_size_chars(value: Any) -> int:
    return len(_json_dumps(value))


_TAGGED_JSON_RESULT_RE = re.compile(
    r"\[(skill_result|capability_result)\]\s*(.*?)\s*\[/\1\]",
    re.DOTALL,
)
_SCRIPT_CONTRACTS_RE = re.compile(
    r"\[script_contracts\]\s*(.*?)\s*\[/script_contracts\]",
    re.DOTALL,
)
_SKILL_CAPABILITIES_RE = re.compile(
    r"\[skill_capabilities\]\s*(.*?)\s*\[/skill_capabilities\]",
    re.DOTALL,
)


def _extract_tagged_json_result(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str):
        return None
    match = _TAGGED_JSON_RESULT_RE.search(value)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(2))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_script_contracts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, str):
        return []
    match = _SCRIPT_CONTRACTS_RE.search(value)
    if match is None:
        return []
    try:
        parsed = json.loads(match.group(1))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _extract_skill_capabilities(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, str):
        return []
    match = _SKILL_CAPABILITIES_RE.search(value)
    if match is None:
        return []
    try:
        parsed = json.loads(match.group(1))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _try_parse_json_text(value: Any) -> Any:
    current = value
    for _ in range(3):
        if not isinstance(current, str):
            return current
        text = current.strip()
        if not text or text[0] not in "[{\"":
            return current
        try:
            parsed = json.loads(text)
        except Exception:
            return current
        if parsed == current:
            return parsed
        current = parsed
    return current


def _normalize_model_result_source(tool_name: str, value: Any) -> Any:
    """Normalize legacy transport wrappers without shortening their contents."""

    tagged = _extract_tagged_json_result(value)
    if tagged is not None:
        return tagged
    parsed = _try_parse_json_text(value)
    if not isinstance(parsed, str) or str(tool_name or "") != "activate_skill":
        return parsed

    text = parsed.strip()
    normalized: dict[str, Any] = {}
    for key in (
        "activation_mode",
        "skill_id",
        "script_name",
        "execution_policy",
        "next_action",
    ):
        match = re.search(rf"(?m)^\s*{re.escape(key)}:\s*([^\n]*?)\s*$", text)
        if match is not None:
            normalized[key] = match.group(1).strip()
    capability_ids_match = re.search(
        r"(?m)^\s*capability_ids:\s*([^\n]*?)\s*$",
        text,
    )
    if capability_ids_match is not None:
        normalized["capability_ids"] = [
            item.strip()
            for item in capability_ids_match.group(1).split(",")
            if item.strip() and item.strip() != "无"
        ]
    skill_capabilities = _extract_skill_capabilities(text)
    if skill_capabilities:
        normalized["skill_capabilities"] = skill_capabilities
    execution_result = _extract_tagged_json_result(text)
    if execution_result is not None:
        normalized["execution_result"] = execution_result
    guide_text = _SKILL_CAPABILITIES_RE.sub(
        "",
        _SCRIPT_CONTRACTS_RE.sub("", text),
    ).strip()
    if guide_text:
        normalized["guide_text"] = guide_text
    return normalized or parsed


def project_task_complete_rejection_for_model(value: Any) -> dict[str, Any] | None:
    """Return an actionable model view while retaining diagnostics elsewhere.

    The complete rejection contract remains the machine result (and durable
    raw resource).  This projection deliberately contains no tool IDs,
    Citation protocol fields, source-ledger internals or retry instructions
    that a model could accidentally repeat to the user.
    """

    parsed = _try_parse_json_text(value)
    if not isinstance(parsed, dict):
        return None
    if parsed.get("ok") is not False and parsed.get("accepted") is not False:
        return None

    error_code = str(parsed.get("error_code") or parsed.get("code") or "").strip().lower()
    citation_binding = parsed.get("citation_binding")
    source_binding_rejected = error_code.startswith("citation_") or isinstance(
        citation_binding,
        dict,
    )
    if source_binding_rejected:
        available_reference_count = 0
        if isinstance(citation_binding, dict):
            try:
                available_reference_count = max(
                    0,
                    int(citation_binding.get("available_reference_count") or 0),
                )
            except (TypeError, ValueError):
                available_reference_count = 0
        if available_reference_count:
            message = (
                "答复中的部分来源还没有正确绑定。请使用本轮已有的有效来源重新整理；"
                "无法绑定的结论请明确说明限制。"
            )
        else:
            message = (
                "还有来源未绑定。请补充可核验的有效来源；如果暂时无法取得，"
                "请删除无法支持的引用，并在最终答复中明确说明限制。"
            )
        next_step = "补充或重新整理有效来源，并如实说明仍无法覆盖的范围。"
    elif error_code == "completion_deliverable_selection_invalid":
        message = (
            "所选交付文件不属于当前任务的权威成功结果，不能继续沿用原选择。"
            "请在当前任务中从已登记输入生成或字节不变复制所需文件并重新登记。"
        )
        next_step = "只选择当前任务成功结果返回的新文件，再结束当前任务。"
    elif parsed.get("missing_delivery_node_ids"):
        message = "当前任务要求的结果文件尚未准备好，请先生成并保存可交付文件。"
        next_step = "完成缺失的结果文件后，再结束当前任务。"
    elif parsed.get("active_node_ids") or parsed.get("unresolved_nodes"):
        message = "当前任务还有未完成或失败的步骤，请先处理这些步骤。"
        next_step = "继续完成剩余步骤；无法完成的部分请在最终答复中说明限制。"
    else:
        message = "当前任务还不能结束，请继续完成缺失内容或明确说明现有限制。"
        next_step = "核对尚未满足的要求，并只补充必要内容。"

    return {
        "ok": False,
        "accepted": False,
        "message": message,
        "next_step": next_step,
    }


def _separate_model_result_controls(tool_name: str, value: Any) -> Any:
    """Keep control, Citation and business projections from duplicating bodies."""

    if not isinstance(value, dict):
        return value
    if str(tool_name or "").strip() == "task_complete":
        rejection_projection = project_task_complete_rejection_for_model(value)
        if rejection_projection is not None:
            return rejection_projection
    result = dict(value)
    for key in (
        "capability_outcome",
        "source_outcome",
        "completion_signal",
        "model_summary",
        "user_summary",
    ):
        result.pop(key, None)
    if str(tool_name or "") == "run_retrieval_subagent" and isinstance(
        result.get("evidence_pack"), dict
    ):
        # Citation publication consumes the verified Source Ledger from the
        # runtime artifact. The model consumes the Evidence Pack. Replaying
        # both copies in one ToolMessage only inflates context.
        result.pop("source_ledger", None)
    return result


def _source_sidecar_unavailable_outcome(
    result: Any,
    source_sidecar: dict[str, Any],
) -> dict[str, Any]:
    """Return a failed source outcome when its durable authority is missing."""

    raw_outcome = None
    if isinstance(result, dict):
        raw_outcome = result.get("source_outcome")
    if raw_outcome is None:
        raw_outcome = source_sidecar.get("source_outcome")
    try:
        previous = SourceOutcome.model_validate(raw_outcome)
    except Exception:
        ledger = source_sidecar.get("source_ledger")
        candidate_count = len(ledger) if isinstance(ledger, list) else 0
        previous = SourceOutcome()
        previous = previous.model_copy(
            update={"candidate_count": candidate_count}
        )
    warning = "source_sidecar_unavailable"
    warnings = list(previous.warnings)
    if warning not in warnings:
        warnings.append(warning)
    return SourceOutcome(
        status=SourceProcessingStatus.FAILED,
        attempted=True,
        candidate_count=previous.candidate_count,
        publishable_count=0,
        rejected_count=previous.candidate_count,
        warnings=warnings,
        retryable=True,
        reason=(
            "source verification completed in memory, but its durable "
            "SourceSidecarRef is unavailable; no Citation may be published"
        ),
    ).model_dump(mode="json")


def normalize_capability_outcome(value: Any) -> dict[str, Any]:
    """Return the one canonical per-call outcome contract, or no contract.

    Legacy tools are deliberately not inferred here.  They either provide the
    contract explicitly or remain semantically unknown at the capability layer.
    """
    parsed = _try_parse_json_text(value)
    if not isinstance(parsed, dict):
        return {}
    try:
        normalized = CapabilityOutcome.model_validate(parsed).model_dump(
            mode="json", exclude_none=True
        )
        if all(
            normalized.get(key) == "unknown"
            for key in ("applicability", "evidence", "coverage", "recovery")
        ):
            return {}
        return normalized
    except Exception:
        return {}


def _extract_capability_outcome(value: Any) -> dict[str, Any]:
    for candidate in _nested_dict_candidates(value):
        outcome = normalize_capability_outcome(candidate.get("capability_outcome"))
        if outcome:
            return outcome
    return {}


def _completion_signal_from_capability_outcome(
    outcome: dict[str, Any],
    *,
    source_tool: str,
) -> dict[str, Any] | None:
    """Project canonical call facts into the legacy next-step hint.

    This never stops a root task.  It only tells the agent whether this exact
    capability path is known to be inapplicable, exhausted, or still open.
    """
    applicability = str(outcome.get("applicability") or "unknown")
    evidence = str(outcome.get("evidence") or "unknown")
    coverage = str(outcome.get("coverage") or "unknown")
    recovery = str(outcome.get("recovery") or "unknown")
    acceptance = {
        "accepted": applicability not in {CapabilityApplicability.NOT_APPLICABLE.value},
        "source_tool": source_tool,
        "capability_outcome": outcome,
        "evidence_permits_external_claims": evidence
        in {
            CapabilityEvidenceState.AVAILABLE.value,
            CapabilityEvidenceState.PARTIAL.value,
        },
    }
    if applicability == CapabilityApplicability.NOT_APPLICABLE.value:
        return {
            "schema_version": "tool_completion_signal_v1",
            "status": "capability_not_applicable",
            "should_stop_tool_loop": False,
            "next_action": "main_agent_evaluate_alternative_capabilities",
            "reason": "当前参数不适用于该能力；这只结束当前能力路径，主 agent 仍需根据根目标评估其他可用能力。",
            "acceptance": acceptance,
        }
    if (
        evidence == CapabilityEvidenceState.NONE.value
        and coverage == CapabilityCoverageState.EXHAUSTED.value
    ):
        return {
            "schema_version": "tool_completion_signal_v1",
            "status": "capability_scope_exhausted",
            "should_stop_tool_loop": False,
            "next_action": "main_agent_evaluate_alternative_capabilities_or_report_limit",
            "reason": "该 Provider 已完成声明的当前查询范围但没有证据；这不表示根任务完成或全局不存在证据。",
            "acceptance": acceptance,
        }
    if recovery == CapabilityRecoveryAction.RETRY_SAME_CALL.value:
        return {
            "schema_version": "tool_completion_signal_v1",
            "status": "capability_retry_advised",
            "should_stop_tool_loop": False,
            "next_action": "main_agent_decide_retry_same_call_or_alternative",
            "reason": "Provider 给出了同参数恢复建议；主 agent 应结合已有轨迹决定是否重试，不能无限循环。",
            "acceptance": acceptance,
        }
    if recovery == CapabilityRecoveryAction.REFINE_INPUT.value:
        return {
            "schema_version": "tool_completion_signal_v1",
            "status": "capability_input_refinement_advised",
            "should_stop_tool_loop": False,
            "next_action": "main_agent_refine_input_or_choose_alternative",
            "reason": "当前能力路径需要更精确的参数或范围；主 agent 决定补充、细化或改用其他能力。",
            "acceptance": acceptance,
        }
    if recovery == CapabilityRecoveryAction.SWITCH_CAPABILITY.value:
        return {
            "schema_version": "tool_completion_signal_v1",
            "status": "capability_alternative_advised",
            "should_stop_tool_loop": False,
            "next_action": "main_agent_evaluate_alternative_capabilities",
            "reason": "当前能力返回了确定的替代路径建议；主 agent 仍需对照根目标自主选择。",
            "acceptance": acceptance,
        }
    if coverage == CapabilityCoverageState.MORE_AVAILABLE.value:
        return {
            "schema_version": "tool_completion_signal_v1",
            "status": "capability_scope_partial",
            "should_stop_tool_loop": False,
            "next_action": "main_agent_evaluate_continue_scope",
            "reason": "当前调用只覆盖了已声明查询范围的一部分；是否继续分页或读取详情由主 agent 按用户目标决定。",
            "acceptance": acceptance,
        }
    if evidence in {
        CapabilityEvidenceState.AVAILABLE.value,
        CapabilityEvidenceState.PARTIAL.value,
    }:
        return {
            "schema_version": "tool_completion_signal_v1",
            "status": "result_available",
            "should_stop_tool_loop": False,
            "next_action": "main_agent_evaluate_requested_fields_and_evidence",
            "reason": "当前能力已返回证据；调用成功和证据可用都不等于根任务完成。",
            "acceptance": acceptance,
        }
    return None


def normalize_tool_completion_signal(value: Any) -> dict[str, Any]:
    """Normalize tool-level acceptance/stop hints into a tiny runtime contract."""
    parsed = _try_parse_json_text(value)
    if not isinstance(parsed, dict):
        return {}

    raw_status = str(parsed.get("status") or parsed.get("state") or "").strip().lower()
    raw_next_action = str(parsed.get("next_action") or parsed.get("recommended_action") or "").strip().lower()
    explicit_stop = parsed.get("should_stop_tool_loop")
    should_stop = (
        bool(explicit_stop)
        or raw_status in READY_TO_ANSWER_STATUSES
        or raw_next_action in READY_TO_ANSWER_NEXT_ACTIONS
    )
    status = raw_status or ("ready_to_answer" if should_stop else "")
    next_action = raw_next_action or ("final_answer" if should_stop else "")
    if not status and not next_action and explicit_stop is None:
        return {}

    out: dict[str, Any] = {
        "schema_version": "tool_completion_signal_v1",
        "status": status or "in_progress",
        "should_stop_tool_loop": bool(should_stop),
        "next_action": next_action or ("final_answer" if should_stop else "continue"),
    }
    for key in ("reason", "acceptance_summary", "missing_requirements"):
        if parsed.get(key) not in (None, "", []):
            out[key] = redact_model_secrets(parsed.get(key))
    acceptance = parsed.get("acceptance")
    if isinstance(acceptance, dict):
        out["acceptance"] = _sanitize_for_model(acceptance)
    deliverables = parsed.get("deliverable_files") or parsed.get("artifacts")
    if isinstance(deliverables, list) and deliverables:
        out["deliverable_files"] = [
            _sanitize_for_model(item)
            for item in deliverables
        ]
    return out


def _nested_dict_candidates(value: Any) -> list[dict[str, Any]]:
    parsed = _try_parse_json_text(value)
    if not isinstance(parsed, dict):
        tagged = _extract_tagged_json_result(value)
        return [tagged] if tagged is not None else []
    candidates = [parsed]
    for key in ("result", "data", "payload", "skill_result"):
        nested = parsed.get(key)
        nested = _try_parse_json_text(nested)
        if isinstance(nested, dict):
            candidates.append(nested)
    return candidates


def _locked_answer_completion_signal(
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Project a declared deterministic answer before generic evidence state."""

    for candidate in candidates:
        if not bool(candidate.get("answer_locked")):
            continue
        output_contract = candidate.get("output_contract")
        output_contract = output_contract if isinstance(output_contract, dict) else {}
        final_answer = output_contract.get("final_answer")
        if final_answer in (None, ""):
            final_answer = candidate.get("answer")
        acceptance: dict[str, Any] = {
            "accepted": True,
            "source_field": "answer_locked",
        }
        if final_answer not in (None, ""):
            acceptance["final_answer"] = redact_model_secrets(final_answer)
        if output_contract:
            acceptance["output_contract"] = _sanitize_for_model(output_contract)
        return {
            "schema_version": "tool_completion_signal_v1",
            "status": "answer_candidate_locked",
            "should_stop_tool_loop": False,
            "next_action": "main_agent_evaluate_root_goal",
            "reason": (
                "工具已返回确定性答案候选。调用任何新工具前，主 agent 必须先对照原始根目标验收："
                "若根目标只是该固定答案且无显式下游产物，直接按 output_contract 答复；"
                "仅当根目标仍要求文件、沙盒或其他节点时继续，不要重复匹配、激活或执行同一能力。"
            ),
            "acceptance": acceptance,
        }
    return None


def _asset_page_evidence_follow_up(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the mechanical next-tool contract for document previews."""

    asset_id = payload.get("asset_id")
    arguments = {"asset_id": asset_id} if asset_id not in (None, "") else {}
    return {
        "recommended_tool_id": "search_milvus_knowledge",
        "load_tool_ids": ["search_milvus_knowledge"],
        "arguments": arguments,
        "missing_required_arguments": [
            *([] if "asset_id" in arguments else ["asset_id"]),
            "query",
        ],
        "argument_contract": {
            "query": {
                "type": "string",
                "minLength": 1,
                "required": True,
                "derive_from": "current_research_question",
            }
        },
        "reading_strategy": {
            "first": "use_matched_chunk_content_and_locator",
            "expand_when": "matched_chunk_lacks_required_context",
            "expand_with": "read_asset_page_parsed_at_matched_page_idx",
            "continue_when": "claim_requires_cross_page_or_section_context",
        },
    }


def infer_tool_completion_signal(
    tool_name: str,
    result: Any,
    *,
    ok: bool | None = None,
    category: str = "",
    artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Infer a strict completion/next-action signal from standard tool payloads.

    This is intentionally conservative: file reads and workspace preparation are
    marked as available/next-tool, while retrieval packs, locked answers, task
    completion and finalized deliverables can stop the tool loop.
    """
    name = str(tool_name or "")
    category_name = str(category or "")
    candidates = _nested_dict_candidates(result)
    effective_ok = bool(ok) if ok is not None else _is_ok(result)
    primary = candidates[0] if candidates else {}
    if name in {"read_asset_summary", "read_asset_parsed"} and effective_ok:
        follow_up = _asset_page_evidence_follow_up(primary)
        is_summary = name == "read_asset_summary"
        return {
            "schema_version": "tool_completion_signal_v1",
            "status": (
                "document_overview_available"
                if is_summary
                else "requires_located_asset_evidence"
            ),
            "should_stop_tool_loop": False,
            "next_action": (
                "main_agent_assess_overview_then_search_if_specific_evidence_needed"
                if is_summary
                else "search_asset_then_assess_matched_content"
            ),
            "reason": (
                "read_asset_summary 已提供文档概览；若根目标需要具体事实或可发布定位，"
                "先在该资产内检索命中原文，命中片段缺少上下文时再读取命中页。"
                if is_summary
                else "read_asset_parsed 只返回文档级概览/正文前段且没有稳定 page/block/chunk 定位；"
                "不能据此发布页级 Citation。"
            ),
            **follow_up,
            "acceptance": {
                "accepted": True,
                "source_tool": name,
                "evidence_scope": "document_preview",
                "evidence_permits_external_claims": False,
                "publishable_page_evidence": False,
            },
        }
    if name == "search_milvus_knowledge" and effective_ok:
        search_payload = next(
            (
                candidate
                for candidate in candidates
                if isinstance(candidate.get("results"), list)
            ),
            primary,
        )
        rows = search_payload.get("results")
        rows = rows if isinstance(rows, list) else []
        if not rows:
            return {
                "schema_version": "tool_completion_signal_v1",
                "status": "knowledge_search_no_match",
                "should_stop_tool_loop": False,
                "next_action": "main_agent_assess_no_match_against_root_goal",
                "reason": (
                    "本次限定范围没有命中原文。先按根目标判断是否应说明未命中或选择另一来源；"
                    "需要扩大范围时，先调整查询或检索范围继续定位。"
                ),
                "acceptance": {
                    "accepted": True,
                    "source_tool": name,
                    "matched_chunks": 0,
                    "evidence_permits_external_claims": False,
                },
            }
        return {
            "schema_version": "tool_completion_signal_v1",
            "status": "located_asset_evidence_available",
            "should_stop_tool_loop": False,
            "next_action": "main_agent_assess_matched_content_against_root_goal",
            "reason": (
                "results[] 已包含命中原文及 asset/page/block/chunk 定位。先判断命中内容能否满足问题；"
                "仅当命中片段缺少必要上下文时，才按同条结果的 asset_id/page_idx 读取整页；"
                "只有结论确实依赖跨页或章节连续性时再扩展相邻页。"
            ),
            "conditional_follow_up": {
                "when": "matched_chunk_lacks_required_context",
                "tool_id": "read_asset_page_parsed",
                "arguments_from_result": {
                    "asset_id": "results[].asset_id",
                    "page_idx": "results[].page_idx",
                },
            },
            "acceptance": {
                "accepted": True,
                "source_tool": name,
                "matched_chunks": len(rows),
                "evidence_permits_external_claims": bool(
                    search_payload.get("_source_sidecar")
                ),
            },
        }
    if name == "read_asset_page_parsed" and effective_ok:
        page_payload = next(
            (
                candidate
                for candidate in candidates
                if isinstance(candidate.get("requested_scope"), dict)
                or isinstance(candidate.get("blocks"), list)
            ),
            primary,
        )
        blocks = page_payload.get("blocks")
        blocks = blocks if isinstance(blocks, list) else []
        valid_page_contract = bool(
            isinstance(page_payload.get("requested_scope"), dict)
            and isinstance(page_payload.get("blocks"), list)
            and page_payload.get("complete") is True
        )
        if not valid_page_contract:
            return {
                "schema_version": "tool_completion_signal_v1",
                "status": "invalid_page_result_contract",
                "should_stop_tool_loop": False,
                "next_action": "main_agent_report_or_try_one_targeted_page_recovery",
                "reason": "分页读取没有返回完整的指定页合同，当前结果不能作为已读页面或可发布证据。",
                "acceptance": {
                    "accepted": False,
                    "source_tool": name,
                    "page_complete": False,
                    "evidence_permits_external_claims": False,
                },
            }
        return {
            "schema_version": "tool_completion_signal_v1",
            "status": "located_page_context_available",
            "should_stop_tool_loop": False,
            "next_action": "main_agent_assess_page_context_against_root_goal",
            "reason": (
                "指定命中页已完整返回。先判断当前页是否补足尚缺上下文；仅在仍有明确跨页或章节"
                "连续性缺口时，再依据 document_navigation 扩展相邻页。"
            ),
            "acceptance": {
                "accepted": True,
                "source_tool": name,
                "page_complete": True,
                "block_count": len(blocks),
                "evidence_permits_external_claims": bool(
                    page_payload.get("_source_sidecar")
                ),
            },
        }
    if name == "run_retrieval_subagent":
        if not effective_ok:
            return {
                "schema_version": "tool_completion_signal_v1",
                "status": "failed",
                "should_stop_tool_loop": False,
                "next_action": "summarize_failure_or_try_one_targeted_recovery",
                "reason": "检索子代理返回失败；主 agent 应说明限制或进行一次有目标的恢复。",
            }
        claims_supported = bool(primary.get("evidence_pack_claims_supported"))
        diagnostics = primary.get("diagnostics") if isinstance(primary.get("diagnostics"), dict) else {}
        source_ledger = primary.get("source_ledger") if isinstance(primary.get("source_ledger"), list) else []
        try:
            source_ledger_count = int(diagnostics.get("source_ledger_count"))
        except (TypeError, ValueError):
            source_ledger_count = len(source_ledger)
        return {
            "schema_version": "tool_completion_signal_v1",
            "status": "retrieval_run_completed",
            "should_stop_tool_loop": False,
            "next_action": "main_agent_evaluate_goal_coverage",
            "reason": (
                "检索子代理调用已结束。evidence_pack_claims_supported 只表示 Evidence Pack 内的"
                "主张已有可发布证据，不表示用户根目标完成；主 agent 自行判断是否答复、补检索或继续下游。"
            ),
            "acceptance": {
                "evidence_pack_claims_supported": claims_supported,
                "source_ledger_count": source_ledger_count,
                "evidence_permits_external_claims": source_ledger_count > 0,
            },
        }
    if name == "web_search":
        if not effective_ok:
            return {
                "schema_version": "tool_completion_signal_v1",
                "status": "failed",
                "should_stop_tool_loop": False,
                "next_action": "summarize_failure_or_try_one_targeted_recovery",
                "reason": "网页检索返回失败；不要无限重试同一路径。",
            }
        return {
            "schema_version": "tool_completion_signal_v1",
            "status": "result_available",
            "should_stop_tool_loop": False,
            "next_action": "continue_or_answer_if_goal_is_satisfied",
            "reason": "网页检索结果已返回；是否足够回答取决于用户目标，不由 web_search 决定任务完成。",
            "acceptance": {
                "accepted": True,
                "source_tool": name,
                "evidence_permits_external_claims": False,
            },
        }
    locked_answer_signal = _locked_answer_completion_signal(candidates)
    if locked_answer_signal is not None:
        return locked_answer_signal

    capability_outcome = _extract_capability_outcome(result)
    projected_outcome_signal = _completion_signal_from_capability_outcome(
        capability_outcome,
        source_tool=name,
    )
    if projected_outcome_signal is not None:
        return projected_outcome_signal

    for candidate in candidates:
        explicit = candidate.get("completion_signal") or candidate.get("completion")
        normalized = normalize_tool_completion_signal(explicit)
        if normalized:
            status = str(normalized.get("status") or "").strip().lower()
            terminal = name == "task_complete" or status == "blocked_needs_user_action"
            if not terminal:
                normalized["should_stop_tool_loop"] = False
                if str(normalized.get("next_action") or "").strip().lower() in READY_TO_ANSWER_NEXT_ACTIONS:
                    normalized["next_action"] = "main_agent_evaluate_root_goal"
            return normalized

    if name == "sandbox_submit" or name.startswith("sandbox_submit_"):
        raw_submit_text = _json_dumps(primary) if isinstance(primary, dict) and primary else str(result or "")
        if "算力点余额不足" in raw_submit_text or "insufficient_balance" in raw_submit_text:
            return {
                "schema_version": "tool_completion_signal_v1",
                "status": "blocked_needs_user_action",
                "should_stop_tool_loop": True,
                "next_action": "final_answer",
                "reason": "沙盒提交被后端拒绝：算力点余额不足。不要重复提交同一任务，也不要改走检索；应向用户说明需要补充算力点后重试。",
                "acceptance": {
                    "accepted": False,
                    "source_tool": "sandbox_submit",
                    "blocker": "insufficient_compute_points",
                },
            }

    if not effective_ok:
        error_kind = ""
        for candidate in candidates:
            error = candidate.get("error")
            if isinstance(error, dict):
                error_kind = str(error.get("kind") or "").strip().lower()
            if not error_kind:
                error_kind = str(candidate.get("error_kind") or "").strip().lower()
            if error_kind:
                break
        unavailable = error_kind in {
            "auth_required",
            "provider_unavailable",
        }
        return {
            "schema_version": "tool_completion_signal_v1",
            "status": "capability_unavailable" if unavailable else "failed",
            "should_stop_tool_loop": False,
            "next_action": (
                "main_agent_report_limitation_or_evaluate_declared_alternative"
                if unavailable
                else "summarize_failure_or_try_one_targeted_recovery"
            ),
            "reason": (
                "当前能力因凭据或 Provider 不可用而未取得证据；不要用模型记忆补写数据库事实，也不要重复同一路径。"
                if unavailable
                else "工具返回失败；不要无限重试同一路径。"
            ),
            "acceptance": {
                "accepted": False,
                "source_tool": name,
                "error_kind": error_kind or "unknown",
                "evidence_permits_external_claims": False,
            },
        }

    if category_name == "bio_database":
        result_state = ""
        total_results: int | None = None
        returned_results: int | None = None
        pagination: dict[str, Any] = {}
        for candidate in candidates:
            candidate_state = str(candidate.get("result_status") or "").strip().lower()
            if candidate_state:
                result_state = candidate_state
            if candidate.get("total_results") is not None:
                try:
                    total_results = max(0, int(candidate.get("total_results")))
                except (TypeError, ValueError):
                    pass
            if candidate.get("returned_results") is not None:
                try:
                    returned_results = max(0, int(candidate.get("returned_results")))
                except (TypeError, ValueError):
                    pass
            if isinstance(candidate.get("pagination"), dict):
                pagination = dict(candidate["pagination"])
            if result_state:
                break

        acceptance = {
            "accepted": result_state not in {"", "empty", "failed"},
            "source_tool": name,
            "result_state": result_state or "unqualified",
            "evidence_permits_external_claims": False,
            **({"total_results": total_results} if total_results is not None else {}),
            **({"returned_results": returned_results} if returned_results is not None else {}),
            **({"pagination": pagination} if pagination else {}),
        }
        if result_state == "empty":
            return {
                "schema_version": "tool_completion_signal_v1",
                "status": "empty_result",
                "should_stop_tool_loop": False,
                "next_action": "evaluate_query_or_try_one_targeted_recovery",
                "reason": (
                    "数据源调用成功，但本次没有返回记录。空结果不等于当前任务节点完成；"
                    "主 agent 应核对查询条件，并在有明确替代路径时进行一次定向恢复。"
                ),
                "acceptance": acceptance,
            }
        if result_state == "partial":
            return {
                "schema_version": "tool_completion_signal_v1",
                "status": "partial_result",
                "should_stop_tool_loop": False,
                "next_action": "evaluate_missing_fields_pagination_or_detail_expansion",
                "reason": (
                    "数据源已返回部分记录，但结果范围仍未完全展开。"
                    "主 agent 应对照用户目标判断是否继续分页、读取详情、去重或统计；"
                    "在缺口补齐前不要把该节点写成 completed。"
                ),
                "acceptance": acceptance,
            }
        if result_state == "available":
            return {
                "schema_version": "tool_completion_signal_v1",
                "status": "result_available",
                "should_stop_tool_loop": False,
                "next_action": "main_agent_evaluate_requested_fields_and_evidence",
                "reason": (
                    "数据源返回了可用记录，但调用成功本身不等于任务完成。"
                    "主 agent 仍需对照用户要求的字段、范围、统计和证据判断节点是否满足。"
                ),
                "acceptance": acceptance,
            }

    for candidate in candidates:
        if bool(candidate.get("answer_ready")) or bool(candidate.get("stop_retrieval")):
            evidence_pack = candidate.get("evidence_pack") if isinstance(candidate.get("evidence_pack"), dict) else {}
            evidence = evidence_pack.get("evidence") if isinstance(evidence_pack, dict) else None
            claims = evidence_pack.get("claims") if isinstance(evidence_pack, dict) else None
            evidence_count = len(evidence) if isinstance(evidence, list) else int(candidate.get("evidence_count") or 0)
            claim_count = len(claims) if isinstance(claims, list) else int(candidate.get("claim_count") or 0)
            return {
                "schema_version": "tool_completion_signal_v1",
                "status": "answer_candidate_ready",
                "should_stop_tool_loop": False,
                "next_action": "main_agent_evaluate_root_goal",
                "reason": "工具已返回可回答候选；主 agent 应对照原始根目标决定答复或继续其他节点。",
                "acceptance": {
                    "accepted": True,
                    "source_field": "answer_ready/stop_retrieval",
                    "claim_count": claim_count,
                    "evidence_count": evidence_count,
                },
            }

    if name == "task_complete":
        signal = {
            "schema_version": "tool_completion_signal_v1",
            "status": "task_complete",
            "should_stop_tool_loop": True,
            "next_action": "final_answer",
            "reason": "任务完成工具已被调用，本轮应输出最终答复并结束。",
            "acceptance": {"accepted": True, "source_tool": "task_complete"},
        }
        deliverables = primary.get("deliverable_files")
        if isinstance(deliverables, list) and deliverables:
            signal["deliverable_files"] = [
                _sanitize_for_model(item)
                for item in deliverables
            ]
        return signal

    if name == "save_execution_plan":
        if primary.get("rejected_as_final_report") or str(primary.get("next_tool") or "") == "generate_markdown_document":
            return {
                "schema_version": "tool_completion_signal_v1",
                "status": "requires_next_tool",
                "should_stop_tool_loop": False,
                "next_action": "generate_markdown_document",
                "reason": "当前调用像最终报告而不是执行计划；不要再次保存计划，改用 generate_markdown_document 生成结果文件。",
                "acceptance": {"accepted": True, "source_tool": name, "noop": True},
            }
        next_action = str(primary.get("next_action") or "").strip() or "continue_task_execution"
        return {
            "schema_version": "tool_completion_signal_v1",
            "status": "plan_saved",
            "should_stop_tool_loop": False,
            "next_action": next_action,
            "reason": "执行计划已保存为任务契约文件；同一任务链继续读取或更新任务树 frontier。",
            "acceptance": {
                "accepted": True,
                "source_tool": name,
                "drawer_section": "plan_file",
                "plan_role": str(primary.get("plan_role") or "mission_contract"),
                "state_authority": str(primary.get("state_authority") or "task_tree"),
            },
        }

    if name == "generate_markdown_document":
        drawer_section = str(primary.get("drawer_section") or "").strip().lower()
        if drawer_section == "temporary_output":
            return {
                "schema_version": "tool_completion_signal_v1",
                "status": "requires_next_tool",
                "should_stop_tool_loop": False,
                "next_action": "continue_task_chain",
                "reason": "临时计划或中间 Markdown 已保存到文件抽屉，继续执行后续任务节点。",
                "acceptance": {"accepted": True, "source_tool": name, "drawer_section": drawer_section},
            }

    if name.startswith("generate_") or category_name == "deliverable":
        file_id = primary.get("conversation_file_id") or primary.get("file_id")
        if file_id or primary.get("file_name") or artifacts:
            signal: dict[str, Any] = {
                "schema_version": "tool_completion_signal_v1",
                "status": "deliverable_node_complete",
                "should_stop_tool_loop": False,
                "next_action": "main_agent_evaluate_root_goal",
                "reason": "结果文件已生成并注册，当前交付节点完成；主 agent 继续核对根目标和其他未完成节点。",
                "acceptance": {"accepted": True, "source_tool": name},
            }
            deliverables = artifacts or _extract_file_artifacts(primary)
            if deliverables:
                signal["deliverable_files"] = deliverables
            return signal

    if name == "sandbox_submit" or name.startswith("sandbox_submit_"):
        raw_text = _json_dumps(primary) if isinstance(primary, dict) and primary else str(result or "")
        if "算力点余额不足" in raw_text or "insufficient_balance" in raw_text:
            return {
                "schema_version": "tool_completion_signal_v1",
                "status": "blocked_needs_user_action",
                "should_stop_tool_loop": True,
                "next_action": "final_answer",
                "reason": "沙盒提交被后端拒绝：算力点余额不足。不要重复提交同一任务，也不要改走检索；应向用户说明需要补充算力点后重试。",
                "acceptance": {
                    "accepted": False,
                    "source_tool": "sandbox_submit",
                    "blocker": "insufficient_compute_points",
                },
            }
        job = primary.get("job") if isinstance(primary.get("job"), dict) else {}
        job_id = str(primary.get("job_id") or primary.get("sandbox_job_id") or job.get("job_id") or "").strip()
        status = str(primary.get("status") or primary.get("sandbox_status") or job.get("status") or "").strip()
        execution_mode = str(primary.get("agent_execution_mode") or "").strip().lower()
        next_action = (
            "sandbox_get_result"
            if execution_mode != "nonblocking_long"
            else "continue_ready_nodes_or_wait_external"
        )
        return {
            "schema_version": "tool_completion_signal_v1",
            "status": "requires_next_tool",
            "should_stop_tool_loop": False,
            "next_action": next_action,
            "reason": (
                "沙盒任务只是提交成功，还没有满足最终目标。"
                + (
                    "这是短等待任务；下一步读取结果，再更新任务树并生成必要交付物。"
                    if execution_mode != "nonblocking_long"
                    else "这是后台任务；继续其他可执行节点，没有可推进节点时等待外部结果。"
                )
            ),
            "acceptance": {
                "accepted": True,
                "source_tool": "sandbox_submit",
                **({"job_id": job_id} if job_id else {}),
                **({"sandbox_status": status} if status else {}),
                **({"agent_execution_mode": execution_mode} if execution_mode else {}),
            },
        }

    if name in {"artifact_finalize", "artifact_finalize_delivery_package", "artifact_register_outputs"}:
        if primary.get("registered_count") or primary.get("package_summary") or primary.get("files"):
            deliverables = artifacts or _extract_file_artifacts(primary)
            return {
                "schema_version": "tool_completion_signal_v1",
                "status": "artifact_node_complete",
                "should_stop_tool_loop": False,
                "next_action": "main_agent_evaluate_root_goal",
                "reason": "artifact 产物已完成 QA/注册，当前产物节点完成；主 agent 继续核对根目标和其他未完成节点。",
                "acceptance": {
                    "accepted": True,
                    "source_tool": name,
                    "registered_count": int(primary.get("registered_count") or 0),
                },
                **({"deliverable_files": deliverables} if deliverables else {}),
            }

    if name == "sandbox_get_result":
        status = str(primary.get("status") or "").strip().lower()
        if status in {"succeeded", "success", "completed", "complete"} and (
            primary.get("artifact_count") or primary.get("registered_count") or primary.get("artifacts")
        ):
            next_action = str(
                primary.get("task_tree_suggested_next_action")
                or "main_agent_evaluate_current_frontier_and_user_goal"
            ).strip()
            return {
                "schema_version": "tool_completion_signal_v1",
                "status": "requires_next_tool",
                "should_stop_tool_loop": False,
                "next_action": next_action,
                "acceptance": {
                    "accepted": True,
                    "source_tool": "sandbox_get_result",
                    "sandbox_status": status,
                    "evidence_complete_for_sandbox_result": True,
                },
            }

    if name == "artifact_workspace_prepare":
        return {
            "schema_version": "tool_completion_signal_v1",
            "status": "requires_next_tool",
            "should_stop_tool_loop": False,
            "next_action": "artifact_run_action",
            "reason": "artifact 工作区只是准备完成；下一步通过统一 artifact_run_action 执行已选择且参数明确的受控动作。",
        }

    if name.startswith("artifact_run_") or name == "artifact_run_action":
        signal: dict[str, Any] = {
            "schema_version": "tool_completion_signal_v1",
            "status": "requires_next_tool",
            "should_stop_tool_loop": False,
            "next_action": "artifact_finalize",
            "reason": "artifact 执行已产生或尝试产生 outputs，下一步应扫描/注册交付物。",
        }
        run_id = str(primary.get("run_id") or "").strip()
        script_path = str(primary.get("script_path") or "").strip()
        language = str(primary.get("language") or "").strip().lower()
        expected_outputs = primary.get("expected_outputs")
        figure_paths = [
            str(path).strip()
            for path in (expected_outputs if isinstance(expected_outputs, list) else [])
            if str(path).strip().lower().endswith((".png", ".svg", ".pdf"))
        ]
        if run_id and script_path and language in {"python", "r"} and figure_paths:
            signal["reason"] = (
                "科研图执行已完成。下一步只调用一次 artifact_finalize，并用下方精确 source 与图像路径"
                "填写 scientific_figure_declaration；不要另写源码文件或 Markdown manifest。"
            )
            signal["scientific_figure_finalize_context"] = {
                "schema_version": "scientific_figure_declaration/v1",
                "source": {
                    "run_id": run_id,
                    "source_script_path": script_path,
                    "language": language,
                },
                "figure_output_paths": figure_paths,
                "required_group_fields": [
                    "figure_key",
                    "display_name",
                    "outputs[].artifact_path",
                    "outputs[].file_format",
                    "outputs[].primary",
                ],
                "constraints": [
                    "每个 figure_key 恰好一个 primary=true",
                    "artifact_path 必须取自 figure_output_paths",
                    "不要把 script_path 放入 outputs；它已是权威可复现源码",
                ],
            }
        return signal

    if name == "conversation_file_read":
        return {
            "schema_version": "tool_completion_signal_v1",
            "status": "result_available",
            "should_stop_tool_loop": False,
            "next_action": "continue_or_answer_if_user_only_asked_to_read_file",
            "reason": "文件内容已读取；是否完成取决于用户目标和当前任务节点。",
            "acceptance": {
                "accepted": True,
                "source_tool": name,
                "evidence_permits_external_claims": False,
            },
        }

    signal = {
        "schema_version": "tool_completion_signal_v1",
        "status": "result_available",
        "should_stop_tool_loop": False,
        "next_action": "continue",
        "reason": "工具返回成功，但没有声明当前任务节点已满足验收条件。",
    }
    verified_source_count = 0
    try:
        from src.subagents.source_ledger import verified_source_candidates_from_output

        verified_source_count = len(
            verified_source_candidates_from_output(
                primary if isinstance(primary, dict) else {}
            )
        )
    except Exception:
        verified_source_count = 0
    if category_name in {
        "bio_database",
        "conversation_files",
        "knowledge_assets",
        "skill_execution",
        "web_search",
    }:
        signal["acceptance"] = {
            "accepted": True,
            "source_tool": name,
            "evidence_permits_external_claims": verified_source_count > 0,
        }
        if verified_source_count:
            signal["acceptance"]["verified_source_count"] = verified_source_count
    return signal


def _sanitize_for_model(value: Any) -> Any:
    """Return complete, secret-redacted control data."""

    value = _try_parse_json_text(value)
    return redact_model_secrets(value)






























def _extract_tool_warning(value: Any, sanitized: Any) -> str:
    for candidate in (sanitized, _try_parse_json_text(value)):
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("severity") or "").strip().lower() == "warning":
            warning = str(candidate.get("warning") or candidate.get("content_note") or "").strip()
            return warning or "工具返回了警告状态。"
        result = candidate.get("result") if isinstance(candidate.get("result"), dict) else None
        if isinstance(result, dict):
            nested = _extract_tool_warning(result, result)
            if nested:
                return nested
    return ""














def _is_ok(value: Any) -> bool:
    tagged = _extract_tagged_json_result(value)
    if tagged is not None:
        if "ok" in tagged:
            return bool(tagged.get("ok"))
        if tagged.get("error") or tagged.get("error_kind"):
            return False
    parsed = _try_parse_json_text(value)
    if isinstance(parsed, dict):
        if "ok" in parsed:
            return bool(parsed.get("ok"))
        if "success" in parsed:
            return bool(parsed.get("success"))
        if parsed.get("error") or parsed.get("error_kind"):
            return False
    if isinstance(parsed, str):
        lowered = parsed.strip().lower()
        if lowered.startswith(("error", "错误", "失败", "exception", "traceback", "timeouterror")):
            return False
    return True


def _extract_web_refs(value: Any) -> list[dict[str, Any]]:
    parsed = _try_parse_json_text(value)
    items = parsed.get("items") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        return []
    refs: list[dict[str, Any]] = []
    for idx, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or item.get("link") or "").strip()
        title = str(item.get("title") or item.get("name") or url).strip()
        if url or title:
            refs.append({"source_type": "web", "source_id": url or title, "locator": url, "title": title, "index": idx})
    return refs


def _extract_knowledge_refs(value: Any) -> list[dict[str, Any]]:
    parsed = _try_parse_json_text(value)
    if not isinstance(parsed, dict):
        return []
    refs: list[dict[str, Any]] = []
    for key in ("results", "items", "records", "data"):
        rows = parsed.get(key)
        if not isinstance(rows, list):
            continue
        for idx, item in enumerate(rows, 1):
            if not isinstance(item, dict):
                continue
            asset_id = item.get("asset_id") or item.get("id") or item.get("file_id") or item.get("document_id")
            file_name = item.get("file_name") or item.get("name") or item.get("title") or ""
            page_idx = item.get("page_idx") if item.get("page_idx") is not None else item.get("page")
            locator = f"page={page_idx}" if page_idx is not None else ""
            refs.append(
                {
                    "source_type": "knowledge_asset",
                    "source_id": str(asset_id or file_name or f"hit_{idx}"),
                    "locator": locator,
                    "title": str(file_name or ""),
                    "index": idx,
                    "score": item.get("score"),
                }
            )
        if refs:
            break
    return refs


def _extract_sandbox_refs(tool_name: str, value: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    text = _json_dumps(value)
    job_ids = []
    for pattern in (r"job_id[=:'\"]+([A-Za-z0-9_.:-]+)", r"sandbox_job_ids=\['([^']+)'\]", r"任务\s+([A-Za-z0-9_.:-]{8,})"):
        for match in re.findall(pattern, text):
            if match not in job_ids:
                job_ids.append(match)
    refs = [
        {"source_type": "sandbox", "source_id": job_id, "locator": f"job_id={job_id}", "title": tool_name}
        for job_id in job_ids
    ]
    artifacts: list[dict[str, Any]] = []
    for file_name in re.findall(r"---\s+([^\n-][^\n]+?)\s+---", text):
        artifacts.append({"title": file_name.strip(), "type": "sandbox_artifact", "artifact_id": ""})
    extracted = {"job_ids": job_ids} if job_ids else {}
    return refs, artifacts, extracted


def _extract_file_artifacts(value: Any) -> list[dict[str, Any]]:
    parsed = _try_parse_json_text(value)
    if not isinstance(parsed, dict):
        parsed = _extract_tagged_json_result(value)
    artifacts: list[dict[str, Any]] = []
    if isinstance(parsed, dict):
        candidates: list[dict[str, Any]] = []
        if parsed.get("file_name") or parsed.get("conversation_file_id") or parsed.get("file_id"):
            candidates.append(parsed)
        files = parsed.get("files")
        if isinstance(files, list):
            candidates.extend(item for item in files if isinstance(item, dict))
        sandbox_artifacts = parsed.get("artifacts")
        if isinstance(sandbox_artifacts, list):
            candidates.extend(item for item in sandbox_artifacts if isinstance(item, dict))
        standardized_outputs = parsed.get("standardized_outputs")
        if isinstance(standardized_outputs, list):
            candidates.extend(item for item in standardized_outputs if isinstance(item, dict))
        registered_artifacts = parsed.get("registered_artifacts")
        if isinstance(registered_artifacts, list):
            candidates.extend(item for item in registered_artifacts if isinstance(item, dict))
        seen: set[tuple[str, str]] = set()
        for item in candidates:
            file_id = item.get("conversation_file_id") or item.get("file_id") or item.get("asset_id") or ""
            file_name = str(item.get("file_name") or item.get("name") or item.get("title") or file_id or "").strip()
            artifact_key = str(item.get("artifact_key") or "").strip()
            artifact_id = str(
                item.get("artifact_id")
                or item.get("native_artifact_id")
                or file_id
                or artifact_key
                or ""
            ).strip()
            if file_id or file_name:
                identity = (
                    artifact_key or str(file_id or artifact_id),
                    file_name,
                )
                if identity in seen:
                    continue
                seen.add(identity)
                artifacts.append(
                    {
                        "artifact_id": artifact_id,
                        "artifact_key": artifact_key,
                        "conversation_file_id": item.get("conversation_file_id") or item.get("file_id"),
                        "file_name": file_name,
                        "title": file_name,
                        "mime": str(item.get("mime_type") or item.get("content_type") or ""),
                        "mime_type": str(item.get("mime_type") or item.get("content_type") or ""),
                        "size_bytes": item.get("size_bytes"),
                        "sha256": str(item.get("sha256") or "").strip().lower() or None,
                        "drawer_section": item.get("drawer_section"),
                        "source_type": item.get("source_type") or "native_tool",
                        "registration_status": item.get("registration_status"),
                        "registered_to_drawer": item.get("registered_to_drawer"),
                        "task_node_id": item.get("task_node_id"),
                        "tool_run_id": item.get("tool_run_id"),
                        "retention_policy": item.get("retention_policy"),
                        "standardized": item.get("standardized"),
                    }
                )
    return artifacts


def _extract_task_refs(value: Any) -> list[dict[str, Any]]:
    parsed = _try_parse_json_text(value)
    if not isinstance(parsed, dict):
        return []
    refs: list[dict[str, Any]] = []
    node_id = parsed.get("node_id")
    run_id = parsed.get("run_id")
    if node_id or run_id:
        refs.append(
            {
                "source_type": "task_tree",
                "source_id": str(node_id or run_id),
                "locator": f"run_id={run_id or ''}",
                "title": str(parsed.get("title") or parsed.get("root_title") or "task_tree"),
            }
        )
    return refs


def _summarize_value(tool_name: str, value: Any, *, ok: bool) -> str:
    parsed = _try_parse_json_text(value)
    if not ok:
        if isinstance(parsed, dict):
            error_kind = str(parsed.get("error_kind") or "tool_error")
            return f"{tool_name} 调用失败（{error_kind}）；完整错误合同见 result。"
        return f"{tool_name} 调用失败；完整错误合同见 result。"
    if isinstance(parsed, dict):
        if isinstance(parsed.get("items"), list):
            return f"{tool_name} 返回 {len(parsed.get('items') or [])} 条 items。"
        if isinstance(parsed.get("results"), list):
            return f"{tool_name} 返回 {len(parsed.get('results') or [])} 条 results。"
        if isinstance(parsed.get("files"), list):
            return f"{tool_name} 返回 {len(parsed.get('files') or [])} 个文件。"
        if isinstance(parsed.get("apps"), list):
            return f"{tool_name} 返回 {len(parsed.get('apps') or [])} 个 apps。"
        if isinstance(parsed.get("submit_fields"), list):
            app = parsed.get("app") if isinstance(parsed.get("app"), dict) else {}
            app_id = app.get("app_id") if isinstance(app, dict) else ""
            prefix = f"{app_id} " if app_id else ""
            return f"{tool_name} 返回 {prefix}{len(parsed.get('submit_fields') or [])} 个 submit_fields。"
        return f"{tool_name} 返回结构化结果；完整字段见 result。"
    if isinstance(parsed, list):
        return f"{tool_name} 返回 {len(parsed)} 条结果。"
    return f"{tool_name} 返回文本结果；完整内容见 result。"


def make_tool_result_envelope(
    tool_name: str,
    result: Any,
    *,
    elapsed_ms: int = 0,
    category: str = "",
    resource_scope: ResourceScope | None = None,
    existing_resource_ref: ResourceRef | None = None,
    model_result_policy: str = "",
) -> dict[str, Any]:
    # Only skip wrapping for a real envelope produced by this module.  A
    # provider payload may legitimately expose ``model_summary`` and a
    # durable ``raw_ref`` (the retrieval subagent does); treating those two
    # fields alone as an envelope bypasses the common size projection and
    # copies the complete provider body back into model context.
    if (
        isinstance(result, dict)
        and result.get("model_summary")
        and str(result.get("tool") or "").strip()
        and "tool_category" in result
        and "result" in result
        and "projection_kind" in result
    ):
        _apply_capability_result_contract(
            tool_name=tool_name,
            category=category,
            raw_result=result.get("result"),
            envelope=result,
            elapsed_ms=int(elapsed_ms or 0),
        )
        return result

    source_projection_input = result
    source_sidecar: dict[str, Any] | None = None
    if isinstance(result, dict) and isinstance(result.get("_source_sidecar"), dict):
        source_sidecar = dict(result["_source_sidecar"])
        source_projection_input = source_sidecar
        result = {
            key: value
            for key, value in result.items()
            if key != "_source_sidecar"
        }

    source_sidecar_ref = (
        persist_source_sidecar_resource(
            scope=resource_scope,
            tool_name=tool_name,
            value=source_sidecar,
        )
        if (
            source_sidecar is not None
            and resource_scope is not None
            and resource_scope.usable
        )
        else None
    )
    source_sidecar_failure_outcome: dict[str, Any] | None = None
    if source_sidecar is not None and source_sidecar_ref is None:
        source_sidecar_failure_outcome = _source_sidecar_unavailable_outcome(
            result,
            source_sidecar,
        )
        source_projection_input = {}
        if isinstance(result, dict):
            result = dict(result)
            result["source_outcome"] = source_sidecar_failure_outcome
            if "source_ledger" in result:
                result["source_ledger"] = []
            identity_projection = result.get("evidence_identity_projection")
            if isinstance(identity_projection, dict):
                identity_projection = dict(identity_projection)
                identity_projection["publishable_source_count"] = 0
                identity_projection["full_evidence_location"] = "unavailable"
                identity_projection["source_sidecar_available"] = False
                result["evidence_identity_projection"] = identity_projection

    ok = _is_ok(result)
    task_complete_rejection = (
        project_task_complete_rejection_for_model(result)
        if str(tool_name or "").strip() == "task_complete" and not ok
        else None
    )
    normalized_model_result_policy = str(model_result_policy or "").strip().lower()
    bounded_model_window = normalized_model_result_policy == "bounded_window"
    raw_chars = _json_size_chars(result)
    model_result_source = _separate_model_result_controls(
        tool_name,
        _normalize_model_result_source(tool_name, result),
    )
    bounded_result_needs_resource = bool(
        bounded_model_window
        and model_result_exceeds_inline_limit(model_result_source)
    )
    resource_ref = existing_resource_ref or (
        persist_tool_result_resource(
            scope=resource_scope,
            tool_name=tool_name,
            value=result,
        )
        if (
            resource_scope is not None
            and resource_scope.usable
            and (not bounded_model_window or bounded_result_needs_resource)
            and tool_name not in {"resource_inspect", "resource_read", "resource_search"}
        )
        else None
    )
    raw_ref = resource_ref.resource_id if resource_ref is not None else None
    raw_stored = resource_ref is not None
    # One generic model contract replaces the former per-tool character caps,
    # first-N lists and cumulative 8K budget. A Provider-declared window may
    # describe exact scope/pagination, but never bypasses the common size
    # boundary. Results stay complete while they fit; a genuinely oversized
    # result becomes a pure ResourceRef navigation result rather than a
    # misleading prefix.
    model_projection = project_tool_result_for_model(
        raw_result=model_result_source,
        resources=[resource_ref] if resource_ref is not None else [],
        tool_name=tool_name,
        provider_bounded=bounded_model_window,
    )
    sanitized = model_projection.result
    projection_contract_failed = (
        model_projection.projection_kind == "resource_unavailable"
    )
    if projection_contract_failed:
        ok = False

    summary_source = sanitized if ok or task_complete_rejection is not None else result
    model_summary = (
        str(task_complete_rejection.get("message") or "").strip()
        if task_complete_rejection is not None
        else _summarize_value(tool_name, summary_source, ok=ok)
    )
    user_summary = model_summary
    evidence_refs: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    extracted: dict[str, Any] = {}

    if tool_name == "web_search" or category == "web_search":
        evidence_refs.extend(_extract_web_refs(result))
    verified_source_refs: list[dict[str, Any]] = []
    if (
        category in {"source", "retrieval_subagent", "knowledge_assets"}
        or tool_name == "run_retrieval_subagent"
    ):
        from src.subagents.source_citation_projection import verified_source_refs_from_result

        verified_source_refs = verified_source_refs_from_result(
            _try_parse_json_text(source_projection_input)
        )
    evidence_refs.extend(verified_source_refs)
    if (
        not verified_source_refs
        and (
            category in {"knowledge_assets", "bio_database"}
            or tool_name.startswith(("search_milvus", "read_asset", "list_knowledge", "list_database"))
        )
    ):
        evidence_refs.extend(_extract_knowledge_refs(result))
    if tool_name.startswith("sandbox_") or category.startswith("sandbox"):
        sandbox_refs, sandbox_artifacts, sandbox_extracted = _extract_sandbox_refs(tool_name, result)
        evidence_refs.extend(sandbox_refs)
        artifacts.extend(sandbox_artifacts)
        extracted.update(sandbox_extracted)
    file_artifacts = _extract_file_artifacts(result)
    if file_artifacts:
        artifacts.extend(file_artifacts)
    if category == "task_tree" or tool_name.startswith("task_tree_"):
        evidence_refs.extend(_extract_task_refs(result))
    capability_outcome = _extract_capability_outcome(result)
    if isinstance(sanitized, dict):
        sanitized = dict(sanitized)
        sanitized.pop("capability_outcome", None)
        for key in ("result", "data", "payload", "skill_result"):
            nested = sanitized.get(key)
            if isinstance(nested, dict) and "capability_outcome" in nested:
                nested_without_outcome = dict(nested)
                nested_without_outcome.pop("capability_outcome", None)
                sanitized[key] = nested_without_outcome
    from src.services.tool_outcome_projection import project_tool_call_outcome

    completion_projection_input = (
        {**result, "_source_sidecar": source_sidecar}
        if (
            isinstance(result, dict)
            and source_sidecar is not None
            and source_sidecar_ref is not None
        )
        else result
    )
    completion_signal = (
        None
        if task_complete_rejection is not None
        else project_tool_call_outcome(
            tool_name,
            completion_projection_input,
            ok=ok,
            category=category,
            artifacts=artifacts,
        )
    )
    sandbox_model_envelope = bool(
        tool_name.startswith("sandbox_") or category.startswith("sandbox")
    )
    envelope: dict[str, Any] = {
        "ok": ok,
        "success": ok,
        "tool": tool_name,
        "tool_category": category or "general",
        "model_summary": model_summary,
        "user_summary": user_summary,
        "summary": model_summary,
        "evidence_refs": evidence_refs,
        "raw_ref": raw_ref,
        "result": sanitized,
    }
    # Sandbox raw results already contain the complete native/standardized
    # artifact manifest.  Result mapping and the runtime-only ToolMessage
    # artifact consume that authoritative machine payload; copying the same
    # rows into the model envelope merely duplicates context and can replay
    # native and normalized views of one file.  Non-sandbox tools retain the
    # existing compact artifact projection.
    if not sandbox_model_envelope:
        envelope["artifacts"] = artifacts
    if completion_signal is not None:
        envelope["completion_signal"] = completion_signal
    if source_sidecar_ref is not None:
        envelope["source_sidecar_ref"] = source_sidecar_ref.resource_id
        envelope["source_sidecar_resource"] = source_sidecar_ref.model_dump(
            mode="json",
            exclude_none=True,
        )
    if source_sidecar_failure_outcome is not None:
        envelope["source_outcome"] = source_sidecar_failure_outcome
    envelope.update(model_projection.contract_payload())
    envelope["ui_projection"] = (
        {
            "ok": False,
            "summary": user_summary,
        }
        if task_complete_rejection is not None
        else build_tool_ui_projection(
            tool_name=tool_name,
            category=category,
            ok=ok,
            summary=user_summary,
        )
    )
    if capability_outcome:
        envelope["capability_outcome"] = capability_outcome
    warning = _extract_tool_warning(result, sanitized)
    if ok and warning:
        envelope["severity"] = "warning"
        envelope["warning"] = redact_model_secrets(warning)
    if _bool_env("EVO_TOOL_ENVELOPE_INCLUDE_DIAGNOSTICS", False):
        envelope["diagnostics"] = {
            "elapsed_ms": int(elapsed_ms or 0),
            "raw_chars": raw_chars,
            "model_chars": _json_size_chars(sanitized),
            "raw_stored": raw_stored,
        }
    if extracted:
        envelope["extracted"] = extracted
    if not ok and task_complete_rejection is not None:
        envelope["message"] = model_summary
    elif not ok:
        envelope["error_kind"] = "tool_error"
        error_payload = result if isinstance(result, dict) else _extract_tagged_json_result(result)
        if isinstance(error_payload, dict):
            raw_error = error_payload.get("error") or error_payload.get("message") or error_payload.get("summary")
            if raw_error:
                envelope["error"] = redact_model_secrets({"error": raw_error})["error"]
            if error_payload.get("error_kind"):
                envelope["error_kind"] = str(error_payload.get("error_kind"))
            if error_payload.get("next_tool"):
                envelope["next_tool"] = str(error_payload.get("next_tool"))
            if error_payload.get("error_code"):
                envelope["error_code"] = str(error_payload.get("error_code"))
            if isinstance(error_payload.get("citation_binding"), dict):
                envelope["citation_binding"] = _sanitize_for_model(
                    error_payload.get("citation_binding")
                )
        if projection_contract_failed:
            envelope["error_kind"] = "contract_violation"
            envelope["error"] = (
                "完整工具结果超过模型安全内联范围，且未能建立可读取的 ResourceRef。"
            )
            envelope["message"] = (
                "该次工具结果交付合同失败，不能把当前调用视为成功；"
                "请修复资源持久化后重试。"
            )
        else:
            envelope["message"] = "该工具未返回可用结果。请基于错误摘要调整下一步，不要无限重试同一路径。"
    if ok and isinstance(result, dict) and (category in {"conversation_files", "deliverable"} or tool_name.startswith(("conversation_file", "generate_", "save_"))):
        for key in ("conversation_file_id", "file_id", "file_name", "mime_type", "size_bytes", "sha256", "registration_status", "artifact_key", "content_path", "status_path", "drawer_section", "retention_policy", "citation_footnote_count", "task_tree_node_updated", "task_tree_node_id", "reused_existing_file", "plan_role", "plan_revision", "state_authority", "next_action", "not_progress_node", "not_final_deliverable", "task_tree_instruction"):
            if key in result and key not in envelope:
                envelope[key] = result.get(key)
    if ok and isinstance(result, dict) and (category == "task_tree" or tool_name.startswith("task_tree_")):
        for key in ("run_id", "node_id", "parent_node_id", "root_node_id", "root_title", "node_count", "status", "result_summary", "sandbox_job_ids", "depends_on_node_ids", "reused_existing_node", "node_refs", "current_frontier", "suggested_next_action"):
            if key in result and key not in envelope:
                envelope[key] = result.get(key)
    _apply_capability_result_contract(
        tool_name=tool_name,
        category=category,
        raw_result=result,
        envelope=envelope,
        elapsed_ms=int(elapsed_ms or 0),
    )
    return envelope


def wrap_tool_function(
    func: Callable[..., Any],
    *,
    tool_name: str,
    category: str = "",
    resource_scope: ResourceScope | None = None,
    model_result_policy: str = "",
) -> Callable[..., dict[str, Any]]:
    def _wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        started = time.monotonic()
        try:
            raw_result = func(*args, **kwargs)
            elapsed_ms = int((time.monotonic() - started) * 1000)
            return make_tool_result_envelope(
                tool_name,
                raw_result,
                elapsed_ms=elapsed_ms,
                category=category,
                resource_scope=resource_scope,
                model_result_policy=model_result_policy,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = int((time.monotonic() - started) * 1000)
            try:
                from src.capabilities.result_mapper import CapabilityResultContractViolation

                error_kind = (
                    "contract_violation"
                    if isinstance(exc, CapabilityResultContractViolation)
                    else "tool_error"
                )
            except Exception:  # pragma: no cover - capability package may be absent in isolated use
                error_kind = "tool_error"
            return make_tool_result_envelope(
                tool_name,
                {
                    "ok": False,
                    "error_kind": error_kind,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                elapsed_ms=elapsed_ms,
                category=category,
                resource_scope=resource_scope,
                model_result_policy=model_result_policy,
            )

    _wrapped.__name__ = getattr(func, "__name__", f"{tool_name}_enveloped")
    _wrapped.__doc__ = getattr(func, "__doc__", None)
    return _wrapped


def wrap_structured_tool(
    tool: StructuredTool,
    *,
    category: str = "",
    resource_scope: ResourceScope | None = None,
) -> StructuredTool:
    raw_func = getattr(tool, "func", None)
    if raw_func is None:
        return tool
    model_result_policy = str(
        getattr(tool, "_evo_model_result_policy", "") or ""
    ).strip()
    wrapped_func = wrap_tool_function(
        raw_func,
        tool_name=str(tool.name),
        category=category,
        resource_scope=resource_scope,
        model_result_policy=model_result_policy,
    )
    kwargs: dict[str, Any] = {
        "func": wrapped_func,
        "name": str(tool.name),
        "description": str(getattr(tool, "description", "") or ""),
        "args_schema": getattr(tool, "args_schema", None),
    }
    if bool(getattr(tool, "return_direct", False)):
        kwargs["return_direct"] = True
    wrapped_tool = StructuredTool.from_function(**kwargs)
    setattr(wrapped_tool, "_evo_raw_tool", getattr(tool, "_evo_raw_tool", tool))
    setattr(wrapped_tool, "_evo_tool_category", str(category or ""))
    declared_output_schema = getattr(tool, "_evo_output_schema", None)
    if isinstance(declared_output_schema, dict) and declared_output_schema:
        setattr(wrapped_tool, "_evo_output_schema", dict(declared_output_schema))
    if model_result_policy:
        setattr(wrapped_tool, "_evo_model_result_policy", model_result_policy)
    return wrapped_tool


def wrap_structured_tools(
    tools: list[StructuredTool],
    *,
    category: str = "",
    resource_scope: ResourceScope | None = None,
) -> list[StructuredTool]:
    return [
        wrap_structured_tool(
            tool,
            category=category,
            resource_scope=resource_scope,
        )
        for tool in tools
    ]
