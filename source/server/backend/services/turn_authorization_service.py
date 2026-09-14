"""Strict HITL validation and one-shot capability authorization on a parent turn."""

from __future__ import annotations

import hashlib
import json
import secrets
import copy
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from database import ProjectConversationTurn


AUTHORIZATION_TTL_SECONDS = 900
OTHER_OPTION_ID = "__other__"


class TurnAuthorizationError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class _QuestionOption(_StrictModel):
    option_id: str = Field(min_length=1, max_length=512)
    label: str = Field(min_length=1, max_length=1000)
    description: str = Field(default="", max_length=4000)


class _Question(_StrictModel):
    question_id: str = Field(min_length=1, max_length=512)
    label: str = Field(min_length=1, max_length=1000)
    description: str = Field(default="", max_length=4000)
    kind: Literal["single", "multi"]
    required: bool = True
    options: list[_QuestionOption] = Field(min_length=1)
    allow_other: bool = False
    other_max_length: int = Field(default=1000, ge=1, le=10000)
    min_select: int | None = Field(default=None, ge=0)
    max_select: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_question(self) -> "_Question":
        option_ids = [item.option_id for item in self.options]
        if len(option_ids) != len(set(option_ids)) or OTHER_OPTION_ID in option_ids:
            raise ValueError("question option IDs must be unique and non-reserved")
        if self.kind == "single":
            if self.min_select not in (None, 0, 1) or self.max_select not in (None, 1):
                raise ValueError("single question has invalid selection bounds")
        elif (
            self.min_select is not None
            and self.max_select is not None
            and self.min_select > self.max_select
        ):
            raise ValueError("multi question has invalid selection bounds")
        return self


class _QuestionBundle(_StrictModel):
    bundle_id: str = Field(min_length=1, max_length=512)
    bundle_title: str = Field(min_length=1, max_length=1000)
    bundle_summary: str = Field(default="", max_length=4000)
    questions: list[_Question] = Field(min_length=1, max_length=5)
    # Runtime appends task progress for UI/resume diagnostics after the strict
    # HumanQuestionBundle has been built.  It is envelope metadata, not part of
    # the authorization challenge and must never affect answer validation.
    task_context: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_ids(self) -> "_QuestionBundle":
        ids = [item.question_id for item in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("question IDs must be unique")
        return self


class _Answer(_StrictModel):
    question_id: str = Field(min_length=1, max_length=512)
    selected_option_ids: list[str] = Field(default_factory=list)
    other_text: str = ""


class _AnswerBundle(_StrictModel):
    bundle_id: str = Field(min_length=1, max_length=512)
    answers: list[_Answer] = Field(default_factory=list, max_length=5)
    additional_notes: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def validate_ids(self) -> "_AnswerBundle":
        ids = [item.question_id for item in self.answers]
        if len(ids) != len(set(ids)):
            raise ValueError("answer question IDs must be unique")
        return self


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def validate_human_answer_bundle(
    question_payload: dict[str, Any],
    answer_payload: dict[str, Any],
) -> tuple[_QuestionBundle, _AnswerBundle]:
    try:
        questions = _QuestionBundle.model_validate(question_payload)
        answers = _AnswerBundle.model_validate(answer_payload)
    except Exception as exc:
        raise TurnAuthorizationError("HITL_ANSWER_INVALID", "人工确认答案格式无效") from exc
    if answers.bundle_id != questions.bundle_id:
        raise TurnAuthorizationError("HITL_BUNDLE_MISMATCH", "问题包与答案包不匹配")
    answer_map = {item.question_id: item for item in answers.answers}
    question_ids = {item.question_id for item in questions.questions}
    if set(answer_map) - question_ids:
        raise TurnAuthorizationError("HITL_ANSWER_INVALID", "答案包含未知问题")
    for question in questions.questions:
        answer = answer_map.get(question.question_id)
        if answer is None:
            if question.required:
                raise TurnAuthorizationError("HITL_ANSWER_INVALID", "缺少必答题答案")
            continue
        selected = [item.strip() for item in answer.selected_option_ids if item.strip()]
        if len(selected) != len(set(selected)):
            raise TurnAuthorizationError("HITL_ANSWER_INVALID", "选项不能重复")
        normal = [item for item in selected if item != OTHER_OPTION_ID]
        allowed = {item.option_id for item in question.options}
        if set(normal) - allowed:
            raise TurnAuthorizationError("HITL_ANSWER_INVALID", "答案包含未知选项")
        if question.kind == "single" and len(selected) > 1:
            raise TurnAuthorizationError("HITL_ANSWER_INVALID", "单选题只能选择一项")
        min_select = (
            question.min_select
            if question.min_select is not None
            else (1 if question.required and question.kind == "multi" else 0)
        )
        if question.kind == "multi" and len(selected) < min_select:
            raise TurnAuthorizationError("HITL_ANSWER_INVALID", "多选题选择数量不足")
        if question.max_select is not None and len(selected) > question.max_select:
            raise TurnAuthorizationError("HITL_ANSWER_INVALID", "选择数量超过限制")
        if question.required and not selected and not answer.other_text:
            raise TurnAuthorizationError("HITL_ANSWER_INVALID", "必答题不能为空")
        if OTHER_OPTION_ID in selected:
            if not question.allow_other or not answer.other_text:
                raise TurnAuthorizationError("HITL_ANSWER_INVALID", "其他选项无效")
        elif answer.other_text:
            raise TurnAuthorizationError("HITL_ANSWER_INVALID", "未选择其他时不能填写补充")
        if len(answer.other_text) > question.other_max_length:
            raise TurnAuthorizationError("HITL_ANSWER_INVALID", "补充内容超过长度限制")
    return questions, answers


def _trace(turn: ProjectConversationTurn) -> dict[str, Any]:
    return (
        copy.deepcopy(turn.thinking_trace_json)
        if isinstance(turn.thinking_trace_json, dict)
        else {"schema_version": "v1"}
    )


def issue_or_reject_authorization_in_session(
    *,
    turn: ProjectConversationTurn,
    actor_user_id: int,
    project_id: str,
    continuation_request_id: str,
    answer_payload: dict[str, Any],
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Validate one answer and update the parent-turn runtime control in-place."""

    trace = _trace(turn)
    meta = trace.get("meta") if isinstance(trace.get("meta"), dict) else {}
    hitl = meta.get("hitl") if isinstance(meta.get("hitl"), dict) else {}
    question_payload = hitl.get("bundle")
    if not isinstance(question_payload, dict):
        raise TurnAuthorizationError("HITL_BUNDLE_MISMATCH", "找不到待确认的问题包")
    question_bundle, answer = validate_human_answer_bundle(
        question_payload,
        answer_payload,
    )
    checkpoint = trace.get("turn_checkpoint")
    if not isinstance(checkpoint, dict):
        return None
    runtime = checkpoint.get("runtime_control")
    pending = (
        runtime.get("pending_capability_authorization")
        if isinstance(runtime, dict)
        else None
    )
    if not isinstance(pending, dict):
        return None
    control = pending.get("needs_input_control")
    call = pending.get("capability_call")
    if not isinstance(control, dict) or not isinstance(call, dict):
        raise TurnAuthorizationError(
            "CAPABILITY_AUTHORIZATION_CHECKPOINT_MISMATCH",
            "能力授权断点无效",
        )
    try:
        control_question_bundle = _QuestionBundle.model_validate(
            control.get("question_bundle")
        )
    except Exception as exc:
        raise TurnAuthorizationError(
            "CAPABILITY_AUTHORIZATION_CHECKPOINT_MISMATCH",
            "能力授权断点问题包无效",
        ) from exc
    question_contract = question_bundle.model_dump(
        mode="json",
        exclude={"task_context"},
    )
    control_question_contract = control_question_bundle.model_dump(
        mode="json",
        exclude={"task_context"},
    )
    if (
        control_question_contract != question_contract
        or str(control.get("pending_call_id") or "") != str(call.get("call_id") or "")
        or str(pending.get("checkpoint_id") or "")
        != str(checkpoint.get("checkpoint_id") or "")
    ):
        raise TurnAuthorizationError(
            "CAPABILITY_AUTHORIZATION_CHECKPOINT_MISMATCH",
            "能力授权断点不匹配",
        )

    runtime_control = (
        dict(trace.get("runtime_control"))
        if isinstance(trace.get("runtime_control"), dict)
        else {}
    )
    records = (
        dict(runtime_control.get("capability_authorizations"))
        if isinstance(runtime_control.get("capability_authorizations"), dict)
        else {}
    )
    pending_call_id = str(call.get("call_id") or "")
    answer_digest = _canonical_digest(answer.model_dump(mode="json"))
    existing = records.get(pending_call_id)
    if isinstance(existing, dict):
        if existing.get("answer_digest") != answer_digest:
            raise TurnAuthorizationError(
                "CAPABILITY_AUTHORIZATION_DECISION_CONFLICT",
                "该授权问题已用不同答案处理",
            )
        return dict(existing.get("resume_control") or {}) or None

    question_id = str(control.get("authorization_question_id") or "")
    approve_option_id = str(control.get("approve_option_id") or "")
    selected = []
    for item in answer.answers:
        if item.question_id == question_id:
            selected = list(item.selected_option_ids)
            break
    approved = selected == [approve_option_id]
    issued_at = now or datetime.now(UTC)
    checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
    grant_id = f"grant_{secrets.token_urlsafe(32)}" if approved else None
    resume_control = {
        "schema_version": "evoengine.capability-resume-control/v1",
        "parent_request_id": str(turn.request_id or ""),
        "continuation_request_id": continuation_request_id,
        "checkpoint_id": checkpoint_id,
        "pending_call_id": pending_call_id,
        "decision": "approved" if approved else "rejected",
        **({"grant_id": grant_id} if grant_id else {}),
    }
    effect_capability_id = str(control.get("effect_capability_id") or "")
    effect_request_digest = str(control.get("effect_request_digest") or "")
    if bool(effect_capability_id) != bool(effect_request_digest):
        raise TurnAuthorizationError(
            "CAPABILITY_AUTHORIZATION_CHECKPOINT_MISMATCH",
            "能力授权 effect 绑定不完整",
        )
    if effect_request_digest:
        resume_control.update(
            effect_capability_id=effect_capability_id,
            effect_request_digest=effect_request_digest,
        )
    record = {
        "schema_version": "evoengine.capability-authorization-grant/v1",
        "grant_id": grant_id,
        "status": "issued" if approved else "rejected",
        "issued_at": issued_at.isoformat(),
        "expires_at": (issued_at + timedelta(seconds=AUTHORIZATION_TTL_SECONDS)).isoformat(),
        "consumed_at": None,
        "actor_user_id": int(actor_user_id),
        "parent_request_id": str(turn.request_id or ""),
        "continuation_request_id": continuation_request_id,
        "project_session_id": int(turn.project_session_id),
        "project_id": project_id,
        "conversation_id": str(turn.conversation_id or ""),
        "bundle_id": answer.bundle_id,
        "checkpoint_id": checkpoint_id,
        "pending_call_id": pending_call_id,
        "capability_id": str(control.get("capability_id") or ""),
        "capability_version": str(control.get("capability_version") or ""),
        "registry_snapshot_id": str(control.get("registry_snapshot_id") or ""),
        "action": str(control.get("action") or ""),
        "target_resource_ref": dict(control.get("target") or {}),
        "target_version_id": control.get("target_version_id"),
        "target_lock_version": control.get("target_lock_version"),
        "target_sha256": control.get("target_sha256"),
        "request_digest": str(control.get("request_digest") or ""),
        "effect_capability_id": effect_capability_id or None,
        "effect_request_digest": effect_request_digest or None,
        "answer_digest": answer_digest,
        "authorization_challenge": str(control.get("authorization_challenge") or ""),
        "effect_receipt": None,
        "resume_control": resume_control,
    }
    records[pending_call_id] = record
    runtime_control["capability_authorizations"] = records
    trace["runtime_control"] = runtime_control
    turn.thinking_trace_json = trace
    return resume_control


def consume_authorized_effect_in_session(
    *,
    turn: ProjectConversationTurn,
    grant_id: str,
    continuation_request_id: str,
    pending_call_id: str,
    capability_id: str,
    action: str,
    request_digest: str,
    effect: Callable[[dict[str, Any]], dict[str, Any]],
    effect_capability_id: str | None = None,
    effect_request_digest: str | None = None,
    now: datetime | None = None,
) -> tuple[dict[str, Any], bool]:
    """Consume a grant and its DB mutation together in the caller's transaction."""

    trace = _trace(turn)
    runtime = trace.get("runtime_control") if isinstance(trace.get("runtime_control"), dict) else {}
    records = runtime.get("capability_authorizations") if isinstance(runtime, dict) else {}
    record = records.get(pending_call_id) if isinstance(records, dict) else None
    if not isinstance(record, dict) or record.get("grant_id") != grant_id:
        raise TurnAuthorizationError("CAPABILITY_AUTHORIZATION_NOT_FOUND", "找不到能力授权")
    expected = {
        "continuation_request_id": continuation_request_id,
        "pending_call_id": pending_call_id,
        "capability_id": capability_id,
        "action": action,
        "request_digest": request_digest,
        "effect_capability_id": effect_capability_id,
        "effect_request_digest": effect_request_digest,
    }
    if any(str(record.get(key) or "") != str(value or "") for key, value in expected.items()):
        raise TurnAuthorizationError("CAPABILITY_AUTHORIZATION_MISMATCH", "能力授权与调用不匹配")
    if record.get("status") == "consumed":
        receipt = record.get("effect_receipt")
        if not isinstance(receipt, dict):
            raise TurnAuthorizationError(
                "CAPABILITY_AUTHORIZATION_ALREADY_CONSUMED",
                "能力授权已经消费",
            )
        return dict(receipt), True
    if record.get("status") != "issued":
        raise TurnAuthorizationError("CAPABILITY_AUTHORIZATION_REQUIRED", "该操作未获授权")
    current = now or datetime.now(UTC)
    try:
        expires_at = datetime.fromisoformat(str(record.get("expires_at") or ""))
    except ValueError as exc:
        raise TurnAuthorizationError("CAPABILITY_AUTHORIZATION_EXPIRED", "能力授权已过期") from exc
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    if current > expires_at:
        raise TurnAuthorizationError("CAPABILITY_AUTHORIZATION_EXPIRED", "能力授权已过期")
    receipt = effect(dict(record))
    if not isinstance(receipt, dict):
        raise TurnAuthorizationError(
            "CAPABILITY_AUTHORIZATION_IDEMPOTENCY_CONFLICT",
            "能力操作未返回可回放结果",
        )
    record["status"] = "consumed"
    record["consumed_at"] = current.isoformat()
    record["effect_receipt"] = dict(receipt)
    records[pending_call_id] = record
    runtime["capability_authorizations"] = records
    trace["runtime_control"] = runtime
    turn.thinking_trace_json = trace
    return dict(receipt), False
