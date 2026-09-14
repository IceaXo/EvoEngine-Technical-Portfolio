from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from database import ProjectConversationTurn
from services.turn_authorization_service import (
    TurnAuthorizationError,
    consume_authorized_effect_in_session,
    issue_or_reject_authorization_in_session,
)


def _bundle() -> dict:
    return {
        "bundle_id": "bundle-1",
        "bundle_title": "确认操作",
        "bundle_summary": "",
        "questions": [
            {
                "question_id": "authorize",
                "label": "是否允许？",
                "description": "",
                "kind": "single",
                "required": True,
                "options": [
                    {"option_id": "approve", "label": "允许", "description": ""},
                    {"option_id": "reject", "label": "拒绝", "description": ""},
                ],
                "allow_other": False,
                "other_max_length": 1000,
                "min_select": None,
                "max_select": None,
            }
        ],
    }


def _answer(option: str) -> dict:
    return {
        "bundle_id": "bundle-1",
        "answers": [
            {
                "question_id": "authorize",
                "selected_option_ids": [option],
                "other_text": "",
            }
        ],
        "additional_notes": "",
    }


def _trace() -> dict:
    bundle = _bundle()
    call = {
        "call_id": "call-1",
        "capability_id": "conversation_file.text_candidate.create",
        "capability_version": "1.0.0",
        "request_id": "parent-1",
        "project_id": "project-1",
        "conversation_id": "conversation-1",
        "user_id": 7,
        "arguments": {},
        "input_refs": [],
        "task_node_id": None,
        "context_fingerprint": "",
        "registry_snapshot_id": "",
        "idempotency_key": "idem-1",
        "requested_at": datetime.now(UTC).isoformat(),
    }
    control = {
        "schema_version": "evoengine.capability-needs-input/v1",
        "question_bundle": bundle,
        "pending_call_id": "call-1",
        "capability_id": call["capability_id"],
        "capability_version": "1.0.0",
        "registry_snapshot_id": "",
        "action": "create_text_candidate",
        "target": {
            "schema_version": "evoengine.resource-ref/v1",
            "resource_id": "resource:conversation-file:19",
            "uri": "conversation-file://19",
            "kind": "conversation_file",
            "storage": "conversation_file",
            "media_type": "text/markdown",
            "size_bytes": 10,
            "sha256": "a" * 64,
            "conversation_file_id": 19,
            "metadata": {},
        },
        "target_version_id": 2,
        "target_lock_version": 1,
        "target_sha256": "a" * 64,
        "request_digest": "b" * 64,
        "authorization_challenge": "c" * 64,
        "authorization_question_id": "authorize",
        "approve_option_id": "approve",
        "reject_option_id": "reject",
    }
    return {
        "schema_version": "v1",
        "meta": {"hitl": {"bundle": bundle, "bundle_status": "waiting"}},
        "turn_checkpoint": {
            "schema_version": "evoengine.turn-checkpoint/v1",
            "checkpoint_id": "checkpoint_" + "d" * 24,
            "runtime_control": {
                "pending_capability_authorization": {
                    "schema_version": "evoengine.pending-capability-authorization/v1",
                    "checkpoint_id": "checkpoint_" + "d" * 24,
                    "tool_name": "deliverable_atom",
                    "tool_call_id": "model-call-1",
                    "capability_call": call,
                    "needs_input_control": control,
                }
            },
        },
    }


@pytest.fixture()
def session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine, tables=[ProjectConversationTurn.__table__])
    with Session(engine) as db:
        turn = ProjectConversationTurn(
            project_session_id=3,
            conversation_session_id=4,
            conversation_id="conversation-1",
            user_id=7,
            request_id="parent-1",
            thinking_trace_json=_trace(),
        )
        db.add(turn)
        db.commit()
        yield db


def _turn(db: Session) -> ProjectConversationTurn:
    return db.exec(
        select(ProjectConversationTurn).where(
            ProjectConversationTurn.request_id == "parent-1"
        )
    ).one()


def test_approve_issue_consume_and_replay_one_effect(session: Session) -> None:
    turn = _turn(session)
    resume = issue_or_reject_authorization_in_session(
        turn=turn,
        actor_user_id=7,
        project_id="project-1",
        continuation_request_id="continuation-1",
        answer_payload=_answer("approve"),
    )
    session.add(turn)
    session.commit()
    assert resume["decision"] == "approved"
    grant_id = resume["grant_id"]
    effects: list[int] = []

    def effect(_record):
        effects.append(1)
        return {"candidate_file_id": 31, "request_digest": "b" * 64}

    turn = _turn(session)
    receipt, replay = consume_authorized_effect_in_session(
        turn=turn,
        grant_id=grant_id,
        continuation_request_id="continuation-1",
        pending_call_id="call-1",
        capability_id="conversation_file.text_candidate.create",
        action="create_text_candidate",
        request_digest="b" * 64,
        effect=effect,
    )
    session.add(turn)
    session.commit()
    assert not replay and receipt["candidate_file_id"] == 31
    turn = _turn(session)
    replayed, replay = consume_authorized_effect_in_session(
        turn=turn,
        grant_id=grant_id,
        continuation_request_id="continuation-1",
        pending_call_id="call-1",
        capability_id="conversation_file.text_candidate.create",
        action="create_text_candidate",
        request_digest="b" * 64,
        effect=effect,
    )
    assert replay and replayed == receipt and effects == [1]


def test_effect_binding_is_persisted_into_grant_and_resume(session: Session) -> None:
    turn = _turn(session)
    trace = dict(turn.thinking_trace_json)
    checkpoint = dict(trace["turn_checkpoint"])
    runtime = dict(checkpoint["runtime_control"])
    pending = dict(runtime["pending_capability_authorization"])
    control = dict(pending["needs_input_control"])
    control["effect_capability_id"] = "office.render.execute"
    control["effect_request_digest"] = "e" * 64
    pending["needs_input_control"] = control
    runtime["pending_capability_authorization"] = pending
    checkpoint["runtime_control"] = runtime
    trace["turn_checkpoint"] = checkpoint
    turn.thinking_trace_json = trace

    resume = issue_or_reject_authorization_in_session(
        turn=turn,
        actor_user_id=7,
        project_id="project-1",
        continuation_request_id="continuation-1",
        answer_payload=_answer("approve"),
    )
    assert resume["effect_capability_id"] == "office.render.execute"
    assert resume["effect_request_digest"] == "e" * 64
    record = turn.thinking_trace_json["runtime_control"][
        "capability_authorizations"
    ]["call-1"]
    assert record["effect_capability_id"] == "office.render.execute"
    assert record["effect_request_digest"] == "e" * 64


def test_effect_binding_tamper_is_zero_effect_and_correct_call_replays(
    session: Session,
) -> None:
    turn = _turn(session)
    trace = dict(turn.thinking_trace_json)
    checkpoint = dict(trace["turn_checkpoint"])
    runtime = dict(checkpoint["runtime_control"])
    pending = dict(runtime["pending_capability_authorization"])
    control = dict(pending["needs_input_control"])
    control["effect_capability_id"] = "office.render.execute"
    control["effect_request_digest"] = "e" * 64
    pending["needs_input_control"] = control
    runtime["pending_capability_authorization"] = pending
    checkpoint["runtime_control"] = runtime
    trace["turn_checkpoint"] = checkpoint
    turn.thinking_trace_json = trace
    resume = issue_or_reject_authorization_in_session(
        turn=turn,
        actor_user_id=7,
        project_id="project-1",
        continuation_request_id="continuation-1",
        answer_payload=_answer("approve"),
    )
    effects = []

    with pytest.raises(TurnAuthorizationError) as mismatch:
        consume_authorized_effect_in_session(
            turn=turn,
            grant_id=resume["grant_id"],
            continuation_request_id="continuation-1",
            pending_call_id="call-1",
            capability_id="conversation_file.text_candidate.create",
            action="create_text_candidate",
            request_digest="b" * 64,
            effect_capability_id="office.render.execute",
            effect_request_digest="f" * 64,
            effect=lambda _record: effects.append(1) or {"ok": True},
        )
    assert mismatch.value.code == "CAPABILITY_AUTHORIZATION_MISMATCH"
    assert effects == []

    receipt, replay = consume_authorized_effect_in_session(
        turn=turn,
        grant_id=resume["grant_id"],
        continuation_request_id="continuation-1",
        pending_call_id="call-1",
        capability_id="conversation_file.text_candidate.create",
        action="create_text_candidate",
        request_digest="b" * 64,
        effect_capability_id="office.render.execute",
        effect_request_digest="e" * 64,
        effect=lambda _record: effects.append(1) or {"ok": True},
    )
    replayed, was_replay = consume_authorized_effect_in_session(
        turn=turn,
        grant_id=resume["grant_id"],
        continuation_request_id="continuation-1",
        pending_call_id="call-1",
        capability_id="conversation_file.text_candidate.create",
        action="create_text_candidate",
        request_digest="b" * 64,
        effect_capability_id="office.render.execute",
        effect_request_digest="e" * 64,
        effect=lambda _record: effects.append(1) or {"ok": False},
    )
    assert replay is False and was_replay is True
    assert replayed == receipt == {"ok": True}
    assert effects == [1]


def test_reject_persists_decision_without_grant(session: Session) -> None:
    turn = _turn(session)
    resume = issue_or_reject_authorization_in_session(
        turn=turn,
        actor_user_id=7,
        project_id="project-1",
        continuation_request_id="continuation-1",
        answer_payload=_answer("reject"),
    )
    assert resume == {
        "schema_version": "evoengine.capability-resume-control/v1",
        "parent_request_id": "parent-1",
        "continuation_request_id": "continuation-1",
        "checkpoint_id": "checkpoint_" + "d" * 24,
        "pending_call_id": "call-1",
        "decision": "rejected",
    }
    record = turn.thinking_trace_json["runtime_control"][
        "capability_authorizations"
    ]["call-1"]
    assert record["status"] == "rejected" and record["grant_id"] is None


def test_runtime_task_context_does_not_change_authorization_challenge(
    session: Session,
) -> None:
    turn = _turn(session)
    trace = dict(turn.thinking_trace_json)
    meta = dict(trace["meta"])
    hitl = dict(meta["hitl"])
    hitl["bundle"] = {
        **dict(hitl["bundle"]),
        "task_context": {"phase": "suspended", "missing_fields": ["authorize"]},
    }
    meta["hitl"] = hitl
    trace["meta"] = meta
    turn.thinking_trace_json = trace

    resume = issue_or_reject_authorization_in_session(
        turn=turn,
        actor_user_id=7,
        project_id="project-1",
        continuation_request_id="continuation-task-context",
        answer_payload=_answer("approve"),
    )

    assert resume is not None
    assert resume["decision"] == "approved"


def test_ordinary_hitl_with_task_context_validates_without_issuing_grant(
    session: Session,
) -> None:
    turn = _turn(session)
    trace = dict(turn.thinking_trace_json)
    meta = dict(trace["meta"])
    hitl = dict(meta["hitl"])
    hitl["bundle"] = {
        **dict(hitl["bundle"]),
        "task_context": {"phase": "suspended"},
    }
    meta["hitl"] = hitl
    trace["meta"] = meta
    trace["turn_checkpoint"] = {
        "schema_version": "evoengine.turn-checkpoint/v1",
        "checkpoint_id": "checkpoint_ordinary",
    }
    turn.thinking_trace_json = trace

    resume = issue_or_reject_authorization_in_session(
        turn=turn,
        actor_user_id=7,
        project_id="project-1",
        continuation_request_id="continuation-ordinary",
        answer_payload=_answer("approve"),
    )

    assert resume is None


def test_invalid_answer_does_not_add_runtime_control(session: Session) -> None:
    turn = _turn(session)
    bad = _answer("approve")
    bad["unexpected"] = True
    with pytest.raises(TurnAuthorizationError) as exc:
        issue_or_reject_authorization_in_session(
            turn=turn,
            actor_user_id=7,
            project_id="project-1",
            continuation_request_id="continuation-1",
            answer_payload=bad,
        )
    assert exc.value.code == "HITL_ANSWER_INVALID"
    assert "runtime_control" not in turn.thinking_trace_json


def test_failed_effect_keeps_grant_issued(session: Session) -> None:
    turn = _turn(session)
    resume = issue_or_reject_authorization_in_session(
        turn=turn,
        actor_user_id=7,
        project_id="project-1",
        continuation_request_id="continuation-1",
        answer_payload=_answer("approve"),
    )
    with pytest.raises(RuntimeError):
        consume_authorized_effect_in_session(
            turn=turn,
            grant_id=resume["grant_id"],
            continuation_request_id="continuation-1",
            pending_call_id="call-1",
            capability_id="conversation_file.text_candidate.create",
            action="create_text_candidate",
            request_digest="b" * 64,
            effect=lambda _record: (_ for _ in ()).throw(RuntimeError("boom")),
        )
    record = turn.thinking_trace_json["runtime_control"][
        "capability_authorizations"
    ]["call-1"]
    assert record["status"] == "issued" and record["effect_receipt"] is None
