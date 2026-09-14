from __future__ import annotations

import hashlib
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi import HTTPException
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.schema import CreateIndex
from sqlmodel import SQLModel, Session, create_engine, select

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import database  # noqa: E402
from routers import task_tree as task_tree_router  # noqa: E402
from services import task_tree_service  # noqa: E402


TASK_TREE_FILE_TABLES = [
    database.ProjectTaskTreeRun.__table__,
    database.ProjectTaskTreeNode.__table__,
    database.ConversationFile.__table__,
]


def _server_registered_artifact_json(
    digest: str,
    *,
    artifact_key: str | None = None,
    path: str | None = None,
) -> dict:
    artifact_metadata = {"sha256": digest}
    if artifact_key is not None:
        artifact_metadata["artifact_key"] = artifact_key
    if path is not None:
        artifact_metadata["path"] = path
        artifact_metadata["display_path"] = path
    return {
        "artifact_metadata": artifact_metadata,
        "_conversation_file_registration": {
            "schema_version": "evoengine.conversation-file-registration/v1",
            "sha256": digest,
            "status": "registered",
        },
    }


@pytest.fixture()
def db_session(tmp_path: Path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'task-tree-frozen-p0.db'}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine, tables=TASK_TREE_FILE_TABLES)
    with Session(engine) as session:
        yield session
    SQLModel.metadata.drop_all(engine, tables=TASK_TREE_FILE_TABLES)
    engine.dispose()


def _create_begun_action(db: Session) -> tuple[dict, dict, str, str]:
    snapshot = task_tree_service.create_task_root(
        db,
        user_id=41,
        project_id="proj-frozen-p0",
        session_id="conv-frozen-p0",
        request_id="req-frozen-p0",
        title="生成严格绑定的结果文件",
    )
    snapshot = task_tree_service.add_task_node(
        db,
        run_id=snapshot["run_id"],
        user_id=41,
        project_id="proj-frozen-p0",
        session_id="conv-frozen-p0",
        parent_node_id=snapshot["root"]["node_id"],
        title="执行产出能力",
        node_kind="biology_tool",
        tool_name="skill_result_run",
        capability_id="skill.result.run",
        capability_version="1.0.0",
        required=True,
    )
    node = snapshot["root"]["children"][0]
    call_id = "call-frozen-p0"
    idempotency_key = "idem-frozen-p0"
    task_tree_service.begin_node_capability(
        db,
        run_id=snapshot["run_id"],
        node_id=node["node_id"],
        call_id=call_id,
        idempotency_key=idempotency_key,
        tool_name="skill_result_run",
        capability_id="skill.result.run",
        capability_version="1.0.0",
    )
    return snapshot, node, call_id, idempotency_key


def _record_success(
    db: Session,
    *,
    snapshot: dict,
    node: dict,
    call_id: str,
    idempotency_key: str,
    conversation_file_id: int,
    artifact_key: str | None = "rd_same_subject_matrix",
    drawer_section: str = "result_file",
) -> dict:
    artifact = {
        "conversation_file_id": conversation_file_id,
        "drawer_section": drawer_section,
        "registration_status": "registered",
    }
    if artifact_key is not None:
        artifact["artifact_key"] = artifact_key
    return task_tree_service.record_node_capability_receipt(
        db,
        run_id=snapshot["run_id"],
        node_id=node["node_id"],
        call_id=call_id,
        idempotency_key=idempotency_key,
        tool_name="skill_result_run",
        capability_id="skill.result.run",
        capability_version="1.0.0",
        status="succeeded",
        summary="结果文件已登记",
        artifacts=[artifact],
        receipt={
            "schema_version": "capability_call_state_v1",
            "request_id": snapshot["request_id"],
            "transport_request_id": snapshot["request_id"],
            "subject_refs": [
                {"kind": "candidate", "id": "candidate_A", "sha256": "a" * 64}
            ]
        },
    )


def test_request_scoped_snapshot_prefers_sealed_authority_over_later_recovery_run(
    db_session: Session,
) -> None:
    first = task_tree_service.create_task_root(
        db_session,
        user_id=41,
        project_id="proj-frozen-p0",
        session_id="conv-frozen-p0",
        request_id="req-sealed-replay",
        title="sealed result",
    )
    sealed = db_session.exec(
        select(database.ProjectTaskTreeRun).where(
            database.ProjectTaskTreeRun.run_id == first["run_id"]
        )
    ).one()
    sealed.status = "blocked"
    sealed.status_reason = "authoritative blocked result"
    sealed.completion_receipt_json = {
        "schema_version": "evoengine.task-completion-receipt/v1",
        "authority": "task_tree_internal_api",
        "receipt_id": "completion_sealed",
    }
    sealed.updated_at = datetime.now(UTC) - timedelta(minutes=1)
    db_session.add(sealed)
    db_session.commit()

    accidental = task_tree_service.create_task_root(
        db_session,
        user_id=41,
        project_id="proj-frozen-p0",
        session_id="conv-frozen-p0",
        request_id="req-sealed-replay",
        title="later accidental recovery",
    )
    task_tree_service.update_mission_status(
        db_session,
        run_id=accidental["run_id"],
        mission_status="cancelled",
        status_reason="cancelled accidental recovery",
    )

    replay = task_tree_service.get_latest_snapshot(
        db_session,
        user_id=41,
        project_id="proj-frozen-p0",
        session_id="conv-frozen-p0",
        request_id="req-sealed-replay",
    )

    assert replay is not None
    assert replay["run_id"] == first["run_id"]
    assert replay["mission_status"] == "blocked"
    assert replay["completion_receipt"]["receipt_id"] == "completion_sealed"


def test_receipt_requires_pre_registered_exact_node_and_begin_lineage(
    db_session: Session,
) -> None:
    snapshot, node, call_id, idempotency_key = _create_begun_action(db_session)
    file_sha256 = hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    file_row = database.ConversationFile(
        user_id=41,
        project_id="proj-frozen-p0",
        conversation_id="conv-frozen-p0",
        request_id="req-frozen-p0",
        source_type="agent_generated",
        drawer_section="result_file",
        file_name="result.csv",
        file_ext="csv",
        mime_type="text/csv",
        size_bytes=8,
        parsed_json=_server_registered_artifact_json(file_sha256),
        content_text="a,b\n1,2\n",
        task_node_id=None,
        tool_name="skill_result_run",
        tool_run_id=None,
    )
    db_session.add(file_row)
    db_session.commit()
    db_session.refresh(file_row)
    assert file_row.id is not None

    with pytest.raises(task_tree_service.TaskTreeContractError) as unbound:
        _record_success(
            db_session,
            snapshot=snapshot,
            node=node,
            call_id=call_id,
            idempotency_key=idempotency_key,
            conversation_file_id=int(file_row.id),
        )
    assert unbound.value.code == "conversation_file_lineage_mismatch"
    db_session.refresh(file_row)
    assert file_row.task_node_id is None
    assert file_row.tool_run_id is None

    file_row.task_node_id = node["node_id"]
    file_row.tool_run_id = "call-wrong"
    db_session.add(file_row)
    db_session.commit()
    with pytest.raises(task_tree_service.TaskTreeContractError) as wrong_call:
        _record_success(
            db_session,
            snapshot=snapshot,
            node=node,
            call_id=call_id,
            idempotency_key=idempotency_key,
            conversation_file_id=int(file_row.id),
        )
    assert wrong_call.value.code == "conversation_file_tool_run_mismatch"

    file_row.tool_run_id = call_id
    db_session.add(file_row)
    db_session.commit()
    completed = _record_success(
        db_session,
        snapshot=snapshot,
        node=node,
        call_id=call_id,
        idempotency_key=idempotency_key,
        conversation_file_id=int(file_row.id),
    )
    assert completed["root"]["children"][0]["status"] == "completed"
    receipt_ref = completed["root"]["children"][0]["capability_receipt_ref"]
    assert receipt_ref["schema_version"] == "evoengine.capability-receipt-ref/v1"
    assert receipt_ref["call_id"] == call_id
    assert receipt_ref["status"] == "succeeded"
    assert receipt_ref["subject_refs"] == [
        {"kind": "candidate", "id": "candidate_A", "sha256": "a" * 64}
    ]
    assert receipt_ref["artifact_refs"] == [
        {
            "artifact_key": "rd_same_subject_matrix",
            "conversation_file_id": str(file_row.id),
            "file_name": "result.csv",
            "mime_type": "text/csv",
            "size_bytes": 8,
            "sha256": hashlib.sha256(b"a,b\n1,2\n").hexdigest(),
            "drawer_section": "result_file",
            "registration_status": "registered",
        }
    ]
    finalized = task_tree_service.complete_task_root(
        db_session,
        run_id=snapshot["run_id"],
        root_node_id=snapshot["root"]["node_id"],
        result_summary="严格文件 lineage 已验收",
    )
    assert finalized["completion_receipt"]["actions"][0]["artifact_refs"] == [
        {
            "artifact_key": "rd_same_subject_matrix",
            "conversation_file_id": str(file_row.id),
            "drawer_section": "result_file",
            "sha256": hashlib.sha256(b"a,b\n1,2\n").hexdigest(),
        }
    ]
    assert finalized["completion_receipt"]["deliverables"] == [
        {
            "conversation_file_id": int(file_row.id),
            "artifact_key": "rd_same_subject_matrix",
            "file_name": "result.csv",
            "mime_type": "text/csv",
            "size_bytes": 8,
            "sha256": hashlib.sha256(b"a,b\n1,2\n").hexdigest(),
            "drawer_section": "result_file",
            "task_node_id": node["node_id"],
            "tool_name": "skill_result_run",
            "tool_run_id": call_id,
        }
    ]


def test_completion_adopts_explicit_late_registered_file_by_exact_lineage(
    db_session: Session,
) -> None:
    snapshot, node, call_id, idempotency_key = _create_begun_action(db_session)
    historical_bytes = b"historical\n"
    historical_sha = hashlib.sha256(historical_bytes).hexdigest()
    historical = database.ConversationFile(
        user_id=41,
        project_id="proj-frozen-p0",
        conversation_id="conv-frozen-p0",
        request_id="req-frozen-p0",
        source_type="artifact_generated",
        drawer_section="result_file",
        file_name="historical.csv",
        file_ext="csv",
        mime_type="text/csv",
        size_bytes=len(historical_bytes),
        parsed_json=_server_registered_artifact_json(
            historical_sha,
            artifact_key="historical_result",
            path="outputs/historical.csv",
        ),
        content_text=historical_bytes.decode("utf-8"),
        task_node_id=node["node_id"],
        tool_name="skill_result_run",
        tool_run_id=call_id,
    )
    db_session.add(historical)
    db_session.commit()
    db_session.refresh(historical)
    assert historical.id is not None
    _record_success(
        db_session,
        snapshot=snapshot,
        node=node,
        call_id=call_id,
        idempotency_key=idempotency_key,
        conversation_file_id=int(historical.id),
        artifact_key="historical_result",
    )

    late_bytes = b"candidate_id,score\nA,1\n"
    late_sha = hashlib.sha256(late_bytes).hexdigest()
    late = database.ConversationFile(
        user_id=41,
        project_id="proj-frozen-p0",
        conversation_id="conv-frozen-p0",
        request_id="req-frozen-p0",
        source_type="artifact_generated",
        drawer_section="result_file",
        file_name="candidate_ranking.csv",
        file_ext="csv",
        mime_type="text/csv",
        size_bytes=len(late_bytes),
        parsed_json=_server_registered_artifact_json(
            late_sha,
            path="outputs/candidate_ranking.csv",
        ),
        content_text=late_bytes.decode("utf-8"),
        task_node_id=node["node_id"],
        tool_name="skill_result_run",
        tool_run_id=call_id,
    )
    db_session.add(late)
    db_session.commit()
    db_session.refresh(late)
    assert late.id is not None

    finalized = task_tree_service.complete_task_root(
        db_session,
        run_id=snapshot["run_id"],
        root_node_id=snapshot["root"]["node_id"],
        result_summary="仅交付显式选择的晚登记结果",
        deliverable_file_ids=[int(late.id)],
    )
    receipt = finalized["completion_receipt"]
    assert [item["conversation_file_id"] for item in receipt["deliverables"]] == [
        int(late.id)
    ]
    assert receipt["deliverables"][0]["artifact_key"] == (
        "outputs/candidate_ranking.csv"
    )
    assert {item["conversation_file_id"] for item in receipt["actions"][0]["artifact_refs"]} == {
        str(historical.id),
        str(late.id),
    }


def test_completion_rejects_late_registered_file_with_wrong_begin_lineage(
    db_session: Session,
) -> None:
    snapshot, node, call_id, idempotency_key = _create_begun_action(db_session)
    task_tree_service.record_node_capability_receipt(
        db_session,
        run_id=snapshot["run_id"],
        node_id=node["node_id"],
        call_id=call_id,
        idempotency_key=idempotency_key,
        tool_name="skill_result_run",
        capability_id="skill.result.run",
        capability_version="1.0.0",
        status="succeeded",
        summary="动作先完成",
        artifacts=[],
        receipt={
            "schema_version": "capability_call_state_v1",
            "request_id": snapshot["request_id"],
            "transport_request_id": snapshot["request_id"],
        },
    )
    file_bytes = b"late\n"
    file_sha = hashlib.sha256(file_bytes).hexdigest()
    late = database.ConversationFile(
        user_id=41,
        project_id="proj-frozen-p0",
        conversation_id="conv-frozen-p0",
        request_id="req-frozen-p0",
        source_type="artifact_generated",
        drawer_section="result_file",
        file_name="late.csv",
        file_ext="csv",
        mime_type="text/csv",
        size_bytes=len(file_bytes),
        parsed_json=_server_registered_artifact_json(
            file_sha,
            artifact_key="late_result",
            path="outputs/late.csv",
        ),
        content_text=file_bytes.decode("utf-8"),
        task_node_id=node["node_id"],
        tool_name="skill_result_run",
        tool_run_id="call-not-the-authoritative-begin",
    )
    db_session.add(late)
    db_session.commit()
    db_session.refresh(late)

    with pytest.raises(task_tree_service.TaskTreeContractError) as rejected:
        task_tree_service.complete_task_root(
            db_session,
            run_id=snapshot["run_id"],
            root_node_id=snapshot["root"]["node_id"],
            result_summary="错误 begin lineage 不得被接纳",
            deliverable_file_ids=[int(late.id)],
        )
    assert rejected.value.code == "completion_deliverable_selection_invalid"


def test_completion_does_not_promote_temporary_outputs(
    db_session: Session,
) -> None:
    snapshot, node, call_id, idempotency_key = _create_begun_action(db_session)
    file_bytes = b"candidate_id,score\nA,1\n"
    file_sha256 = hashlib.sha256(file_bytes).hexdigest()
    file_row = database.ConversationFile(
        user_id=41,
        project_id="proj-frozen-p0",
        conversation_id="conv-frozen-p0",
        request_id="req-frozen-p0",
        source_type="agent_generated",
        drawer_section="temporary_output",
        file_name="candidate_pool.csv",
        file_ext="csv",
        mime_type="text/csv",
        size_bytes=len(file_bytes),
        parsed_json=_server_registered_artifact_json(file_sha256),
        content_text=file_bytes.decode("utf-8"),
        task_node_id=node["node_id"],
        tool_name="skill_result_run",
        tool_run_id=call_id,
    )
    db_session.add(file_row)
    db_session.commit()
    db_session.refresh(file_row)
    assert file_row.id is not None

    _record_success(
        db_session,
        snapshot=snapshot,
        node=node,
        call_id=call_id,
        idempotency_key=idempotency_key,
        conversation_file_id=int(file_row.id),
        artifact_key="candidate_pool",
        drawer_section="temporary_output",
    )
    with pytest.raises(task_tree_service.TaskTreeContractError) as rejected:
        task_tree_service.complete_task_root(
            db_session,
            run_id=snapshot["run_id"],
            root_node_id=snapshot["root"]["node_id"],
            result_summary="不得在完成阶段改变文件状态",
            deliverable_file_ids=[int(file_row.id)],
        )

    db_session.refresh(file_row)
    assert rejected.value.code == "completion_deliverable_selection_invalid"
    assert file_row.drawer_section == "temporary_output"


def test_optional_unstarted_attempt_does_not_block_completion(
    db_session: Session,
) -> None:
    snapshot = task_tree_service.create_task_root(
        db_session,
        user_id=41,
        project_id="proj-optional-attempt",
        session_id="conv-optional-attempt",
        request_id="req-optional-attempt",
        title="可选尝试不是完成义务",
    )
    task_tree_service.add_task_node(
        db_session,
        run_id=snapshot["run_id"],
        user_id=41,
        project_id="proj-optional-attempt",
        session_id="conv-optional-attempt",
        parent_node_id=snapshot["root"]["node_id"],
        title="尚未执行的展示尝试",
        node_kind="biology_tool",
        tool_name="skill_optional_run",
        capability_id="skill.optional.run",
        capability_version="1.0.0",
        required=False,
    )

    completed = task_tree_service.complete_task_root(
        db_session,
        run_id=snapshot["run_id"],
        root_node_id=snapshot["root"]["node_id"],
        result_summary="没有未满足的必要义务",
    )

    assert completed["mission_status"] == "completed"
    assert completed["completion_receipt"]["actions"] == []


def _capability_outcome(
    *,
    evidence: str,
    coverage: str,
    applicability: str = "applicable",
    recovery: str = "unknown",
) -> dict:
    return {
        "schema_version": "evoengine.capability-outcome/v1",
        "applicability": applicability,
        "applicability_basis": "validated provider input scope",
        "evidence": evidence,
        "evidence_basis": "deterministic provider result",
        "coverage": coverage,
        "coverage_scope": (
            "the exact bounded provider query" if coverage == "exhausted" else ""
        ),
        "coverage_basis": (
            "the exact bounded provider query was exhausted"
            if coverage != "unknown"
            else ""
        ),
        "recovery": recovery,
        "recovery_basis": (
            "provider recovery guidance" if recovery != "unknown" else ""
        ),
    }


def test_negative_result_skips_dependent_and_still_allows_complete_delivery(
    db_session: Session,
) -> None:
    snapshot = task_tree_service.create_task_root(
        db_session,
        user_id=41,
        project_id="proj-negative-resolution",
        session_id="conv-negative-resolution",
        request_id="req-negative-resolution",
        title="允许阴性科研结论完整交付",
    )
    snapshot = task_tree_service.add_task_node(
        db_session,
        run_id=snapshot["run_id"],
        user_id=41,
        project_id="proj-negative-resolution",
        session_id="conv-negative-resolution",
        parent_node_id=snapshot["root"]["node_id"],
        title="检索候选",
        node_kind="biology_tool",
        tool_name="skill_candidate_search",
        capability_id="skill.candidate.search",
        capability_version="1.0.0",
        required=True,
    )
    source = snapshot["root"]["children"][0]
    snapshot = task_tree_service.add_task_node(
        db_session,
        run_id=snapshot["run_id"],
        user_id=41,
        project_id="proj-negative-resolution",
        session_id="conv-negative-resolution",
        parent_node_id=snapshot["root"]["node_id"],
        title="分析候选结构",
        node_kind="biology_tool",
        tool_name="skill_candidate_structure",
        capability_id="skill.candidate.structure",
        capability_version="1.0.0",
        required=True,
        depends_on_node_ids=[source["node_id"]],
        dependency_requirement="achieved",
    )
    dependent = next(
        item
        for item in snapshot["root"]["children"]
        if item["title"] == "分析候选结构"
    )
    task_tree_service.begin_node_capability(
        db_session,
        run_id=snapshot["run_id"],
        node_id=source["node_id"],
        call_id="call-negative-source",
        idempotency_key="idem-negative-source",
        tool_name="skill_candidate_search",
        capability_id="skill.candidate.search",
        capability_version="1.0.0",
    )
    resolved = task_tree_service.record_node_capability_receipt(
        db_session,
        run_id=snapshot["run_id"],
        node_id=source["node_id"],
        call_id="call-negative-source",
        idempotency_key="idem-negative-source",
        tool_name="skill_candidate_search",
        capability_id="skill.candidate.search",
        capability_version="1.0.0",
        status="succeeded",
        summary="检索正常完成，在声明范围内未找到候选",
        receipt={
            "schema_version": "capability_call_state_v1",
            "request_id": snapshot["request_id"],
            "transport_request_id": snapshot["request_id"],
            "capability_outcome": _capability_outcome(
                evidence="none",
                coverage="exhausted",
                recovery="do_not_retry",
            ),
        },
    )
    resolved_nodes = {
        item["node_id"]: item for item in resolved["root"]["children"]
    }
    assert resolved_nodes[source["node_id"]]["status"] == "completed"
    assert resolved_nodes[source["node_id"]]["resolution_outcome"] == "negative"
    assert resolved_nodes[dependent["node_id"]]["status"] == "skipped"
    assert resolved_nodes[dependent["node_id"]]["resolution_outcome"] == "skipped"
    assert (
        resolved_nodes[dependent["node_id"]]["resolution_reason_code"]
        == "upstream_not_achieved"
    )

    completed = task_tree_service.complete_task_root(
        db_session,
        run_id=snapshot["run_id"],
        root_node_id=snapshot["root"]["node_id"],
        result_summary="检索与验收流程已完成；声明范围内没有可进入结构分析的候选。",
    )
    receipt = completed["completion_receipt"]
    assert receipt["terminal_status"] == "completed"
    assert receipt["goal_outcome"] == "not_achieved"
    assert {item["outcome"] for item in receipt["obligation_resolutions"]} == {
        "negative",
        "skipped",
    }
    assert receipt["actions"][0]["resolution_outcome"] == "negative"


def test_negative_result_does_not_skip_ordering_only_report(
    db_session: Session,
) -> None:
    snapshot = task_tree_service.create_task_root(
        db_session,
        user_id=41,
        project_id="proj-negative-report",
        session_id="conv-negative-report",
        request_id="req-negative-report",
        title="阴性结果仍需进入报告",
    )
    snapshot = task_tree_service.add_task_node(
        db_session,
        run_id=snapshot["run_id"],
        user_id=41,
        project_id="proj-negative-report",
        session_id="conv-negative-report",
        parent_node_id=snapshot["root"]["node_id"],
        title="检索候选",
        node_kind="biology_tool",
        tool_name="skill_candidate_search",
        capability_id="skill.candidate.search",
        capability_version="1.0.0",
        required=True,
    )
    source = snapshot["root"]["children"][0]
    snapshot = task_tree_service.add_task_node(
        db_session,
        run_id=snapshot["run_id"],
        user_id=41,
        project_id="proj-negative-report",
        session_id="conv-negative-report",
        parent_node_id=snapshot["root"]["node_id"],
        title="生成阴性报告",
        node_kind="biology_tool",
        tool_name="skill_report",
        capability_id="skill.report",
        capability_version="1.0.0",
        required=True,
        depends_on_node_ids=[source["node_id"]],
        dependency_requirement="completed",
    )
    report = next(
        item for item in snapshot["root"]["children"] if item["title"] == "生成阴性报告"
    )
    task_tree_service.begin_node_capability(
        db_session,
        run_id=snapshot["run_id"],
        node_id=source["node_id"],
        call_id="call-negative-report-source",
        idempotency_key="idem-negative-report-source",
        tool_name="skill_candidate_search",
        capability_id="skill.candidate.search",
        capability_version="1.0.0",
    )
    resolved = task_tree_service.record_node_capability_receipt(
        db_session,
        run_id=snapshot["run_id"],
        node_id=source["node_id"],
        call_id="call-negative-report-source",
        idempotency_key="idem-negative-report-source",
        tool_name="skill_candidate_search",
        capability_id="skill.candidate.search",
        capability_version="1.0.0",
        status="succeeded",
        receipt={
            "schema_version": "capability_call_state_v1",
            "request_id": snapshot["request_id"],
            "transport_request_id": snapshot["request_id"],
            "capability_outcome": _capability_outcome(
                evidence="none",
                coverage="exhausted",
                recovery="do_not_retry",
            ),
        },
    )
    report_after = next(
        item for item in resolved["root"]["children"] if item["node_id"] == report["node_id"]
    )
    assert report_after["status"] == "pending"
    assert report_after["resolution_outcome"] is None


def test_failed_required_action_allows_blocked_delivery_but_not_completed(
    db_session: Session,
) -> None:
    snapshot, node, call_id, idempotency_key = _create_begun_action(db_session)
    task_tree_service.record_node_capability_receipt(
        db_session,
        run_id=snapshot["run_id"],
        node_id=node["node_id"],
        call_id=call_id,
        idempotency_key=idempotency_key,
        tool_name="skill_result_run",
        capability_id="skill.result.run",
        capability_version="1.0.0",
        status="failed",
        summary="Provider 暂不可用，安全重试已耗尽",
        receipt={
            "schema_version": "capability_call_state_v1",
            "request_id": snapshot["request_id"],
            "transport_request_id": snapshot["request_id"],
            "error_kind": "provider_unavailable",
            "error_retryable": False,
        },
    )

    with pytest.raises(task_tree_service.TaskTreeContractError) as completed:
        task_tree_service.complete_task_root(
            db_session,
            run_id=snapshot["run_id"],
            root_node_id=snapshot["root"]["node_id"],
            result_summary="不能把工具失败冒充正常完成。",
        )
    assert completed.value.code == "required_capability_receipts_missing"

    blocked = task_tree_service.complete_task_root(
        db_session,
        run_id=snapshot["run_id"],
        root_node_id=snapshot["root"]["node_id"],
        result_summary=(
            "已交付当前可得结论；Provider 暂不可用，无法完成剩余分析。"
        ),
        terminal_status="blocked",
    )
    receipt = blocked["completion_receipt"]
    assert blocked["mission_status"] == "blocked"
    assert blocked["root"]["status"] == "partial"
    assert receipt["terminal_status"] == "blocked"
    assert receipt["goal_outcome"] == "not_achieved"
    assert receipt["obligation_resolutions"] == [
        {
            "obligation_id": node["execution_contract"]["obligation_id"],
            "node_id": node["node_id"],
            "outcome": "blocked",
            "source": "capability_receipt",
            "call_id": call_id,
            "reason_code": "provider_unavailable",
        }
    ]


def test_inconclusive_success_resolves_obligation_without_claiming_goal_achieved(
    db_session: Session,
) -> None:
    snapshot, node, call_id, idempotency_key = _create_begun_action(db_session)
    task_tree_service.record_node_capability_receipt(
        db_session,
        run_id=snapshot["run_id"],
        node_id=node["node_id"],
        call_id=call_id,
        idempotency_key=idempotency_key,
        tool_name="skill_result_run",
        capability_id="skill.result.run",
        capability_version="1.0.0",
        status="succeeded",
        summary="分析已完成，但现有证据不足以支持阳性结论",
        receipt={
            "schema_version": "capability_call_state_v1",
            "request_id": snapshot["request_id"],
            "transport_request_id": snapshot["request_id"],
            "capability_outcome": _capability_outcome(
                evidence="partial",
                coverage="exhausted",
                recovery="refine_input",
            ),
        },
    )

    completed = task_tree_service.complete_task_root(
        db_session,
        run_id=snapshot["run_id"],
        root_node_id=snapshot["root"]["node_id"],
        result_summary="分析完整结束，结论为证据不足。",
    )

    receipt = completed["completion_receipt"]
    assert receipt["goal_outcome"] == "not_achieved"
    assert receipt["obligation_resolutions"] == [
        {
            "obligation_id": node["execution_contract"]["obligation_id"],
            "node_id": node["node_id"],
            "outcome": "inconclusive",
            "source": "capability_receipt",
            "call_id": call_id,
        }
    ]


def test_server_rejects_inconsistent_capability_outcome(
    db_session: Session,
) -> None:
    snapshot, node, call_id, idempotency_key = _create_begun_action(db_session)
    invalid = _capability_outcome(
        applicability="not_applicable",
        evidence="available",
        coverage="unknown",
    )
    with pytest.raises(task_tree_service.TaskTreeContractError) as rejected:
        task_tree_service.record_node_capability_receipt(
            db_session,
            run_id=snapshot["run_id"],
            node_id=node["node_id"],
            call_id=call_id,
            idempotency_key=idempotency_key,
            tool_name="skill_result_run",
            capability_id="skill.result.run",
            capability_version="1.0.0",
            status="succeeded",
            receipt={
                "schema_version": "capability_call_state_v1",
                "request_id": snapshot["request_id"],
                "transport_request_id": snapshot["request_id"],
                "capability_outcome": invalid,
            },
        )
    assert rejected.value.code == "capability_outcome_invalid"


def test_completion_rejects_unowned_explicit_deliverable(
    db_session: Session,
) -> None:
    snapshot, node, call_id, idempotency_key = _create_begun_action(db_session)
    task_tree_service.record_node_capability_receipt(
        db_session,
        run_id=snapshot["run_id"],
        node_id=node["node_id"],
        call_id=call_id,
        idempotency_key=idempotency_key,
        tool_name="skill_result_run",
        capability_id="skill.result.run",
        capability_version="1.0.0",
        status="succeeded",
        summary="无文件结果",
        artifacts=[],
        receipt={
            "schema_version": "capability_call_state_v1",
            "request_id": snapshot["request_id"],
            "transport_request_id": snapshot["request_id"],
        },
    )

    with pytest.raises(task_tree_service.TaskTreeContractError) as rejected:
        task_tree_service.complete_task_root(
            db_session,
            run_id=snapshot["run_id"],
            root_node_id=snapshot["root"]["node_id"],
            result_summary="不得提升无权威归属的文件",
            deliverable_file_ids=[999999],
        )
    assert rejected.value.code == "completion_deliverable_selection_invalid"


def test_completion_rejects_result_file_without_artifact_key(
    db_session: Session,
) -> None:
    snapshot, node, call_id, idempotency_key = _create_begun_action(db_session)
    file_bytes = b"a,b\n1,2\n"
    file_row = database.ConversationFile(
        user_id=41,
        project_id="proj-frozen-p0",
        conversation_id="conv-frozen-p0",
        request_id="req-frozen-p0",
        source_type="agent_generated",
        drawer_section="result_file",
        file_name="result.csv",
        file_ext="csv",
        mime_type="text/csv",
        size_bytes=len(file_bytes),
        parsed_json=_server_registered_artifact_json(
            hashlib.sha256(file_bytes).hexdigest()
        ),
        content_text=file_bytes.decode("utf-8"),
        task_node_id=node["node_id"],
        tool_name="skill_result_run",
        tool_run_id=call_id,
    )
    db_session.add(file_row)
    db_session.commit()
    db_session.refresh(file_row)
    assert file_row.id is not None

    _record_success(
        db_session,
        snapshot=snapshot,
        node=node,
        call_id=call_id,
        idempotency_key=idempotency_key,
        conversation_file_id=int(file_row.id),
        artifact_key=None,
    )

    with pytest.raises(task_tree_service.TaskTreeContractError) as missing_key:
        task_tree_service.complete_task_root(
            db_session,
            run_id=snapshot["run_id"],
            root_node_id=snapshot["root"]["node_id"],
            result_summary="不得生成缺少交付语义身份的 CompletionReceipt",
        )
    assert missing_key.value.code == "completion_deliverable_artifact_key_missing"


def test_completion_replay_requires_exact_summary_and_router_returns_409(
    db_session: Session,
) -> None:
    snapshot = task_tree_service.create_task_root(
        db_session,
        user_id=52,
        project_id="proj-summary-replay",
        session_id="conv-summary-replay",
        request_id="req-summary-replay",
        title="冻结完成摘要",
    )
    completed = task_tree_service.complete_task_root(
        db_session,
        run_id=snapshot["run_id"],
        root_node_id=snapshot["root"]["node_id"],
        result_summary="首次权威摘要",
    )
    replayed = task_tree_service.complete_task_root(
        db_session,
        run_id=snapshot["run_id"],
        root_node_id=snapshot["root"]["node_id"],
        result_summary="首次权威摘要",
    )
    assert replayed["completion_receipt"] == completed["completion_receipt"]

    with pytest.raises(task_tree_service.TaskTreeContractError) as conflict:
        task_tree_service.complete_task_root(
            db_session,
            run_id=snapshot["run_id"],
            root_node_id=snapshot["root"]["node_id"],
            result_summary="不同的重放摘要",
        )
    assert conflict.value.code == "completion_summary_conflict"
    db_session.rollback()

    with pytest.raises(HTTPException) as http_conflict:
        task_tree_router.complete_task_root(
            snapshot["run_id"],
            task_tree_router.CompleteTaskRootReq(
                root_node_id=snapshot["root"]["node_id"],
                result_summary="不同的重放摘要",
            ),
            db_session,
            True,
        )
    assert http_conflict.value.status_code == 409
    assert http_conflict.value.detail["code"] == "completion_summary_conflict"


def test_snapshot_fails_closed_for_root_or_run_completed_outside_finalizer(
    db_session: Session,
) -> None:
    snapshot = task_tree_service.create_task_root(
        db_session,
        user_id=63,
        project_id="proj-fail-closed",
        session_id="conv-fail-closed",
        request_id="req-fail-closed",
        title="拒绝伪造终态",
    )
    root = db_session.exec(
        select(database.ProjectTaskTreeNode).where(
            database.ProjectTaskTreeNode.run_id == snapshot["run_id"],
            database.ProjectTaskTreeNode.node_id == snapshot["root"]["node_id"],
        )
    ).one()
    run = db_session.exec(
        select(database.ProjectTaskTreeRun).where(
            database.ProjectTaskTreeRun.run_id == snapshot["run_id"]
        )
    ).one()
    now = datetime.now(UTC).replace(tzinfo=None)
    root.status = "completed"
    root.completed_at = now
    db_session.add(root)
    db_session.commit()

    with pytest.raises(task_tree_service.TaskTreeContractError) as root_only:
        task_tree_service.build_snapshot(db_session, run_id=snapshot["run_id"])
    assert root_only.value.code == "root_completion_without_finalize"

    run.status = "completed"
    run.status_reason = "伪造完成"
    run.completed_at = now
    db_session.add(run)
    db_session.commit()
    with pytest.raises(task_tree_service.TaskTreeContractError) as missing_receipt:
        task_tree_service.build_snapshot(db_session, run_id=snapshot["run_id"])
    assert missing_receipt.value.code == "completion_receipt_missing"


def test_active_run_unique_index_and_conflict_winner_recovery(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = next(
        item
        for item in database.ProjectTaskTreeRun.__table__.indexes
        if item.name == "uq_project_task_tree_runs_active_session"
    )
    postgres_ddl = str(CreateIndex(index).compile(dialect=postgresql.dialect()))
    sqlite_ddl = str(CreateIndex(index).compile(dialect=sqlite.dialect()))
    assert "CREATE UNIQUE INDEX uq_project_task_tree_runs_active_session" in postgres_ddl
    assert "WHERE status IN" in postgres_ddl
    assert "CREATE UNIQUE INDEX uq_project_task_tree_runs_active_session" in sqlite_ddl
    assert "WHERE status IN" in sqlite_ddl

    first = task_tree_service.create_task_root(
        db_session,
        user_id=74,
        project_id="proj-active-winner",
        session_id="conv-active-winner",
        request_id="req-active-winner-a",
        title="活动树 winner",
    )
    real_find_active_run = task_tree_service._find_active_run
    find_calls = 0

    def simulate_lost_race(*args, **kwargs):
        nonlocal find_calls
        find_calls += 1
        if find_calls == 1:
            return None
        return real_find_active_run(*args, **kwargs)

    monkeypatch.setattr(
        task_tree_service,
        "_find_active_run",
        simulate_lost_race,
    )
    recovered = task_tree_service.create_task_root(
        db_session,
        user_id=74,
        project_id="proj-active-winner",
        session_id="conv-active-winner",
        request_id="req-active-winner-b",
        title="并发 loser 不得新建",
    )
    assert recovered["run_id"] == first["run_id"]
    assert find_calls == 2
    runs = db_session.exec(
        select(database.ProjectTaskTreeRun).where(
            database.ProjectTaskTreeRun.user_id == 74,
            database.ProjectTaskTreeRun.project_id == "proj-active-winner",
            database.ProjectTaskTreeRun.session_id == "conv-active-winner",
        )
    ).all()
    assert len(runs) == 1


def test_postgres_creation_lock_is_scope_stable_and_sqlite_is_noop() -> None:
    class FakeDialect:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeBind:
        def __init__(self, name: str) -> None:
            self.dialect = FakeDialect(name)

    class FakeSession:
        def __init__(self, name: str) -> None:
            self.bind = FakeBind(name)
            self.statements: list[str] = []

        def get_bind(self):
            return self.bind

        def exec(self, statement):
            self.statements.append(str(statement))

    postgres_session = FakeSession("postgresql")
    task_tree_service._acquire_active_run_lock(
        postgres_session,
        user_id=81,
        project_id="proj-lock",
        session_id="conv-lock",
    )
    assert postgres_session.statements == [
        "SELECT pg_advisory_xact_lock(:lock_key)"
    ]
    lock_key = task_tree_service._active_run_lock_key(
        user_id=81,
        project_id="proj-lock",
        session_id="conv-lock",
    )
    assert lock_key == task_tree_service._active_run_lock_key(
        user_id=81,
        project_id="proj-lock",
        session_id="conv-lock",
    )
    assert lock_key != task_tree_service._active_run_lock_key(
        user_id=81,
        project_id="proj-lock",
        session_id="conv-other",
    )

    sqlite_session = FakeSession("sqlite")
    task_tree_service._acquire_active_run_lock(
        sqlite_session,
        user_id=81,
        project_id="proj-lock",
        session_id="conv-lock",
    )
    assert sqlite_session.statements == []
