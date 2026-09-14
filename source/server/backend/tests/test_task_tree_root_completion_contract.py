from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from sqlmodel import SQLModel, Session, create_engine, select

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import database  # noqa: E402,F401
from services import task_tree_service  # noqa: E402


TASK_TREE_TABLES = [
    database.ProjectTaskTreeRun.__table__,
    database.ProjectTaskTreeNode.__table__,
]


def _add_action_with_receipt(
    db: Session,
    *,
    run_id: str,
    user_id: int,
    project_id: str,
    session_id: str,
    request_id: str,
    parent_node_id: str,
    title: str,
    receipt_status: str,
) -> dict:
    snapshot = task_tree_service.add_task_node(
        db,
        run_id=run_id,
        user_id=user_id,
        project_id=project_id,
        session_id=session_id,
        parent_node_id=parent_node_id,
        title=title,
        node_kind="biology_tool",
        tool_name="skill_demo_run",
        capability_id="skill.demo.run",
        capability_version="1.0.0",
        required=True,
    )
    node = next(
        item
        for item in snapshot["root"]["children"][0].get("children", [])
        if item["title"] == title
    )
    call_id = f"call-{node['node_id']}"
    idempotency_key = f"idem-{node['node_id']}"
    task_tree_service.begin_node_capability(
        db,
        run_id=run_id,
        node_id=node["node_id"],
        call_id=call_id,
        idempotency_key=idempotency_key,
        tool_name="skill_demo_run",
        capability_id="skill.demo.run",
        capability_version="1.0.0",
    )
    return task_tree_service.record_node_capability_receipt(
        db,
        run_id=run_id,
        node_id=node["node_id"],
        call_id=call_id,
        idempotency_key=idempotency_key,
        tool_name="skill_demo_run",
        capability_id="skill.demo.run",
        capability_version="1.0.0",
        status=receipt_status,
        summary=f"{title}: {receipt_status}",
        receipt={
            "schema_version": "capability_call_state_v1",
            "request_id": request_id,
            "transport_request_id": request_id,
        },
    )


def test_child_rollup_never_completes_root_without_explicit_agent_update(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'task-tree-root-contract.db'}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine, tables=TASK_TREE_TABLES)
    with Session(engine) as db:
        snapshot = task_tree_service.create_task_root(
            db,
            user_id=7,
            project_id="proj-root-contract",
            session_id="conv-root-contract",
            request_id="req-root-contract",
            title="分析序列并形成报告",
        )
        run_id = snapshot["run_id"]
        root_id = snapshot["root"]["node_id"]

        phase_snapshot = task_tree_service.add_task_node(
            db,
            run_id=run_id,
            user_id=7,
            project_id="proj-root-contract",
            session_id="conv-root-contract",
            parent_node_id=root_id,
            title="执行序列分析",
            node_kind="phase",
        )
        phase_id = phase_snapshot["root"]["children"][0]["node_id"]
        child_completed = _add_action_with_receipt(
            db,
            run_id=run_id,
            user_id=7,
            project_id="proj-root-contract",
            session_id="conv-root-contract",
            request_id="req-root-contract",
            parent_node_id=phase_id,
            title="形成已通过门禁的交付文档",
            receipt_status="succeeded",
        )

        assert child_completed["root"]["children"][0]["status"] == "completed"
        assert child_completed["root"]["status"] == "running"
        assert child_completed["mission_status"] == "active"
        root_row = db.exec(
            select(database.ProjectTaskTreeNode).where(
                database.ProjectTaskTreeNode.run_id == run_id,
                database.ProjectTaskTreeNode.node_id == root_id,
            )
        ).one()
        run_row = db.exec(
            select(database.ProjectTaskTreeRun).where(
                database.ProjectTaskTreeRun.run_id == run_id
            )
        ).one()
        assert root_row.status == "running"
        assert root_row.completed_at is None
        assert run_row.status == "active"
        assert run_row.completed_at is None

        with pytest.raises(ValueError, match="task-complete endpoint"):
            task_tree_service.update_task_node(
                db,
                run_id=run_id,
                node_id=root_id,
                patch={"status": "completed", "result_summary": "绕过完成门禁"},
            )
        unchanged = task_tree_service.build_snapshot(db, run_id=run_id)
        assert unchanged["root"]["status"] == "running"
        assert unchanged["mission_status"] == "active"

        completion_summary = "Agent 已完成报告与复核\n候选 α。\n"
        agent_completed = task_tree_service.complete_task_root(
            db,
            run_id=run_id,
            root_node_id=root_id,
            result_summary=completion_summary,
        )

        assert agent_completed["root"]["status"] == "completed"
        assert agent_completed["mission_status"] == "completed"
        assert agent_completed["status_reason"] == completion_summary
        expected_summary_sha = hashlib.sha256(
            json.dumps(
                completion_summary,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        assert (
            agent_completed["completion_receipt"]["summary_sha256"]
            == expected_summary_sha
        )
        action_node = child_completed["root"]["children"][0]["children"][0]
        actions = agent_completed["completion_receipt"]["actions"]
        assert len(actions) == 1
        assert actions[0]["node_id"] == action_node["node_id"]
        assert actions[0]["tool_name"] == "skill_demo_run"
        assert actions[0]["capability_id"] == "skill.demo.run"
        assert actions[0]["capability_version"] == "1.0.0"
        assert actions[0]["status"] == "succeeded"
        assert agent_completed["completion_receipt"]["deliverables"] == []
        assert agent_completed["completion_receipt"]["supersessions"] == []
        replayed = task_tree_service.complete_task_root(
            db,
            run_id=run_id,
            root_node_id=root_id,
            result_summary=completion_summary,
        )
        assert replayed["completion_receipt"] == agent_completed["completion_receipt"]
        with pytest.raises(task_tree_service.TaskTreeContractError) as conflict:
            task_tree_service.complete_task_root(
                db,
                run_id=run_id,
                root_node_id=root_id,
                result_summary="different summary",
            )
        assert conflict.value.code == "completion_summary_conflict"
        db.expire_all()
        completed_root_row = db.exec(
            select(database.ProjectTaskTreeNode).where(
                database.ProjectTaskTreeNode.run_id == run_id,
                database.ProjectTaskTreeNode.node_id == root_id,
            )
        ).one()
        completed_run_row = db.exec(
            select(database.ProjectTaskTreeRun).where(
                database.ProjectTaskTreeRun.run_id == run_id
            )
        ).one()
        assert completed_root_row.status == "completed"
        assert completed_root_row.completed_at is not None
        assert completed_run_row.status == "completed"
        assert completed_run_row.completed_at is not None

    SQLModel.metadata.drop_all(engine, tables=TASK_TREE_TABLES)


def test_partial_child_is_user_visible_without_server_overriding_agent_completion(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'task-tree-partial-contract.db'}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine, tables=TASK_TREE_TABLES)
    with Session(engine) as db:
        snapshot = task_tree_service.create_task_root(
            db,
            user_id=9,
            project_id="proj-partial",
            session_id="conv-partial",
            request_id="req-partial",
            title="查询公共数据库",
        )
        run_id = snapshot["run_id"]
        root_id = snapshot["root"]["node_id"]
        snapshot = task_tree_service.add_task_node(
            db,
            run_id=run_id,
            user_id=9,
            project_id="proj-partial",
            session_id="conv-partial",
            parent_node_id=root_id,
            title="查询 ClinVar 阶段",
            node_kind="phase",
            status="partial",
        )

        assert snapshot["root"]["children"][0]["status"] == "partial"
        assert snapshot["root"]["status"] == "running"
        assert snapshot["mission_status"] == "active"

    SQLModel.metadata.drop_all(engine, tables=TASK_TREE_TABLES)


@pytest.mark.parametrize("terminal_status", ["failed", "cancelled"])
def test_mixed_terminal_children_never_mark_phase_completed(tmp_path, terminal_status):
    engine = create_engine(
        f"sqlite:///{tmp_path / f'task-tree-mixed-{terminal_status}.db'}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine, tables=TASK_TREE_TABLES)
    with Session(engine) as db:
        snapshot = task_tree_service.create_task_root(
            db,
            user_id=8,
            project_id=f"proj-mixed-{terminal_status}",
            session_id=f"conv-mixed-{terminal_status}",
            request_id=f"req-mixed-{terminal_status}",
            title="执行两个输入分支",
        )
        run_id = snapshot["run_id"]
        root_id = snapshot["root"]["node_id"]
        snapshot = task_tree_service.add_task_node(
            db,
            run_id=run_id,
            user_id=8,
            project_id=f"proj-mixed-{terminal_status}",
            session_id=f"conv-mixed-{terminal_status}",
            parent_node_id=root_id,
            title="并行处理阶段",
            node_kind="phase",
        )
        phase_id = snapshot["root"]["children"][0]["node_id"]
        for title, status in (("成功分支", "succeeded"), ("异常分支", terminal_status)):
            snapshot = _add_action_with_receipt(
                db,
                run_id=run_id,
                user_id=8,
                project_id=f"proj-mixed-{terminal_status}",
                session_id=f"conv-mixed-{terminal_status}",
                request_id=f"req-mixed-{terminal_status}",
                parent_node_id=phase_id,
                title=title,
                receipt_status=status,
            )

        phase = snapshot["root"]["children"][0]
        assert phase["status"] == terminal_status
        assert snapshot["root"]["status"] == "running"
        assert snapshot["mission_status"] == "active"

    SQLModel.metadata.drop_all(engine, tables=TASK_TREE_TABLES)
