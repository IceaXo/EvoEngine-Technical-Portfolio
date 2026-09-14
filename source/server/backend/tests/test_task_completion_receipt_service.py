from __future__ import annotations

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from database import ProjectTaskTreeRun
from services import task_completion_receipt_service


def _engine():
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def _receipt() -> dict:
    receipt = {
        "schema_version": "evoengine.task-completion-receipt/v1",
        "authority": "task_tree_internal_api",
        "receipt_id": "",
        "run_id": "run-1",
        "root_node_id": "root-1",
        "request_id": "request-1",
        "project_id": "project-1",
        "session_id": "session-1",
        "user_id": 7,
        "completed_at": "2026-08-14T00:00:00Z",
        "tree_state_sha256": "a" * 64,
        "summary_sha256": "b" * 64,
        "actions": [],
        "deliverables": [],
        "supersessions": [],
    }
    receipt["receipt_id"] = task_completion_receipt_service.completion_receipt_id(
        receipt
    )
    return receipt


def test_completion_receipt_resolves_only_exact_persisted_authority() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)
    receipt = _receipt()
    with Session(engine) as db:
        db.add(
            ProjectTaskTreeRun(
                run_id="run-1",
                user_id=7,
                project_id="project-1",
                session_id="session-1",
                request_id="request-1",
                title="test",
                status="completed",
                root_node_id="root-1",
                completion_receipt_json=receipt,
            )
        )
        db.commit()

        assert task_completion_receipt_service.resolve_authoritative_completion_receipt(
            db,
            received=receipt,
            request_id="request-1",
            project_id="project-1",
            session_id="session-1",
            user_id=7,
        ) == receipt

        tampered = dict(receipt, summary_sha256="c" * 64)
        tampered["receipt_id"] = task_completion_receipt_service.completion_receipt_id(
            tampered
        )
        try:
            task_completion_receipt_service.resolve_authoritative_completion_receipt(
                db,
                received=tampered,
                request_id="request-1",
                project_id="project-1",
                session_id="session-1",
                user_id=7,
            )
        except ValueError as exc:
            assert str(exc) == "agent_completion_receipt_authority_mismatch"
        else:  # pragma: no cover - contract assertion
            raise AssertionError("tampered receipt must be rejected")


def test_chat_completion_requires_receipt_only_when_exact_tree_exists() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)
    with Session(engine) as db:
        assert task_completion_receipt_service.resolve_chat_completion_receipt(
            db,
            received=None,
            request_id="plain-request",
            project_id="project-1",
            session_id="session-1",
            user_id=7,
        ) is None
        db.add(
            ProjectTaskTreeRun(
                run_id="run-active",
                user_id=7,
                project_id="project-1",
                session_id="session-1",
                request_id="tree-request",
                title="tree",
                status="active",
                root_node_id="root-active",
            )
        )
        db.commit()

        try:
            task_completion_receipt_service.resolve_chat_completion_receipt(
                db,
                received=None,
                request_id="tree-request",
                project_id="project-1",
                session_id="session-1",
                user_id=7,
            )
        except ValueError as exc:
            assert str(exc) == "agent_completion_receipt_missing_for_task_tree"
        else:  # pragma: no cover - contract assertion
            raise AssertionError("TaskTree completion must carry its receipt")


def test_blocked_completion_receipt_is_a_valid_terminal_authority() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)
    receipt = {
        **_receipt(),
        "terminal_status": "blocked",
        "goal_outcome": "partially_achieved",
    }
    receipt["receipt_id"] = task_completion_receipt_service.completion_receipt_id(
        receipt
    )
    with Session(engine) as db:
        db.add(
            ProjectTaskTreeRun(
                run_id="run-1",
                user_id=7,
                project_id="project-1",
                session_id="session-1",
                request_id="request-1",
                title="blocked but deliverable",
                status="blocked",
                root_node_id="root-1",
                completion_receipt_json=receipt,
            )
        )
        db.commit()

        assert task_completion_receipt_service.resolve_authoritative_completion_receipt(
            db,
            received=receipt,
            request_id="request-1",
            project_id="project-1",
            session_id="session-1",
            user_id=7,
        ) == receipt
