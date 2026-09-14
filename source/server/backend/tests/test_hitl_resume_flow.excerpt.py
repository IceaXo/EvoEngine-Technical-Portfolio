# Reading excerpt; functions/classes retain their original bodies.
# Origin: server/backend/tests/test_hitl_resume_flow.py @ 8ee96f3171f20eb56c08e4a8aba6a43ab02f0f69
# Module setup, unselected helpers and service wiring are omitted.
# See source_manifest.json and docs/DEPENDENCIES.md. Do not execute this slice.

# ORIGINAL L1-L1
from __future__ import annotations

# ORIGINAL L3-L3
import asyncio

# ORIGINAL L4-L4
import inspect

# ORIGINAL L5-L5
import json

# ORIGINAL L6-L6
import os

# ORIGINAL L7-L7
import sys

# ORIGINAL L8-L8
from pathlib import Path

# ORIGINAL L9-L9
from types import SimpleNamespace

# ORIGINAL L11-L11
import httpx

# ORIGINAL L12-L12
import pytest

# ORIGINAL L13-L13
from fastapi.testclient import TestClient

# ORIGINAL L27-L30
def _bind_session_test_database(monkeypatch) -> None:
    """Keep every test on the engine selected before modules were imported."""

    monkeypatch.setenv("DATABASE_URL", _SESSION_TEST_DATABASE_URL)

# ORIGINAL L33-L40
def _ensure_test_tables(server_backend):
    global _TEST_TABLES_READY
    if _TEST_TABLES_READY:
        return
    import database as database_module

    database_module.SQLModel.metadata.create_all(database_module.engine)
    _TEST_TABLES_READY = True

# ORIGINAL L43-L57
def _ensure_user(server_backend, user_id: int):
    with server_backend.Session(server_backend.engine) as db:
        user = db.get(server_backend.User, user_id)
        if user is None:
            user = server_backend.User(
                id=user_id,
                phone=f"1880000{user_id:04d}",
                hashed_password="<REDACTED_SECRET>",
                username=f"test-user-{user_id}",
                is_active=True,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        return user

# ORIGINAL L60-L67
def _timeline_answer(block_id: str, text: str, order: int) -> dict:
    return {
        "block_id": block_id,
        "kind": "answer",
        "status": "done",
        "order": order,
        "payload": {"text": text},
    }

# ORIGINAL L70-L85
def _hitl_lineage(
    continuation_request_id: str,
    parent_request_id: str,
    *,
    task_authority_request_id: str | None = None,
    task_run_id: str | None = None,
) -> dict:
    return {
        "schema_version": "evoengine.hitl-continuation-lineage/v1",
        "continuation_request_id": continuation_request_id,
        "parent_request_id": parent_request_id,
        "task_authority_request_id": (
            task_authority_request_id or continuation_request_id
        ),
        "task_run_id": task_run_id,
    }

# ORIGINAL L88-L95
def _same_request_task_authority(request_id: str) -> dict:
    return {
        "schema_version": "evoengine.task-authority-control/v1",
        "transport_request_id": request_id,
        "task_authority_request_id": request_id,
        "task_run_id": None,
        "continuation_parent_request_id": None,
    }

# ORIGINAL L98-L131
def _create_completed_task_tree_receipt(
    server_backend,
    *,
    project_session,
    request_id: str,
    conversation_id: str,
    summary: str,
) -> dict:
    import database as database_module

    with server_backend.Session(server_backend.engine) as db:
        created = server_backend.task_tree_service.create_task_root(
            db,
            user_id=int(project_session.user_id),
            project_id=str(project_session.project_id),
            session_id=conversation_id,
            request_id=request_id,
            title="checkpoint completion authority",
        )
        completed = server_backend.task_tree_service.complete_task_root(
            db,
            run_id=str(created["run_id"]),
            root_node_id=str(created["root"]["node_id"]),
            result_summary=summary,
        )
        receipt = dict(completed["completion_receipt"])
        run = db.exec(
            server_backend.select(database_module.ProjectTaskTreeRun).where(
                database_module.ProjectTaskTreeRun.run_id == created["run_id"]
            )
        ).one()
        assert run.status == "completed"
        assert dict(run.completion_receipt_json or {}) == receipt
        return receipt

# ORIGINAL L134-L149
def test_pending_sandbox_job_resume_state_is_not_silently_capped(monkeypatch, tmp_path):
    _bind_session_test_database(monkeypatch)
    monkeypatch.setenv("SECRET_KEY", os.environ.get("SECRET_KEY", "test-secret"))
    import server_backend

    jobs = [
        {"job_id": f"job-{index:02d}", "app_id": "blast", "status": "queued"}
        for index in range(25)
    ]
    jobs.append(dict(jobs[4]))

    normalized = server_backend._normalize_pending_sandbox_jobs(jobs)

    assert len(normalized) == 25
    assert normalized[0]["job_id"] == "job-00"
    assert normalized[-1]["job_id"] == "job-24"

# ORIGINAL L185-L269
def test_turn_checkpoint_is_database_authority_after_runtime_state_is_gone(
    monkeypatch,
    tmp_path,
):
    _bind_session_test_database(monkeypatch)
    monkeypatch.setenv("SECRET_KEY", os.environ.get("SECRET_KEY", "test-secret"))
    import server_backend

    _ensure_test_tables(server_backend)
    project_session = server_backend._get_or_create_project_session(
        user_id=1,
        project_id="proj-turn-checkpoint-authority",
    )
    with server_backend.Session(server_backend.engine) as db:
        conversation = server_backend._ensure_project_default_conversation(
            db,
            project_session=project_session,
        )
        turn = server_backend.ProjectConversationTurn(
            project_session_id=int(project_session.id or 0),
            conversation_session_id=int(conversation.id or 0),
            conversation_id=conversation.conversation_id,
            user_id=1,
            request_id="req-turn-checkpoint-authority",
            input_text="保存后等待人工选择",
            status="waiting_human",
            thinking_trace_json={"schema_version": "v1", "meta": {}},
        )
        db.add(turn)
        db.commit()

    checkpoint = {
        "schema_version": "evoengine.turn-checkpoint/v1",
        "checkpoint_id": "checkpoint_database_authority",
        "request_id": "req-turn-checkpoint-authority",
        "status": "waiting_human",
        "user_goal": "保存后等待人工选择",
        "task_state": {"phase": "suspended"},
        "tool_outcomes": [
            {
                "tool": "save_execution_plan",
                "ok": True,
                "result_status": "succeeded",
            }
        ],
        "side_effect_ledger": {
            "items": [
                {
                    "tool": "save_execution_plan",
                    "effect_identity": "effect-save-plan",
                }
            ]
        },
        "protocol_messages": [
            {"type": "human", "content": "保存后等待人工选择"},
            {
                "type": "tool",
                "name": "save_execution_plan",
                "tool_call_id": "call-save-plan",
                "content": '{"ok":true,"tail":"DURABLE_CHECKPOINT_TAIL"}',
            },
        ],
        "pending_question_bundle": {"bundle_id": "bundle-durable"},
    }
    server_backend._persist_turn_checkpoint_payload(
        project_session,
        "req-turn-checkpoint-authority",
        checkpoint,
    )

    # Simulate loss of process-local runtime state.  Resume must come from the
    # persisted parent turn, not an active websocket/Agent registry entry.
    server_backend._active_turn_registry.pop(
        "req-turn-checkpoint-authority", None
    )
    loaded = server_backend._load_turn_checkpoint_payload(
        int(project_session.id or 0),
        "req-turn-checkpoint-authority",
    )

    assert loaded == checkpoint
    assert (
        loaded["protocol_messages"][-1]["content"]
        == '{"ok":true,"tail":"DURABLE_CHECKPOINT_TAIL"}'
    )

# ORIGINAL L571-L641
def test_hitl_continuation_keeps_full_parent_user_request(monkeypatch, tmp_path):
    _bind_session_test_database(monkeypatch)
    monkeypatch.setenv("SECRET_KEY", os.environ.get("SECRET_KEY", "test-secret"))
    import server_backend

    _ensure_test_tables(server_backend)
    project_session = server_backend._get_or_create_project_session(
        user_id=1,
        project_id="proj-hitl-parent-input",
    )
    project_session_id = int(project_session.id or 0)
    parent_text = "原始精确输入：" + ("A" * 420) + "\n>protein_D\nMSDLKDKAKELEKQLEEANKKLAEQAERYDDMAAAMKAVTEQGHELSNEERNLLSVAYKN"
    with server_backend.Session(server_backend.engine) as db:
        conversation = server_backend._ensure_project_default_conversation(
            db,
            project_session=project_session,
        )
        conversation_id = int(conversation.id or 0)
        turn = server_backend.ProjectConversationTurn(
            project_session_id=project_session_id,
            conversation_session_id=conversation_id,
            conversation_id=conversation.conversation_id,
            user_id=1,
            request_id="req-hitl-parent-input",
            input_text=parent_text,
            status="waiting_human",
        )
        db.add(turn)
        db.commit()
        db.refresh(turn)
        turn_id = int(turn.id or 0)
    with server_backend.Session(server_backend.engine) as db:
        stored = db.get(server_backend.ProjectConversationTurn, turn_id)
        stored.thinking_trace_json = {
            "meta": {
                "hitl": {
                    "bundle": {
                        "bundle_id": "bundle-parent-input",
                        "bundle_title": "确认阈值",
                        "questions": [
                            {
                                "question_id": "identity_threshold",
                                "label": "序列一致性阈值",
                                "options": [{"option_id": "c_0.9", "label": "0.9"}],
                            }
                        ],
                    }
                }
            }
        }
        db.add(stored)
        db.commit()

    continuation = server_backend._format_human_answer_as_new_turn(
        project_session_id,
        "req-hitl-parent-input",
        "bundle-parent-input",
        [
            {
                "question_id": "identity_threshold",
                "selected_option_ids": ["c_0.9"],
                "other_text": "",
            }
        ],
        "",
    )

    assert "[HITL_PARENT_USER_REQUEST]" in continuation
    assert parent_text in continuation
    assert "MSDLKDKAKELEKQLEEANKKLAEQAERYDDMAAAMKAVTEQGHELSNEERNLLSVAYKN" in continuation
    assert "选择：0.9" in continuation

# ORIGINAL L644-L667
def test_build_agent_http_payload_keeps_identity_when_custom_payload_present(monkeypatch, tmp_path):
    _bind_session_test_database(monkeypatch)
    monkeypatch.setenv("SECRET_KEY", os.environ.get("SECRET_KEY", "test-secret"))
    import server_backend

    payload = server_backend._build_agent_http_payload(
        text="请继续",
        attachments=[],
        request_id="req-agent-payload",
        history=[],
        project_summary="summary",
        user_id=9,
        project_id="proj-9",
        agent_payload={
            "request_id": "req-agent-payload",
            "bundle_id": "bundle-1",
            "answer_bundle": {"bundle_id": "bundle-1", "answers": []},
        },
    )

    assert payload["user_id"] == 9
    assert payload["project_id"] == "proj-9"
    assert payload["bundle_id"] == "bundle-1"
    assert payload["answer_bundle"]["bundle_id"] == "bundle-1"

# ORIGINAL L1064-L1105
def test_hitl_parent_scope_resolves_original_conversation(monkeypatch, tmp_path):
    _bind_session_test_database(monkeypatch)
    monkeypatch.setenv("SECRET_KEY", os.environ.get("SECRET_KEY", "test-secret"))
    import server_backend

    _ensure_test_tables(server_backend)
    project_session = server_backend._get_or_create_project_session(user_id=1, project_id="proj-hitl-scope")
    with server_backend.Session(server_backend.engine) as db:
        parent_conversation = server_backend._create_conversation_session(
            db,
            project_session=project_session,
            title="呃呃",
            first_input_text="呃呃",
        )
        other_conversation = server_backend._create_conversation_session(
            db,
            project_session=project_session,
            title="上一个对话",
            first_input_text="上一个对话",
        )
        parent_conversation_id = parent_conversation.conversation_id
        other_conversation_id = other_conversation.conversation_id
        db.add(
            server_backend.ProjectConversationTurn(
                project_session_id=int(project_session.id or 0),
                conversation_session_id=int(parent_conversation.id or 0),
                conversation_id=parent_conversation_id,
                user_id=1,
                request_id="req-hitl-parent",
                input_text="提交前先让我确认",
                status="success",
            )
        )
        db.commit()

    scope = server_backend._get_turn_scope_for_user(1, "req-hitl-parent")

    assert scope is not None
    assert scope["project_session_id"] == project_session.id
    assert scope["project_id"] == project_session.project_id
    assert scope["conversation_id"] == parent_conversation_id
    assert scope["conversation_id"] != other_conversation_id

# ORIGINAL L1332-L1463
def test_stream_from_agent_http_keeps_waiting_sandbox_non_terminal(monkeypatch, tmp_path):
    _bind_session_test_database(monkeypatch)
    monkeypatch.setenv("SECRET_KEY", os.environ.get("SECRET_KEY", "test-secret"))
    import server_backend
    import database as database_module

    _ensure_test_tables(server_backend)

    pushed_messages: list[dict] = []
    request_id = "req-sandbox-waiting"

    class _FakeResponse:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            yield json.dumps(
                {
                    "type": "timeline_turn_suspended",
                    "request_id": request_id,
                    "payload": {
                        "status": "waiting_sandbox",
                        "reason": "awaiting_sandbox_result",
                        "pending_sandbox_jobs": [
                            {
                                "job_id": "job-sandbox-waiting",
                                "app_id": "blast",
                                "status": "queued",
                            }
                        ],
                        "usage": {},
                        "citations": [],
                        "block_timeline": [],
                    },
                },
                ensure_ascii=False,
            )

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, *args, **kwargs):
            return _FakeResponse()

        async def aclose(self):
            return None

    async def _fake_safe_ws_send_json(_ws, message):
        pushed_messages.append(message)
        return True

    monkeypatch.setattr(server_backend, "_safe_ws_send_json", _fake_safe_ws_send_json)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    project_session = server_backend._get_or_create_project_session(user_id=1, project_id="proj-sandbox")
    with server_backend.Session(server_backend.engine) as db:
        conversation = server_backend._ensure_project_default_conversation(
            db,
            project_session=project_session,
        )
        conversation_id = str(conversation.conversation_id)
    turn_id = server_backend._ensure_turn(
        project_session=project_session,
        conversation_session=conversation,
        user_id=1,
        request_id=request_id,
        input_text="帮我跑一个 blast 任务",
        attachments_meta=[],
    )
    assert turn_id > 0

    outcome = asyncio.run(
        server_backend._stream_from_agent_http(
            frontend_ws=object(),
            project_session=project_session,
            text="帮我跑一个 blast 任务",
            attachments=[],
            conversation_files=[],
            request_id=request_id,
            history=[],
            user_id=1,
            project_id="proj-sandbox",
            cancel_event=asyncio.Event(),
            ws_alive_flag={"alive": True},
            agent_payload={
                "request_id": request_id,
                "session_id": conversation_id,
            },
        )
    )

    with server_backend.Session(server_backend.engine) as db:
        turn = db.exec(
            server_backend.select(server_backend.ProjectConversationTurn).where(
                server_backend.ProjectConversationTurn.project_session_id == project_session.id,
                server_backend.ProjectConversationTurn.request_id == request_id,
            )
        ).first()

    assert outcome.waiting_sandbox is True
    assert outcome.pending_sandbox_jobs == [
        {
            "job_id": "job-sandbox-waiting",
            "app_id": "blast",
            "status": "queued",
        }
    ]
    assert outcome.natural_complete is False
    assert pushed_messages[-1]["type"] == "timeline_turn_suspended"
    assert turn is not None
    assert turn.status == "running"
    assert not str(turn.final_reply_text or "").strip()
    assert turn.thinking_trace_json["meta"]["phase"] == "waiting_sandbox"
    assert turn.thinking_trace_json["meta"]["pending_sandbox_jobs"] == [
        {
            "job_id": "job-sandbox-waiting",
            "app_id": "blast",
            "status": "queued",
        }
    ]

# ORIGINAL L1466-L1569
def test_sandbox_terminal_callback_runs_agent_before_finalizing(monkeypatch, tmp_path):
    _bind_session_test_database(monkeypatch)
    monkeypatch.setenv("SECRET_KEY", os.environ.get("SECRET_KEY", "test-secret"))
    import server_backend
    import database as database_module

    _ensure_test_tables(server_backend)
    project_session = server_backend._get_or_create_project_session(
        user_id=1,
        project_id="proj-sandbox-auto",
    )
    with server_backend.Session(server_backend.engine) as db:
        conversation = server_backend._ensure_project_default_conversation(
            db,
            project_session=project_session,
        )
        conversation_id = str(conversation.conversation_id)
    request_id = "req-sandbox-auto"
    server_backend._ensure_turn(
        project_session=project_session,
        conversation_session=conversation,
        user_id=1,
        request_id=request_id,
        input_text="运行并总结沙盒结果",
        attachments_meta=[],
    )
    server_backend._persist_turn_progress(
        project_session,
        request_id,
        partial_reply="",
        turn_phase="waiting_sandbox",
        usage=_usage_ledger(request_id, 10, 2),
        pending_sandbox_jobs=[
            {"job_id": "job-auto-1", "app_id": "blast", "status": "queued"}
        ],
    )
    persisted_checkpoint = {
        "schema_version": "evoengine.turn-checkpoint/v1",
        "checkpoint_id": "checkpoint-sandbox-auto",
        "request_id": request_id,
        "status": "waiting_sandbox",
        "pending_sandbox_jobs": [
            {"job_id": "job-auto-1", "app_id": "blast", "status": "queued"}
        ],
    }
    server_backend._persist_turn_checkpoint_payload(
        project_session,
        request_id,
        persisted_checkpoint,
    )

    observed_payloads: list[dict] = []

    async def _fake_stream(*args, **kwargs):
        observed_payloads.append(dict(kwargs.get("agent_payload") or {}))
        return server_backend.AgentStreamOutcome(
            final_text="沙盒结果已完成并总结。",
            usage=_usage_ledger(request_id, 30, 6),
            natural_complete=True,
            timeline_blocks=[],
            ws_alive=False,
        )

    monkeypatch.setattr(server_backend, "_stream_from_agent_http", _fake_stream)

    asyncio.run(
        server_backend._continue_sandbox_turn(
            {
                "job_id": "job-auto-1",
                "request_id": request_id,
                "project_id": "proj-sandbox-auto",
                "session_id": conversation_id,
                "user_id": 1,
                "app_id": "blast",
                "status": "succeeded",
            }
        )
    )

    with server_backend.Session(server_backend.engine) as db:
        turn = db.exec(
            server_backend.select(server_backend.ProjectConversationTurn).where(
                server_backend.ProjectConversationTurn.project_session_id == project_session.id,
                server_backend.ProjectConversationTurn.request_id == request_id,
            )
        ).first()

    assert len(observed_payloads) == 1
    assert observed_payloads[0]["sandbox_resume_payload"] == {
        "jobs": [
            {
                "job_id": "job-auto-1",
                "app_id": "blast",
                "status": "succeeded",
            }
        ]
    }
    assert observed_payloads[0]["turn_checkpoint"] == persisted_checkpoint
    assert observed_payloads[0]["text"] == "运行并总结沙盒结果"
    assert observed_payloads[0]["usage_ledger"]["input_tokens"] == 10
    assert turn is not None
    assert turn.status == "success"
    assert turn.final_reply_text == "沙盒结果已完成并总结。"
    assert "pending_sandbox_jobs" not in turn.thinking_trace_json["meta"]

# ORIGINAL L1572-L1612
def test_sandbox_finalize_callback_includes_turn_owner(monkeypatch, tmp_path):
    _bind_session_test_database(monkeypatch)
    monkeypatch.setenv("SECRET_KEY", os.environ.get("SECRET_KEY", "test-secret"))
    import database as database_module
    from routers import sandbox as sandbox_router
    import server_backend

    _ensure_test_tables(server_backend)
    captured: list[dict] = []
    monkeypatch.setattr(sandbox_router, "_job_terminal_callback", lambda job: captured.append(dict(job)))

    with server_backend.Session(server_backend.engine) as db:
        db.add(
            database_module.SandboxJob(
                job_id="job-callback-contract",
                user_id=42,
                project_id="proj-callback-contract",
                request_id="req-callback-contract",
                session_id="conv-callback-contract",
                app_id="mafft",
                app_name="MAFFT",
                job_name="callback contract",
                status=database_module.SandboxJobStatus.QUEUED,
                billing_status=database_module.SandboxBillingStatus.PENDING,
            )
        )
        db.commit()
        asyncio.run(
            sandbox_router.finalize_job(
                "job-callback-contract",
                sandbox_router.SandboxFinalizeReq(status="cancelled"),
                db,
                sandbox_router.SANDBOX_SCHEDULER_TOKEN,
            )
        )

    assert len(captured) == 1
    assert captured[0]["job_id"] == "job-callback-contract"
    assert captured[0]["request_id"] == "req-callback-contract"
    assert captured[0]["session_id"] == "conv-callback-contract"
    assert captured[0]["user_id"] == 42

# ORIGINAL L1615-L1643
def test_sandbox_terminal_callbacks_are_serialized_per_turn(monkeypatch, tmp_path):
    _bind_session_test_database(monkeypatch)
    monkeypatch.setenv("SECRET_KEY", os.environ.get("SECRET_KEY", "test-secret"))
    import server_backend

    observed: list[str] = []

    async def _fake_continue(job):
        observed.append(str(job["job_id"]))
        await asyncio.sleep(0)

    monkeypatch.setattr(server_backend, "_continue_sandbox_turn", _fake_continue)
    server_backend._sandbox_continuation_tasks.clear()
    server_backend._sandbox_continuation_queue.clear()
    server_backend._sandbox_continuation_inflight.clear()

    async def _run():
        first = {"job_id": "job-1", "request_id": "req-1", "status": "succeeded"}
        second = {"job_id": "job-2", "request_id": "req-1", "status": "succeeded"}
        server_backend._schedule_sandbox_turn_continuation(first)
        server_backend._schedule_sandbox_turn_continuation(first)
        server_backend._schedule_sandbox_turn_continuation(second)
        task = server_backend._sandbox_continuation_tasks["req-1"]
        await task
        await asyncio.sleep(0)

    asyncio.run(_run())

    assert observed == ["job-1", "job-2"]

# ORIGINAL L1668-L1757
def test_hitl_lineage_mints_exact_parent_run_and_nested_authority(
    monkeypatch,
    tmp_path,
) -> None:
    _bind_session_test_database(monkeypatch)
    monkeypatch.setenv("SECRET_KEY", os.environ.get("SECRET_KEY", "test-secret"))
    import server_backend

    _ensure_test_tables(server_backend)
    project_session = server_backend._get_or_create_project_session(
        user_id=1,
        project_id="proj-hitl-task-authority-lineage",
    )
    with server_backend.Session(server_backend.engine) as db:
        conversation = server_backend._ensure_project_default_conversation(
            db,
            project_session=project_session,
        )
        parent = server_backend.ProjectConversationTurn(
            project_session_id=int(project_session.id),
            conversation_session_id=int(conversation.id),
            conversation_id=conversation.conversation_id,
            user_id=1,
            request_id="req-lineage-parent",
            input_text="父任务",
            status="waiting_human",
        )
        prefix_parent = server_backend.ProjectConversationTurn(
            project_session_id=int(project_session.id),
            conversation_session_id=int(conversation.id),
            conversation_id=conversation.conversation_id,
            user_id=1,
            request_id="req-lineage-parent-prefix-but-distinct",
            input_text="无任务树父任务",
            status="waiting_human",
        )
        db.add(parent)
        db.add(prefix_parent)
        db.commit()
        tree = server_backend.task_tree_service.create_task_root(
            db,
            user_id=1,
            project_id=project_session.project_id,
            session_id=conversation.conversation_id,
            request_id=parent.request_id,
            title="lineage authority",
        )
        lineage = server_backend._mint_hitl_continuation_lineage_in_session(
            db,
            project_session=project_session,
            parent_turn=parent,
            continuation_request_id="req_hitl_continue_lineage_child",
        )
        assert lineage["task_authority_request_id"] == parent.request_id
        assert lineage["task_run_id"] == tree["run_id"]
        child = server_backend.ProjectConversationTurn(
            project_session_id=int(project_session.id),
            conversation_session_id=int(conversation.id),
            conversation_id=conversation.conversation_id,
            user_id=1,
            request_id=lineage["continuation_request_id"],
            input_text="一级继续",
            status="waiting_human",
            thinking_trace_json={
                "schema_version": "v1",
                "meta": {"hitl_continuation": lineage},
            },
        )
        db.add(child)
        db.commit()
        nested = server_backend._mint_hitl_continuation_lineage_in_session(
            db,
            project_session=project_session,
            parent_turn=child,
            continuation_request_id="req_hitl_continue_lineage_grandchild",
        )
        assert nested["parent_request_id"] == child.request_id
        assert nested["task_authority_request_id"] == parent.request_id
        assert nested["task_run_id"] == tree["run_id"]

        no_tree = server_backend._mint_hitl_continuation_lineage_in_session(
            db,
            project_session=project_session,
            parent_turn=prefix_parent,
            continuation_request_id="req_hitl_continue_no_tree",
        )
        assert no_tree["task_authority_request_id"] == no_tree[
            "continuation_request_id"
        ]
        assert no_tree["task_run_id"] is None
