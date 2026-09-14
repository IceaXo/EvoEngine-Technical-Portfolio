from __future__ import annotations

import requests
import pytest

from src.services import conversation_file_registry


class _Response:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _existing_file(file_id: int, file_name: str) -> dict:
    return {
        "conversation_file_id": file_id,
        "file_name": file_name,
        "source_type": "agent_generated",
        "source_ref_id": "generate_markdown_document",
        "drawer_section": "result_file",
        "tool_name": "generate_markdown_document",
    }


def test_text_registration_reconciles_timeout_without_second_write(monkeypatch) -> None:
    post_calls: list[dict] = []

    def _post(*_args, **kwargs):
        post_calls.append(kwargs)
        raise requests.Timeout("COS response timed out after server accepted upload")

    monkeypatch.setattr(conversation_file_registry, "INTERNAL_TOKEN", "token")
    monkeypatch.setattr(conversation_file_registry.requests, "post", _post)
    monkeypatch.setattr(
        conversation_file_registry,
        "find_existing_conversation_file",
        lambda **_kwargs: _existing_file(1052, "report.md"),
    )

    result = conversation_file_registry.register_conversation_file_text(
        user_id=2,
        project_id="project_1",
        conversation_id="conversation_1",
        request_id="request_1",
        file_name="report.md",
        content_text="# Report",
        source_ref_id="generate_markdown_document",
        drawer_section="result_file",
        tool_name="generate_markdown_document",
    )

    assert result == {
        "ok": True,
        "file": _existing_file(1052, "report.md"),
        "reconciled_after_registration_error": True,
    }
    assert len(post_calls) == 1
    assert post_calls[0]["timeout"] == 75


def test_bound_registration_inherits_exact_capability_authority(monkeypatch) -> None:
    post_calls: list[dict] = []

    def _post(*_args, **kwargs):
        post_calls.append(kwargs)
        return _Response(
            {
                "conversation_file_id": 77,
                "file_name": "report.md",
                "sha256": "a" * 64,
                "registration_status": "registered",
            }
        )

    monkeypatch.setattr(conversation_file_registry, "INTERNAL_TOKEN", "token")
    monkeypatch.setattr(conversation_file_registry.requests, "post", _post)

    with conversation_file_registry.bind_file_registration_authority(
        request_id="task-request",
        transport_request_id="continuation-request",
        task_node_id="task-node-1",
        tool_run_id="begin-call-1",
        tool_name="generate_markdown_document",
    ):
        result = conversation_file_registry.register_conversation_file_text(
            user_id=2,
            project_id="project_1",
            conversation_id="conversation_1",
            request_id="task-request",
            file_name="report.md",
            content_text="# Report",
            drawer_section="result_file",
            tool_name="generate_markdown_document",
        )

    assert result and result["conversation_file_id"] == 77
    assert post_calls[0]["json"] == {
        "user_id": 2,
        "project_id": "project_1",
        "conversation_id": "conversation_1",
        "request_id": "task-request",
        "transport_request_id": "continuation-request",
        "file_name": "report.md",
        "mime_type": "text/plain; charset=utf-8",
        "source_type": "agent_generated",
        "source_ref_id": None,
        "content_text": "# Report",
        "display_path": None,
        "qa_status": None,
        "generation_method": None,
        "artifact_metadata": None,
        "drawer_section": "result_file",
        "archive_status": None,
        "task_node_id": "task-node-1",
        "tool_name": "generate_markdown_document",
        "tool_run_id": "begin-call-1",
        "raw_ref": None,
        "trace_ref": None,
        "sandbox_job_id": None,
        "retention_policy": None,
        "is_visible": None,
    }


def test_bound_registration_rejects_conflicting_explicit_lineage(monkeypatch) -> None:
    monkeypatch.setattr(conversation_file_registry, "INTERNAL_TOKEN", "token")
    with conversation_file_registry.bind_file_registration_authority(
        request_id="task-request",
        transport_request_id="transport-request",
        task_node_id="task-node-1",
        tool_run_id="begin-call-1",
        tool_name="artifact_finalize",
    ):
        with pytest.raises(ValueError, match="task_node_id conflicts"):
            conversation_file_registry.register_conversation_file_bytes(
                user_id=2,
                project_id="project_1",
                conversation_id="conversation_1",
                request_id="task-request",
                file_name="result.csv",
                content_bytes=b"value\n1\n",
                task_node_id="other-node",
                tool_run_id="begin-call-1",
                tool_name="artifact_finalize",
            )


def test_content_read_is_scoped_to_user_project_and_conversation(monkeypatch) -> None:
    get_calls: list[tuple[str, dict]] = []

    def _get(url: str, **kwargs):
        get_calls.append((url, kwargs))
        return _Response({"file": {"conversation_file_id": 42}, "content_text": "# Report"})

    monkeypatch.setattr(conversation_file_registry, "INTERNAL_TOKEN", "token")
    monkeypatch.setattr(conversation_file_registry.requests, "get", _get)

    result = conversation_file_registry.read_conversation_file_content_internal(
        user_id=7,
        project_id="project_1",
        conversation_id="conversation_1",
        file_id=42,
    )

    assert result == {"file": {"conversation_file_id": 42}, "content_text": "# Report"}
    assert get_calls == [
        (
            f"{conversation_file_registry.SERVER_BASE_URL}/api/v1/conversation-files/internal/42/content",
            {
                "params": {
                    "user_id": 7,
                    "project_id": "project_1",
                    "conversation_id": "conversation_1",
                },
                "headers": {"X-Internal-Token": "token"},
                "timeout": 75.0,
            },
        )
    ]


def test_batch_registration_reconciles_every_file_before_fallback_write(monkeypatch) -> None:
    post_calls: list[dict] = []

    def _post(*_args, **kwargs):
        post_calls.append(kwargs)
        raise requests.Timeout("batch response timed out")

    existing = {
        "a.pdb": _existing_file(11, "a.pdb"),
        "b.pdb": _existing_file(12, "b.pdb"),
    }
    monkeypatch.setattr(conversation_file_registry, "INTERNAL_TOKEN", "token")
    monkeypatch.setattr(conversation_file_registry.requests, "post", _post)
    monkeypatch.setattr(
        conversation_file_registry,
        "find_existing_conversation_file",
        lambda **kwargs: existing.get(str(kwargs.get("file_name") or "")),
    )

    result = conversation_file_registry.register_conversation_file_batch(
        user_id=2,
        project_id="project_1",
        conversation_id="conversation_1",
        request_id="request_1",
        files=[
            {
                "file_name": "a.pdb",
                "content_bytes": b"a",
                "source_type": "skill_artifact_generated",
                "source_ref_id": "call:a",
                "drawer_section": "temporary_output",
                "tool_name": "execute_skill_script",
            },
            {
                "file_name": "b.pdb",
                "content_bytes": b"b",
                "source_type": "skill_artifact_generated",
                "source_ref_id": "call:b",
                "drawer_section": "temporary_output",
                "tool_name": "execute_skill_script",
            },
        ],
    )

    assert result is not None
    assert result["ok"] is True
    assert result["registered_count"] == 2
    assert result["failed_count"] == 0
    assert all(item["reconciled_after_registration_error"] for item in result["results"])
    assert len(post_calls) == 1
    assert post_calls[0]["timeout"] == 90


def test_plan_revision_fingerprint_uses_latest_plan_file(monkeypatch) -> None:
    files = [
        {
            "conversation_file_id": 10,
            "file_name": "plan-old.md",
            "drawer_section": "plan_file",
            "tool_name": "save_execution_plan",
            "request_id": "request_old",
            "created_at": "2026-07-12T10:00:00Z",
            "updated_at": "2026-07-12T10:00:00Z",
        },
        {
            "conversation_file_id": 20,
            "file_name": "plan-new.md",
            "drawer_section": "plan_file",
            "tool_name": "save_execution_plan",
            "request_id": "request_new",
            "created_at": "2026-07-13T10:00:00Z",
            "updated_at": "2026-07-13T10:00:00Z",
        },
    ]
    monkeypatch.setattr(
        conversation_file_registry,
        "list_conversation_files_internal",
        lambda **_kwargs: {"files": files},
    )

    latest_fingerprint = conversation_file_registry.get_plan_revision_fingerprint(
        user_id=2,
        project_id="project_1",
        conversation_id="conversation_1",
    )
    files[1]["updated_at"] = "2026-07-13T11:00:00Z"
    updated_fingerprint = conversation_file_registry.get_plan_revision_fingerprint(
        user_id=2,
        project_id="project_1",
        conversation_id="conversation_1",
    )

    assert len(latest_fingerprint) == 64
    assert latest_fingerprint != updated_fingerprint


def test_plan_revision_fingerprint_is_empty_without_plan(monkeypatch) -> None:
    monkeypatch.setattr(
        conversation_file_registry,
        "list_conversation_files_internal",
        lambda **_kwargs: {"files": [{"drawer_section": "result_file"}]},
    )

    assert conversation_file_registry.get_plan_revision_fingerprint(
        user_id=2,
        project_id="project_1",
        conversation_id="conversation_1",
    ) == ""
