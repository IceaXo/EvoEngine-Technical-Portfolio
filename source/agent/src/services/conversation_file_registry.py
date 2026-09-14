from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from collections.abc import Iterator
from typing import Any

import requests

SERVER_BASE_URL = os.getenv("EVOENGINE_SERVER_BASE_URL", "http://backend:8000")
INTERNAL_TOKEN = os.getenv("INTERNAL_API_TOKEN")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileRegistrationAuthority:
    request_id: str
    transport_request_id: str
    task_node_id: str = ""
    tool_run_id: str = ""
    tool_name: str = ""


_FILE_REGISTRATION_AUTHORITY: ContextVar[FileRegistrationAuthority | None] = (
    ContextVar("evoengine_file_registration_authority", default=None)
)


@contextmanager
def bind_file_registration_authority(
    *,
    request_id: str,
    transport_request_id: str,
    task_node_id: str | None = None,
    tool_run_id: str | None = None,
    tool_name: str | None = None,
) -> Iterator[None]:
    """Bind immutable capability lineage for nested file producers.

    ContextVar keeps parallel tool calls isolated and is copied by
    ``asyncio.to_thread``.  Registration helpers still validate any explicit
    identity supplied by a producer, so this is authority propagation rather
    than a fallback that can silently retag files.
    """

    normalized_request_id = str(request_id or "").strip()
    normalized_transport_id = str(transport_request_id or "").strip()
    normalized_node_id = str(task_node_id or "").strip()
    normalized_tool_run_id = str(tool_run_id or "").strip()
    if not normalized_request_id or not normalized_transport_id:
        raise ValueError("file registration authority requires request identities")
    if bool(normalized_node_id) != bool(normalized_tool_run_id):
        raise ValueError("file registration authority requires node and begin call together")
    authority = FileRegistrationAuthority(
        request_id=normalized_request_id,
        transport_request_id=normalized_transport_id,
        task_node_id=normalized_node_id,
        tool_run_id=normalized_tool_run_id,
        tool_name=str(tool_name or "").strip(),
    )
    token = _FILE_REGISTRATION_AUTHORITY.set(authority)
    try:
        yield
    finally:
        _FILE_REGISTRATION_AUTHORITY.reset(token)


def _resolve_registration_authority(
    *,
    request_id: str | None,
    transport_request_id: str | None,
    task_node_id: str | None,
    tool_run_id: str | None,
    tool_name: str | None,
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    explicit = {
        "request_id": str(request_id or "").strip(),
        "transport_request_id": str(transport_request_id or "").strip(),
        "task_node_id": str(task_node_id or "").strip(),
        "tool_run_id": str(tool_run_id or "").strip(),
        "tool_name": str(tool_name or "").strip(),
    }
    authority = _FILE_REGISTRATION_AUTHORITY.get()
    if authority is None:
        return tuple(value or None for value in explicit.values())  # type: ignore[return-value]

    authoritative = {
        "request_id": authority.request_id,
        "transport_request_id": authority.transport_request_id,
        "task_node_id": authority.task_node_id,
        "tool_run_id": authority.tool_run_id,
        "tool_name": authority.tool_name,
    }
    for key, supplied in explicit.items():
        expected = authoritative[key]
        if supplied and expected and supplied != expected:
            raise ValueError(f"file registration {key} conflicts with capability authority")
    resolved = {
        key: (explicit[key] or authoritative[key] or None)
        for key in explicit
    }
    if bool(resolved["task_node_id"]) != bool(resolved["tool_run_id"]):
        raise ValueError("file registration requires node and begin call together")
    return (
        resolved["request_id"],
        resolved["transport_request_id"],
        resolved["task_node_id"],
        resolved["tool_run_id"],
        resolved["tool_name"],
    )


def _content_read_timeout_seconds() -> float:
    raw = str(os.getenv("EVO_CONVERSATION_FILE_READ_TIMEOUT_SECONDS", "75")).strip()
    try:
        configured = float(raw)
    except (TypeError, ValueError):
        configured = 75.0
    return max(30.0, min(600.0, configured))


def _registration_timeout_seconds(kind: str, *, file_count: int = 1) -> int:
    env_name = f"EVO_CONVERSATION_FILE_{kind.upper()}_TIMEOUT_SECONDS"
    defaults = {
        "text": 75,
        "bytes": 90,
        "batch": max(90, min(300, 30 + 30 * max(1, file_count))),
    }
    try:
        configured = int(str(os.getenv(env_name) or defaults[kind]).strip())
    except (KeyError, TypeError, ValueError):
        configured = defaults.get(kind, 90)
    return max(30, min(600, configured))


def _reconcile_registered_file(
    *,
    user_id: int,
    project_id: str,
    conversation_id: str,
    request_id: str | None,
    file_name: str,
    source_type: str | None,
    source_ref_id: str | None,
    drawer_section: str | None,
    tool_name: str | None,
    task_node_id: str | None,
    tool_run_id: str | None,
) -> dict[str, Any] | None:
    existing = find_existing_conversation_file(
        user_id=user_id,
        project_id=project_id,
        conversation_id=conversation_id,
        request_id=request_id,
        file_name=file_name,
        source_type=source_type,
        source_ref_id=source_ref_id,
        drawer_section=drawer_section,
        tool_name=tool_name,
        task_node_id=task_node_id,
        tool_run_id=tool_run_id,
    )
    if not isinstance(existing, dict):
        return None
    return {
        "ok": True,
        "file": existing,
        "reconciled_after_registration_error": True,
    }


def list_conversation_files_internal(
    *,
    user_id: int | None,
    project_id: str | None,
    conversation_id: str | None,
) -> dict[str, Any] | None:
    if user_id is None or not project_id or not conversation_id or not INTERNAL_TOKEN:
        return None
    try:
        res = requests.get(
            f"{SERVER_BASE_URL}/api/v1/conversation-files/internal/list",
            params={
                "user_id": int(user_id),
                "project_id": str(project_id),
                "conversation_id": str(conversation_id),
            },
            headers={"X-Internal-Token": INTERNAL_TOKEN},
            timeout=20,
        )
        res.raise_for_status()
        return res.json()
    except Exception:
        return None


def read_conversation_file_content_internal(
    *,
    user_id: int | None,
    project_id: str | None,
    conversation_id: str | None,
    file_id: int,
) -> dict[str, Any] | None:
    """Read a full conversation file through the server's ownership-scoped endpoint."""
    if user_id is None or not project_id or not conversation_id or not INTERNAL_TOKEN:
        return None
    try:
        resolved_file_id = int(file_id)
    except (TypeError, ValueError):
        return None
    if resolved_file_id <= 0:
        return None
    try:
        res = requests.get(
            f"{SERVER_BASE_URL}/api/v1/conversation-files/internal/{resolved_file_id}/content",
            params={
                "user_id": int(user_id),
                "project_id": str(project_id),
                "conversation_id": str(conversation_id),
            },
            headers={"X-Internal-Token": INTERNAL_TOKEN},
            timeout=_content_read_timeout_seconds(),
        )
        res.raise_for_status()
        payload = res.json()
        return payload if isinstance(payload, dict) else None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "conversation_file_content_read_failed file_id=%s project_id=%s conversation_id=%s error=%s",
            resolved_file_id,
            project_id,
            conversation_id,
            exc,
        )
        return None


def inspect_conversation_file_internal(
    *,
    user_id: int | None,
    project_id: str | None,
    conversation_id: str | None,
    file_id: int,
) -> dict[str, Any] | None:
    """Read ownership-scoped metadata without loading the file body."""
    if user_id is None or not project_id or not conversation_id or not INTERNAL_TOKEN:
        return None
    try:
        resolved_file_id = int(file_id)
    except (TypeError, ValueError):
        return None
    if resolved_file_id <= 0:
        return None
    try:
        res = requests.get(
            f"{SERVER_BASE_URL}/api/v1/conversation-files/internal/{resolved_file_id}",
            params={
                "user_id": int(user_id),
                "project_id": str(project_id),
                "conversation_id": str(conversation_id),
            },
            headers={"X-Internal-Token": INTERNAL_TOKEN},
            timeout=20,
        )
        res.raise_for_status()
        payload = res.json()
        return payload if isinstance(payload, dict) else None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "conversation_file_inspect_failed file_id=%s error=%s",
            resolved_file_id,
            exc,
        )
        return None


def read_conversation_file_window_internal(
    *,
    user_id: int | None,
    project_id: str | None,
    conversation_id: str | None,
    file_id: int,
    offset: int = 0,
    max_chars: int = 4000,
) -> dict[str, Any] | None:
    """Read one bounded text window and preserve explicit continuation state."""
    if user_id is None or not project_id or not conversation_id or not INTERNAL_TOKEN:
        return None
    try:
        resolved_file_id = int(file_id)
    except (TypeError, ValueError):
        return None
    if resolved_file_id <= 0:
        return None
    try:
        res = requests.get(
            f"{SERVER_BASE_URL}/api/v1/conversation-files/internal/{resolved_file_id}/read-window",
            params={
                "user_id": int(user_id),
                "project_id": str(project_id),
                "conversation_id": str(conversation_id),
                "offset": max(0, int(offset or 0)),
                "max_chars": max(500, min(int(max_chars or 4000), 12000)),
            },
            headers={"X-Internal-Token": INTERNAL_TOKEN},
            timeout=_content_read_timeout_seconds(),
        )
        res.raise_for_status()
        payload = res.json()
        return payload if isinstance(payload, dict) else None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "conversation_file_window_read_failed file_id=%s offset=%s error=%s",
            resolved_file_id,
            offset,
            exc,
        )
        return None


def search_conversation_file_internal(
    *,
    user_id: int | None,
    project_id: str | None,
    conversation_id: str | None,
    file_id: int,
    query: str,
    cursor: int = 0,
    limit: int = 8,
    context_chars: int = 240,
) -> dict[str, Any] | None:
    """Search an authoritative file server-side and return bounded match windows."""
    if user_id is None or not project_id or not conversation_id or not INTERNAL_TOKEN:
        return None
    try:
        resolved_file_id = int(file_id)
    except (TypeError, ValueError):
        return None
    needle = str(query or "").strip()
    if resolved_file_id <= 0 or not needle:
        return None
    try:
        res = requests.get(
            f"{SERVER_BASE_URL}/api/v1/conversation-files/internal/{resolved_file_id}/search",
            params={
                "user_id": int(user_id),
                "project_id": str(project_id),
                "conversation_id": str(conversation_id),
                "query": needle,
                "cursor": max(0, int(cursor or 0)),
                "limit": max(1, min(int(limit or 8), 20)),
                "context_chars": max(40, min(int(context_chars or 240), 1000)),
            },
            headers={"X-Internal-Token": INTERNAL_TOKEN},
            timeout=_content_read_timeout_seconds(),
        )
        res.raise_for_status()
        payload = res.json()
        return payload if isinstance(payload, dict) else None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "conversation_file_search_failed file_id=%s query_chars=%s error=%s",
            resolved_file_id,
            len(needle),
            exc,
        )
        return None


def find_existing_conversation_file(
    *,
    user_id: int | None,
    project_id: str | None,
    conversation_id: str | None,
    file_name: str | None = None,
    source_type: str | None = None,
    source_ref_id: str | None = None,
    request_id: str | None = None,
    drawer_section: str | None = None,
    tool_name: str | None = None,
    task_node_id: str | None = None,
    tool_run_id: str | None = None,
) -> dict[str, Any] | None:
    payload = list_conversation_files_internal(
        user_id=user_id,
        project_id=project_id,
        conversation_id=conversation_id,
    )
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, list):
        return None
    expected = {
        "file_name": str(file_name or "").strip(),
        "source_type": str(source_type or "").strip(),
        "source_ref_id": str(source_ref_id or "").strip(),
        "request_id": str(request_id or "").strip(),
        "drawer_section": str(drawer_section or "").strip(),
        "tool_name": str(tool_name or "").strip(),
        "task_node_id": str(task_node_id or "").strip(),
        "tool_run_id": str(tool_run_id or "").strip(),
    }
    for item in files:
        if not isinstance(item, dict):
            continue
        if expected["file_name"] and str(item.get("file_name") or "").strip() != expected["file_name"]:
            continue
        if expected["source_type"] and str(item.get("source_type") or "").strip() != expected["source_type"]:
            continue
        if expected["source_ref_id"] and str(item.get("source_ref_id") or "").strip() != expected["source_ref_id"]:
            continue
        if expected["request_id"] and str(item.get("request_id") or "").strip() != expected["request_id"]:
            continue
        if expected["drawer_section"] and str(item.get("drawer_section") or "").strip() != expected["drawer_section"]:
            continue
        if expected["tool_name"] and str(item.get("tool_name") or "").strip() != expected["tool_name"]:
            continue
        if expected["task_node_id"] and str(item.get("task_node_id") or "").strip() != expected["task_node_id"]:
            continue
        if expected["tool_run_id"] and str(item.get("tool_run_id") or "").strip() != expected["tool_run_id"]:
            continue
        return item
    return None


def get_plan_revision_fingerprint(
    *,
    user_id: int | None,
    project_id: str | None,
    conversation_id: str | None,
) -> str:
    payload = list_conversation_files_internal(
        user_id=user_id,
        project_id=project_id,
        conversation_id=conversation_id,
    )
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, list):
        return ""
    plans = [
        item
        for item in files
        if isinstance(item, dict)
        and (
            str(item.get("drawer_section") or "").strip() == "plan_file"
            or str(item.get("tool_name") or "").strip() == "save_execution_plan"
        )
    ]
    if not plans:
        return ""

    def _sort_key(item: dict[str, Any]) -> tuple[str, str, int]:
        raw_id = item.get("conversation_file_id") or item.get("file_id") or item.get("id") or 0
        try:
            file_id = int(raw_id)
        except (TypeError, ValueError):
            file_id = 0
        return (
            str(item.get("updated_at") or item.get("created_at") or ""),
            str(item.get("created_at") or ""),
            file_id,
        )

    latest = sorted(plans, key=_sort_key)[-1]
    identity = {
        key: latest.get(key)
        for key in (
            "conversation_file_id",
            "file_id",
            "id",
            "file_name",
            "size_bytes",
            "sha256",
            "content_path",
            "request_id",
            "created_at",
            "updated_at",
        )
        if latest.get(key) not in (None, "")
    }
    return hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def register_conversation_file_text(
    *,
    user_id: int | None,
    project_id: str | None,
    conversation_id: str | None,
    request_id: str | None,
    transport_request_id: str | None = None,
    file_name: str,
    content_text: str,
    mime_type: str = "text/plain; charset=utf-8",
    source_type: str = "agent_generated",
    source_ref_id: str | None = None,
    display_path: str | None = None,
    qa_status: str | None = None,
    generation_method: str | None = None,
    artifact_metadata: dict[str, Any] | None = None,
    drawer_section: str | None = None,
    archive_status: str | None = None,
    task_node_id: str | None = None,
    tool_name: str | None = None,
    tool_run_id: str | None = None,
    raw_ref: str | None = None,
    trace_ref: str | None = None,
    sandbox_job_id: str | None = None,
    retention_policy: str | None = None,
    is_visible: bool | None = None,
) -> dict[str, Any] | None:
    if user_id is None or not project_id or not conversation_id or not INTERNAL_TOKEN:
        return None
    (
        request_id,
        transport_request_id,
        task_node_id,
        tool_run_id,
        tool_name,
    ) = _resolve_registration_authority(
        request_id=request_id,
        transport_request_id=transport_request_id,
        task_node_id=task_node_id,
        tool_run_id=tool_run_id,
        tool_name=tool_name,
    )
    payload = {
        "user_id": int(user_id),
        "project_id": str(project_id),
        "conversation_id": str(conversation_id),
        "request_id": str(request_id or "").strip() or None,
        "transport_request_id": str(transport_request_id or "").strip() or None,
        "file_name": str(file_name or "").strip() or "agent_output.txt",
        "mime_type": str(mime_type or "").strip() or "text/plain; charset=utf-8",
        "source_type": str(source_type or "agent_generated").strip() or "agent_generated",
        "source_ref_id": str(source_ref_id or "").strip() or None,
        "content_text": str(content_text or ""),
        "display_path": str(display_path or "").strip() or None,
        "qa_status": str(qa_status or "").strip() or None,
        "generation_method": str(generation_method or "").strip() or None,
        "artifact_metadata": artifact_metadata if isinstance(artifact_metadata, dict) else None,
        "drawer_section": str(drawer_section or "").strip() or None,
        "archive_status": str(archive_status or "").strip() or None,
        "task_node_id": str(task_node_id or "").strip() or None,
        "tool_name": str(tool_name or "").strip() or None,
        "tool_run_id": str(tool_run_id or "").strip() or None,
        "raw_ref": str(raw_ref or "").strip() or None,
        "trace_ref": str(trace_ref or "").strip() or None,
        "sandbox_job_id": str(sandbox_job_id or "").strip() or None,
        "retention_policy": str(retention_policy or "").strip() or None,
        "is_visible": is_visible,
    }
    try:
        res = requests.post(
            f"{SERVER_BASE_URL}/api/v1/conversation-files/internal/register",
            json=payload,
            headers={"X-Internal-Token": INTERNAL_TOKEN},
            timeout=_registration_timeout_seconds("text"),
        )
        res.raise_for_status()
        return res.json()
    except Exception as exc:  # noqa: BLE001
        reconciled = _reconcile_registered_file(
            user_id=int(user_id),
            project_id=str(project_id),
            conversation_id=str(conversation_id),
            request_id=str(request_id or "").strip() or None,
            file_name=str(payload["file_name"]),
            source_type=str(payload.get("source_type") or "").strip() or None,
            source_ref_id=str(payload.get("source_ref_id") or "").strip() or None,
            drawer_section=str(payload.get("drawer_section") or "").strip() or None,
            tool_name=str(payload.get("tool_name") or "").strip() or None,
            task_node_id=str(payload.get("task_node_id") or "").strip() or None,
            tool_run_id=str(payload.get("tool_run_id") or "").strip() or None,
        )
        if reconciled:
            return reconciled
        logger.warning("conversation_file_text_registration_failed file=%s error=%s", payload["file_name"], exc)
        return None


def register_conversation_file_bytes(
    *,
    user_id: int | None,
    project_id: str | None,
    conversation_id: str | None,
    request_id: str | None,
    transport_request_id: str | None = None,
    file_name: str,
    content_bytes: bytes,
    mime_type: str = "application/octet-stream",
    source_type: str = "agent_download",
    source_ref_id: str | None = None,
    content_text: str | None = None,
    display_path: str | None = None,
    qa_status: str | None = None,
    generation_method: str | None = None,
    artifact_metadata: dict[str, Any] | None = None,
    drawer_section: str | None = None,
    archive_status: str | None = None,
    task_node_id: str | None = None,
    tool_name: str | None = None,
    tool_run_id: str | None = None,
    raw_ref: str | None = None,
    trace_ref: str | None = None,
    sandbox_job_id: str | None = None,
    retention_policy: str | None = None,
    is_visible: bool | None = None,
) -> dict[str, Any] | None:
    if user_id is None or not project_id or not conversation_id or not INTERNAL_TOKEN:
        return None
    (
        request_id,
        transport_request_id,
        task_node_id,
        tool_run_id,
        tool_name,
    ) = _resolve_registration_authority(
        request_id=request_id,
        transport_request_id=transport_request_id,
        task_node_id=task_node_id,
        tool_run_id=tool_run_id,
        tool_name=tool_name,
    )
    payload = {
        "user_id": int(user_id),
        "project_id": str(project_id),
        "conversation_id": str(conversation_id),
        "request_id": str(request_id or "").strip() or None,
        "transport_request_id": str(transport_request_id or "").strip() or None,
        "file_name": str(file_name or "").strip() or "download.bin",
        "mime_type": str(mime_type or "").strip() or "application/octet-stream",
        "source_type": str(source_type or "agent_download").strip() or "agent_download",
        "source_ref_id": str(source_ref_id or "").strip() or None,
        "content_base64": base64.b64encode(content_bytes).decode("ascii"),
        "content_text": str(content_text or "").strip() or None,
        "display_path": str(display_path or "").strip() or None,
        "qa_status": str(qa_status or "").strip() or None,
        "generation_method": str(generation_method or "").strip() or None,
        "artifact_metadata": artifact_metadata if isinstance(artifact_metadata, dict) else None,
        "drawer_section": str(drawer_section or "").strip() or None,
        "archive_status": str(archive_status or "").strip() or None,
        "task_node_id": str(task_node_id or "").strip() or None,
        "tool_name": str(tool_name or "").strip() or None,
        "tool_run_id": str(tool_run_id or "").strip() or None,
        "raw_ref": str(raw_ref or "").strip() or None,
        "trace_ref": str(trace_ref or "").strip() or None,
        "sandbox_job_id": str(sandbox_job_id or "").strip() or None,
        "retention_policy": str(retention_policy or "").strip() or None,
        "is_visible": is_visible,
    }
    try:
        res = requests.post(
            f"{SERVER_BASE_URL}/api/v1/conversation-files/internal/register",
            json=payload,
            headers={"X-Internal-Token": INTERNAL_TOKEN},
            timeout=_registration_timeout_seconds("bytes"),
        )
        res.raise_for_status()
        return res.json()
    except Exception as exc:  # noqa: BLE001
        reconciled = _reconcile_registered_file(
            user_id=int(user_id),
            project_id=str(project_id),
            conversation_id=str(conversation_id),
            request_id=str(request_id or "").strip() or None,
            file_name=str(payload["file_name"]),
            source_type=str(payload.get("source_type") or "").strip() or None,
            source_ref_id=str(payload.get("source_ref_id") or "").strip() or None,
            drawer_section=str(payload.get("drawer_section") or "").strip() or None,
            tool_name=str(payload.get("tool_name") or "").strip() or None,
            task_node_id=str(payload.get("task_node_id") or "").strip() or None,
            tool_run_id=str(payload.get("tool_run_id") or "").strip() or None,
        )
        if reconciled:
            return reconciled
        response_detail = ""
        response = getattr(exc, "response", None)
        if response is not None:
            response_detail = str(getattr(response, "text", "") or "").strip()[:500]
        logger.warning(
            "conversation_file_bytes_registration_failed file=%s error=%s detail=%s",
            payload["file_name"],
            exc,
            response_detail,
        )
        return None



def register_conversation_file_batch(
    *,
    user_id: int | None,
    project_id: str | None,
    conversation_id: str | None,
    request_id: str | None,
    files: list[dict[str, Any]],
    transport_request_id: str | None = None,
) -> dict[str, Any] | None:
    if user_id is None or not project_id or not conversation_id or not INTERNAL_TOKEN:
        return None
    request_id, transport_request_id, _, _, _ = _resolve_registration_authority(
        request_id=request_id,
        transport_request_id=transport_request_id,
        task_node_id=None,
        tool_run_id=None,
        tool_name=None,
    )
    normalized_files: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        (
            _,
            _,
            resolved_task_node_id,
            resolved_tool_run_id,
            resolved_tool_name,
        ) = _resolve_registration_authority(
            request_id=request_id,
            transport_request_id=transport_request_id,
            task_node_id=item.get("task_node_id"),
            tool_run_id=item.get("tool_run_id"),
            tool_name=item.get("tool_name"),
        )
        content_bytes = item.get("content_bytes")
        content_base64 = item.get("content_base64")
        if isinstance(content_bytes, bytes):
            content_base64 = base64.b64encode(content_bytes).decode("ascii")
        normalized_files.append(
            {
                "file_name": str(item.get("file_name") or "").strip() or "artifact.bin",
                "mime_type": str(item.get("mime_type") or "").strip() or "application/octet-stream",
                "source_type": str(item.get("source_type") or "artifact_generated").strip() or "artifact_generated",
                "source_ref_id": str(item.get("source_ref_id") or "").strip() or None,
                "content_base64": str(content_base64 or "").strip() or None,
                "content_text": str(item.get("content_text") or "").strip() or None,
                "display_path": str(item.get("display_path") or "").strip() or None,
                "qa_status": str(item.get("qa_status") or "").strip() or None,
                "generation_method": str(item.get("generation_method") or "").strip() or None,
                "artifact_metadata": item.get("artifact_metadata") if isinstance(item.get("artifact_metadata"), dict) else None,
                "drawer_section": str(item.get("drawer_section") or "").strip() or None,
                "archive_status": str(item.get("archive_status") or "").strip() or None,
                "task_node_id": resolved_task_node_id,
                "tool_name": resolved_tool_name,
                "tool_run_id": resolved_tool_run_id,
                "raw_ref": str(item.get("raw_ref") or "").strip() or None,
                "trace_ref": str(item.get("trace_ref") or "").strip() or None,
                "sandbox_job_id": str(item.get("sandbox_job_id") or "").strip() or None,
                "retention_policy": str(item.get("retention_policy") or "").strip() or None,
                "is_visible": item.get("is_visible") if isinstance(item.get("is_visible"), bool) else None,
            }
        )
    if not normalized_files:
        return None
    payload = {
        "user_id": int(user_id),
        "project_id": str(project_id),
        "conversation_id": str(conversation_id),
        "request_id": str(request_id or "").strip() or None,
        "transport_request_id": (
            str(transport_request_id or "").strip() or None
        ),
        "files": normalized_files,
    }
    try:
        res = requests.post(
            f"{SERVER_BASE_URL}/api/v1/conversation-files/internal/register-batch",
            json=payload,
            headers={"X-Internal-Token": INTERNAL_TOKEN},
            timeout=_registration_timeout_seconds("batch", file_count=len(normalized_files)),
        )
        res.raise_for_status()
        return res.json()
    except Exception:
        results: list[dict[str, Any]] = []
        registered_count = 0
        for idx, item in enumerate(normalized_files):
            reconciled = _reconcile_registered_file(
                user_id=int(user_id),
                project_id=str(project_id),
                conversation_id=str(conversation_id),
                request_id=str(request_id or "").strip() or None,
                file_name=str(item.get("file_name") or "artifact.bin"),
                source_type=str(item.get("source_type") or "").strip() or None,
                source_ref_id=str(item.get("source_ref_id") or "").strip() or None,
                drawer_section=str(item.get("drawer_section") or "").strip() or None,
                tool_name=str(item.get("tool_name") or "").strip() or None,
                task_node_id=str(item.get("task_node_id") or "").strip() or None,
                tool_run_id=str(item.get("tool_run_id") or "").strip() or None,
            )
            if reconciled:
                results.append(
                    {
                        "ok": True,
                        "index": idx,
                        "file": reconciled["file"],
                        "reconciled_after_registration_error": True,
                    }
                )
                registered_count += 1
                continue
            try:
                single_payload = {
                    "user_id": int(user_id),
                    "project_id": str(project_id),
                    "conversation_id": str(conversation_id),
                    "request_id": str(request_id or "").strip() or None,
                    **item,
                }
                res = requests.post(
                    f"{SERVER_BASE_URL}/api/v1/conversation-files/internal/register",
                    json=single_payload,
                    headers={"X-Internal-Token": INTERNAL_TOKEN},
                    timeout=_registration_timeout_seconds("text"),
                )
                res.raise_for_status()
                payload_item = res.json()
                results.append({"ok": True, "index": idx, "file": payload_item.get("file") if isinstance(payload_item, dict) else payload_item})
                registered_count += 1
            except Exception as exc:  # noqa: BLE001
                results.append({"ok": False, "index": idx, "error": str(exc)})
        return {
            "ok": registered_count == len(normalized_files),
            "registered_count": registered_count,
            "failed_count": len(normalized_files) - registered_count,
            "results": results,
            "fallback": "per_file_register",
        }
