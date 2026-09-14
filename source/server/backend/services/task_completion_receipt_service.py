from __future__ import annotations

import hashlib
import json
from typing import Any

from sqlmodel import Session, select

from database import ProjectTaskTreeRun


COMPLETION_RECEIPT_SCHEMA = "evoengine.task-completion-receipt/v1"
COMPLETION_RECEIPT_AUTHORITY = "task_tree_internal_api"


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def validate_completion_summary(
    receipt: dict[str, Any],
    *,
    summary: str,
) -> None:
    expected = str(receipt.get("summary_sha256") or "").strip().lower()
    observed = canonical_json_sha256(str(summary))
    if not expected or expected != observed:
        raise ValueError("agent_completion_receipt_summary_mismatch")


def completion_receipt_id(receipt: dict[str, Any]) -> str:
    unsigned = dict(receipt)
    unsigned.pop("receipt_id", None)
    digest = canonical_json_sha256(unsigned)
    return f"completion_{digest}"


def resolve_authoritative_completion_receipt(
    db: Session,
    *,
    received: dict[str, Any] | None,
    request_id: str,
    project_id: str,
    session_id: str,
    user_id: int,
) -> dict[str, Any]:
    """Resolve an Agent-carried receipt against TaskTree persistent authority.

    The digest is an integrity check, not a signature.  Authority comes from
    an exact authenticated run lookup plus byte-canonical equality with the
    immutable receipt stored by the atomic task finalizer.
    """

    candidate = dict(received or {})
    if candidate.get("schema_version") != COMPLETION_RECEIPT_SCHEMA:
        raise ValueError("agent_completion_receipt_schema_invalid")
    if candidate.get("authority") != COMPLETION_RECEIPT_AUTHORITY:
        raise ValueError("agent_completion_receipt_authority_invalid")
    if str(candidate.get("receipt_id") or "").strip() != completion_receipt_id(
        candidate
    ):
        raise ValueError("agent_completion_receipt_digest_invalid")
    run_id = str(candidate.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("agent_completion_receipt_run_id_missing")
    expected_scope = {
        "request_id": str(request_id),
        "project_id": str(project_id),
        "session_id": str(session_id),
        "user_id": int(user_id),
    }
    for field, expected in expected_scope.items():
        observed = candidate.get(field)
        if field == "user_id":
            try:
                observed = int(observed)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "agent_completion_receipt_scope_mismatch:user_id"
                ) from exc
        else:
            observed = str(observed or "")
        if observed != expected:
            raise ValueError(f"agent_completion_receipt_scope_mismatch:{field}")

    run = db.exec(
        select(ProjectTaskTreeRun).where(
            ProjectTaskTreeRun.run_id == run_id,
            ProjectTaskTreeRun.user_id == int(user_id),
            ProjectTaskTreeRun.project_id == str(project_id),
            ProjectTaskTreeRun.session_id == str(session_id),
            ProjectTaskTreeRun.request_id == str(request_id),
            ProjectTaskTreeRun.status.in_(("completed", "blocked")),
        )
    ).first()
    if run is None:
        raise ValueError("agent_completion_receipt_authority_not_found")
    authoritative = dict(run.completion_receipt_json or {})
    if not authoritative:
        raise ValueError("agent_completion_receipt_authority_missing")
    if _canonical_json(candidate) != _canonical_json(authoritative):
        raise ValueError("agent_completion_receipt_authority_mismatch")
    return authoritative


def resolve_chat_completion_receipt(
    db: Session,
    *,
    received: dict[str, Any] | None,
    request_id: str,
    project_id: str,
    session_id: str,
    user_id: int,
) -> dict[str, Any] | None:
    """Accept receipt-less completion only when this turn created no TaskTree.

    A missing envelope is valid for a simple no-tree conversation.  Once an
    exact scoped TaskTree run exists, its atomic finalizer is the only
    completion authority and the public terminal must carry that persisted
    receipt.
    """

    if isinstance(received, dict):
        return resolve_authoritative_completion_receipt(
            db,
            received=received,
            request_id=request_id,
            project_id=project_id,
            session_id=session_id,
            user_id=user_id,
        )
    scoped_run = db.exec(
        select(ProjectTaskTreeRun).where(
            ProjectTaskTreeRun.user_id == int(user_id),
            ProjectTaskTreeRun.project_id == str(project_id),
            ProjectTaskTreeRun.session_id == str(session_id),
            ProjectTaskTreeRun.request_id == str(request_id),
        )
    ).first()
    if scoped_run is not None:
        raise ValueError("agent_completion_receipt_missing_for_task_tree")
    return None
