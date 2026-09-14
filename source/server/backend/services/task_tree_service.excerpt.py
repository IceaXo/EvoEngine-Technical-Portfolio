# Reading excerpt; functions/classes retain their original bodies.
# Origin: server/backend/services/task_tree_service.py @ 8ee96f3171f20eb56c08e4a8aba6a43ab02f0f69
# Module setup, unselected helpers and service wiring are omitted.
# See source_manifest.json and docs/DEPENDENCIES.md. Do not execute this slice.

# ORIGINAL L1-L1
from __future__ import annotations

# ORIGINAL L3-L3
import hashlib

# ORIGINAL L4-L4
import json

# ORIGINAL L5-L5
import re

# ORIGINAL L6-L6
import uuid

# ORIGINAL L7-L7
from datetime import UTC, datetime, timedelta

# ORIGINAL L8-L8
from typing import Any

# ORIGINAL L10-L10
from sqlalchemy import case

# ORIGINAL L11-L11
from sqlalchemy.exc import IntegrityError

# ORIGINAL L12-L12
from sqlalchemy.sql import text as sql_text

# ORIGINAL L13-L13
from sqlmodel import Session, select

# ORIGINAL L15-L21
from database import (
    ConversationFile,
    ProjectConversationTurn,
    ProjectSession,
    ProjectTaskTreeNode,
    ProjectTaskTreeRun,
)

# ORIGINAL L22-L22
from domain.task_tree import TaskTreeNode, TaskTreeSnapshot

# ORIGINAL L23-L26
from services.conversation_file_service import (
    _conversation_file_registration_authority,
    _raw_artifact_metadata,
)

# ORIGINAL L3180-L3364
def _validate_tree_completion(
    *,
    rows: list[ProjectTaskTreeNode],
    root_node_id: str,
    terminal_status: str = "completed",
) -> dict[str, Any]:
    normalized_terminal = str(terminal_status or "completed").strip().lower()
    if normalized_terminal not in {"completed", "blocked"}:
        raise TaskTreeContractError(
            code="task_terminal_status_invalid",
            message="task terminal status must be completed or blocked",
        )
    required_obligation_nodes: dict[str, list[str]] = {}
    resolution_candidates: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.node_id) == str(root_node_id):
            continue
        arguments = dict(row.tool_arguments_json or {})
        contract_value = arguments.get(EXECUTION_CONTRACT_KEY)
        contract = contract_value if isinstance(contract_value, dict) else None
        if contract is None:
            continue

        contract = _require_exact_execution_identity(
            row,
            tool_name=str(contract.get("tool_name") or ""),
            capability_id=str(contract.get("capability_id") or ""),
            capability_version=str(contract.get("capability_version") or ""),
        )
        obligation_id = str(contract.get("obligation_id") or "").strip()
        if bool(contract.get("required")):
            required_obligation_nodes.setdefault(obligation_id, []).append(
                str(row.node_id)
            )
        system_resolution = arguments.get(TASK_RESOLUTION_KEY)
        if system_resolution is not None:
            projection = _validated_system_task_resolution(
                row=row,
                contract=contract,
                resolution=system_resolution,
            )
            if obligation_id:
                resolution_candidates.setdefault(obligation_id, []).append(
                    {
                        "obligation_id": obligation_id,
                        "node_id": str(row.node_id),
                        "outcome": projection["outcome"],
                        "source": (
                            "system_block"
                            if projection["outcome"] == "blocked"
                            else "system_skip"
                        ),
                        "reason_code": projection["reason_code"],
                        "dependency_node_ids": projection[
                            "dependency_node_ids"
                        ],
                    }
                )
            continue
        begin = arguments.get(CAPABILITY_BEGIN_KEY)
        receipt = arguments.get(CAPABILITY_RECEIPT_KEY)
        if not isinstance(receipt, dict):
            continue
        begin_identity = _validate_authoritative_begin(contract=contract, begin=begin)
        receipt_status = _validate_authoritative_receipt(
            contract=contract,
            begin_identity=begin_identity,
            receipt=receipt,
        )
        expected_node_status = _RECEIPT_STATUS_TO_NODE_STATUS[receipt_status]
        node_status = str(row.status or "").strip().lower()
        if node_status != expected_node_status:
            raise TaskTreeContractError(
                code="capability_receipt_node_status_mismatch",
                message="task node status does not match its authoritative capability receipt",
                details={
                    "node_id": str(row.node_id),
                    "node_status": node_status,
                    "receipt_status": receipt_status,
                },
            )
        if receipt_status == "succeeded" and obligation_id:
            outcome = str(
                receipt.get("resolution_outcome") or "achieved"
            ).strip().lower()
            resolution_candidates.setdefault(obligation_id, []).append(
                {
                    "obligation_id": obligation_id,
                    "node_id": str(row.node_id),
                    "outcome": outcome,
                    "source": "capability_receipt",
                    "call_id": begin_identity["call_id"],
                }
            )
        elif (
            normalized_terminal == "blocked"
            and receipt_status in {"failed", "cancelled"}
            and obligation_id
        ):
            receipt_payload = (
                receipt.get("receipt")
                if isinstance(receipt.get("receipt"), dict)
                else {}
            )
            resolution_candidates.setdefault(obligation_id, []).append(
                {
                    "obligation_id": obligation_id,
                    "node_id": str(row.node_id),
                    "outcome": "blocked",
                    "source": "capability_receipt",
                    "call_id": begin_identity["call_id"],
                    "reason_code": (
                        str(receipt_payload.get("error_kind") or "").strip()
                        or (
                            "capability_cancelled"
                            if receipt_status == "cancelled"
                            else "capability_failed"
                        )
                    ),
                }
            )

    missing_obligation_ids = sorted(
        set(required_obligation_nodes) - set(resolution_candidates)
    )
    if missing_obligation_ids:
        raise TaskTreeContractError(
            code="required_capability_receipts_missing",
            message=(
                "each required obligation needs an authoritative resolved "
                "capability result or a Server-derived dependency skip"
            ),
            details={
                "obligation_ids": missing_obligation_ids,
                "node_ids": sorted(
                    {
                        node_id
                        for obligation_id in missing_obligation_ids
                        for node_id in required_obligation_nodes[obligation_id]
                    }
                ),
            },
        )

    outcome_priority = {
        "achieved": 0,
        "inconclusive": 1,
        "negative": 2,
        "not_applicable": 3,
        "skipped": 4,
        "blocked": 5,
    }
    obligation_resolutions: list[dict[str, Any]] = []
    for obligation_id in sorted(required_obligation_nodes):
        candidates = resolution_candidates[obligation_id]
        selected = min(
            candidates,
            key=lambda item: (
                outcome_priority.get(str(item.get("outcome") or ""), 99),
                str(item.get("node_id") or ""),
            ),
        )
        obligation_resolutions.append(selected)
    required_outcomes = {
        str(item.get("outcome") or "") for item in obligation_resolutions
    }
    if normalized_terminal == "blocked" and "blocked" not in required_outcomes:
        raise TaskTreeContractError(
            code="blocked_terminal_without_blocker",
            message=(
                "blocked finalization requires an authoritative failed or "
                "cancelled required capability, or a hard dependent blocked by it"
            ),
        )
    if not obligation_resolutions or required_outcomes == {"achieved"}:
        goal_outcome = "achieved"
    elif "achieved" in required_outcomes:
        goal_outcome = "partially_achieved"
    else:
        goal_outcome = "not_achieved"
    return {
        "terminal_status": normalized_terminal,
        "goal_outcome": goal_outcome,
        "obligation_resolutions": obligation_resolutions,
    }

# ORIGINAL L3379-L3488
def _validated_completion_receipt(
    run: ProjectTaskTreeRun,
) -> dict[str, Any]:
    receipt = dict(run.completion_receipt_json or {})
    if not receipt:
        raise TaskTreeContractError(
            code="completion_receipt_missing",
            message="completed task run has no authoritative CompletionReceipt",
            details={"run_id": str(run.run_id or "")},
        )

    invalid_fields: list[str] = []
    expected_scope: dict[str, Any] = {
        "schema_version": COMPLETION_RECEIPT_SCHEMA,
        "authority": CAPABILITY_AUTHORITY,
        "run_id": str(run.run_id),
        "root_node_id": str(run.root_node_id),
        "request_id": str(run.request_id or "") or None,
        "project_id": str(run.project_id),
        "session_id": str(run.session_id),
        "user_id": int(run.user_id),
    }
    for field, expected in expected_scope.items():
        observed = receipt.get(field)
        if field == "request_id":
            observed = str(observed or "") or None
        elif field == "user_id":
            try:
                observed = int(observed)
            except (TypeError, ValueError, OverflowError):
                observed = None
        else:
            observed = str(observed or "")
        if observed != expected:
            invalid_fields.append(field)

    for field in ("tree_state_sha256", "summary_sha256"):
        digest = str(receipt.get(field) or "").strip().lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            invalid_fields.append(field)
    if str(receipt.get("summary_sha256") or "").strip().lower() != _canonical_sha256(
        str(run.status_reason or "")
    ):
        invalid_fields.append("summary_sha256")
    resolution_fields_present = any(
        field in receipt
        for field in ("terminal_status", "goal_outcome", "obligation_resolutions")
    )
    if str(run.status or "").strip().lower() == "blocked" and not resolution_fields_present:
        invalid_fields.append("terminal_status")
    if resolution_fields_present:
        receipt_terminal = str(receipt.get("terminal_status") or "").strip().lower()
        if receipt_terminal not in {"completed", "blocked"}:
            invalid_fields.append("terminal_status")
        if receipt_terminal != str(run.status or "").strip().lower():
            invalid_fields.append("terminal_status")
        if receipt.get("goal_outcome") not in {
            "achieved",
            "partially_achieved",
            "not_achieved",
        }:
            invalid_fields.append("goal_outcome")
        resolutions = receipt.get("obligation_resolutions")
        if not isinstance(resolutions, list):
            invalid_fields.append("obligation_resolutions")
        else:
            obligation_ids: list[str] = []
            for item in resolutions:
                if (
                    not isinstance(item, dict)
                    or not str(item.get("obligation_id") or "").strip()
                    or not str(item.get("node_id") or "").strip()
                    or str(item.get("outcome") or "").strip().lower()
                    not in _RESOLUTION_OUTCOMES
                    or str(item.get("source") or "").strip()
                    not in {
                        "capability_receipt",
                        "system_skip",
                        "system_block",
                    }
                ):
                    invalid_fields.append("obligation_resolutions")
                    break
                obligation_ids.append(str(item.get("obligation_id")))
            if len(obligation_ids) != len(set(obligation_ids)):
                invalid_fields.append("obligation_resolutions")
    expected_completed_at = (
        run.completed_at.isoformat() + "Z" if run.completed_at is not None else ""
    )
    if str(receipt.get("completed_at") or "").strip() != expected_completed_at:
        invalid_fields.append("completed_at")

    unsigned = dict(receipt)
    observed_receipt_id = str(unsigned.pop("receipt_id", "") or "").strip()
    expected_receipt_id = "completion_" + _canonical_sha256(unsigned)
    if observed_receipt_id != expected_receipt_id:
        invalid_fields.append("receipt_id")

    if invalid_fields:
        raise TaskTreeContractError(
            code="completion_receipt_invalid",
            message="completed task run has an invalid authoritative CompletionReceipt",
            details={
                "run_id": str(run.run_id or ""),
                "invalid_fields": sorted(set(invalid_fields)),
            },
        )
    return receipt

# ORIGINAL L3491-L3861
def _build_completion_receipt(
    db: Session,
    *,
    run: ProjectTaskTreeRun,
    rows: list[ProjectTaskTreeNode],
    root_node_id: str,
    result_summary: str,
    completed_at: datetime,
    completion_resolution: dict[str, Any],
    deliverable_file_ids: list[int] | None = None,
) -> dict[str, Any]:
    requested_deliverable_ids = [int(item) for item in (deliverable_file_ids or [])]
    if (
        any(item < 1 for item in requested_deliverable_ids)
        or len(requested_deliverable_ids) != len(set(requested_deliverable_ids))
        or len(requested_deliverable_ids) > 64
    ):
        raise TaskTreeContractError(
            code="completion_deliverable_selection_invalid",
            message="completion deliverable file IDs must be unique positive integers",
        )
    requested_deliverable_id_set = set(requested_deliverable_ids)
    valid_supersessions = _valid_execution_supersessions(rows)
    actions: list[dict[str, Any]] = []
    action_projection_by_node_id: dict[str, dict[str, Any]] = {}
    succeeded_action_by_node_id: dict[str, ProjectTaskTreeNode] = {}
    result_file_refs: dict[int, tuple[ProjectTaskTreeNode, dict[str, Any]]] = {}
    tree_state_nodes: list[dict[str, Any]] = []
    row_by_id = {str(row.node_id): row for row in rows}
    terminal_status = str(
        completion_resolution.get("terminal_status") or "completed"
    ).strip().lower()

    for row in sorted(rows, key=lambda item: str(item.node_id)):
        arguments = dict(row.tool_arguments_json or {})
        contract = arguments.get(EXECUTION_CONTRACT_KEY)
        tree_state_nodes.append(
            {
                "node_id": str(row.node_id),
                "parent_node_id": str(row.parent_node_id or ""),
                "node_kind": str(row.node_kind or ""),
                "status": (
                    ("completed" if terminal_status == "completed" else "partial")
                    if str(row.node_id) == str(root_node_id)
                    else str(row.status or "")
                ),
                "depends_on_node_ids": sorted(
                    str(item)
                    for item in (row.depends_on_node_ids_json or [])
                    if str(item).strip()
                ),
                "execution_contract": contract if isinstance(contract, dict) else None,
                "capability_receipt_sha256": (
                    _canonical_sha256(arguments.get(CAPABILITY_RECEIPT_KEY))
                    if isinstance(arguments.get(CAPABILITY_RECEIPT_KEY), dict)
                    else None
                ),
                "task_resolution_sha256": (
                    _canonical_sha256(arguments.get(TASK_RESOLUTION_KEY))
                    if isinstance(arguments.get(TASK_RESOLUTION_KEY), dict)
                    else None
                ),
                "artifacts": sorted(
                    [dict(item) for item in (row.artifacts_json or []) if isinstance(item, dict)],
                    key=lambda item: (
                        str(item.get("conversation_file_id") or ""),
                        str(item.get("artifact_id") or ""),
                    ),
                ),
                "superseded_by_node_id": valid_supersessions.get(str(row.node_id)),
            }
        )
        if not isinstance(contract, dict):
            continue
        receipt = arguments.get(CAPABILITY_RECEIPT_KEY)
        if not isinstance(receipt, dict):
            # Optional/informational attempts may remain unstarted or pending;
            # they are execution history, not completion obligations.
            continue
        begin_identity = _validate_authoritative_begin(
            contract=contract,
            begin=arguments.get(CAPABILITY_BEGIN_KEY),
        )
        receipt_status = _validate_authoritative_receipt(
            contract=contract,
            begin_identity=begin_identity,
            receipt=receipt,
        )
        receipt_projection = (
            receipt.get("receipt") if isinstance(receipt, dict) else {}
        )
        subject_refs = _sanitize_subject_refs(
            receipt_projection.get("subject_refs")
            if isinstance(receipt_projection, dict)
            else None
        )
        artifact_refs: list[dict[str, Any]] = []
        for artifact in row.artifacts_json or []:
            if not isinstance(artifact, dict):
                continue
            projection = {
                key: artifact[key]
                for key in (
                    "artifact_id",
                    "artifact_key",
                    "conversation_file_id",
                    "drawer_section",
                    "sha256",
                )
                if artifact.get(key) not in (None, "")
            }
            if projection:
                artifact_refs.append(projection)
            if receipt_status != "succeeded" or str(
                artifact.get("drawer_section") or ""
            ) != "result_file":
                continue
            artifact_key = _bounded_string(
                artifact.get("artifact_key"),
                limit=_MAX_RECEIPT_ID_CHARS,
            )
            if not artifact_key.strip():
                raise TaskTreeContractError(
                    code="completion_deliverable_artifact_key_missing",
                    message=(
                        "result_file artifacts require a non-empty artifact_key "
                        "for CompletionReceipt delivery identity"
                    ),
                )
            try:
                file_id = int(artifact.get("conversation_file_id"))
            except (TypeError, ValueError) as exc:
                raise TaskTreeContractError(
                    code="result_file_identity_missing",
                    message="result_file artifacts require an exact conversation_file_id",
                ) from exc
            existing_result_file = result_file_refs.get(file_id)
            if (
                existing_result_file is not None
                and str(existing_result_file[0].node_id) != str(row.node_id)
            ):
                raise TaskTreeContractError(
                    code="completion_deliverable_lineage_conflict",
                    message=(
                        "one result_file ConversationFile cannot be owned by "
                        "multiple action receipts"
                    ),
                    details={"conversation_file_id": file_id},
                )
            result_file_refs[file_id] = (row, artifact)
        action_projection = {
                "node_id": str(row.node_id),
                "obligation_id": str(contract.get("obligation_id") or ""),
                "tool_name": str(contract.get("tool_name") or ""),
                "capability_id": str(contract.get("capability_id") or ""),
                "capability_version": str(contract.get("capability_version") or ""),
                "call_id": begin_identity["call_id"],
                "idempotency_key": begin_identity["idempotency_key"],
                "status": receipt_status,
                "resolution_outcome": (
                    str(receipt.get("resolution_outcome") or "achieved")
                    if receipt_status == "succeeded"
                    else None
                ),
                "capability_receipt_sha256": _canonical_sha256(receipt),
                "subject_refs": subject_refs,
                "artifact_refs": sorted(
                    artifact_refs,
                    key=lambda item: (
                        str(item.get("conversation_file_id") or ""),
                        str(item.get("artifact_id") or ""),
                    ),
                ),
            }
        actions.append(action_projection)
        action_projection_by_node_id[str(row.node_id)] = action_projection
        if (
            receipt_status == "succeeded"
            and str(row.status or "").strip().lower() == "completed"
            and str(row.node_id) not in valid_supersessions
        ):
            succeeded_action_by_node_id[str(row.node_id)] = row

    # A file can finish durable registration after the capability receipt was
    # sealed (for example, a long artifact batch whose ConversationFile writes
    # drain after the provider result).  The immutable capability receipt must
    # not be rewritten, but an explicitly selected deliverable may still be
    # adopted when the Server can prove that it belongs to the exact succeeded
    # action and authoritative begin call.  This is an ID/lineage join, never a
    # filename or "latest file" heuristic.
    unresolved_deliverable_ids = requested_deliverable_id_set - set(result_file_refs)
    if unresolved_deliverable_ids:
        late_file_rows = list(
            db.exec(
                select(ConversationFile)
                .where(ConversationFile.id.in_(sorted(unresolved_deliverable_ids)))
                .with_for_update()
            ).all()
        )
        for file_row in late_file_rows:
            if file_row.id is None:
                continue
            node_id = str(file_row.task_node_id or "").strip()
            action_node = succeeded_action_by_node_id.get(node_id)
            if action_node is None:
                continue
            metadata = _raw_artifact_metadata(file_row)
            artifact_key = _bounded_string(
                metadata.get("artifact_key") or metadata.get("path"),
                limit=_MAX_RECEIPT_ID_CHARS,
            )
            if not artifact_key.strip():
                continue
            authority = _conversation_file_registration_authority(file_row)
            artifact = {
                "artifact_key": artifact_key,
                "conversation_file_id": str(int(file_row.id)),
                "file_name": str(file_row.file_name or ""),
                "mime_type": str(
                    file_row.mime_type or "application/octet-stream"
                ),
                "size_bytes": int(file_row.size_bytes or 0),
                "sha256": str(authority.get("sha256") or ""),
                "drawer_section": "result_file",
                "registration_status": str(
                    authority.get("registration_status") or ""
                ),
            }
            try:
                _validate_registered_artifacts(
                    db,
                    run=run,
                    node=action_node,
                    artifacts=[dict(artifact)],
                )
            except TaskTreeContractError:
                continue
            if str(file_row.drawer_section or "").strip() != "result_file":
                continue
            result_file_refs[int(file_row.id)] = (action_node, artifact)
            action_projection = action_projection_by_node_id[node_id]
            action_projection["artifact_refs"] = sorted(
                [*list(action_projection.get("artifact_refs") or []), {
                    key: artifact[key]
                    for key in (
                        "artifact_key",
                        "conversation_file_id",
                        "drawer_section",
                        "sha256",
                    )
                }],
                key=lambda item: (
                    str(item.get("conversation_file_id") or ""),
                    str(item.get("artifact_id") or ""),
                ),
            )

    unresolved_deliverable_ids = requested_deliverable_id_set - set(result_file_refs)
    if unresolved_deliverable_ids:
        raise TaskTreeContractError(
            code="completion_deliverable_selection_invalid",
            message=(
                "selected completion files must belong to authoritative "
                "succeeded action receipts"
            ),
            details={"conversation_file_ids": sorted(unresolved_deliverable_ids)},
        )

    # When the caller supplies an explicit selection, that set is the entire
    # delivery contract.  Other historical result_file artifacts remain in the
    # action audit trail but must not leak into CompletionReceipt.deliverables.
    if requested_deliverable_id_set:
        result_file_refs = {
            file_id: value
            for file_id, value in result_file_refs.items()
            if file_id in requested_deliverable_id_set
        }

    deliverables: list[dict[str, Any]] = []
    if result_file_refs:
        file_rows = list(
            db.exec(
                select(ConversationFile)
                .where(ConversationFile.id.in_(sorted(result_file_refs)))
                .with_for_update()
            ).all()
        )
        files_by_id = {
            int(row.id): row for row in file_rows if row.id is not None
        }
        if set(files_by_id) != set(result_file_refs):
            raise TaskTreeContractError(
                code="completion_deliverable_missing",
                message="one or more completion deliverables no longer exist",
            )
        for file_id in sorted(result_file_refs):
            action_node, artifact = result_file_refs[file_id]
            file_row = files_by_id[file_id]
            _validate_registered_artifacts(
                db,
                run=run,
                node=action_node,
                artifacts=[dict(artifact)],
            )
            deliverables.append(
                {
                    "conversation_file_id": file_id,
                    "artifact_key": _bounded_string(
                        artifact.get("artifact_key"),
                        limit=_MAX_RECEIPT_ID_CHARS,
                    ),
                    "file_name": str(file_row.file_name or ""),
                    "mime_type": str(
                        file_row.mime_type or "application/octet-stream"
                    ),
                    "size_bytes": int(file_row.size_bytes or 0),
                    "sha256": _conversation_file_sha256(db, file_row),
                    "drawer_section": "result_file",
                    "task_node_id": str(action_node.node_id),
                    "tool_name": str(file_row.tool_name or "") or None,
                    "tool_run_id": str(file_row.tool_run_id or "") or None,
                }
            )

    supersessions: list[dict[str, Any]] = []
    for source_node_id, replacement_node_id in sorted(valid_supersessions.items()):
        source = row_by_id[source_node_id]
        source_arguments = dict(source.tool_arguments_json or {})
        source_contract = source_arguments.get(EXECUTION_CONTRACT_KEY) or {}
        record = _authoritative_supersession_record(source) or {}
        supersessions.append(
            {
                "source_node_id": source_node_id,
                "replacement_node_id": replacement_node_id,
                "obligation_id": str(source_contract.get("obligation_id") or ""),
                "reason_sha256": _canonical_sha256(
                    str(record.get("reason") or "")
                ),
                "superseded_at": str(record.get("superseded_at") or ""),
            }
        )

    unsigned = {
        "schema_version": COMPLETION_RECEIPT_SCHEMA,
        "authority": CAPABILITY_AUTHORITY,
        "run_id": str(run.run_id),
        "root_node_id": str(root_node_id),
        "request_id": str(run.request_id or "") or None,
        "project_id": str(run.project_id),
        "session_id": str(run.session_id),
        "user_id": int(run.user_id),
        "completed_at": completed_at.isoformat() + "Z",
        "tree_state_sha256": _canonical_sha256(tree_state_nodes),
        "summary_sha256": _canonical_sha256(str(result_summary or "")),
        "terminal_status": str(
            terminal_status
        ),
        "goal_outcome": str(
            completion_resolution.get("goal_outcome") or "achieved"
        ),
        "obligation_resolutions": list(
            completion_resolution.get("obligation_resolutions") or []
        ),
        "actions": sorted(actions, key=lambda item: str(item["node_id"])),
        "deliverables": deliverables,
        "supersessions": supersessions,
    }
    return {
        **unsigned,
        "receipt_id": "completion_" + _canonical_sha256(unsigned),
    }

# ORIGINAL L3864-L3965
def complete_task_root(
    db: Session,
    *,
    run_id: str,
    root_node_id: str,
    result_summary: str,
    deliverable_file_ids: list[int] | None = None,
    terminal_status: str = "completed",
) -> dict[str, Any]:
    run = db.exec(
        select(ProjectTaskTreeRun)
        .where(ProjectTaskTreeRun.run_id == run_id)
        .with_for_update()
    ).first()
    if run is None:
        raise ValueError(f"task tree run not found: {run_id}")
    if str(run.root_node_id or "") != str(root_node_id or ""):
        raise ValueError("task tree root node mismatch")
    mission_status = str(run.status or "").strip().lower()
    requested_terminal = str(terminal_status or "completed").strip().lower()
    if requested_terminal not in {"completed", "blocked"}:
        raise TaskTreeContractError(
            code="task_terminal_status_invalid",
            message="task terminal status must be completed or blocked",
        )
    if mission_status in {"completed", "blocked"}:
        existing_completion_receipt = _validated_completion_receipt(run)
        if mission_status != requested_terminal:
            raise TaskTreeContractError(
                code="completion_terminal_status_conflict",
                message=(
                    "terminal task run cannot be replayed with a different "
                    "terminal status"
                ),
                details={"run_id": str(run_id)},
            )
        replay_summary_sha256 = _canonical_sha256(str(result_summary or ""))
        if replay_summary_sha256 != str(
            existing_completion_receipt.get("summary_sha256") or ""
        ):
            raise TaskTreeContractError(
                code="completion_summary_conflict",
                message=(
                    "completed task run cannot be replayed with a different "
                    "result summary"
                ),
                details={"run_id": str(run_id)},
            )
        requested_ids = {int(item) for item in (deliverable_file_ids or [])}
        persisted_ids = {
            int(item.get("conversation_file_id"))
            for item in (existing_completion_receipt.get("deliverables") or [])
            if isinstance(item, dict) and item.get("conversation_file_id") is not None
        }
        if requested_ids and requested_ids != persisted_ids:
            raise TaskTreeContractError(
                code="completion_deliverable_selection_conflict",
                message=(
                    "completed task run cannot be replayed with different "
                    "deliverable file IDs"
                ),
                details={"run_id": str(run_id)},
            )
        db.commit()
        return build_snapshot(db, run_id=run_id)
    if mission_status in TERMINAL_MISSION_STATUSES:
        raise _terminal_run_error(run)
    rows = _lock_run_nodes(db, run_id=run_id)
    row_by_id = {str(row.node_id): row for row in rows}
    root = row_by_id.get(str(root_node_id))
    if root is None:
        raise ValueError(f"task node not found: {root_node_id}")
    completion_resolution = _validate_tree_completion(
        rows=rows,
        root_node_id=root_node_id,
        terminal_status=requested_terminal,
    )

    now = _now()
    completion_receipt = _build_completion_receipt(
        db,
        run=run,
        rows=rows,
        root_node_id=root_node_id,
        result_summary=result_summary,
        completed_at=now,
        completion_resolution=completion_resolution,
        deliverable_file_ids=deliverable_file_ids,
    )
    root.status = "completed" if requested_terminal == "completed" else "partial"
    root.result_summary = str(result_summary or "") or None
    root.completed_at = now
    root.updated_at = now
    run.status = requested_terminal
    run.status_reason = str(result_summary or "")
    run.completion_receipt_json = completion_receipt
    run.completed_at = now
    run.updated_at = now
    db.add(root)
    db.add(run)
    db.commit()
    return build_snapshot(db, run_id=run_id)
