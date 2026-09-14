# Reading excerpt; functions/classes retain their original bodies.
# Origin: server/backend/services/sandbox_service.py @ 8ee96f3171f20eb56c08e4a8aba6a43ab02f0f69
# Module setup, unselected helpers and service wiring are omitted.
# See source_manifest.json and docs/DEPENDENCIES.md. Do not execute this slice.

# ORIGINAL L1-L1
from __future__ import annotations

# ORIGINAL L3-L3
import io

# ORIGINAL L4-L4
import hashlib

# ORIGINAL L5-L5
import json

# ORIGINAL L6-L6
import logging

# ORIGINAL L7-L7
import math

# ORIGINAL L8-L8
import os

# ORIGINAL L9-L9
import re

# ORIGINAL L10-L10
import uuid

# ORIGINAL L11-L11
import zipfile

# ORIGINAL L12-L12
from datetime import UTC, datetime

# ORIGINAL L13-L13
from pathlib import Path

# ORIGINAL L14-L14
from types import SimpleNamespace

# ORIGINAL L15-L15
from urllib.parse import quote

# ORIGINAL L16-L16
from typing import Any

# ORIGINAL L18-L18
import gemmi

# ORIGINAL L19-L19
from fastapi import HTTPException

# ORIGINAL L20-L20
from redis import Redis

# ORIGINAL L21-L21
from sqlalchemy import MetaData, Table, func, inspect, text

# ORIGINAL L22-L22
from sqlmodel import Session, select

# ORIGINAL L24-L40
from database import (
    AssetFolder,
    AssetScope,
    AssetStatus,
    ConversationFile,
    ProjectAsset,
    RDRunStatus,
    RDTrafficLabel,
    RDWorkflowRun,
    SandboxApp,
    SandboxArtifact,
    SandboxBillingStatus,
    SandboxJob,
    SandboxJobStatus,
    User,
    WalletType,
)

# ORIGINAL L41-L41
from services import compute_point_service, wallet_service

# ORIGINAL L42-L42
from services import sandbox_pricing_service

# ORIGINAL L43-L43
from services import asset_upload_service

# ORIGINAL L44-L49
from sandbox_manifest_contract import (
    find_forbidden_option_tokens,
    manifest_version_for_app,
    output_contract_for_app,
    validate_submit_payload,
)

# ORIGINAL L50-L50
from sandbox.tool_registry import get_runtime_registry

# ORIGINAL L51-L51
from sandbox.submit_adapters import adapt_submit_params

# ORIGINAL L52-L52
from utils.cos_client import cos_helper

# ORIGINAL L53-L53
from utils.posthog_client import capture as ph_capture

# ORIGINAL L54-L60
from sandbox.converters import (
    BLAST_ARTIFACT_FORMATS,
    CONVERTERS,
    artifact_formats_for_key,
    blast_artifact_file_name,
    sandbox_artifact_file_name,
)

# ORIGINAL L705-L712
def build_job_terminal_callback_payload(db: Session, *, job_id: str) -> dict[str, Any]:
    """Build the internal callback contract without exposing owner IDs publicly."""
    job = db.exec(select(SandboxJob).where(SandboxJob.job_id == job_id)).first()
    if job is None:
        return {}
    payload = _serialize_job(job)
    payload["user_id"] = int(job.user_id)
    return payload

# ORIGINAL L1156-L1164
def get_job_or_404(db: Session, *, user_id: int, job_id: str) -> Any:
    if _sandbox_jobs_has_resume_columns(db):
        row = db.exec(select(SandboxJob).where(SandboxJob.job_id == job_id, SandboxJob.user_id == user_id)).first()
    else:
        rows = _load_legacy_sandbox_job_projection(db, user_id=user_id, job_id=job_id)
        row = rows[0] if rows else None
    if not row:
        raise HTTPException(status_code=404, detail="作业不存在")
    return row

# ORIGINAL L1225-L1234
def _normalize_job_timeout_seconds(value: Any) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="任务超时设置必须是整数秒") from exc
    if normalized < MIN_JOB_TIMEOUT_SECONDS:
        raise HTTPException(status_code=400, detail=f"任务超时设置不能小于 {MIN_JOB_TIMEOUT_SECONDS} 秒")
    if normalized > MAX_JOB_TIMEOUT_SECONDS:
        raise HTTPException(status_code=400, detail=f"任务超时设置不能超过 {MAX_JOB_TIMEOUT_SECONDS} 秒")
    return normalized

# ORIGINAL L1261-L1292
def update_job_timeout(
    db: Session,
    *,
    user_id: int,
    job_id: str,
    max_run_duration_seconds: Any,
) -> dict[str, Any]:
    normalized_timeout = _normalize_job_timeout_seconds(max_run_duration_seconds)
    job = get_job_or_404(db, user_id=user_id, job_id=job_id)
    current_status = _normalized_status_text(getattr(job, "status", "") or "queued")
    if current_status not in ACTIVE_JOB_STATUSES:
        raise HTTPException(status_code=400, detail="仅排队中或运行中的任务可以修改超时设置")
    now = _now()
    if _sandbox_jobs_has_resume_columns(db) and isinstance(job, SandboxJob):
        job.max_run_duration_seconds = normalized_timeout
        job.updated_at = now
        db.add(job)
        db.flush()
        db.refresh(job)
        return {"job": _serialize_job(job)}

    bind = db.get_bind()
    table = Table("sandbox_jobs", MetaData(), autoload_with=bind)
    db.execute(
        table.update()
        .where(table.c.job_id == job_id, table.c.user_id == int(user_id))
        .values(max_run_duration_seconds=normalized_timeout, updated_at=now)
    )
    payload = dict(vars(job))
    payload["max_run_duration_seconds"] = normalized_timeout
    payload["updated_at"] = now
    return {"job": _serialize_job_projection(SimpleNamespace(**payload))}
