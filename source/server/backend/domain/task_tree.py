from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_serializer


def _utc_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    aware = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return aware.isoformat().replace("+00:00", "Z")

TaskNodeStatus = Literal[
    "pending",
    "running",
    "waiting",
    "partial",
    "completed",
    "skipped",
    "failed",
    "cancelled",
]
MissionStatus = Literal[
    "active",
    "waiting_human",
    "waiting_external",
    "completed",
    "blocked",
    "cancelled",
]
TaskNodeKind = Literal[
    "goal",
    "phase",
    "public_retrieval",
    "user_asset_retrieval",
    "biology_tool",
    "sandbox",
    "document",
    "review",
]
TaskExecutionMode = Literal["immediate", "blocking_short", "nonblocking_long"]


class TaskTreeNode(BaseModel):
    node_id: str
    parent_node_id: str | None = None
    title: str
    description: str = ""
    node_kind: TaskNodeKind = "phase"
    status: TaskNodeStatus = "pending"
    tool_name: str | None = None
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    execution_contract: dict[str, Any] | None = None
    required: bool = False
    superseded_by_node_id: str | None = None
    supersession_valid: bool = False
    execution_mode: TaskExecutionMode = "immediate"
    depends_on_node_ids: list[str] = Field(default_factory=list)
    skill_id: str | None = None
    mcp_server: str | None = None
    sandbox_job_ids: list[str] = Field(default_factory=list)
    result_summary: str | None = None
    artifacts: list[dict] = Field(default_factory=list)
    capability_receipt_ref: dict[str, Any] | None = None
    resolution_outcome: str | None = None
    resolution_reason_code: str | None = None
    assigned_session_id: str | None = None
    created_by_session_id: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    children: list["TaskTreeNode"] = Field(default_factory=list)

    @field_serializer("created_at", "started_at", "completed_at", when_used="json")
    def serialize_utc_times(self, value: datetime | None) -> str | None:
        return _utc_iso(value)


class TaskTreeSnapshot(BaseModel):
    run_id: str
    request_id: str | None = None
    project_id: str
    session_id: str
    title: str
    mission_status: MissionStatus = "active"
    status: MissionStatus = "active"
    goal_summary: str = ""
    status_reason: str = ""
    completion_receipt: dict[str, Any] | None = None
    root: TaskTreeNode
    updated_at: datetime | None = None

    @field_serializer("updated_at", when_used="json")
    def serialize_updated_at(self, value: datetime | None) -> str | None:
        return _utc_iso(value)


TaskTreeNode.model_rebuild()
