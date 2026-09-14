# Reading excerpt; functions/classes retain their original bodies.
# Origin: server/backend/database.py @ 8ee96f3171f20eb56c08e4a8aba6a43ab02f0f69
# Module setup, unselected helpers and service wiring are omitted.
# See source_manifest.json and docs/DEPENDENCIES.md. Do not execute this slice.

# ORIGINAL L1-L1
import os

# ORIGINAL L2-L2
import enum

# ORIGINAL L3-L3
from datetime import datetime

# ORIGINAL L4-L4
from decimal import Decimal

# ORIGINAL L5-L5
from pathlib import Path

# ORIGINAL L6-L6
from typing import Optional

# ORIGINAL L7-L7
from sqlmodel import Field, SQLModel, create_engine, Session

# ORIGINAL L8-L8
from dotenv import load_dotenv

# ORIGINAL L9-L18
from sqlalchemy import (
    CheckConstraint,
    Column,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    UniqueConstraint,
)

# ORIGINAL L19-L19
from sqlalchemy.sql import text as sql_text

# ORIGINAL L741-L781
class ProjectTaskTreeRun(SQLModel, table=True):
    __tablename__ = "project_task_tree_runs"
    __table_args__ = (
        UniqueConstraint("run_id", name="uq_project_task_tree_run_id"),
        Index("ix_project_task_tree_runs_user_project_session", "user_id", "project_id", "session_id"),
        Index(
            "uq_project_task_tree_runs_active_session",
            "user_id",
            "project_id",
            "session_id",
            unique=True,
            postgresql_where=sql_text(
                "status IN ('active', 'waiting_human', 'waiting_external', "
                "'running', 'waiting', 'pending')"
            ),
            sqlite_where=sql_text(
                "status IN ('active', 'waiting_human', 'waiting_external', "
                "'running', 'waiting', 'pending')"
            ),
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True, description="任务树运行 ID")
    user_id: int = Field(index=True, description="所属用户 ID")
    project_id: str = Field(index=True, description="项目 ID")
    session_id: str = Field(index=True, description="稳定会话 ID")
    request_id: Optional[str] = Field(default=None, index=True, description="触发本次运行的请求 ID")
    title: str = Field(default="执行任务树", description="根任务标题")
    status: str = Field(default="running", index=True, description="任务树整体状态")
    goal_summary: str = Field(default="", description="总体目标摘要")
    status_reason: str = Field(default="", description="任务树状态原因说明")
    completion_receipt_json: dict = Field(
        default_factory=dict,
        sa_column=Column(JSON),
        description="服务端原子完成后生成的不可变 CompletionReceipt",
    )
    root_node_id: str = Field(index=True, description="根节点 ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)

# ORIGINAL L849-L905
class ProjectConversationTurn(SQLModel, table=True):
    __tablename__ = "project_conversation_turns"
    __table_args__ = (
        UniqueConstraint("project_session_id", "request_id", name="uq_project_turn_request"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_session_id: int = Field(index=True, description="关联 project_sessions.id")
    conversation_session_id: Optional[int] = Field(default=None, index=True, description="关联 project_conversation_sessions.id")
    conversation_id: Optional[str] = Field(default=None, index=True, description="会话 ID")
    user_id: int = Field(index=True, description="所属用户 ID")
    request_id: str = Field(index=True, description="链路请求 ID")
    mode: str = Field(default="chat", description="Legacy 字段：旧版对话模式，仅保留历史数据兼容")
    input_text: str = Field(default="", description="用户输入")
    final_reply_text: Optional[str] = Field(default=None, description="最终回复")
    status: str = Field(default="running", description="轮次状态 running/success/failed/cancelled")
    error_code: Optional[str] = Field(default=None, description="失败码")
    error_message: Optional[str] = Field(default=None, description="失败信息")
    consumed_credits: Decimal = Field(
        default=Decimal("0.0000"),
        sa_column=Column(Numeric(18, 4), nullable=False, default=Decimal("0.0000")),
        description="本轮算力消耗",
    )
    output_char_count: int = Field(default=0, description="本轮智能体输出字数（审计用，不参与计费）")
    # Legacy ratio snapshot kept for read-compat with old records; new records use token fields.
    token_ratio_snapshot: Decimal = Field(
        default=Decimal("0.0000"),
        sa_column=Column(Numeric(18, 4), nullable=False, default=Decimal("0.0000")),
        description="旧版字符计费系数快照（已废弃，仅旧记录兼容）",
    )
    billing_version: str = Field(default="token_weighted_v1", description="计费算法版本")
    # token_weighted_v1 billing fields
    input_tokens: int = Field(default=0, description="本轮 LLM API 累计 input token")
    output_tokens: int = Field(default=0, description="本轮 LLM API 累计 output token")
    model_name: Optional[str] = Field(default=None, max_length=128, description="计费所用模型名称")
    billing_w_in_snapshot: Decimal = Field(
        default=Decimal("0.0000"),
        sa_column=Column(Numeric(18, 4), nullable=False, default=Decimal("0.0000")),
        description="计费时 W_in 权重快照",
    )
    billing_w_out_snapshot: Decimal = Field(
        default=Decimal("0.0000"),
        sa_column=Column(Numeric(18, 4), nullable=False, default=Decimal("0.0000")),
        description="计费时 W_out 权重快照",
    )
    model_weight_snapshot: Decimal = Field(
        default=Decimal("1.0000"),
        sa_column=Column(Numeric(18, 4), nullable=False, default=Decimal("1.0000")),
        description="计费时 ModelWeight 快照",
    )
    thinking_trace_json: Optional[dict] = Field(
        default=None,
        sa_column=Column(JSON),
        description="思考过程时间轴 JSON：schema_version + events",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
