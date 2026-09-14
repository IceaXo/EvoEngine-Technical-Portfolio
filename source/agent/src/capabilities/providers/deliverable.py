"""Stateless deliverable planner backed by a frozen child capability executor."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import Field, model_validator

from src.capabilities.executor import CapabilityExecutor
from src.capabilities.idempotency import build_idempotency_key
from src.capabilities.models import (
    CapabilityCall,
    CapabilityNeedsInput,
    CapabilitySpec,
    CapabilityStatus,
    DiscoveryPolicy,
    Permission,
    ProviderType,
    ResourceRef,
    StrictModel,
)
from src.capabilities.providers.base import ProviderContext, ProviderInvocationResult
from src.capabilities.registry import CapabilityRegistry
from src.schemas.image_region_edits import ImageRegionEditSpec


DELIVERABLE_TOOL_NAME = "deliverable_task_submit"
DELIVERABLE_CAPABILITY_ID = "deliverable.task.submit"
_ATOM_IDENTITIES = (
    "office.blueprint.create@1.0.0",
    "office.blueprint.patch@1.0.0",
    "office.render.execute@1.0.0",
    "conversation_file.text_candidate.create@1.0.0",
    "conversation_file.image_region_candidate.create@1.0.0",
    "scientific_figure.context.read@1.0.0",
    "scientific_figure.run.submit@1.0.0",
    "scientific_figure.run.status@1.0.0",
)


class DeliverableGuardedLineEdit(StrictModel):
    operation: Literal["replace", "delete", "insert_before", "insert_after"] = Field(
        description="Exact line operation; never use JSON Patch op/path/value fields."
    )
    start_line: int = Field(ge=1, description="1-based first source line.")
    end_line: int | None = Field(
        default=None,
        ge=1,
        description="Required for replace/delete and must be >= start_line.",
    )
    expected_text: str = Field(
        max_length=500_000,
        description=(
            "Exact current source slice including indentation. The terminal line delimiter may be omitted; "
            "the service preserves it for replacements."
        ),
    )
    replacement_text: str = Field(
        default="",
        max_length=500_000,
        description="Exact replacement text; empty only for delete.",
    )

    @model_validator(mode="after")
    def validate_edit(self) -> "DeliverableGuardedLineEdit":
        if self.operation in {"replace", "delete"}:
            if self.end_line is None or self.end_line < self.start_line:
                raise ValueError("replace/delete require end_line >= start_line")
        elif self.end_line is not None and self.end_line != self.start_line:
            raise ValueError("insert operations accept only the anchor start_line")
        if self.operation == "delete" and self.replacement_text:
            raise ValueError("delete cannot contain replacement_text")
        if self.operation in {"insert_before", "insert_after"} and not self.replacement_text:
            raise ValueError("insert operations require replacement_text")
        return self


class DeliverableScientificFigureContext(StrictModel):
    family_id: int = Field(ge=1)
    family_lock_version: int = Field(ge=0)
    source_lock_version: int = Field(ge=0)
    source_version_file_id: int = Field(ge=1)
    source_resource: ResourceRef
    figure_key: str = Field(min_length=1, max_length=120)
    display_name: str = Field(min_length=1, max_length=300)


class DeliverableOfficeAsset(StrictModel):
    binding_key: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")
    conversation_file_id: int = Field(ge=1)
    sha256: str = Field(default="", pattern=r"^(?:|[a-f0-9]{64})$")
    caption: str = Field(default="", max_length=4000)
    alt_text: str = Field(default="", max_length=4000)


class DeliverableOfficeBlock(StrictModel):
    type: Literal["heading", "paragraph", "bullets", "list", "table", "image", "figure", "page_break"]
    text: str = Field(default="", max_length=100_000)
    level: int | None = Field(default=None, ge=1, le=6)
    items: list[str] = Field(default_factory=list, max_length=500)
    headers: list[str] = Field(default_factory=list, max_length=100)
    rows: list[list[str]] = Field(default_factory=list, max_length=1000)
    binding_key: str = Field(default="", max_length=120)
    asset_binding: str = Field(default="", max_length=120)
    caption: str = Field(default="", max_length=4000)


class DeliverableOfficeSlide(StrictModel):
    layout: Literal["title", "bullets", "figure"] = "bullets"
    title: str = Field(default="", max_length=10_000)
    subtitle: str = Field(default="", max_length=10_000)
    body: str = Field(default="", max_length=100_000)
    bullets: list[str] = Field(default_factory=list, max_length=100)
    image_key: str = Field(default="", max_length=120)
    caption: str = Field(default="", max_length=4000)


class DeliverableOfficeSourceSpec(StrictModel):
    kind: Literal["docx", "pptx"] | None = None
    title: str = Field(default="", max_length=10_000)
    blocks: list[DeliverableOfficeBlock] = Field(default_factory=list, max_length=2000)
    slides: list[DeliverableOfficeSlide] = Field(default_factory=list, max_length=60)


class DeliverableTaskInput(StrictModel):
    action: Literal[
        "create_office_blueprint",
        "patch_office_blueprint",
        "render_office",
        "create_text_candidate",
        "create_image_region_candidate",
        "read_scientific_figure_context",
        "submit_scientific_figure_run",
        "read_scientific_figure_status",
    ]
    target: ResourceRef | None = Field(
        default=None,
        description=(
            "For text/image candidate actions: the exact active ConversationFile ResourceRef. "
            "For an image region, copy target from resource_read(selection resource)."
        ),
    )
    selection: ResourceRef | None = Field(
        default=None,
        description=(
            "For image candidate or Office blueprint patch: the canonical selection ResourceRef "
            "from the current turn reference catalog."
        ),
    )
    image_edit: ImageRegionEditSpec | None = Field(
        default=None,
        description=(
            "Only for create_image_region_candidate. Choose one fixed operation; arbitrary code "
            "or browser workspace payloads are never accepted."
        ),
    )
    target_conversation_file_id: int | None = Field(
        default=None,
        ge=1,
        description="Only for read_scientific_figure_context: the managed figure file ID.",
    )
    scientific_context: DeliverableScientificFigureContext | None = Field(
        default=None,
        description=(
            "Only for submit_scientific_figure_run: copy the entire context returned by "
            "read_scientific_figure_context without splitting out its fields."
        ),
    )
    content_text: str | None = Field(default=None, max_length=500_000)
    summary: str | None = Field(default=None, max_length=1000)
    expected_lock_version: int | None = Field(default=None, ge=0)
    family_id: int | None = Field(default=None, ge=1)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)
    edits: list[DeliverableGuardedLineEdit] = Field(
        default_factory=list,
        max_length=100,
        description=(
            "Exact guarded source-line edits using operation, start_line, end_line, "
            "expected_text and replacement_text."
        ),
    )
    run_id: str | None = Field(default=None, pattern=r"^sfr_[a-f0-9]{32}$")
    deliverable_kind: Literal["docx", "pptx"] | None = None
    name: str | None = Field(default=None, min_length=1, max_length=180)
    blueprint_markdown: str | None = Field(default=None, min_length=1, max_length=250_000)
    blueprint_root_file_id: int | None = Field(default=None, ge=1)
    expected_blueprint_lock_version: int | None = Field(default=None, ge=0)
    source_spec: DeliverableOfficeSourceSpec | None = None
    assets: list[DeliverableOfficeAsset] = Field(default_factory=list, max_length=100)
    output_file_name: str | None = Field(default=None, max_length=180)
    office_root_file_id: int | None = Field(default=None, ge=1)
    expected_office_version_file_id: int | None = Field(default=None, ge=1)
    expected_office_lock_version: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_action(self) -> "DeliverableTaskInput":
        required = {
            "create_office_blueprint": (
                "deliverable_kind", "name", "blueprint_markdown", "idempotency_key",
            ),
            "patch_office_blueprint": (
                "target", "selection", "expected_lock_version", "idempotency_key",
            ),
            "render_office": (
                "target", "blueprint_root_file_id", "expected_blueprint_lock_version",
                "deliverable_kind", "name", "source_spec", "idempotency_key",
            ),
            "create_text_candidate": (
                "target", "content_text", "expected_lock_version", "idempotency_key"
            ),
            "create_image_region_candidate": (
                "target", "selection", "image_edit", "expected_lock_version", "idempotency_key"
            ),
            "read_scientific_figure_context": ("target_conversation_file_id",),
            "submit_scientific_figure_run": (
                "scientific_context", "idempotency_key",
            ),
            "read_scientific_figure_status": ("family_id", "run_id"),
        }[self.action]
        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise ValueError("missing action fields: " + ", ".join(missing))
        if self.action in {"submit_scientific_figure_run", "patch_office_blueprint"} and not self.edits:
            raise ValueError("this action requires guarded edits")
        if self.action not in {"submit_scientific_figure_run", "patch_office_blueprint"} and self.edits:
            raise ValueError("edits are only valid for source or blueprint patches")
        if self.action == "read_scientific_figure_context" and self.target is not None:
            raise ValueError("scientific figure context read accepts only target_conversation_file_id")
        if self.action != "read_scientific_figure_context" and self.target_conversation_file_id is not None:
            raise ValueError("target_conversation_file_id is only valid for scientific figure context read")
        if self.action != "submit_scientific_figure_run" and self.scientific_context is not None:
            raise ValueError("scientific_context is only valid for scientific figure submission")
        if self.action == "submit_scientific_figure_run" and self.target is not None:
            raise ValueError("scientific figure submission accepts scientific_context, not target")
        if self.action not in {"create_image_region_candidate", "patch_office_blueprint"} and self.selection is not None:
            raise ValueError("selection is only valid for image or Office blueprint patches")
        if self.action != "create_image_region_candidate" and self.image_edit is not None:
            raise ValueError("image_edit is only valid for image region candidates")
        if self.action == "create_image_region_candidate" and self.content_text is not None:
            raise ValueError("image region candidates do not accept content_text")
        return self


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atom_plan(task: DeliverableTaskInput) -> tuple[str, dict[str, Any]]:
    payload = task.model_dump(mode="json", exclude_none=True)
    action = payload.pop("action")
    if action == "create_office_blueprint":
        return "office.blueprint.create", {
            "deliverable_kind": task.deliverable_kind,
            "name": task.name,
            "blueprint_markdown": task.blueprint_markdown,
            "idempotency_key": task.idempotency_key,
        }
    if action == "patch_office_blueprint":
        office_patch_payload = {
            "target": task.target.model_dump(mode="json") if task.target else None,
            "selection": task.selection.model_dump(mode="json") if task.selection else None,
            "expected_lock_version": task.expected_lock_version,
            "idempotency_key": task.idempotency_key,
            "summary": task.summary,
            "edits": [item.model_dump(mode="json") for item in task.edits],
        }
        return "office.blueprint.patch", {
            key: value for key, value in office_patch_payload.items() if value is not None
        }
    if action == "render_office":
        office_payload = {
            "target": task.target.model_dump(mode="json") if task.target else None,
            "blueprint_root_file_id": task.blueprint_root_file_id,
            "expected_blueprint_lock_version": task.expected_blueprint_lock_version,
            "deliverable_kind": task.deliverable_kind,
            "name": task.name,
            "source_spec": task.source_spec.model_dump(mode="json", exclude_none=True) if task.source_spec else {},
            "assets": [item.model_dump(mode="json") for item in task.assets],
            "output_file_name": task.output_file_name,
            "office_root_file_id": task.office_root_file_id,
            "expected_office_version_file_id": task.expected_office_version_file_id,
            "expected_office_lock_version": task.expected_office_lock_version,
            "idempotency_key": task.idempotency_key,
        }
        return "office.render.execute", {
            key: value for key, value in office_payload.items() if value is not None
        }
    if action == "create_text_candidate":
        for name in (
            "family_id", "scientific_context", "edits", "run_id", "selection", "image_edit", "assets",
        ):
            payload.pop(name, None)
        return "conversation_file.text_candidate.create", payload
    if action == "create_image_region_candidate":
        edit = payload.pop("image_edit")
        payload["edit"] = edit
        for name in (
            "content_text", "family_id", "scientific_context", "edits", "run_id",
            "target_conversation_file_id", "assets",
        ):
            payload.pop(name, None)
        return "conversation_file.image_region_candidate.create", payload
    if action == "read_scientific_figure_context":
        return "scientific_figure.context.read", {
            "conversation_file_id": payload["target_conversation_file_id"]
        }
    if action == "submit_scientific_figure_run":
        scientific_context = payload.pop("scientific_context")
        payload["target"] = scientific_context["source_resource"]
        payload["family_id"] = scientific_context["family_id"]
        payload["expected_family_lock_version"] = scientific_context["family_lock_version"]
        payload["expected_source_lock_version"] = scientific_context["source_lock_version"]
        payload["expected_source_version_file_id"] = scientific_context["source_version_file_id"]
        payload.pop("content_text", None)
        payload.pop("summary", None)
        payload.pop("expected_lock_version", None)
        payload.pop("run_id", None)
        payload.pop("assets", None)
        return "scientific_figure.run.submit", payload
    return "scientific_figure.run.status", {
        "family_id": int(task.family_id or 0),
        "run_id": str(task.run_id or ""),
    }


def deliverable_capability_spec(
    *,
    discovery: DiscoveryPolicy | dict[str, Any],
) -> CapabilitySpec:
    normalized_discovery = (
        discovery
        if isinstance(discovery, DiscoveryPolicy)
        else DiscoveryPolicy.model_validate(discovery)
    )
    return CapabilitySpec(
        capability_id=DELIVERABLE_CAPABILITY_ID,
        version="1.0.0",
        display_name="Deliverable task submit",
        description=(
            "Revise an existing Markdown/text ResourceRef, apply one fixed operation to an "
            "authoritative ordinary-image region, or submit/read a "
            "managed scientific figure run. For a figure edit, first read_scientific_figure_context "
            "with the referenced figure conversation_file_id, then resource_read the returned source ResourceRef, and "
            "finally submit_scientific_figure_run with guarded edits and the returned locks. "
            "It does not create a new target-less document. "
            "Candidates are never automatically selected or accepted."
        ),
        provider_type=ProviderType.SUBAGENT,
        provider_ref={"tool_name": DELIVERABLE_TOOL_NAME, "handler_kind": "deliverable_planner"},
        input_schema=DeliverableTaskInput.model_json_schema(),
        input_refs_schema={"type": "array", "maxItems": 0},
        output_schema={"type": "object"},
        permission=Permission.WRITE,
        side_effects=["plans_and_delegates_candidate_operation"],
        requires_confirmation=False,
        discovery=normalized_discovery,
    )


def build_deliverable_subagent_tool() -> StructuredTool:
    def _unreachable(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("deliverable tool must execute through SubagentProvider")

    tool = StructuredTool.from_function(
        func=_unreachable,
        name=DELIVERABLE_TOOL_NAME,
        description=(
            "基于既有 ResourceRef 提交文本候选、普通图片权威选区的固定操作候选，或科研图受管重跑；"
            "图片选区先 resource_read 获取 target/lock，只允许 blur、pixelate、grayscale、fill、crop。"
            "不能从零生成文件；仅创建候选，不会自动选择、接受或删除文件。"
        ),
        args_schema=DeliverableTaskInput,
    )
    setattr(tool, "_evo_provider_type", ProviderType.SUBAGENT.value)
    setattr(tool, "_evo_capability_id", DELIVERABLE_CAPABILITY_ID)
    setattr(tool, "_evo_tool_category", "deliverable")
    return tool


def build_deliverable_handler(
    *,
    executor: CapabilityExecutor,
    registry: CapabilityRegistry,
):
    snapshot = registry.freeze(_ATOM_IDENTITIES)
    scoped = executor.scoped(snapshot)

    async def _handler(
        arguments: dict[str, Any],
        input_refs: list[dict[str, Any]],
        context: ProviderContext,
    ) -> ProviderInvocationResult:
        if input_refs:
            raise ValueError("deliverable planner does not accept input_refs")
        task = DeliverableTaskInput.model_validate(arguments)
        atom_id, atom_arguments = _atom_plan(task)
        atom_call_id = f"{context.metadata.get('call_id')}:atom"
        atom_call = CapabilityCall(
            call_id=atom_call_id,
            capability_id=atom_id,
            capability_version="1.0.0",
            request_id=context.request_id,
            project_id=context.project_id,
            conversation_id=context.conversation_id,
            user_id=context.user_id,
            arguments=atom_arguments,
            registry_snapshot_id=snapshot.snapshot_id,
            idempotency_key=build_idempotency_key(
                capability_id=atom_id,
                capability_version="1.0.0",
                arguments=atom_arguments,
                project_id=context.project_id,
                conversation_id=context.conversation_id,
                registry_snapshot_id=snapshot.snapshot_id,
            ),
        )
        outer_authorization = context.metadata.get("capability_authorization")
        child_metadata: dict[str, Any] = {}
        if isinstance(outer_authorization, dict):
            child_metadata["capability_authorization"] = {
                **outer_authorization,
                "authorized_capability_id": DELIVERABLE_CAPABILITY_ID,
                "authorized_action": task.action,
                "authorized_request_digest": str(
                    outer_authorization.get("request_digest") or ""
                ),
            }
        result = await scoped.invoke(atom_call, provider_metadata=child_metadata)
        if result.needs_input_control is not None:
            child = result.needs_input_control
            outer_digest = _canonical_digest(arguments)
            mapped = CapabilityNeedsInput(
                question_bundle=child.question_bundle,
                pending_call_id=str(context.metadata.get("call_id") or ""),
                capability_id=DELIVERABLE_CAPABILITY_ID,
                capability_version=context.capability_version,
                registry_snapshot_id=str(context.metadata.get("registry_snapshot_id") or ""),
                action=task.action,
                target=child.target,
                target_version_id=child.target_version_id,
                target_lock_version=child.target_lock_version,
                target_sha256=child.target_sha256,
                request_digest=outer_digest,
                effect_capability_id=child.capability_id,
                effect_request_digest=child.request_digest,
                authorization_challenge=_canonical_digest(
                    {
                        "outer_call_id": context.metadata.get("call_id"),
                        "atom_snapshot_id": snapshot.snapshot_id,
                        "atom_id": atom_id,
                        "target": child.target.model_dump(mode="json"),
                        "request_digest": outer_digest,
                        "effect_capability_id": child.capability_id,
                        "effect_request_digest": child.request_digest,
                    }
                ),
                authorization_question_id=child.authorization_question_id,
                approve_option_id=child.approve_option_id,
                reject_option_id=child.reject_option_id,
            )
            return ProviderInvocationResult(
                status=CapabilityStatus.NEEDS_INPUT,
                summary="Deliverable operation requires user confirmation",
                error=result.error,
                needs_input_control=mapped,
            )
        return ProviderInvocationResult(
            status=result.status,
            summary=result.summary,
            data={
                "brief": {
                    "schema_version": "evoengine.deliverable-action-brief/v1",
                    "action": task.action,
                    "atom_capability_id": atom_id,
                    "target_resource_id": (
                        task.target.resource_id
                        if task.target
                        else (
                            task.scientific_context.source_resource.resource_id
                            if task.scientific_context
                            else (
                                f"resource:conversation-file:{task.target_conversation_file_id}"
                                if task.target_conversation_file_id
                                else None
                            )
                        )
                    ),
                },
                "result": result.data,
            },
            resources=list(result.resources),
            artifacts=list(result.artifacts),
            error=result.error,
            idempotency_reused=result.idempotency_reused,
        )

    return _handler
