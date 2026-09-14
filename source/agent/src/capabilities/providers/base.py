"""Minimal provider interface used by the capability executor."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import Field, model_validator

from src.capabilities.models import (
    AgentMode,
    ArtifactRef,
    ArtifactSpec,
    CapabilityError,
    CapabilityErrorKind,
    CapabilityNeedsInput,
    CapabilityOutcome,
    CapabilityStatus,
    EvidenceRef,
    ProviderType,
    ResourceRef,
    SourceCandidateRef,
    SourceOutcome,
    StrictModel,
)
from src.context.task_state import TaskPhase


class ProviderContext(StrictModel):
    # request_id is the immutable TaskTree/business authority.  The transport
    # request is carried separately so HITL/sandbox continuations never retag
    # task-bound files, sources or receipts.
    request_id: str = Field(min_length=1, max_length=512)
    transport_request_id: str = Field(min_length=1, max_length=512)
    task_phase: TaskPhase = TaskPhase.EXECUTE
    agent_mode: AgentMode | None = Field(default=None, exclude=True)
    capability_id: str = Field(min_length=1, max_length=512)
    capability_version: str = Field(min_length=1, max_length=128)
    project_id: str | None = Field(default=None, max_length=512)
    conversation_id: str | None = Field(default=None, max_length=512)
    user_id: int | None = Field(default=None, ge=1)
    provider_ref: dict[str, Any] = Field(default_factory=dict)
    evidence_policy: dict[str, Any] = Field(default_factory=dict)
    artifact_specs: list[ArtifactSpec] = Field(default_factory=list)
    timeout_seconds: int = Field(default=120, ge=1, le=86_400)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderArtifactSource(StrictModel):
    artifact_id: str = Field(min_length=1, max_length=512)
    local_path: str = Field(min_length=1, max_length=4000)


class ProviderInvocationResult(StrictModel):
    status: CapabilityStatus
    summary: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    contract_data: dict[str, Any] | None = Field(default=None, exclude=True)
    citation_projection: list[dict[str, Any]] = Field(default_factory=list)
    source_candidates: list[SourceCandidateRef] = Field(default_factory=list)
    resources: list[ResourceRef] = Field(default_factory=list)
    complete: bool = True
    has_more: bool = False
    cursor: str | None = Field(default=None, max_length=4000)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    capability_outcome: CapabilityOutcome = Field(default_factory=CapabilityOutcome)
    source_outcome: SourceOutcome = Field(default_factory=SourceOutcome)
    error: CapabilityError | None = None
    needs_input_control: CapabilityNeedsInput | None = Field(default=None, exclude=True)
    raw_ref: str | None = Field(default=None, max_length=4000)
    idempotency_reused: bool = False
    legacy_envelope: dict[str, Any] | None = Field(default=None, exclude=True)
    artifact_sources: list[ProviderArtifactSource] = Field(default_factory=list, exclude=True)

    @model_validator(mode="after")
    def validate_error_consistency(self) -> "ProviderInvocationResult":
        successful = self.status in {CapabilityStatus.SUCCEEDED, CapabilityStatus.PENDING}
        if successful and self.error is not None:
            raise ValueError("successful provider results must not include error")
        if not successful and self.error is None:
            raise ValueError("failed provider results require error")
        if self.status == CapabilityStatus.NEEDS_INPUT and (
            self.error is None or self.error.kind != CapabilityErrorKind.NEEDS_INPUT
        ):
            raise ValueError("needs_input provider results require a needs_input error")
        if self.needs_input_control is not None and self.status != CapabilityStatus.NEEDS_INPUT:
            raise ValueError("needs_input_control requires needs_input status")
        if self.status == CapabilityStatus.NEEDS_INPUT and self.error is not None:
            if not self.error.missing_fields and self.needs_input_control is None:
                raise ValueError(
                    "needs_input requires missing_fields or a typed needs_input_control"
                )
        if self.status == CapabilityStatus.CANCELLED and (
            self.error is None or self.error.kind != CapabilityErrorKind.CANCELLED
        ):
            raise ValueError("cancelled provider results require a cancelled error")
        artifact_ids = [item.artifact_id for item in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("provider artifacts require unique artifact_id values")
        unknown_sources = [
            item.artifact_id for item in self.artifact_sources if item.artifact_id not in artifact_ids
        ]
        if unknown_sources:
            raise ValueError("artifact source does not match a provider artifact")
        if self.complete and self.has_more:
            raise ValueError("complete provider results cannot also have_more")
        if self.complete and self.cursor is not None:
            raise ValueError("complete provider results cannot expose a continuation cursor")
        if self.has_more and not self.cursor:
            raise ValueError("has_more provider results require a continuation cursor")
        if not self.complete and not self.has_more and successful:
            raise ValueError("incomplete provider results must declare has_more")
        resource_ids = [item.resource_id for item in self.resources]
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("provider resources require unique resource_id values")
        candidate_ids = [item.candidate_id for item in self.source_candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("provider source candidates require unique candidate_id values")
        missing_candidate_resources = [
            item.result_resource_id
            for item in self.source_candidates
            if item.result_resource_id not in resource_ids
        ]
        if missing_candidate_resources:
            raise ValueError("provider source candidate must reference a returned resource")
        return self


@runtime_checkable
class CapabilityProvider(Protocol):
    provider_type: ProviderType

    async def invoke(
        self,
        *,
        arguments: dict[str, Any],
        input_refs: list[dict[str, Any]],
        context: ProviderContext,
    ) -> ProviderInvocationResult: ...
