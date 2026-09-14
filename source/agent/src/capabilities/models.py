"""Pydantic models shared by all capability providers."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.context.task_state import TaskPhase
from src.schemas.hitl import HumanQuestionBundle


_CAPABILITY_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]*$")
_DISCOVERY_TAXONOMY_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]*$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


REFERENCE_RESOURCE_STORAGE_SCHEMES: dict[str, str] = {
    "conversation_file": "conversation-file",
    "conversation_file_selection": "conversation-file-selection",
    "project_asset": "project-asset",
    "conversation_turn": "conversation-turn",
    "conversation_turn_message": "conversation-turn-message",
    "asset_folder": "asset-folder",
    "reference_manifest": "reference-manifest",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ProviderType(StrEnum):
    NATIVE = "native"
    SKILL_SCRIPT = "skill_script"
    SANDBOX = "sandbox"
    SUBAGENT = "subagent"
    MCP = "mcp"
    INTERNAL_SERVER = "internal_server"


class DiscoveryVisibility(StrEnum):
    """Where a capability may be exposed to an agent or user."""

    PUBLIC = "public"
    INTERNAL = "internal"
    HIDDEN = "hidden"
    RETIRED = "retired"


class DiscoveryMatchMode(StrEnum):
    """How a capability may participate in unified discovery."""

    NATURAL = "natural"
    EXACT_ONLY = "exact_only"
    HIDDEN = "hidden"


class DiscoveryFallbackCondition(StrEnum):
    """Mechanical gate for a lower-priority execution provider."""

    PRIMARY_UNAVAILABLE_OR_UNSUPPORTED = "primary_unavailable_or_unsupported"


class AgentMode(StrEnum):
    """Deprecated request compatibility values; never used for execution policy."""
    ASK = "ask"
    PLAN = "plan"
    AGENT = "agent"


class Permission(StrEnum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    CONTROL = "control"


class CapabilityStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PENDING = "pending"
    NEEDS_INPUT = "needs_input"
    CANCELLED = "cancelled"


class CapabilityErrorKind(StrEnum):
    NEEDS_INPUT = "needs_input"
    VALIDATION_ERROR = "validation_error"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    AUTH_REQUIRED = "auth_required"
    TIMEOUT = "timeout"
    EXECUTION_ERROR = "execution_error"
    CONTRACT_VIOLATION = "contract_violation"
    UNSATISFIED_RESULT = "unsatisfied_result"
    CANCELLED = "cancelled"


class CapabilityApplicability(StrEnum):
    """Whether this exact call is suitable for the selected capability."""

    UNKNOWN = "unknown"
    APPLICABLE = "applicable"
    PARTIALLY_APPLICABLE = "partially_applicable"
    NOT_APPLICABLE = "not_applicable"


class CapabilityEvidenceState(StrEnum):
    """Evidence yielded by this exact provider invocation."""

    UNKNOWN = "unknown"
    AVAILABLE = "available"
    PARTIAL = "partial"
    NONE = "none"


class CapabilityCoverageState(StrEnum):
    """Coverage only within the Provider-declared query scope."""

    UNKNOWN = "unknown"
    MORE_AVAILABLE = "more_available"
    EXHAUSTED = "exhausted"
    NOT_APPLICABLE = "not_applicable"


class CapabilityRecoveryAction(StrEnum):
    """Mechanical advice for the same capability path, never root-task control."""

    UNKNOWN = "unknown"
    RETRY_SAME_CALL = "retry_same_call"
    REFINE_INPUT = "refine_input"
    SWITCH_CAPABILITY = "switch_capability"
    DO_NOT_RETRY = "do_not_retry"
    WAIT = "wait"


class SourceProcessingStatus(StrEnum):
    """State of optional source processing for one capability result.

    This status is deliberately independent from ``CapabilityStatus`` and
    ``CapabilityOutcome``.  A source projection may fail while the underlying
    business capability result remains valid and usable.
    """

    NOT_REQUESTED = "not_requested"
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class DrawerSection(StrEnum):
    USER_UPLOAD = "user_upload"
    TEMPORARY_OUTPUT = "temporary_output"
    RESULT_FILE = "result_file"
    PLAN_FILE = "plan_file"


class InputSourceType(StrEnum):
    CONVERSATION_FILE = "conversation_file"
    DATABASE_ASSET = "database_asset"
    TASK_ARTIFACT = "task_artifact"
    USER_MESSAGE = "user_message"
    PLAN_FILE = "plan_file"
    SANDBOX_JOB = "sandbox_job"


class InputRefHandling(StrEnum):
    MATERIALIZE = "materialize"
    PASS_THROUGH = "pass_through"


class RegistrationStatus(StrEnum):
    REGISTERED = "registered"
    PENDING = "pending"
    FAILED = "failed"


class AsyncMode(StrEnum):
    BLOCKING = "blocking"
    NONBLOCKING_LONG = "nonblocking_long"
    POLL = "poll"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CapabilityOutcome(StrictModel):
    """Facts about one capability call, independent from root-task completion.

    Providers may only assert non-unknown values from deterministic contract or
    response facts.  The scope is intentionally mandatory when a provider
    claims exhaustion, so an empty page cannot be mistaken for global absence.
    """

    schema_version: Literal["evoengine.capability-outcome/v1"] = (
        "evoengine.capability-outcome/v1"
    )
    applicability: CapabilityApplicability = CapabilityApplicability.UNKNOWN
    applicability_basis: str = ""
    evidence: CapabilityEvidenceState = CapabilityEvidenceState.UNKNOWN
    evidence_basis: str = ""
    coverage: CapabilityCoverageState = CapabilityCoverageState.UNKNOWN
    coverage_scope: str = ""
    coverage_basis: str = ""
    recovery: CapabilityRecoveryAction = CapabilityRecoveryAction.UNKNOWN
    recovery_basis: str = ""

    @model_validator(mode="after")
    def validate_semantic_consistency(self) -> "CapabilityOutcome":
        if (
            self.applicability != CapabilityApplicability.UNKNOWN
            and not self.applicability_basis
        ):
            raise ValueError("non-unknown applicability requires applicability_basis")
        if self.evidence != CapabilityEvidenceState.UNKNOWN and not self.evidence_basis:
            raise ValueError("non-unknown evidence requires evidence_basis")
        if self.coverage != CapabilityCoverageState.UNKNOWN and not self.coverage_basis:
            raise ValueError("non-unknown coverage requires coverage_basis")
        if self.recovery != CapabilityRecoveryAction.UNKNOWN and not self.recovery_basis:
            raise ValueError("non-unknown recovery requires recovery_basis")
        if self.applicability == CapabilityApplicability.NOT_APPLICABLE:
            if self.evidence in {
                CapabilityEvidenceState.AVAILABLE,
                CapabilityEvidenceState.PARTIAL,
            }:
                raise ValueError("not_applicable cannot claim available evidence")
            if self.coverage not in {
                CapabilityCoverageState.UNKNOWN,
                CapabilityCoverageState.NOT_APPLICABLE,
            }:
                raise ValueError("not_applicable cannot claim query coverage")
        if self.coverage == CapabilityCoverageState.NOT_APPLICABLE and (
            self.applicability != CapabilityApplicability.NOT_APPLICABLE
        ):
            raise ValueError("coverage not_applicable requires applicability not_applicable")
        if self.coverage in {
            CapabilityCoverageState.MORE_AVAILABLE,
            CapabilityCoverageState.EXHAUSTED,
        } and (
            not self.coverage_scope or not self.coverage_basis
        ):
            raise ValueError(
                "more_available/exhausted coverage requires coverage_scope and coverage_basis"
            )
        if self.recovery == CapabilityRecoveryAction.DO_NOT_RETRY and (
            self.applicability != CapabilityApplicability.NOT_APPLICABLE
            and self.coverage != CapabilityCoverageState.EXHAUSTED
        ):
            raise ValueError("do_not_retry requires not_applicable or exhausted coverage")
        return self


class SourceOutcome(StrictModel):
    """Facts about source registration/publication for one capability call.

    The contract reports only the source sidecar.  It never changes whether
    the capability itself succeeded and never asserts root-task completion.
    """

    schema_version: Literal["evoengine.source-outcome/v1"] = (
        "evoengine.source-outcome/v1"
    )
    status: SourceProcessingStatus = SourceProcessingStatus.NOT_REQUESTED
    attempted: bool = False
    candidate_count: int = Field(default=0, ge=0)
    publishable_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)
    retryable: bool = False
    reason: str = ""

    @model_validator(mode="after")
    def validate_source_processing(self) -> "SourceOutcome":
        if self.publishable_count > self.candidate_count:
            raise ValueError("publishable_count cannot exceed candidate_count")
        if self.rejected_count > self.candidate_count:
            raise ValueError("rejected_count cannot exceed candidate_count")
        if self.status == SourceProcessingStatus.NOT_REQUESTED:
            if self.attempted:
                raise ValueError("not_requested source processing cannot be attempted")
            if any(
                (
                    self.candidate_count,
                    self.publishable_count,
                    self.rejected_count,
                    len(self.warnings),
                )
            ) or self.reason or self.retryable:
                raise ValueError("not_requested source processing cannot carry results")
            return self
        if not self.attempted:
            raise ValueError("source processing results require attempted=true")
        if self.status == SourceProcessingStatus.FAILED and not self.reason:
            raise ValueError("failed source processing requires a reason")
        return self


class DiscoveryFallbackPolicy(StrictModel):
    """Declare one lower-priority provider for the same business contract.

    Discovery may expose this route naturally only when none of the declared
    primary capabilities is currently available.  An Agent may also load it
    explicitly after a primary call reports provider unavailability or that
    the primary contract does not cover the requested operation.
    """

    for_capability_ids: list[str] = Field(min_length=1)
    condition: DiscoveryFallbackCondition = (
        DiscoveryFallbackCondition.PRIMARY_UNAVAILABLE_OR_UNSUPPORTED
    )

    @field_validator("for_capability_ids")
    @classmethod
    def validate_primary_capability_ids(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for raw_value in values:
            value = str(raw_value or "").strip()
            if (
                not value
                or value != value.lower()
                or not _CAPABILITY_ID_RE.fullmatch(value)
            ):
                raise ValueError(
                    "fallback primary capability IDs must be lowercase stable identifiers"
                )
            if value not in normalized:
                normalized.append(value)
        return normalized


class DiscoveryPolicy(StrictModel):
    """Declarative discovery and execution-entry contract for a capability.

    The default deliberately remains exact-identifier only.  This lets legacy
    ``CapabilitySpec`` documents continue to load while preventing contracts
    that have not yet declared structured discovery semantics from competing
    in natural-language recall.
    """

    visibility: DiscoveryVisibility = DiscoveryVisibility.PUBLIC
    match_mode: DiscoveryMatchMode = DiscoveryMatchMode.EXACT_ONLY
    operations: list[str] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)
    object_types: list[str] = Field(default_factory=list)
    input_types: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    search_terms: list[str] = Field(default_factory=list)
    execution_entry_id: str | None = Field(default=None, max_length=256)
    load_tool_ids: list[str] = Field(default_factory=list)
    fallback: DiscoveryFallbackPolicy | None = None

    @field_validator("operations", "source_types", "object_types", "input_types")
    @classmethod
    def validate_taxonomy_values(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for raw_value in values:
            value = str(raw_value or "").strip()
            if not value or value != value.lower() or not _DISCOVERY_TAXONOMY_RE.fullmatch(value):
                raise ValueError(
                    "discovery taxonomy values must be lowercase stable identifiers"
                )
            if value not in normalized:
                normalized.append(value)
        return normalized

    @field_validator("aliases", "search_terms")
    @classmethod
    def normalize_search_values(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        identities: set[str] = set()
        for raw_value in values:
            value = str(raw_value or "").strip()
            if not value:
                raise ValueError("discovery aliases and search terms cannot be empty")
            identity = value.casefold()
            if identity in identities:
                continue
            identities.add(identity)
            normalized.append(value)
        return normalized

    @field_validator("execution_entry_id")
    @classmethod
    def validate_execution_entry_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if normalized != normalized.lower() or not _CAPABILITY_ID_RE.fullmatch(normalized):
            raise ValueError("execution_entry_id must be a lowercase stable identifier")
        return normalized

    @field_validator("load_tool_ids")
    @classmethod
    def validate_load_tool_ids(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for raw_value in values:
            value = str(raw_value or "").strip()
            if not value or value != value.lower() or not _CAPABILITY_ID_RE.fullmatch(value):
                raise ValueError("load_tool_ids must contain lowercase stable identifiers")
            if value not in normalized:
                normalized.append(value)
        return normalized

    @model_validator(mode="after")
    def validate_discovery_boundary(self) -> "DiscoveryPolicy":
        has_execution_entry = bool(self.execution_entry_id)
        has_load_tools = bool(self.load_tool_ids)
        if has_execution_entry != has_load_tools:
            raise ValueError(
                "execution_entry_id and load_tool_ids must be declared together"
            )
        if self.execution_entry_id and self.execution_entry_id not in self.load_tool_ids:
            raise ValueError("load_tool_ids must include execution_entry_id")
        if self.match_mode == DiscoveryMatchMode.NATURAL:
            if self.visibility != DiscoveryVisibility.PUBLIC:
                raise ValueError("natural discovery requires public visibility")
            missing: list[str] = []
            if not self.execution_entry_id:
                missing.append("execution_entry_id")
            if not self.load_tool_ids:
                missing.append("load_tool_ids")
            if missing:
                raise ValueError(
                    "natural discovery requires a complete execution contract: "
                    + ", ".join(missing)
                )
        if self.visibility in {
            DiscoveryVisibility.HIDDEN,
            DiscoveryVisibility.RETIRED,
        } and self.match_mode != DiscoveryMatchMode.HIDDEN:
            raise ValueError("hidden and retired capabilities require hidden match_mode")
        if self.visibility == DiscoveryVisibility.RETIRED and (
            self.execution_entry_id or self.load_tool_ids
        ):
            raise ValueError("retired capabilities cannot expose execution entries")
        return self


class RetryPolicy(StrictModel):
    max_attempts: int = Field(default=1, ge=1, le=3)
    retryable_kinds: list[CapabilityErrorKind] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_retryable_kinds(self) -> "RetryPolicy":
        forbidden = {
            CapabilityErrorKind.NEEDS_INPUT,
            CapabilityErrorKind.VALIDATION_ERROR,
            CapabilityErrorKind.AUTH_REQUIRED,
            CapabilityErrorKind.CONTRACT_VIOLATION,
            CapabilityErrorKind.UNSATISFIED_RESULT,
            CapabilityErrorKind.CANCELLED,
        }
        invalid = sorted({kind.value for kind in self.retryable_kinds if kind in forbidden})
        if invalid:
            raise ValueError(f"non-retryable error kinds declared retryable: {', '.join(invalid)}")
        if self.max_attempts == 1 and self.retryable_kinds:
            raise ValueError("retryable_kinds require max_attempts greater than 1")
        return self


class ExecutionPolicy(StrictModel):
    timeout_seconds: int = Field(default=120, ge=1, le=86_400)
    idempotent: bool = True
    async_mode: AsyncMode = AsyncMode.BLOCKING
    retry: RetryPolicy = Field(default_factory=RetryPolicy)


class ArtifactSpec(StrictModel):
    artifact_key: str = Field(min_length=1, max_length=160)
    required: bool = False
    file_name: str | None = Field(default=None, max_length=512)
    file_types: list[str] = Field(default_factory=list)
    drawer_section: DrawerSection
    retention_policy: str = Field(default="keep", pattern=r"^(ephemeral|keep)$")

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
            raise ValueError("artifact file_name must be a plain file name")
        return normalized

    @field_validator("file_types")
    @classmethod
    def normalize_file_types(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            item = str(value or "").strip().lower().lstrip(".")
            if item and item not in normalized:
                normalized.append(item)
        return normalized

    @model_validator(mode="after")
    def validate_file_name_type(self) -> "ArtifactSpec":
        if self.file_name and "." in self.file_name and self.file_types:
            extension = self.file_name.rsplit(".", 1)[-1].lower()
            if extension not in self.file_types:
                raise ValueError("artifact file_name extension must match file_types")
        return self


class InputRef(StrictModel):
    source_type: InputSourceType
    source_id: str = Field(min_length=1, max_length=512)
    input_key: str | None = Field(default=None, min_length=1, max_length=160)
    file_name: str | None = Field(default=None, max_length=512)
    sha256: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_conversation_file_resource(cls, value: Any) -> Any:
        """Accept an owned conversation-file ``ResourceRef`` as a file input.

        Capability results expose durable ResourceRefs, while providers consume
        the narrower InputRef contract.  Normalize that hand-off here so a
        later capability never needs the model to transcribe the underlying
        conversation-file ID.  Non-file resources remain invalid rather than
        being guessed into a different input source.
        """

        normalized = normalize_resource_ref_input(value)
        if normalized is None:
            return value
        wrapper = value if isinstance(value, dict) else {}
        resource = normalized["resource"]
        metadata = resource.metadata if isinstance(resource.metadata, dict) else {}
        return {
            "source_type": InputSourceType.CONVERSATION_FILE.value,
            "source_id": str(resource.conversation_file_id),
            "input_key": str(wrapper.get("input_key") or "").strip() or None,
            "file_name": (
                str(wrapper.get("file_name") or metadata.get("file_name") or "").strip()
                or None
            ),
            "sha256": str(wrapper.get("sha256") or resource.sha256 or "").strip() or None,
        }

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        normalized = value.strip().lower()
        if not _SHA256_RE.fullmatch(normalized):
            raise ValueError("sha256 must contain exactly 64 lowercase hexadecimal characters")
        return normalized


def normalize_resource_ref_input(value: Any) -> dict[str, Any] | None:
    """Return one validated conversation-file ResourceRef input projection."""

    if isinstance(value, InputRef):
        return None
    wrapper = value if isinstance(value, dict) else {}
    raw_resource = wrapper.get("resource_ref") if "resource_ref" in wrapper else value
    is_resource = isinstance(raw_resource, ResourceRef) or (
        isinstance(raw_resource, dict)
        and (
            str(raw_resource.get("schema_version") or "")
            == "evoengine.resource-ref/v1"
            or all(
                key in raw_resource
                for key in ("resource_id", "uri", "storage", "sha256")
            )
        )
    )
    if not is_resource:
        return None
    resource = (
        raw_resource
        if isinstance(raw_resource, ResourceRef)
        else ResourceRef.model_validate(raw_resource)
    )
    if resource.storage != "conversation_file" or resource.conversation_file_id is None:
        raise ValueError(
            "only conversation_file ResourceRefs can be used as capability file inputs"
        )
    return {
        "resource": resource,
        "input_key": str(wrapper.get("input_key") or "").strip() or None,
        "file_name": str(wrapper.get("file_name") or "").strip() or None,
        "sha256": str(wrapper.get("sha256") or "").strip() or None,
    }


class ArtifactRef(StrictModel):
    artifact_id: str = Field(min_length=1, max_length=512)
    artifact_key: str = Field(default="", max_length=160)
    conversation_file_id: int | None = Field(default=None, ge=1)
    file_name: str = Field(default="", max_length=512)
    mime_type: str = Field(default="", max_length=255)
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = None
    drawer_section: DrawerSection | None = None
    source_type: str = Field(default="", max_length=160)
    registration_status: RegistrationStatus = RegistrationStatus.PENDING

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        normalized = value.strip().lower()
        if not _SHA256_RE.fullmatch(normalized):
            raise ValueError("sha256 must contain exactly 64 lowercase hexadecimal characters")
        return normalized

    @model_validator(mode="after")
    def validate_registration(self) -> "ArtifactRef":
        if self.registration_status == RegistrationStatus.REGISTERED and self.conversation_file_id is None:
            raise ValueError("registered artifacts require conversation_file_id")
        return self


class EvidenceRef(StrictModel):
    title: str = Field(default="", max_length=1000)
    url: str | None = Field(default=None, max_length=4000)
    doi: str | None = Field(default=None, max_length=512)
    pmid: str | None = Field(default=None, max_length=128)
    database: str | None = Field(default=None, max_length=256)
    record_id: str | None = Field(default=None, max_length=512)
    source_id: str | None = Field(default=None, max_length=512)
    locator: str | None = Field(default=None, max_length=4000)
    claim_ids: list[str] = Field(default_factory=list)
    dataset_id: str | None = Field(default=None, max_length=256)
    dataset_version: str | None = Field(default=None, max_length=128)
    source: str | None = Field(default=None, max_length=128)
    query_time: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def validate_locator(self) -> "EvidenceRef":
        has_location = any(
            str(value or "").strip()
            for value in (self.url, self.doi, self.pmid, self.record_id, self.source_id, self.locator)
        )
        if not has_location:
            raise ValueError("evidence references require at least one stable locator")
        return self


class ResourceRef(StrictModel):
    """Durable reference to complete capability data kept outside model context.

    ``ResourceRef`` describes identity and storage only.  It does not imply that
    the model-facing projection contains the complete resource body.
    """

    schema_version: Literal["evoengine.resource-ref/v1"] = (
        "evoengine.resource-ref/v1"
    )
    resource_id: str = Field(min_length=1, max_length=512)
    uri: str = Field(min_length=1, max_length=4000)
    kind: str = Field(default="tool_result", min_length=1, max_length=160)
    storage: str = Field(default="conversation_file", min_length=1, max_length=160)
    media_type: str = Field(default="application/json", max_length=255)
    size_bytes: int = Field(ge=0)
    sha256: str
    conversation_file_id: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sha256")
    @classmethod
    def validate_resource_sha256(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if not _SHA256_RE.fullmatch(normalized):
            raise ValueError(
                "resource sha256 must contain exactly 64 lowercase hexadecimal characters"
            )
        return normalized

    @model_validator(mode="after")
    def validate_resource_location(self) -> "ResourceRef":
        expected_scheme = REFERENCE_RESOURCE_STORAGE_SCHEMES.get(self.storage)
        if expected_scheme is None:
            raise ValueError("resource storage is not supported")
        parsed = urlparse(self.uri)
        if parsed.scheme != expected_scheme or not (parsed.netloc or parsed.path.lstrip("/")):
            raise ValueError(
                f"{self.storage} resource uri must use {expected_scheme}:// with a stable target"
            )
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("resource uri must be an identity URI without credentials, cursor or fragment")
        expected_resource_id = (
            f"resource:{expected_scheme}:"
            f"{parsed.netloc}{parsed.path}".rstrip("/")
        )
        if self.resource_id != expected_resource_id:
            raise ValueError("resource uri must match resource_id and storage")
        if self.storage == "conversation_file":
            if self.conversation_file_id is None:
                raise ValueError(
                    "conversation_file resources require conversation_file_id"
                )
            expected_uri = f"conversation-file://{self.conversation_file_id}"
            if self.uri != expected_uri:
                raise ValueError(
                    "conversation_file resource uri must match conversation_file_id"
                )
            if self.resource_id != f"resource:conversation-file:{self.conversation_file_id}":
                raise ValueError(
                    "conversation_file resource_id must match conversation_file_id"
                )
        elif self.conversation_file_id is not None:
            raise ValueError(
                "non-conversation-file resources cannot carry conversation_file_id"
            )
        return self


class SourceCandidateRef(StrictModel):
    """Reference to a business result that may be verified as a source later."""

    schema_version: Literal["evoengine.source-candidate-ref/v1"] = (
        "evoengine.source-candidate-ref/v1"
    )
    candidate_id: str = Field(min_length=1, max_length=512)
    capability_id: str = Field(min_length=1, max_length=512)
    capability_version: str = Field(min_length=1, max_length=128)
    result_resource_id: str = Field(min_length=1, max_length=512)
    result_type: str = Field(min_length=1, max_length=256)
    payload_path: str = Field(min_length=1, max_length=512)

    @field_validator("capability_id")
    @classmethod
    def validate_source_capability_id(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if normalized != normalized.lower() or not _CAPABILITY_ID_RE.fullmatch(normalized):
            raise ValueError("invalid source candidate capability_id")
        return normalized

    @field_validator("result_resource_id")
    @classmethod
    def validate_result_resource_id(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not re.fullmatch(
            r"(?:resource:conversation-file:|conversation-file://)[1-9][0-9]*",
            normalized,
        ):
            raise ValueError("source candidate requires a conversation-file ResourceRef")
        return normalized


class CapabilitySpec(StrictModel):
    capability_id: str
    version: str = Field(min_length=1, max_length=128)
    display_name: str = Field(min_length=1, max_length=512)
    description: str = Field(default="", max_length=4000)
    provider_type: ProviderType
    discovery: DiscoveryPolicy = Field(default_factory=DiscoveryPolicy)
    provider_ref: dict[str, Any] = Field(default_factory=dict)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    input_refs_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "array"})
    input_ref_handling: InputRefHandling = InputRefHandling.MATERIALIZE
    output_schema: dict[str, Any] = Field(default_factory=dict)
    # Declarative acceptance predicate over the same authoritative machine
    # result validated by ``output_schema``.  The runtime only evaluates JSON
    # Schema here; it does not know business field names, titles, or Case IDs.
    acceptance_schema: dict[str, Any] = Field(default_factory=dict)
    permission: Permission
    side_effects: list[str] = Field(default_factory=list)
    allowed_modes: list[AgentMode] = Field(
        default_factory=lambda: [AgentMode.AGENT],
        exclude=True,
        deprecated="agent modes are compatibility-only and do not gate capabilities",
    )
    requires_confirmation: bool = False
    produces: list[ArtifactSpec] = Field(default_factory=list)
    evidence_policy: dict[str, Any] = Field(default_factory=dict)
    execution_policy: ExecutionPolicy = Field(default_factory=ExecutionPolicy)
    enabled: bool = True
    availability: str = Field(default="available", pattern=r"^(available|unavailable|degraded)$")

    @field_validator("capability_id")
    @classmethod
    def validate_capability_id(cls, value: str) -> str:
        stripped = value.strip()
        if stripped != stripped.lower():
            raise ValueError("capability_id must be lowercase")
        normalized = stripped
        if not _CAPABILITY_ID_RE.fullmatch(normalized):
            raise ValueError("capability_id must use lowercase letters, digits, '.', ':', '_' or '-'")
        return normalized

    @field_validator("allowed_modes")
    @classmethod
    def validate_allowed_modes(cls, values: list[AgentMode]) -> list[AgentMode]:
        return list(dict.fromkeys(values or [AgentMode.AGENT]))

    @model_validator(mode="after")
    def validate_artifact_keys(self) -> "CapabilitySpec":
        # Validate schemas at spec construction as a fail-fast safety boundary;
        # imports stay local to avoid a module cycle with validation models.
        from src.capabilities.validation import check_json_schema

        for label, schema in (
            ("input_schema", self.input_schema),
            ("input_refs_schema", self.input_refs_schema),
            ("output_schema", self.output_schema),
            ("acceptance_schema", self.acceptance_schema),
        ):
            report = check_json_schema(schema)
            if not report.valid:
                details = "; ".join(
                    f"{issue.path}: {issue.message}" for issue in report.issues
                )
                raise ValueError(f"invalid {label}: {details}")
        artifact_keys = [item.artifact_key for item in self.produces]
        if len(artifact_keys) != len(set(artifact_keys)):
            raise ValueError("capability artifact_key values must be unique")
        exact_file_names = [item.file_name for item in self.produces if item.file_name]
        if len(exact_file_names) != len(set(exact_file_names)):
            raise ValueError("capability artifact file_name values must be unique")
        return self


def capability_spec_digest(spec: CapabilitySpec) -> tuple[str, str]:
    """Return the immutable canonical payload and digest for one capability spec."""

    payload = json.dumps(
        spec.model_dump(mode="json", exclude_none=False),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CapabilityRegistrySnapshotItem(StrictModel):
    """One exact capability contract captured for a parent turn."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        frozen=True,
    )

    capability_id: str
    version: str = Field(min_length=1, max_length=128)
    spec_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    provider_type: ProviderType
    permission: Permission
    availability: str = Field(pattern=r"^(available|unavailable|degraded)$")
    enabled: bool
    spec_json: str = Field(min_length=2)

    @classmethod
    def from_spec(cls, spec: CapabilitySpec) -> "CapabilityRegistrySnapshotItem":
        frozen_spec = spec.model_copy(deep=True)
        spec_json, spec_digest = capability_spec_digest(frozen_spec)
        return cls(
            capability_id=frozen_spec.capability_id,
            version=frozen_spec.version,
            spec_digest=spec_digest,
            provider_type=frozen_spec.provider_type,
            permission=frozen_spec.permission,
            availability=frozen_spec.availability,
            enabled=frozen_spec.enabled,
            spec_json=spec_json,
        )

    def to_spec(self) -> CapabilitySpec:
        return CapabilitySpec.model_validate_json(self.spec_json).model_copy(deep=True)

    @model_validator(mode="after")
    def validate_frozen_spec(self) -> "CapabilityRegistrySnapshotItem":
        spec = CapabilitySpec.model_validate_json(self.spec_json)
        canonical_json, digest = capability_spec_digest(spec)
        if canonical_json != self.spec_json or digest != self.spec_digest:
            raise ValueError("snapshot spec payload or digest is not canonical")
        if (
            self.capability_id != spec.capability_id
            or self.version != spec.version
            or self.provider_type != spec.provider_type
            or self.permission != spec.permission
            or self.availability != spec.availability
            or self.enabled != spec.enabled
        ):
            raise ValueError("snapshot item metadata does not match frozen spec")
        return self


class CapabilityRegistrySnapshot(StrictModel):
    """Serializable immutable capability view for a parent turn and its children."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        frozen=True,
    )

    schema_version: Literal["evoengine.capability-registry-snapshot/v1"] = (
        "evoengine.capability-registry-snapshot/v1"
    )
    snapshot_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    items: tuple[CapabilityRegistrySnapshotItem, ...] = Field(min_length=1)

    @staticmethod
    def _snapshot_digest(items: tuple[CapabilityRegistrySnapshotItem, ...]) -> str:
        payload = {
            "schema_version": "evoengine.capability-registry-snapshot/v1",
            "items": [
                {
                    "capability_id": item.capability_id,
                    "version": item.version,
                    "spec_digest": item.spec_digest,
                    "provider_type": item.provider_type.value,
                    "permission": item.permission.value,
                    "availability": item.availability,
                    "enabled": item.enabled,
                }
                for item in items
            ],
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_specs(cls, specs: list[CapabilitySpec]) -> "CapabilityRegistrySnapshot":
        items = tuple(
            sorted(
                (CapabilityRegistrySnapshotItem.from_spec(spec) for spec in specs),
                key=lambda item: (item.capability_id, item.version),
            )
        )
        if not items:
            raise ValueError("capability registry snapshot requires at least one capability")
        return cls(snapshot_id=cls._snapshot_digest(items), items=items)

    def resolve_spec(self, capability_id: str, version: str) -> CapabilitySpec | None:
        normalized_id = str(capability_id or "").strip().lower()
        normalized_version = str(version or "").strip()
        for item in self.items:
            if item.capability_id == normalized_id and item.version == normalized_version:
                return item.to_spec()
        return None

    @model_validator(mode="after")
    def validate_snapshot_identity(self) -> "CapabilityRegistrySnapshot":
        identities = [(item.capability_id, item.version) for item in self.items]
        if identities != sorted(identities):
            raise ValueError("snapshot items must use canonical identity order")
        if len(identities) != len(set(identities)):
            raise ValueError("snapshot capability identities must be unique")
        if self.snapshot_id != self._snapshot_digest(self.items):
            raise ValueError("snapshot_id does not match frozen capability specs")
        return self


class CapabilityCall(StrictModel):
    call_id: str = Field(min_length=1, max_length=512)
    capability_id: str
    capability_version: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=512)
    project_id: str | None = Field(default=None, max_length=512)
    conversation_id: str | None = Field(default=None, max_length=512)
    user_id: int | None = Field(default=None, ge=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    input_refs: list[InputRef] = Field(default_factory=list)
    task_node_id: str | None = Field(default=None, max_length=512)
    # Runtime orchestration contract.  This is deliberately separate from the
    # capability's business input_schema: a live task tree may require the
    # Agent to select an exact execution node without teaching every Skill or
    # provider about TaskTree fields.
    task_tree_binding_required: bool = False
    context_fingerprint: str = Field(default="", max_length=128)
    registry_snapshot_id: str = Field(default="", pattern=r"^(?:|[a-f0-9]{64})$")
    idempotency_key: str = Field(min_length=1, max_length=128)
    requested_at: datetime = Field(default_factory=utc_now)

    @field_validator("capability_id")
    @classmethod
    def validate_capability_id(cls, value: str) -> str:
        stripped = value.strip()
        if stripped != stripped.lower():
            raise ValueError("capability_id must be lowercase")
        normalized = stripped
        if not _CAPABILITY_ID_RE.fullmatch(normalized):
            raise ValueError("invalid capability_id")
        return normalized

    @field_validator("requested_at")
    @classmethod
    def validate_requested_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("requested_at must include a timezone")
        return value.astimezone(timezone.utc)


class CapabilityError(StrictModel):
    kind: CapabilityErrorKind
    message: str = Field(min_length=1)
    retryable: bool = False
    missing_fields: list[str] = Field(default_factory=list)
    details_ref: str | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def validate_retryability(self) -> "CapabilityError":
        never_retry = {
            CapabilityErrorKind.NEEDS_INPUT,
            CapabilityErrorKind.VALIDATION_ERROR,
            CapabilityErrorKind.AUTH_REQUIRED,
            CapabilityErrorKind.CONTRACT_VIOLATION,
            CapabilityErrorKind.UNSATISFIED_RESULT,
            CapabilityErrorKind.CANCELLED,
        }
        if self.kind in never_retry and self.retryable:
            raise ValueError(f"{self.kind.value} errors cannot be retryable")
        return self


class CapabilityNeedsInput(StrictModel):
    """Private authorization control emitted by one exact capability call."""

    schema_version: Literal["evoengine.capability-needs-input/v1"] = (
        "evoengine.capability-needs-input/v1"
    )
    question_bundle: HumanQuestionBundle
    pending_call_id: str = Field(min_length=1, max_length=512)
    capability_id: str = Field(min_length=1, max_length=512)
    capability_version: str = Field(min_length=1, max_length=128)
    registry_snapshot_id: str = Field(default="", pattern=r"^(?:|[a-f0-9]{64})$")
    action: str = Field(min_length=1, max_length=256)
    target: ResourceRef
    target_version_id: int | None = Field(default=None, ge=1)
    target_lock_version: int | None = Field(default=None, ge=0)
    target_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    request_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    effect_capability_id: str | None = Field(default=None, min_length=1, max_length=512)
    effect_request_digest: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    authorization_challenge: str = Field(pattern=r"^[a-f0-9]{64}$")
    authorization_question_id: str = Field(min_length=1, max_length=512)
    approve_option_id: str = Field(min_length=1, max_length=512)
    reject_option_id: str = Field(min_length=1, max_length=512)

    @field_validator("capability_id")
    @classmethod
    def validate_control_capability_id(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if normalized != normalized.lower() or not _CAPABILITY_ID_RE.fullmatch(normalized):
            raise ValueError("invalid authorization capability_id")
        return normalized

    @model_validator(mode="after")
    def validate_authorization_question(self) -> "CapabilityNeedsInput":
        questions = list(self.question_bundle.questions)
        if len(questions) != 1:
            raise ValueError("authorization requires exactly one question")
        question = questions[0]
        if (
            question.kind != "single"
            or not question.required
            or question.allow_other
            or question.question_id != self.authorization_question_id
        ):
            raise ValueError("authorization question must be one required single-select")
        option_ids = {item.option_id for item in question.options}
        if (
            self.approve_option_id == self.reject_option_id
            or self.approve_option_id not in option_ids
            or self.reject_option_id not in option_ids
        ):
            raise ValueError("authorization question requires distinct approve/reject options")
        if self.pending_call_id == "":
            raise ValueError("pending_call_id is required")
        if (self.effect_capability_id is None) != (self.effect_request_digest is None):
            raise ValueError("effect capability and digest must be provided together")
        return self


class CapabilityResult(StrictModel):
    ok: bool
    status: CapabilityStatus
    call_id: str = Field(min_length=1, max_length=512)
    capability_id: str
    capability_version: str = Field(min_length=1, max_length=128)
    provider_type: ProviderType | None
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
    source_sidecar_resource: ResourceRef | None = None
    error: CapabilityError | None = None
    needs_input_control: CapabilityNeedsInput | None = Field(default=None, exclude=True)
    raw_ref: str | None = Field(default=None, max_length=4000)
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    elapsed_ms: int | None = Field(default=None, ge=0)
    idempotency_reused: bool = False
    legacy_envelope: dict[str, Any] | None = Field(default=None, exclude=True)

    @field_validator("capability_id")
    @classmethod
    def validate_capability_id(cls, value: str) -> str:
        stripped = value.strip()
        if stripped != stripped.lower():
            raise ValueError("capability_id must be lowercase")
        normalized = stripped
        if not _CAPABILITY_ID_RE.fullmatch(normalized):
            raise ValueError("invalid capability_id")
        return normalized

    @field_validator("started_at", "finished_at")
    @classmethod
    def validate_datetime(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must include a timezone")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_status_consistency(self) -> "CapabilityResult":
        successful_statuses = {CapabilityStatus.SUCCEEDED, CapabilityStatus.PENDING}
        if self.ok != (self.status in successful_statuses):
            raise ValueError("ok must be true only for succeeded or pending results")
        if self.ok and self.provider_type is None:
            raise ValueError("successful results require provider_type")
        if self.ok and self.error is not None:
            raise ValueError("successful results must not include error")
        if not self.ok and self.error is None:
            raise ValueError("failed, needs_input and cancelled results require error")
        if self.status == CapabilityStatus.NEEDS_INPUT and (
            self.error is None or self.error.kind != CapabilityErrorKind.NEEDS_INPUT
        ):
            raise ValueError("needs_input results require a needs_input error")
        if self.needs_input_control is not None and self.status != CapabilityStatus.NEEDS_INPUT:
            raise ValueError("needs_input_control requires needs_input status")
        if self.status == CapabilityStatus.NEEDS_INPUT:
            if self.error is None or (
                not self.error.missing_fields and self.needs_input_control is None
            ):
                raise ValueError(
                    "needs_input requires missing_fields or a typed needs_input_control"
                )
            if self.needs_input_control is not None and (
                self.needs_input_control.pending_call_id != self.call_id
                or self.needs_input_control.capability_id != self.capability_id
                or self.needs_input_control.capability_version != self.capability_version
            ):
                raise ValueError("needs_input_control does not match capability result")
        if self.status == CapabilityStatus.CANCELLED and (
            self.error is None or self.error.kind != CapabilityErrorKind.CANCELLED
        ):
            raise ValueError("cancelled results require a cancelled error")
        if self.status == CapabilityStatus.FAILED and self.error is not None and self.error.kind in {
            CapabilityErrorKind.NEEDS_INPUT,
            CapabilityErrorKind.CANCELLED,
        }:
            raise ValueError("failed results cannot use needs_input or cancelled errors")
        candidate_ids = [item.candidate_id for item in self.source_candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("source candidates require unique candidate_id values")
        resource_ids = {item.resource_id for item in self.resources}
        if any(
            item.result_resource_id not in resource_ids
            for item in self.source_candidates
        ):
            raise ValueError("source candidate must reference a result resource")
        if self.status == CapabilityStatus.PENDING and self.finished_at is not None:
            raise ValueError("pending results must not have finished_at")
        if self.status != CapabilityStatus.PENDING and self.finished_at is None:
            raise ValueError("terminal results require finished_at")
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at cannot be earlier than started_at")
        if self.complete and self.has_more:
            raise ValueError("complete results cannot also have_more")
        if self.complete and self.cursor is not None:
            raise ValueError("complete results cannot expose a continuation cursor")
        if self.has_more and not self.cursor:
            raise ValueError("has_more results require a continuation cursor")
        if (
            not self.complete
            and not self.has_more
            and self.status in successful_statuses
        ):
            raise ValueError("incomplete results must declare has_more")
        resource_ids = [item.resource_id for item in self.resources]
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("capability resources require unique resource_id values")
        return self
