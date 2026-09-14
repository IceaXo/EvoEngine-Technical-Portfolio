from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.capabilities import (
    AgentMode,
    ArtifactRef,
    ArtifactSpec,
    CapabilityCall,
    CapabilityApplicability,
    CapabilityCoverageState,
    CapabilityError,
    CapabilityErrorKind,
    CapabilityEvidenceState,
    CapabilityOutcome,
    CapabilityRecoveryAction,
    CapabilityResult,
    CapabilitySpec,
    CapabilityStatus,
    DrawerSection,
    EvidenceRef,
    ExecutionPolicy,
    InputRef,
    InputSourceType,
    Permission,
    ProviderType,
    RetryPolicy,
    ResourceRef,
    build_idempotency_key,
    canonical_json,
    check_json_schema,
    validate_json_instance,
)


def _finished_result(**overrides):
    now = datetime.now(timezone.utc)
    payload = {
        "ok": True,
        "status": CapabilityStatus.SUCCEEDED,
        "call_id": "call_1",
        "capability_id": "native.file.read",
        "capability_version": "1.0.0",
        "provider_type": ProviderType.NATIVE,
        "summary": "read complete",
        "started_at": now,
        "finished_at": now,
    }
    payload.update(overrides)
    return CapabilityResult(**payload)


def test_capability_spec_accepts_typed_contract_and_normalizes_file_types():
    spec = CapabilitySpec(
        capability_id="skill.structure.download",
        version="1.0.0",
        display_name="Structure download",
        provider_type=ProviderType.SKILL_SCRIPT,
        provider_ref={"skill_id": "structure-download", "script_name": "download.py"},
        input_schema={
            "type": "object",
            "required": ["accession_ids"],
            "properties": {"accession_ids": {"type": "array", "items": {"type": "string"}}},
        },
        output_schema={"type": "object"},
        permission=Permission.EXECUTE,
        allowed_modes=[AgentMode.AGENT, AgentMode.AGENT],
        produces=[
            ArtifactSpec(
                artifact_key="structures",
                required=True,
                file_types=[".PDB", "pdb", "CIF"],
                drawer_section=DrawerSection.TEMPORARY_OUTPUT,
                retention_policy="ephemeral",
            )
        ],
    )

    assert spec.capability_id == "skill.structure.download"
    assert "allowed_modes" not in spec.model_dump(mode="json")
    assert spec.produces[0].file_types == ["pdb", "cif"]


def test_artifact_spec_accepts_exact_file_name_and_rejects_paths():
    spec = ArtifactSpec(
        artifact_key="qc_report",
        file_name="pdb_qc_report.tsv",
        file_types=["tsv"],
        drawer_section=DrawerSection.TEMPORARY_OUTPUT,
    )
    assert spec.file_name == "pdb_qc_report.tsv"

    with pytest.raises(ValidationError, match="plain file name"):
        ArtifactSpec(
            artifact_key="qc_report",
            file_name="outputs/pdb_qc_report.tsv",
            drawer_section=DrawerSection.TEMPORARY_OUTPUT,
        )

    with pytest.raises(ValidationError, match="extension must match"):
        ArtifactSpec(
            artifact_key="qc_report",
            file_name="pdb_qc_report.tsv",
            file_types=["json"],
            drawer_section=DrawerSection.TEMPORARY_OUTPUT,
        )


def test_capability_spec_rejects_ambiguous_artifact_keys_and_file_names():
    shared = {
        "file_types": ["tsv"],
        "drawer_section": DrawerSection.TEMPORARY_OUTPUT,
    }
    with pytest.raises(ValidationError, match="artifact_key values must be unique"):
        CapabilitySpec(
            capability_id="skill.qc.run",
            version="1",
            display_name="QC",
            provider_type=ProviderType.SKILL_SCRIPT,
            permission=Permission.EXECUTE,
            produces=[
                ArtifactSpec(artifact_key="report", file_name="a.tsv", **shared),
                ArtifactSpec(artifact_key="report", file_name="b.tsv", **shared),
            ],
        )


@pytest.mark.parametrize("capability_id", ["", "Skill.Bad", "contains space", "/bad"])
def test_capability_spec_rejects_unstable_ids(capability_id: str):
    with pytest.raises(ValidationError):
        CapabilitySpec(
            capability_id=capability_id,
            version="1",
            display_name="bad",
            provider_type=ProviderType.NATIVE,
            permission=Permission.READ,
        )


def test_retry_policy_rejects_non_retryable_contract_errors():
    with pytest.raises(ValidationError, match="non-retryable"):
        RetryPolicy(
            max_attempts=2,
            retryable_kinds=[CapabilityErrorKind.CONTRACT_VIOLATION],
        )


def test_execution_policy_allows_bounded_timeout_retry():
    policy = ExecutionPolicy(
        timeout_seconds=30,
        retry=RetryPolicy(
            max_attempts=2,
            retryable_kinds=[CapabilityErrorKind.TIMEOUT],
        ),
    )
    assert policy.timeout_seconds == 30
    assert policy.retry.max_attempts == 2


def test_capability_call_requires_timezone_and_stable_idempotency_key():
    ref = InputRef(
        source_type=InputSourceType.CONVERSATION_FILE,
        source_id="101",
        file_name="input.tsv",
        sha256="a" * 64,
    )
    key = build_idempotency_key(
        capability_id="sandbox.analysis.run",
        capability_version="1",
        arguments={"threads": 4},
        input_refs=[ref],
        project_id="project_1",
        conversation_id="conv_1",
    )
    call = CapabilityCall(
        call_id="call_1",
        capability_id="sandbox.analysis.run",
        capability_version="1",
        request_id="req_1",
        project_id="project_1",
        conversation_id="conv_1",
        user_id=2,
        arguments={"threads": 4},
        input_refs=[ref],
        idempotency_key=key,
    )
    assert len(call.idempotency_key) == 64
    assert call.requested_at.tzinfo is not None

    with pytest.raises(ValidationError, match="timezone"):
        call.model_copy(update={"requested_at": datetime(2026, 7, 12)}).model_dump()
        CapabilityCall(**{**call.model_dump(), "requested_at": datetime(2026, 7, 12)})


def test_conversation_file_resource_ref_is_a_lossless_next_capability_input():
    resource = ResourceRef(
        resource_id="resource:conversation-file:401",
        uri="conversation-file://401",
        kind="tool_result",
        storage="conversation_file",
        media_type="application/json",
        size_bytes=128,
        sha256="c" * 64,
        conversation_file_id=401,
        metadata={"file_name": "records.json"},
    )

    direct = InputRef.model_validate(resource.model_dump(mode="json"))
    slotted = InputRef.model_validate(
        {"resource_ref": resource.model_dump(mode="json"), "input_key": "records"}
    )

    assert direct.source_type == InputSourceType.CONVERSATION_FILE
    assert direct.source_id == "401"
    assert direct.file_name == "records.json"
    assert direct.sha256 == "c" * 64
    assert slotted.source_id == "401"
    assert slotted.input_key == "records"

    with pytest.raises(ValidationError, match="only conversation_file"):
        InputRef.model_validate(
            ResourceRef(
                resource_id="resource:project-asset:knowledge/41",
                uri="project-asset://knowledge/41",
                storage="project_asset",
                size_bytes=12,
                sha256="d" * 64,
            ).model_dump(mode="json")
        )


def test_idempotency_key_is_order_independent_but_changes_with_inputs():
    refs = [
        InputRef(source_type=InputSourceType.DATABASE_ASSET, source_id="asset_2", sha256="b" * 64),
        InputRef(source_type=InputSourceType.CONVERSATION_FILE, source_id="file_1", sha256="a" * 64),
    ]
    first = build_idempotency_key(
        capability_id="sandbox.align.run",
        capability_version="1",
        arguments={"b": 2, "a": 1},
        input_refs=refs,
        project_id="p",
        conversation_id="c",
    )
    reordered = build_idempotency_key(
        capability_id="sandbox.align.run",
        capability_version="1",
        arguments={"a": 1, "b": 2},
        input_refs=list(reversed(refs)),
        project_id="p",
        conversation_id="c",
    )
    changed = build_idempotency_key(
        capability_id="sandbox.align.run",
        capability_version="1",
        arguments={"a": 1, "b": 3},
        input_refs=refs,
        project_id="p",
        conversation_id="c",
    )
    assert first == reordered
    assert first != changed
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'


def test_capability_result_enforces_status_and_never_accepts_task_completion_fields():
    assert _finished_result().ok is True

    with pytest.raises(ValidationError, match="ok must be true"):
        _finished_result(ok=True, status=CapabilityStatus.FAILED)

    with pytest.raises(ValidationError, match="require error"):
        _finished_result(ok=False, status=CapabilityStatus.FAILED)

    with pytest.raises(ValidationError, match="extra"):
        CapabilityResult(
            **_finished_result().model_dump(),
            answer_ready=True,
        )


def test_capability_semantic_feedback_is_not_cut_at_legacy_text_limits():
    long_text = "HEAD|" + ("x" * 6_000) + "|TAIL_FACT"
    outcome = CapabilityOutcome(
        applicability=CapabilityApplicability.APPLICABLE,
        applicability_basis=long_text,
    )
    error = CapabilityError(
        kind=CapabilityErrorKind.VALIDATION_ERROR,
        message=long_text,
    )
    result = _finished_result(summary=long_text, capability_outcome=outcome)

    assert outcome.applicability_basis.endswith("|TAIL_FACT")
    assert error.message.endswith("|TAIL_FACT")
    assert result.summary.endswith("|TAIL_FACT")


def test_capability_result_resource_continuation_contract_is_strict():
    resource = ResourceRef(
        resource_id="resource:conversation-file:401",
        uri="conversation-file://401",
        kind="tool_result",
        storage="conversation_file",
        media_type="application/json",
        size_bytes=10_000_000,
        sha256="4" * 64,
        conversation_file_id=401,
    )
    result = _finished_result(
        resources=[resource],
        complete=False,
        has_more=True,
        cursor="conversation-file://401?offset=0",
    )

    assert result.resources[0].conversation_file_id == 401
    assert result.complete is False
    assert result.has_more is True

    with pytest.raises(ValidationError, match="incomplete results must declare has_more"):
        _finished_result(complete=False, has_more=False)
    with pytest.raises(ValidationError, match="has_more results require"):
        _finished_result(complete=False, has_more=True, cursor=None)
    with pytest.raises(ValidationError, match="uri must match"):
        ResourceRef(
            resource_id="resource:conversation-file:401",
            uri="conversation-file://999",
            size_bytes=1,
            sha256="4" * 64,
            conversation_file_id=401,
        )


@pytest.mark.parametrize(
    ("storage", "scheme", "target"),
    [
        ("project_asset", "project-asset", "knowledge/41"),
        ("conversation_turn", "conversation-turn", "conv-1/turn-8"),
        ("conversation_turn_message", "conversation-turn-message", "41/assistant"),
        ("asset_folder", "asset-folder", "database/root"),
        ("reference_manifest", "reference-manifest", "conv-1/turn-8"),
    ],
)
def test_resource_ref_accepts_only_matching_reference_storage_and_identity(
    storage: str,
    scheme: str,
    target: str,
) -> None:
    resource = ResourceRef(
        resource_id=f"resource:{scheme}:{target}",
        uri=f"{scheme}://{target}",
        storage=storage,
        size_bytes=42,
        sha256="a" * 64,
    )

    assert resource.storage == storage
    assert resource.conversation_file_id is None

    with pytest.raises(ValidationError, match="must match"):
        ResourceRef(
            resource_id=f"resource:{scheme}:other",
            uri=f"{scheme}://{target}",
            storage=storage,
            size_bytes=42,
            sha256="a" * 64,
        )
    with pytest.raises(ValidationError, match="cannot carry"):
        ResourceRef(
            resource_id=f"resource:{scheme}:{target}",
            uri=f"{scheme}://{target}",
            storage=storage,
            size_bytes=42,
            sha256="a" * 64,
            conversation_file_id=41,
        )


def test_capability_outcome_defaults_to_unknown_and_requires_scoped_exhaustion():
    result = _finished_result()
    assert result.capability_outcome.applicability == CapabilityApplicability.UNKNOWN
    assert result.capability_outcome.evidence == CapabilityEvidenceState.UNKNOWN
    assert result.capability_outcome.coverage == CapabilityCoverageState.UNKNOWN
    assert result.capability_outcome.recovery == CapabilityRecoveryAction.UNKNOWN

    exhausted = CapabilityOutcome(
        applicability=CapabilityApplicability.APPLICABLE,
        applicability_basis="typed_operation_contract",
        evidence=CapabilityEvidenceState.NONE,
        evidence_basis="provider_returned_zero_records",
        coverage=CapabilityCoverageState.EXHAUSTED,
        coverage_scope="provider query for exact identifier",
        coverage_basis="provider_completed_exact_query",
        recovery=CapabilityRecoveryAction.DO_NOT_RETRY,
        recovery_basis="same_query_is_deterministically_empty",
    )
    assert exhausted.coverage == CapabilityCoverageState.EXHAUSTED

    with pytest.raises(ValidationError, match="coverage_scope"):
        CapabilityOutcome(
            applicability=CapabilityApplicability.APPLICABLE,
            applicability_basis="typed_operation_contract",
            evidence=CapabilityEvidenceState.NONE,
            evidence_basis="provider_returned_zero_records",
            coverage=CapabilityCoverageState.EXHAUSTED,
            coverage_basis="provider_completed_exact_query",
        )


def test_capability_outcome_rejects_global_claims_from_inapplicable_call():
    with pytest.raises(ValidationError, match="cannot claim available evidence"):
        CapabilityOutcome(
            applicability=CapabilityApplicability.NOT_APPLICABLE,
            applicability_basis="operation_not_declared",
            evidence=CapabilityEvidenceState.AVAILABLE,
            evidence_basis="invalid_test_claim",
        )


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"applicability": "applicable"}, "applicability_basis"),
        ({"evidence": "available"}, "evidence_basis"),
        (
            {"coverage": "more_available", "coverage_basis": "next_page"},
            "coverage_scope",
        ),
        ({"recovery": "retry_same_call"}, "recovery_basis"),
    ],
)
def test_capability_outcome_requires_basis_for_non_unknown_facts(kwargs, expected):
    with pytest.raises(ValidationError, match=expected):
        CapabilityOutcome(**kwargs)


def test_pending_result_has_no_finished_timestamp():
    result = _finished_result(
        status=CapabilityStatus.PENDING,
        finished_at=None,
    )
    assert result.status == CapabilityStatus.PENDING

    with pytest.raises(ValidationError, match="pending results"):
        _finished_result(status=CapabilityStatus.PENDING)


def test_needs_input_result_requires_non_retryable_error_and_missing_fields():
    now = datetime.now(timezone.utc)
    result = CapabilityResult(
        ok=False,
        status=CapabilityStatus.NEEDS_INPUT,
        call_id="call_missing",
        capability_id="skill.analysis.run",
        capability_version="1",
        provider_type=ProviderType.SKILL_SCRIPT,
        error=CapabilityError(
            kind=CapabilityErrorKind.NEEDS_INPUT,
            message="missing sample sheet",
            missing_fields=["sample_sheet"],
        ),
        started_at=now,
        finished_at=now,
    )
    assert result.error is not None
    assert result.error.retryable is False


def test_registered_artifact_requires_conversation_file_id():
    with pytest.raises(ValidationError, match="conversation_file_id"):
        ArtifactRef(
            artifact_id="artifact_1",
            file_name="report.md",
            registration_status="registered",
        )


def test_evidence_reference_requires_stable_locator():
    with pytest.raises(ValidationError, match="stable locator"):
        EvidenceRef(title="unlocatable")
    assert EvidenceRef(title="record", database="PDB", record_id="4HHB").record_id == "4HHB"


def test_json_schema_validation_reports_stable_nested_paths_and_formats():
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["records"],
        "properties": {
            "records": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["url", "score"],
                    "properties": {
                        "url": {"type": "string", "format": "uri"},
                        "score": {"type": "number", "minimum": 0},
                    },
                },
            }
        },
    }
    assert check_json_schema(schema).valid is True
    valid = validate_json_instance({"records": [{"url": "https://example.org", "score": 1}]}, schema)
    assert valid.valid is True

    invalid = validate_json_instance({"records": [{"url": "not a url", "score": -1}]}, schema)
    assert invalid.valid is False
    assert {issue.path for issue in invalid.issues} == {"$.records[0].score", "$.records[0].url"}


def test_invalid_json_schema_is_reported_without_validating_instance():
    report = check_json_schema({"type": "definitely-not-a-json-schema-type"})
    assert report.valid is False
    assert report.issues[0].kind == "schema_error"
