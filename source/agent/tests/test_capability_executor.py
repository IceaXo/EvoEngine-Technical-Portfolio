from __future__ import annotations

import asyncio

import pytest

from src.capabilities import (
    AgentMode,
    ArtifactRef,
    CapabilityCall,
    CapabilityOutcome,
    CapabilityError,
    CapabilityErrorKind,
    CapabilityExecutor,
    CapabilityRegistry,
    CapabilitySpec,
    CapabilityStatus,
    ExecutionPolicy,
    DrawerSection,
    InputRef,
    InputSourceType,
    MaterializedInput,
    Permission,
    ProviderType,
    RegistrationStatus,
    RetryPolicy,
)
from src.capabilities.providers import (
    ProviderArtifactSource,
    ProviderContext,
    ProviderInvocationResult,
)
from src.capabilities.idempotency import build_idempotency_key
from src.capabilities.state_store import CapabilityRecovery


class _FakeProvider:
    provider_type = ProviderType.NATIVE

    def __init__(self, result: ProviderInvocationResult | None = None, error: Exception | None = None):
        self.result = result or ProviderInvocationResult(
            status=CapabilityStatus.SUCCEEDED,
            summary="done",
            data={"value": 7},
        )
        self.error = error
        self.calls = 0
        self.last_arguments = None
        self.last_input_refs = None
        self.last_context = None

    async def invoke(self, *, arguments, input_refs, context: ProviderContext):
        self.calls += 1
        self.last_arguments = arguments
        self.last_input_refs = input_refs
        self.last_context = context
        if self.error is not None:
            raise self.error
        return self.result


def _spec(**updates) -> CapabilitySpec:
    payload = {
        "capability_id": "native.example.run",
        "version": "1.0.0",
        "display_name": "Example",
        "provider_type": ProviderType.NATIVE,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["query"],
            "properties": {"query": {"type": "string", "minLength": 1}},
        },
        "output_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["value"],
            "properties": {"value": {"type": "integer"}},
        },
        "permission": Permission.EXECUTE,
        "allowed_modes": [AgentMode.AGENT],
    }
    payload.update(updates)
    return CapabilitySpec(**payload)


def _call(**updates) -> CapabilityCall:
    payload = {
        "call_id": "call_1",
        "capability_id": "native.example.run",
        "capability_version": "1.0.0",
        "request_id": "req_1",
        "project_id": "project_1",
        "conversation_id": "conv_1",
        "user_id": 2,
        "arguments": {"query": "example"},
        "idempotency_key": "a" * 64,
    }
    payload.update(updates)
    return CapabilityCall(**payload)


def _executor(spec: CapabilitySpec | None = None, provider: _FakeProvider | None = None):
    registry = CapabilityRegistry()
    if spec is not None:
        registry.register_spec(spec)
    if provider is not None:
        registry.register_provider(provider)
    return CapabilityExecutor(registry), registry


def test_executor_validates_and_invokes_provider_with_context():
    provider = _FakeProvider()
    executor, _registry = _executor(_spec(), provider)
    result = asyncio.run(executor.invoke(_call(), agent_mode=AgentMode.AGENT))

    assert result.status == CapabilityStatus.SUCCEEDED
    assert result.data == {"value": 7}
    assert result.provider_type == ProviderType.NATIVE
    assert provider.calls == 1
    assert provider.last_arguments == {"query": "example"}
    assert provider.last_context.request_id == "req_1"
    assert provider.last_context.transport_request_id == "req_1"
    assert provider.last_context.capability_id == "native.example.run"
    assert provider.last_context.capability_version == "1.0.0"
    assert provider.last_context.timeout_seconds == 120
    assert provider.last_context.metadata["call_id"] == "call_1"


def test_executor_keeps_task_authority_separate_from_hitl_transport():
    provider = _FakeProvider()
    spec = _spec()
    state_store = _RecoveryStateStore(None)
    registry = CapabilityRegistry()
    registry.register_spec(spec)
    registry.register_provider(provider)
    executor = CapabilityExecutor(registry, execution_state_store=state_store)
    executor.bind_transport_request_id("req_hitl_continue_exact")

    result = asyncio.run(
        executor.invoke(
            _call(request_id="req_task_authority"),
            agent_mode=AgentMode.AGENT,
        )
    )

    assert result.status == CapabilityStatus.SUCCEEDED
    assert provider.last_context.request_id == "req_task_authority"
    assert provider.last_context.transport_request_id == "req_hitl_continue_exact"
    recorded_call, _recorded_result, recorded_transport = state_store.record_calls[0]
    assert recorded_call.request_id == "req_task_authority"
    assert recorded_transport == "req_hitl_continue_exact"
    with pytest.raises(ValueError, match="immutable"):
        executor.bind_transport_request_id("req_hitl_continue_prefix_other")


def test_executor_rejects_well_formed_but_unsatisfied_result():
    provider = _FakeProvider(
        ProviderInvocationResult(
            status=CapabilityStatus.SUCCEEDED,
            summary="well-formed but not accepted",
            data={"value": 7, "accepted": False},
            contract_data={"value": 7, "accepted": False},
        )
    )
    spec = _spec(
        output_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["value", "accepted"],
            "properties": {
                "value": {"type": "integer"},
                "accepted": {"type": "boolean"},
            },
        },
        acceptance_schema={
            "type": "object",
            "required": ["accepted"],
            "properties": {"accepted": {"const": True}},
        },
    )
    registry = CapabilityRegistry()
    registry.register_spec(spec)
    registry.register_provider(provider)
    executor = CapabilityExecutor(registry)

    result = asyncio.run(executor.invoke(_call(), agent_mode=AgentMode.AGENT))

    assert result.status == CapabilityStatus.FAILED
    assert result.error is not None
    assert result.error.kind == CapabilityErrorKind.UNSATISFIED_RESULT
    assert result.data == {"value": 7, "accepted": False}


def test_executor_acceptance_schema_is_declarative_and_optional():
    provider = _FakeProvider(
        ProviderInvocationResult(
            status=CapabilityStatus.SUCCEEDED,
            summary="accepted",
            data={"value": 7, "accepted": True},
            contract_data={"value": 7, "accepted": True},
        )
    )
    executor, _registry = _executor(
        _spec(
            output_schema={"type": "object"},
            acceptance_schema={
                "type": "object",
                "required": ["accepted"],
                "properties": {"accepted": {"const": True}},
            },
        ),
        provider,
    )

    accepted = asyncio.run(executor.invoke(_call(), agent_mode=AgentMode.AGENT))

    assert accepted.status == CapabilityStatus.SUCCEEDED
    assert accepted.data["accepted"] is True


def test_executor_preserves_provider_capability_outcome():
    provider = _FakeProvider(
        ProviderInvocationResult(
            status=CapabilityStatus.SUCCEEDED,
            summary="no matching records",
            data={"value": 7},
            capability_outcome=CapabilityOutcome(
                applicability="applicable",
                applicability_basis="typed_operation_contract",
                evidence="none",
                evidence_basis="provider_zero_records",
                coverage="exhausted",
                coverage_scope="exact test query",
                coverage_basis="provider_completed_query",
                recovery="do_not_retry",
                recovery_basis="same_query_empty",
            ),
        )
    )
    executor, _registry = _executor(_spec(), provider)

    result = asyncio.run(executor.invoke(_call(), agent_mode=AgentMode.AGENT))

    assert result.capability_outcome.coverage.value == "exhausted"
    assert result.capability_outcome.recovery.value == "do_not_retry"


def test_executor_does_not_upgrade_arbitrary_contract_data_to_citations():
    verified_record = {
        "ref_id": "R-PMID-19114008",
        "evidence_id": "E-PMID-19114008",
        "source_id": "pmid:19114008",
        "source_type": "literature",
        "provider": "ncbi_eutils",
        "provider_record_id": "19114008",
        "pmid": "19114008",
        "title": "WGCNA",
        "url": "https://pubmed.ncbi.nlm.nih.gov/19114008/",
        "canonical_url": "https://pubmed.ncbi.nlm.nih.gov/19114008/",
        "snippet": "WGCNA is an R package.",
        "verification": {
            "identity_status": "verified",
            "link_status": "reachable",
            "content_status": "abstract_only",
            "content_identity": "matched",
        },
        "claim_evidence_links": [
            {
                "claim_id": "C-PMID-19114008",
                "evidence_id": "E-PMID-19114008",
                "claim": "WGCNA is an R package.",
                "support_level": "primary",
                "verification_state": "completed",
                "verdict": "supported",
            }
        ],
    }
    contract_data = {"value": 7, "source_ledger": [verified_record]}
    provider = _FakeProvider(
        ProviderInvocationResult(
            status=CapabilityStatus.SUCCEEDED,
            summary="done",
            data={"value": 7},
            contract_data=contract_data,
        )
    )
    spec = _spec(
        output_schema={
            "type": "object",
            "additionalProperties": True,
            "required": ["value"],
            "properties": {"value": {"type": "integer"}},
        }
    )
    executor, _registry = _executor(spec, provider)

    result = asyncio.run(executor.invoke(_call(), agent_mode=AgentMode.AGENT))

    assert result.status == CapabilityStatus.SUCCEEDED
    assert result.citation_projection == []


def test_executor_passes_internal_provider_metadata_without_adding_it_to_arguments():
    provider = _FakeProvider()
    executor, _registry = _executor(_spec(provider_ref={"entrypoint": "run"}), provider)
    result = asyncio.run(
        executor.invoke(
            _call(),
            agent_mode=AgentMode.AGENT,
            provider_metadata={"artifact_output_dir": "/tmp/provider-output"},
        )
    )

    assert result.status == CapabilityStatus.SUCCEEDED
    assert provider.last_arguments == {"query": "example"}
    assert provider.last_context.provider_ref == {"entrypoint": "run"}
    assert provider.last_context.metadata["artifact_output_dir"] == "/tmp/provider-output"


def test_executor_materializes_file_refs_before_provider_invocation():
    class _Materializer:
        async def materialize(self, *, input_refs, context):
            assert input_refs[0].source_id == "101"
            assert context.capability_id == "native.example.run"
            return [
                MaterializedInput(
                    source_type=InputSourceType.CONVERSATION_FILE,
                    source_id="101",
                    file_name="input.txt",
                    local_path="/tmp/capability-inputs/input.txt",
                    mime_type="text/plain",
                    size_bytes=4,
                    sha256="a" * 64,
                )
            ]

    provider = _FakeProvider()
    registry = CapabilityRegistry()
    registry.register_spec(_spec())
    registry.register_provider(provider)
    executor = CapabilityExecutor(registry, input_materializer=_Materializer())
    call = _call(
        input_refs=[
            InputRef(
                source_type=InputSourceType.CONVERSATION_FILE,
                source_id="101",
                file_name="input.txt",
            )
        ]
    )

    result = asyncio.run(executor.invoke(call, agent_mode=AgentMode.AGENT))

    assert result.status == CapabilityStatus.SUCCEEDED
    assert provider.calls == 1
    materialized = provider.last_context.metadata["materialized_inputs"]
    assert materialized[0]["source_id"] == "101"
    assert materialized[0]["local_path"] == "/tmp/capability-inputs/input.txt"
    assert provider.last_input_refs[0]["source_type"] == "conversation_file"


def test_executor_does_not_invoke_provider_when_file_materializer_is_missing():
    provider = _FakeProvider()
    executor, _registry = _executor(_spec(), provider)
    call = _call(
        input_refs=[
            InputRef(source_type=InputSourceType.DATABASE_ASSET, source_id="202")
        ]
    )

    result = asyncio.run(executor.invoke(call, agent_mode=AgentMode.AGENT))

    assert result.status == CapabilityStatus.FAILED
    assert result.error is not None
    assert result.error.kind == CapabilityErrorKind.PROVIDER_UNAVAILABLE
    assert provider.calls == 0


def test_executor_returns_needs_input_without_invoking_provider():
    provider = _FakeProvider()
    executor, _registry = _executor(_spec(), provider)
    result = asyncio.run(
        executor.invoke(_call(arguments={}), agent_mode=AgentMode.AGENT)
    )
    assert result.status == CapabilityStatus.NEEDS_INPUT
    assert result.error is not None
    assert result.error.kind == CapabilityErrorKind.NEEDS_INPUT
    assert result.error.missing_fields == ["query"]
    assert provider.calls == 0


def test_executor_returns_needs_input_for_shortest_anyof_required_set():
    provider = _FakeProvider()
    spec = _spec(
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "question_text": {"type": "string", "minLength": 1},
                "phenotype": {"type": "string", "minLength": 1},
                "candidate_genes": {"type": "array", "minItems": 1},
            },
            "anyOf": [
                {"required": ["question_text"]},
                {"required": ["phenotype", "candidate_genes"]},
            ],
        }
    )
    executor, _registry = _executor(spec, provider)

    result = asyncio.run(executor.invoke(_call(arguments={}), agent_mode=AgentMode.AGENT))

    assert result.status == CapabilityStatus.NEEDS_INPUT
    assert result.error is not None
    assert result.error.missing_fields == ["question_text"]
    assert provider.calls == 0


def test_executor_keeps_anyof_wrong_type_as_validation_error():
    provider = _FakeProvider()
    spec = _spec(
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "question_text": {"type": "string", "minLength": 1},
                "phenotype": {"type": "string", "minLength": 1},
                "candidate_genes": {"type": "array", "minItems": 1},
            },
            "anyOf": [
                {"required": ["question_text"]},
                {"required": ["phenotype", "candidate_genes"]},
            ],
        }
    )
    executor, _registry = _executor(spec, provider)

    result = asyncio.run(
        executor.invoke(_call(arguments={"question_text": 123}), agent_mode=AgentMode.AGENT)
    )

    assert result.status == CapabilityStatus.FAILED
    assert result.error is not None
    assert result.error.kind == CapabilityErrorKind.VALIDATION_ERROR
    assert provider.calls == 0


def test_executor_returns_needs_input_when_required_file_ref_is_missing():
    provider = _FakeProvider()
    spec = _spec(
        input_refs_schema={
            "type": "array",
            "minItems": 1,
            "maxItems": 1,
            "items": {"type": "object"},
        }
    )
    executor, _registry = _executor(spec, provider)

    result = asyncio.run(executor.invoke(_call(), agent_mode=AgentMode.AGENT))

    assert result.status == CapabilityStatus.NEEDS_INPUT
    assert result.error is not None
    assert result.error.kind == CapabilityErrorKind.NEEDS_INPUT
    assert result.error.missing_fields == ["input_refs"]
    assert provider.calls == 0


def test_executor_rejects_file_ref_source_not_allowed_by_capability():
    provider = _FakeProvider()
    spec = _spec(
        input_refs_schema={
            "type": "array",
            "minItems": 1,
            "maxItems": 1,
            "items": {
                "type": "object",
                "required": ["source_type", "source_id"],
                "properties": {
                    "source_type": {"const": "conversation_file"},
                    "source_id": {"type": "string"},
                },
            },
        }
    )
    executor, _registry = _executor(spec, provider)
    call = _call(
        input_refs=[
            InputRef(source_type=InputSourceType.DATABASE_ASSET, source_id="202")
        ]
    )

    result = asyncio.run(executor.invoke(call, agent_mode=AgentMode.AGENT))

    assert result.status == CapabilityStatus.FAILED
    assert result.error is not None
    assert result.error.kind == CapabilityErrorKind.VALIDATION_ERROR
    assert provider.calls == 0


def test_executor_returns_validation_error_for_wrong_input_type():
    provider = _FakeProvider()
    executor, _registry = _executor(_spec(), provider)
    result = asyncio.run(
        executor.invoke(_call(arguments={"query": 123}), agent_mode=AgentMode.AGENT)
    )
    assert result.status == CapabilityStatus.FAILED
    assert result.error is not None
    assert result.error.kind == CapabilityErrorKind.VALIDATION_ERROR
    assert provider.calls == 0


class _ProjectionRecordingStateStore:
    def __init__(self) -> None:
        self.project_calls = []

    async def project_call(self, *, call, spec):
        self.project_calls.append((call, spec))
        return call


def test_executor_does_not_project_invalid_business_input_into_task_tree():
    provider = _FakeProvider()
    state_store = _ProjectionRecordingStateStore()
    registry = CapabilityRegistry()
    registry.register_spec(_spec())
    registry.register_provider(provider)
    executor = CapabilityExecutor(
        registry,
        execution_state_store=state_store,
    )

    result = asyncio.run(
        executor.invoke(
            _call(
                arguments={"query": 123},
                task_tree_binding_required=True,
            ),
            agent_mode=AgentMode.AGENT,
        )
    )

    assert result.status == CapabilityStatus.FAILED
    assert result.error is not None
    assert result.error.kind == CapabilityErrorKind.VALIDATION_ERROR
    assert state_store.project_calls == []
    assert provider.calls == 0


def test_executor_ignores_legacy_mode_and_uses_single_execution_policy():
    provider = _FakeProvider()
    executor, _registry = _executor(_spec(), provider)
    result = asyncio.run(executor.invoke(_call(), agent_mode=AgentMode.PLAN))
    assert result.ok is True
    assert result.error is None
    assert provider.calls == 1


def test_executor_reports_unregistered_capability_and_provider():
    executor, _registry = _executor()
    missing_capability = asyncio.run(executor.invoke(_call(), agent_mode=AgentMode.AGENT))
    assert missing_capability.error is not None
    assert missing_capability.error.kind == CapabilityErrorKind.PROVIDER_UNAVAILABLE
    assert missing_capability.provider_type is None

    executor, _registry = _executor(_spec(), None)
    missing_provider = asyncio.run(executor.invoke(_call(), agent_mode=AgentMode.AGENT))
    assert missing_provider.error is not None
    assert missing_provider.error.kind == CapabilityErrorKind.PROVIDER_UNAVAILABLE


def test_executor_rejects_unavailable_spec_without_provider_call():
    provider = _FakeProvider()
    executor, _registry = _executor(_spec(availability="unavailable"), provider)
    result = asyncio.run(executor.invoke(_call(), agent_mode=AgentMode.AGENT))
    assert result.error is not None
    assert result.error.kind == CapabilityErrorKind.PROVIDER_UNAVAILABLE
    assert provider.calls == 0


def test_executor_converts_output_schema_mismatch_to_contract_violation():
    provider = _FakeProvider(
        ProviderInvocationResult(
            status=CapabilityStatus.SUCCEEDED,
            summary="bad output",
            data={"value": "not-an-integer"},
            raw_ref="tool-result://raw/bad-output",
        )
    )
    executor, _registry = _executor(_spec(), provider)
    result = asyncio.run(executor.invoke(_call(), agent_mode=AgentMode.AGENT))
    assert result.status == CapabilityStatus.FAILED
    assert result.error is not None
    assert result.error.kind == CapabilityErrorKind.CONTRACT_VIOLATION
    assert result.error.retryable is False
    assert result.raw_ref == "tool-result://raw/bad-output"


def test_executor_preserves_provider_failure_without_applying_success_output_schema():
    provider = _FakeProvider(
        ProviderInvocationResult(
            status=CapabilityStatus.FAILED,
            summary="remote failed",
            data={},
            error=CapabilityError(
                kind=CapabilityErrorKind.EXECUTION_ERROR,
                message="remote failed",
            ),
        )
    )
    executor, _registry = _executor(_spec(), provider)
    result = asyncio.run(executor.invoke(_call(), agent_mode=AgentMode.AGENT))
    assert result.status == CapabilityStatus.FAILED
    assert result.error is not None
    assert result.error.kind == CapabilityErrorKind.EXECUTION_ERROR


def test_executor_preserves_pending_without_final_output_validation():
    provider = _FakeProvider(
        ProviderInvocationResult(
            status=CapabilityStatus.PENDING,
            summary="queued",
            data={"job_id": "job_1"},
        )
    )
    executor, _registry = _executor(_spec(), provider)
    result = asyncio.run(executor.invoke(_call(), agent_mode=AgentMode.AGENT))
    assert result.status == CapabilityStatus.PENDING
    assert result.finished_at is None
    assert result.data == {"job_id": "job_1"}


def test_executor_marks_timeout_retryable_only_when_spec_allows_it():
    policy = ExecutionPolicy(
        timeout_seconds=5,
        retry=RetryPolicy(
            max_attempts=2,
            retryable_kinds=[CapabilityErrorKind.TIMEOUT],
        ),
    )
    provider = _FakeProvider(error=TimeoutError())
    executor, _registry = _executor(_spec(execution_policy=policy), provider)
    result = asyncio.run(executor.invoke(_call(), agent_mode=AgentMode.AGENT))
    assert result.error is not None
    assert result.error.kind == CapabilityErrorKind.TIMEOUT
    assert result.error.retryable is True


def test_executor_isolates_provider_exception_without_leaking_message():
    provider = _FakeProvider(error=RuntimeError("secret provider detail"))
    executor, _registry = _executor(_spec(), provider)
    result = asyncio.run(executor.invoke(_call(), agent_mode=AgentMode.AGENT))
    assert result.error is not None
    assert result.error.kind == CapabilityErrorKind.EXECUTION_ERROR
    assert "secret provider detail" not in result.error.message


def test_executor_tolerates_wall_clock_step_back(monkeypatch):
    from datetime import datetime, timedelta, timezone

    from src.capabilities import executor as executor_module

    started = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
    timestamps = iter([started, started - timedelta(milliseconds=20)])
    monkeypatch.setattr(executor_module, "utc_now", lambda: next(timestamps))
    provider = _FakeProvider(error=RuntimeError("failed"))
    executor, _registry = _executor(_spec(), provider)

    result = asyncio.run(executor.invoke(_call(), agent_mode=AgentMode.AGENT))

    assert result.started_at == started
    assert result.finished_at == started


def test_scoped_executor_recomputes_action_scoped_idempotency_key():
    provider = _FakeProvider()
    executor, registry = _executor(_spec(), provider)
    snapshot = registry.freeze(["native.example.run@1.0.0"])
    scoped = executor.scoped(snapshot)
    bound_key = build_idempotency_key(
        capability_id="native.example.run",
        capability_version="1.0.0",
        arguments={"query": "example"},
        project_id="project_1",
        conversation_id="conv_1",
        registry_snapshot_id=snapshot.snapshot_id,
        task_node_id="node-action-1",
    )
    call = _call(
        registry_snapshot_id=snapshot.snapshot_id,
        task_node_id="node-action-1",
        idempotency_key=bound_key,
    )

    accepted = asyncio.run(scoped.invoke(call, agent_mode=AgentMode.AGENT))

    assert accepted.ok is True
    assert provider.calls == 1

    legacy_unscoped_key = build_idempotency_key(
        capability_id="native.example.run",
        capability_version="1.0.0",
        arguments={"query": "example"},
        project_id="project_1",
        conversation_id="conv_1",
        registry_snapshot_id=snapshot.snapshot_id,
    )
    rejected = asyncio.run(
        scoped.invoke(
            call.model_copy(update={"idempotency_key": legacy_unscoped_key}),
            agent_mode=AgentMode.AGENT,
        )
    )

    assert rejected.ok is False
    assert rejected.error is not None
    assert rejected.error.kind == CapabilityErrorKind.CONTRACT_VIOLATION
    assert provider.calls == 1


class _RecoveryStateStore:
    def __init__(self, recovery: CapabilityRecovery | None) -> None:
        self.recovery = recovery
        self.begin_calls = []
        self.record_calls = []

    async def validate_binding(self, *, call, spec) -> None:
        return None

    async def restore(self, *, call, spec):
        return self.recovery

    async def begin(self, *, call, spec) -> None:
        self.begin_calls.append(call)

    async def record(self, *, call, spec, result, transport_request_id) -> None:
        self.record_calls.append((call, result, transport_request_id))


class _ReconcilingProvider(_FakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.reconcile_calls = 0

    async def reconcile(self, *, arguments, input_refs, context):
        self.reconcile_calls += 1
        self.last_arguments = arguments
        self.last_input_refs = input_refs
        self.last_context = context
        return self.result


class _CapturingArtifactRegistrar:
    def __init__(self) -> None:
        self.contexts = []

    async def register(self, *, artifacts, sources, context, raw_ref):
        self.contexts.append(context)
        return [
            artifact.model_copy(
                update={
                    "conversation_file_id": 77,
                    "registration_status": RegistrationStatus.REGISTERED,
                }
            )
            for artifact in artifacts
        ]


def test_bound_non_idempotent_action_without_reconciler_fails_before_provider():
    provider = _FakeProvider()
    spec = _spec(execution_policy=ExecutionPolicy(idempotent=False))
    executor, _registry = _executor(spec, provider)

    result = asyncio.run(
        executor.invoke(
            _call(task_node_id="node-action-1"),
            agent_mode=AgentMode.AGENT,
        )
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.kind == CapabilityErrorKind.CONTRACT_VIOLATION
    assert "provider_reconcile_v1" in result.error.message
    assert provider.calls == 0


def test_bound_non_idempotent_recovery_uses_authoritative_begin_locator():
    provider = _ReconcilingProvider()
    spec = _spec(
        provider_ref={
            "tool_name": "native.example.run",
            "reconciliation_adapter": "provider_reconcile_v1",
        },
        execution_policy=ExecutionPolicy(idempotent=False),
    )
    recovery = CapabilityRecovery(
        provider_metadata={
            "reconciliation_required": True,
            "recovery_action": "reconcile_authoritative_begin",
            "reconciliation_locator": {
                "call_id": "call-original",
                "task_node_id": "node-action-1",
                "idempotency_key": "a" * 64,
            },
        },
        task_node_id="node-action-1",
        begin_call_id="call-original",
    )
    state_store = _RecoveryStateStore(recovery)
    registry = CapabilityRegistry()
    registry.register_spec(spec)
    registry.register_provider(provider)
    executor = CapabilityExecutor(
        registry,
        execution_state_store=state_store,
    )

    result = asyncio.run(
        executor.invoke(
            _call(
                call_id="call-retry",
                task_node_id="node-action-1",
                task_tree_binding_required=True,
            ),
            agent_mode=AgentMode.AGENT,
            provider_metadata={
                "call_id": "spoofed-call",
                "task_node_id": "spoofed-node",
            },
        )
    )

    assert provider.calls == 0
    assert provider.reconcile_calls == 1
    assert provider.last_context.metadata["call_id"] == "call-original"
    assert provider.last_context.metadata["task_node_id"] == "node-action-1"
    assert provider.last_context.metadata["reconciliation_locator"]["call_id"] == (
        "call-original"
    )
    assert state_store.begin_calls[0].call_id == "call-original"
    assert state_store.record_calls[0][0].call_id == "call-original"
    assert state_store.record_calls[0][1].call_id == "call-original"
    assert result.call_id == "call-original"


def test_recovered_action_keeps_authoritative_call_id_for_file_lineage():
    provider = _FakeProvider(
        ProviderInvocationResult(
            status=CapabilityStatus.SUCCEEDED,
            summary="generated",
            data={"value": 7},
            artifacts=[
                ArtifactRef(
                    artifact_id="artifact-report",
                    artifact_key="report",
                    file_name="report.md",
                    mime_type="text/markdown",
                    sha256="b" * 64,
                    drawer_section=DrawerSection.RESULT_FILE,
                )
            ],
            artifact_sources=[
                ProviderArtifactSource(
                    artifact_id="artifact-report",
                    local_path="/tmp/report.md",
                )
            ],
        )
    )
    recovery = CapabilityRecovery(
        provider_metadata={
            "reconciliation_required": True,
            "recovery_action": "resume_same_idempotent_call",
            "reconciliation_locator": {
                "call_id": "call-original",
                "task_node_id": "node-action-1",
                "idempotency_key": "a" * 64,
            },
        },
        task_node_id="node-action-1",
        begin_call_id="call-original",
    )
    state_store = _RecoveryStateStore(recovery)
    registrar = _CapturingArtifactRegistrar()
    registry = CapabilityRegistry()
    registry.register_spec(_spec())
    registry.register_provider(provider)
    executor = CapabilityExecutor(
        registry,
        artifact_registrar=registrar,
        execution_state_store=state_store,
    )

    result = asyncio.run(
        executor.invoke(
            _call(
                call_id="call-retry",
                task_node_id="node-action-1",
                task_tree_binding_required=True,
            ),
            agent_mode=AgentMode.AGENT,
        )
    )

    assert registrar.contexts[0].metadata["call_id"] == "call-original"
    assert registrar.contexts[0].metadata["task_node_id"] == "node-action-1"
    assert result.call_id == "call-original"
    assert result.artifacts[0].conversation_file_id == 77
    recorded_call, recorded_result, recorded_transport_request_id = (
        state_store.record_calls[0]
    )
    assert recorded_call.call_id == "call-original"
    assert recorded_result.call_id == "call-original"
    assert recorded_result.artifacts[0].conversation_file_id == 77
    assert recorded_transport_request_id == recorded_call.request_id
