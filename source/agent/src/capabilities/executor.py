"""Mechanical capability execution with contract validation."""

from __future__ import annotations

import asyncio
import time
import logging
from typing import Any

from src.capabilities.artifacts import CapabilityArtifactRegistrar
from src.capabilities.inputs import (
    CapabilityInputMaterializer,
    InputMaterializationError,
    requires_input_materialization,
)
from src.capabilities.models import (
    AgentMode,
    CapabilityCall,
    CapabilityError,
    CapabilityErrorKind,
    CapabilityOutcome,
    CapabilityRegistrySnapshot,
    CapabilityResult,
    CapabilitySpec,
    CapabilityStatus,
    InputRefHandling,
    ProviderType,
    RegistrationStatus,
    utc_now,
)
from src.capabilities.idempotency import build_idempotency_key
from src.capabilities.providers.base import ProviderContext, ProviderInvocationResult
from src.capabilities.registry import (
    CapabilityNotFoundError,
    CapabilityProviderNotFoundError,
    CapabilityRegistry,
)
from src.capabilities.validation import ContractValidationReport, validate_json_instance
from src.capabilities.state_store import (
    CapabilityExecutionBlocked,
    CapabilityExecutionStateStore,
    CapabilityTaskNodeBindingError,
    PendingSandboxPromotion,
)
from src.runtime.worker_diagnostics import emit_worker_checkpoint
from src.context.task_state import TaskPhase


logger = logging.getLogger("evoengine-agent")
_PROVIDER_RECONCILIATION_ADAPTER = "provider_reconcile_v1"


def _declared_provider_reconciler(spec: CapabilitySpec, provider: Any):
    """Return the explicitly contracted provider reconciliation entrypoint."""

    declaration = str(
        spec.provider_ref.get("reconciliation_adapter") or ""
    ).strip()
    reconciler = getattr(provider, "reconcile", None)
    if declaration == _PROVIDER_RECONCILIATION_ADAPTER and callable(reconciler):
        return reconciler
    return None


def _missing_fields(report: ContractValidationReport) -> list[str]:
    fields: list[str] = []
    for issue in report.issues:
        candidates: list[Any] = []
        if issue.validator == "required":
            candidates = list(issue.details.get("missing_fields", []))
        elif issue.validator in {"anyOf", "oneOf"}:
            alternatives = [
                list(item)
                for item in issue.details.get("missing_field_sets", [])
                if isinstance(item, list) and item
            ]
            if alternatives:
                candidates = min(alternatives, key=lambda item: (len(item), alternatives.index(item)))
            else:
                return []
        else:
            # Missing fields are actionable only when they are the complete
            # validation failure. Unknown fields, wrong types, or constraints
            # must remain validation errors even if another schema branch is
            # also missing inputs.
            return []
        for field in candidates:
            normalized = str(field or "").strip()
            if normalized and normalized not in fields:
                fields.append(normalized)
    return fields


def _failed_result(
    *,
    call: CapabilityCall,
    spec: CapabilitySpec | None,
    started_at,
    started_monotonic: float,
    error: CapabilityError,
    summary: str,
    raw_ref: str | None = None,
    status: CapabilityStatus = CapabilityStatus.FAILED,
    capability_outcome: CapabilityOutcome | None = None,
    data: dict[str, Any] | None = None,
    contract_data: dict[str, Any] | None = None,
    artifacts: list[Any] | None = None,
    resources: list[Any] | None = None,
    evidence_refs: list[Any] | None = None,
) -> CapabilityResult:
    finished_at = max(utc_now(), started_at)
    return CapabilityResult(
        ok=False,
        status=status,
        call_id=call.call_id,
        capability_id=call.capability_id,
        capability_version=call.capability_version,
        provider_type=spec.provider_type if spec is not None else None,
        summary=summary,
        data=dict(data or {}),
        contract_data=contract_data,
        artifacts=list(artifacts or []),
        resources=list(resources or []),
        evidence_refs=list(evidence_refs or []),
        error=error,
        raw_ref=raw_ref,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_ms=max(0, int((time.monotonic() - started_monotonic) * 1000)),
        capability_outcome=capability_outcome or CapabilityOutcome(),
    )


class CapabilityExecutor:
    def __init__(
        self,
        registry: CapabilityRegistry,
        *,
        artifact_registrar: CapabilityArtifactRegistrar | None = None,
        input_materializer: CapabilityInputMaterializer | None = None,
        execution_state_store: CapabilityExecutionStateStore | None = None,
        _bound_registry_snapshot_id: str = "",
        _transport_binding: dict[str, str] | None = None,
    ):
        self._registry = registry
        self._artifact_registrar = artifact_registrar
        self._input_materializer = input_materializer
        self._execution_state_store = execution_state_store
        self._bound_registry_snapshot_id = str(_bound_registry_snapshot_id or "").strip()
        self._transport_binding = (
            _transport_binding if _transport_binding is not None else {}
        )

    def bind_transport_request_id(self, request_id: str) -> None:
        normalized = str(request_id or "").strip()
        if not normalized:
            raise ValueError("capability transport request_id is required")
        existing = str(self._transport_binding.get("request_id") or "").strip()
        if existing and existing != normalized:
            raise ValueError("capability executor transport binding is immutable")
        self._transport_binding["request_id"] = normalized

    def _transport_request_id(self, call: CapabilityCall) -> str:
        normalized = str(self._transport_binding.get("request_id") or "").strip()
        return normalized or str(call.request_id or "").strip()

    def scoped(self, snapshot: CapabilityRegistrySnapshot) -> "ScopedCapabilityExecutor":
        """Create a child executor over one frozen capability view."""

        return ScopedCapabilityExecutor(self, snapshot)

    async def promote_sandbox_job(
        self,
        *,
        job_id: str,
        request_id: str,
        project_id: str,
        conversation_id: str,
        user_id: int,
        task_phase: TaskPhase | str = TaskPhase.EXECUTE,
    ) -> tuple[CapabilityResult, PendingSandboxPromotion]:
        """Mechanically resume the original submit action bound to ``job_id``.

        This is the only task-tree sandbox promotion path.  The caller does
        not supply a node, capability, begin call, arguments or idempotency
        identity; all of them are recovered from the authoritative pending
        receipt and revalidated before provider access.
        """

        resolver = (
            getattr(self._execution_state_store, "resolve_sandbox_promotion", None)
            if self._execution_state_store is not None
            else None
        )
        if not callable(resolver):
            raise CapabilityTaskNodeBindingError(
                "sandbox promotion requires a task-tree execution state store"
            )
        promotion = await resolver(
            job_id=job_id,
            request_id=request_id,
            project_id=project_id,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        try:
            spec = self._registry.resolve(
                promotion.call.capability_id,
                version=promotion.call.capability_version,
            )
        except CapabilityNotFoundError as exc:
            raise CapabilityTaskNodeBindingError(
                "sandbox promotion capability is not present in this frozen runtime"
            ) from exc
        if (
            spec.provider_type != ProviderType.SANDBOX
            or "creates_sandbox_job" not in set(spec.side_effects)
            or str(spec.provider_ref.get("tool_name") or spec.display_name).strip()
            != promotion.tool_name
        ):
            raise CapabilityTaskNodeBindingError(
                "sandbox promotion receipt does not identify a sandbox submit action"
            )
        if self._transport_request_id(promotion.call) != str(request_id or "").strip():
            raise CapabilityTaskNodeBindingError(
                "sandbox wakeup transport does not match the authoritative submit transport"
            )
        result = await self.invoke(
            promotion.call,
            task_phase=task_phase,
            provider_metadata={
                "sandbox_promotion_job_id": promotion.job_id,
                "sandbox_promotion_required": True,
            },
        )
        return result, promotion

    async def _record_result(
        self,
        *,
        call: CapabilityCall,
        spec: CapabilitySpec,
        result: CapabilityResult,
    ) -> CapabilityResult:
        logger.info(
            "capability_call_finished call_id=%s capability_id=%s provider_type=%s status=%s ok=%s elapsed_ms=%s idempotency_reused=%s artifact_count=%s job_id=%s",
            result.call_id,
            result.capability_id,
            result.provider_type.value if result.provider_type is not None else "",
            result.status.value,
            result.ok,
            result.elapsed_ms,
            result.idempotency_reused,
            len(result.artifacts),
            str(result.data.get("job_id") or ""),
        )
        emit_worker_checkpoint(
            "capability_call_finished",
            call_id=result.call_id,
            capability_id=result.capability_id,
            provider_type=result.provider_type.value if result.provider_type is not None else "",
            status=result.status.value,
            ok=result.ok,
            elapsed_ms=result.elapsed_ms,
            idempotency_reused=result.idempotency_reused,
            artifact_count=len(result.artifacts),
            job_id=str(result.data.get("job_id") or ""),
        )
        if self._execution_state_store is None or (
            not spec.execution_policy.idempotent and not call.task_node_id
        ):
            return result
        try:
            # Once provider execution has produced a terminal value, receipt
            # finalization is the state-machine commit boundary.  Shield that
            # bounded HTTP write so cancellation cannot leave a terminal
            # provider result with only an authoritative begin record.
            receipt_task = asyncio.create_task(
                self._execution_state_store.record(
                    call=call,
                    spec=spec,
                    result=result,
                    transport_request_id=self._transport_request_id(call),
                )
            )
            try:
                await asyncio.shield(receipt_task)
            except asyncio.CancelledError as cancelled:
                try:
                    await receipt_task
                except Exception as exc:  # noqa: BLE001 - preserve cancellation
                    logger.warning(
                        "capability_state_record_failed_during_cancel call_id=%s error_type=%s",
                        call.call_id,
                        exc.__class__.__name__,
                    )
                raise cancelled
        except Exception as exc:  # noqa: BLE001 - bound receipts fail closed below
            logger.warning(
                "capability_state_record_failed call_id=%s error_type=%s",
                call.call_id,
                exc.__class__.__name__,
            )
            if call.task_tree_binding_required:
                return result.model_copy(
                    update={
                        "ok": True,
                        "status": CapabilityStatus.PENDING,
                        "summary": "capability result is awaiting authoritative task-node receipt reconciliation",
                        "error": None,
                        "finished_at": None,
                        "complete": False,
                        "has_more": True,
                        "cursor": f"task-node-receipt:{call.call_id}",
                    }
                )
        return result

    async def invoke(
        self,
        call: CapabilityCall,
        *,
        task_phase: TaskPhase | str = TaskPhase.EXECUTE,
        agent_mode: AgentMode | str | None = None,
        provider_metadata: dict[str, Any] | None = None,
    ) -> CapabilityResult:
        started_at = utc_now()
        started_monotonic = time.monotonic()
        normalized_phase = TaskPhase(task_phase)
        if call.registry_snapshot_id != self._bound_registry_snapshot_id:
            return _failed_result(
                call=call,
                spec=None,
                started_at=started_at,
                started_monotonic=started_monotonic,
                error=CapabilityError(
                    kind=CapabilityErrorKind.CONTRACT_VIOLATION,
                    message=(
                        "registry_snapshot_id requires the matching scoped capability executor"
                    ),
                ),
                summary="capability call registry snapshot is not bound to this executor",
            )
        try:
            spec = self._registry.resolve(call.capability_id, version=call.capability_version)
        except CapabilityNotFoundError:
            return _failed_result(
                call=call,
                spec=None,
                started_at=started_at,
                started_monotonic=started_monotonic,
                error=CapabilityError(
                    kind=CapabilityErrorKind.PROVIDER_UNAVAILABLE,
                    message=f"capability is not registered: {call.capability_id}@{call.capability_version}",
                ),
                summary="capability is not registered",
            )

        logger.info(
            "capability_call_started call_id=%s capability_id=%s provider_type=%s request_id=%s task_node_id=%s",
            call.call_id,
            spec.capability_id,
            spec.provider_type.value,
            call.request_id or "",
            call.task_node_id or "",
        )
        emit_worker_checkpoint(
            "capability_call_started",
            call_id=call.call_id,
            capability_id=spec.capability_id,
            provider_type=spec.provider_type.value,
            request_id=call.request_id or "",
            task_node_id=call.task_node_id or "",
        )

        if not spec.enabled or spec.availability == "unavailable":
            return _failed_result(
                call=call,
                spec=spec,
                started_at=started_at,
                started_monotonic=started_monotonic,
                error=CapabilityError(
                    kind=CapabilityErrorKind.PROVIDER_UNAVAILABLE,
                    message=f"capability is unavailable: {spec.capability_id}",
                ),
                summary="capability is unavailable",
            )

        input_report = validate_json_instance(call.arguments, spec.input_schema)
        if not input_report.valid:
            missing = _missing_fields(input_report)
            if missing:
                return _failed_result(
                    call=call,
                    spec=spec,
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                    error=CapabilityError(
                        kind=CapabilityErrorKind.NEEDS_INPUT,
                        message="missing required capability inputs",
                        missing_fields=missing,
                    ),
                    summary="capability requires additional input",
                    status=CapabilityStatus.NEEDS_INPUT,
                )
            return _failed_result(
                call=call,
                spec=spec,
                started_at=started_at,
                started_monotonic=started_monotonic,
                error=CapabilityError(
                    kind=CapabilityErrorKind.VALIDATION_ERROR,
                    message="; ".join(
                        f"{issue.path}: {issue.message}"
                        for issue in input_report.issues
                    ),
                ),
                summary="capability input validation failed",
            )

        input_refs_payload = [
            item.model_dump(mode="json", exclude_none=True) for item in call.input_refs
        ]
        input_refs_report = validate_json_instance(input_refs_payload, spec.input_refs_schema)
        if not input_refs_report.valid:
            if not call.input_refs:
                return _failed_result(
                    call=call,
                    spec=spec,
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                    error=CapabilityError(
                        kind=CapabilityErrorKind.NEEDS_INPUT,
                        message="missing required capability file inputs",
                        missing_fields=["input_refs"],
                    ),
                    summary="capability requires file input",
                    status=CapabilityStatus.NEEDS_INPUT,
                )
            return _failed_result(
                call=call,
                spec=spec,
                started_at=started_at,
                started_monotonic=started_monotonic,
                error=CapabilityError(
                    kind=CapabilityErrorKind.VALIDATION_ERROR,
                    message="; ".join(
                        f"{issue.path}: {issue.message}"
                        for issue in input_refs_report.issues
                    ),
                ),
                summary="capability file input validation failed",
            )

        try:
            provider = self._registry.get_provider(spec.provider_type)
        except CapabilityProviderNotFoundError:
            return _failed_result(
                call=call,
                spec=spec,
                started_at=started_at,
                started_monotonic=started_monotonic,
                error=CapabilityError(
                    kind=CapabilityErrorKind.PROVIDER_UNAVAILABLE,
                    message=f"provider is not registered: {spec.provider_type.value}",
                ),
                summary="capability provider is unavailable",
            )

        project_call = (
            getattr(self._execution_state_store, "project_call", None)
            if self._execution_state_store is not None
            else None
        )
        task_node_id_before_projection = str(call.task_node_id or "").strip()
        idempotency_key_before_projection = str(call.idempotency_key or "")
        if callable(project_call):
            try:
                call = await project_call(call=call, spec=spec)
            except Exception as exc:  # noqa: BLE001 - projection is downstream
                logger.warning(
                    "capability_task_projection_failed call_id=%s error_type=%s",
                    call.call_id,
                    exc.__class__.__name__,
                )

        projected_task_node_id = str(call.task_node_id or "").strip()
        if (
            projected_task_node_id
            and not task_node_id_before_projection
            and call.idempotency_key == idempotency_key_before_projection
        ):
            # Dynamic TaskTree projection happens after the model tool call is
            # adapted.  Rebind provider idempotency to the mechanically chosen
            # action before begin/restore/provider access; otherwise two
            # actions with the same business arguments share an unbound key and
            # the later sandbox promotion cannot reconstruct the begin identity.
            call.idempotency_key = build_idempotency_key(
                capability_id=call.capability_id,
                capability_version=call.capability_version,
                arguments=call.arguments,
                input_refs=call.input_refs,
                project_id=call.project_id,
                conversation_id=call.conversation_id,
                context_fingerprint=call.context_fingerprint,
                registry_snapshot_id=call.registry_snapshot_id,
                task_node_id=projected_task_node_id,
            )

        validate_binding = (
            getattr(self._execution_state_store, "validate_binding", None)
            if self._execution_state_store is not None
            else None
        )
        if callable(validate_binding):
            try:
                await validate_binding(call=call, spec=spec)
            except CapabilityTaskNodeBindingError as exc:
                return _failed_result(
                    call=call,
                    spec=spec,
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                    error=CapabilityError(
                        kind=CapabilityErrorKind.CONTRACT_VIOLATION,
                        message=str(exc),
                    ),
                    summary="capability task-node binding is invalid",
                )
            except Exception as exc:  # noqa: BLE001 - explicit legacy binding
                logger.warning(
                    "capability_task_node_binding_check_failed call_id=%s error_type=%s",
                    call.call_id,
                    exc.__class__.__name__,
                )
                return _failed_result(
                    call=call,
                    spec=spec,
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                    error=CapabilityError(
                        kind=CapabilityErrorKind.CONTRACT_VIOLATION,
                        message="task-node projection identity could not be verified",
                    ),
                    summary="capability task-node projection is invalid",
                )

        provider_reconciler = _declared_provider_reconciler(spec, provider)
        bound_non_idempotent = bool(call.task_node_id) and not bool(
            spec.execution_policy.idempotent
        )
        if bound_non_idempotent and (
            self._execution_state_store is None or provider_reconciler is None
        ):
            return _failed_result(
                call=call,
                spec=spec,
                started_at=started_at,
                started_monotonic=started_monotonic,
                error=CapabilityError(
                    kind=CapabilityErrorKind.CONTRACT_VIOLATION,
                    message=(
                        "task-bound non-idempotent capability requires an explicit "
                        "provider_reconcile_v1 adapter"
                    ),
                ),
                summary="unsafe non-idempotent task action was rejected before begin",
            )

        effective_provider_metadata = dict(provider_metadata or {})
        recovered_task_node_id = ""
        begin_call = call
        reconciliation_required = False
        if self._execution_state_store is not None and (
            spec.execution_policy.idempotent or provider_reconciler is not None
        ):
            try:
                recovery = await self._execution_state_store.restore(call=call, spec=spec)
            except Exception as exc:  # noqa: BLE001 - provider idempotency remains fallback
                logger.warning(
                    "capability_state_restore_failed call_id=%s error_type=%s",
                    call.call_id,
                    exc.__class__.__name__,
                )
                recovery = None
            if recovery is not None:
                if recovery.result is not None:
                    return recovery.result
                effective_provider_metadata.update(recovery.provider_metadata)
                reconciliation_required = bool(
                    recovery.provider_metadata.get("reconciliation_required")
                )
                recovered_task_node_id = str(recovery.task_node_id or "")
                if recovery.begin_call_id:
                    begin_call = call.model_copy(
                        update={"call_id": recovery.begin_call_id}
                    )
        execution_call = begin_call
        use_provider_reconciler = bool(
            reconciliation_required and not spec.execution_policy.idempotent
        )
        if reconciliation_required and not str(
            (effective_provider_metadata.get("reconciliation_locator") or {}).get(
                "call_id"
            )
            if isinstance(
                effective_provider_metadata.get("reconciliation_locator"), dict
            )
            else ""
        ).strip():
            return _failed_result(
                call=execution_call,
                spec=spec,
                started_at=started_at,
                started_monotonic=started_monotonic,
                error=CapabilityError(
                    kind=CapabilityErrorKind.CONTRACT_VIOLATION,
                    message="provider reconciliation requires an authoritative begin locator",
                ),
                summary="capability reconciliation locator is unavailable",
            )

        begin = (
            getattr(self._execution_state_store, "begin", None)
            if self._execution_state_store is not None
            else None
        )
        if callable(begin):
            try:
                await begin(call=execution_call, spec=spec)
            except CapabilityExecutionBlocked as exc:
                return _failed_result(
                    call=execution_call,
                    spec=spec,
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                    error=CapabilityError(
                        kind=CapabilityErrorKind.EXECUTION_ERROR,
                        message=str(exc),
                        retryable=True,
                    ),
                    summary="capability is waiting for task dependencies",
                )
            except CapabilityTaskNodeBindingError as exc:
                return _failed_result(
                    call=execution_call,
                    spec=spec,
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                    error=CapabilityError(
                        kind=CapabilityErrorKind.CONTRACT_VIOLATION,
                        message=str(exc),
                    ),
                    summary="capability task-node begin was rejected",
                )
            except Exception as exc:  # noqa: BLE001 - state projection is non-critical
                if execution_call.task_tree_binding_required:
                    return _failed_result(
                        call=execution_call,
                        spec=spec,
                        started_at=started_at,
                        started_monotonic=started_monotonic,
                        error=CapabilityError(
                            kind=CapabilityErrorKind.CONTRACT_VIOLATION,
                            message="required task-node begin could not be verified",
                        ),
                        summary="capability task-node begin is unavailable",
                    )
                logger.warning(
                    "capability_state_begin_failed call_id=%s error_type=%s",
                    execution_call.call_id,
                    exc.__class__.__name__,
                )

        context = ProviderContext(
            request_id=execution_call.request_id,
            transport_request_id=self._transport_request_id(execution_call),
            task_phase=normalized_phase,
            agent_mode=AgentMode(agent_mode) if agent_mode is not None else None,
            capability_id=spec.capability_id,
            capability_version=spec.version,
            project_id=execution_call.project_id,
            conversation_id=execution_call.conversation_id,
            user_id=execution_call.user_id,
            provider_ref=dict(spec.provider_ref),
            evidence_policy=dict(spec.evidence_policy),
            artifact_specs=list(spec.produces),
            timeout_seconds=spec.execution_policy.timeout_seconds,
            metadata={
                **effective_provider_metadata,
                "call_id": execution_call.call_id,
                "idempotency_key": (
                    execution_call.idempotency_key
                    if spec.execution_policy.idempotent
                    or provider_reconciler is not None
                    else ""
                ),
                "registry_snapshot_id": execution_call.registry_snapshot_id,
                "task_node_id": execution_call.task_node_id
                or recovered_task_node_id,
            },
        )
        if (
            spec.input_ref_handling == InputRefHandling.MATERIALIZE
            and requires_input_materialization(execution_call.input_refs)
        ):
            if self._input_materializer is None:
                return await self._record_result(call=execution_call, spec=spec, result=_failed_result(
                    call=execution_call,
                    spec=spec,
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                    error=CapabilityError(
                        kind=CapabilityErrorKind.PROVIDER_UNAVAILABLE,
                        message="capability file input materializer is unavailable",
                    ),
                    summary="capability file input materializer is unavailable",
                ))
            try:
                materialized_inputs = await self._input_materializer.materialize(
                    input_refs=list(execution_call.input_refs),
                    context=context,
                )
            except InputMaterializationError as exc:
                return await self._record_result(call=execution_call, spec=spec, result=_failed_result(
                    call=execution_call,
                    spec=spec,
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                    error=CapabilityError(kind=exc.error_kind, message=str(exc)),
                    summary="capability file input materialization failed",
                ))
            context = context.model_copy(
                update={
                    "metadata": {
                        **context.metadata,
                        "materialized_inputs": [
                            item.model_dump(mode="json") for item in materialized_inputs
                        ],
                    }
                }
            )
        try:
            provider_entrypoint = (
                provider_reconciler
                if use_provider_reconciler
                else provider.invoke
            )
            provider_result = await provider_entrypoint(
                arguments=dict(execution_call.arguments),
                input_refs=input_refs_payload,
                context=context,
            )
            provider_result = ProviderInvocationResult.model_validate(provider_result)
        except asyncio.CancelledError:
            await self._record_result(call=execution_call, spec=spec, result=_failed_result(
                call=execution_call,
                spec=spec,
                started_at=started_at,
                started_monotonic=started_monotonic,
                error=CapabilityError(
                    kind=CapabilityErrorKind.CANCELLED,
                    message="capability provider invocation was cancelled",
                ),
                summary="capability provider invocation was cancelled",
                status=CapabilityStatus.CANCELLED,
            ))
            raise
        except TimeoutError:
            retryable = (
                CapabilityErrorKind.TIMEOUT in spec.execution_policy.retry.retryable_kinds
                and spec.execution_policy.retry.max_attempts > 1
            )
            return await self._record_result(call=execution_call, spec=spec, result=_failed_result(
                call=execution_call,
                spec=spec,
                started_at=started_at,
                started_monotonic=started_monotonic,
                error=CapabilityError(
                    kind=CapabilityErrorKind.TIMEOUT,
                    message="capability provider timed out",
                    retryable=retryable,
                ),
                summary="capability provider timed out",
            ))
        except Exception as exc:  # noqa: BLE001 - provider failures are isolated
            return await self._record_result(call=execution_call, spec=spec, result=_failed_result(
                call=execution_call,
                spec=spec,
                started_at=started_at,
                started_monotonic=started_monotonic,
                error=CapabilityError(
                    kind=CapabilityErrorKind.EXECUTION_ERROR,
                    message=f"capability provider failed: {exc.__class__.__name__}",
                ),
                summary="capability provider failed",
            ))

        needs_input_control = provider_result.needs_input_control
        if needs_input_control is not None and (
            needs_input_control.pending_call_id != execution_call.call_id
            or needs_input_control.capability_id != spec.capability_id
            or needs_input_control.capability_version != spec.version
            or needs_input_control.registry_snapshot_id
            != execution_call.registry_snapshot_id
        ):
            return await self._record_result(call=execution_call, spec=spec, result=_failed_result(
                call=execution_call,
                spec=spec,
                started_at=started_at,
                started_monotonic=started_monotonic,
                error=CapabilityError(
                    kind=CapabilityErrorKind.CONTRACT_VIOLATION,
                    message="typed needs_input control does not match the capability call",
                ),
                summary="capability authorization control is invalid",
            ))

        output_report = (
            validate_json_instance(
                provider_result.contract_data
                if provider_result.contract_data is not None
                else provider_result.data,
                spec.output_schema,
            )
            if provider_result.status == CapabilityStatus.SUCCEEDED
            else None
        )
        if output_report is not None and not output_report.valid:
            return await self._record_result(call=execution_call, spec=spec, result=_failed_result(
                call=execution_call,
                spec=spec,
                started_at=started_at,
                started_monotonic=started_monotonic,
                error=CapabilityError(
                    kind=CapabilityErrorKind.CONTRACT_VIOLATION,
                    message="; ".join(
                        f"{issue.path}: {issue.message}"
                        for issue in output_report.issues
                    ),
                    details_ref=provider_result.raw_ref,
                ),
                summary="capability output contract validation failed",
                raw_ref=provider_result.raw_ref,
            ))

        required_artifact_keys = {
            item.artifact_key for item in spec.produces if item.required
        }
        if provider_result.status == CapabilityStatus.SUCCEEDED:
            missing_artifact_keys = sorted(
                required_artifact_keys
                - {artifact.artifact_key for artifact in provider_result.artifacts}
            )
            if missing_artifact_keys:
                return await self._record_result(call=execution_call, spec=spec, result=_failed_result(
                    call=execution_call,
                    spec=spec,
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                    error=CapabilityError(
                        kind=CapabilityErrorKind.CONTRACT_VIOLATION,
                        message="required capability artifacts are missing: "
                        + ", ".join(missing_artifact_keys),
                        details_ref=provider_result.raw_ref,
                    ),
                    summary="required capability artifacts are missing",
                    raw_ref=provider_result.raw_ref,
                ))

        if (
            provider_result.status == CapabilityStatus.SUCCEEDED
            and provider_result.artifact_sources
            and self._artifact_registrar is not None
        ):
            try:
                registered_artifacts = await self._artifact_registrar.register(
                    artifacts=list(provider_result.artifacts),
                    sources=list(provider_result.artifact_sources),
                    context=context,
                    raw_ref=provider_result.raw_ref,
                )
            except Exception as exc:  # noqa: BLE001 - registration is an execution boundary
                return await self._record_result(call=execution_call, spec=spec, result=_failed_result(
                    call=execution_call,
                    spec=spec,
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                    error=CapabilityError(
                        kind=CapabilityErrorKind.EXECUTION_ERROR,
                        message=f"capability artifact registration failed: {exc.__class__.__name__}",
                        details_ref=provider_result.raw_ref,
                    ),
                    summary="capability artifact registration failed",
                    raw_ref=provider_result.raw_ref,
                ))
            provider_result = provider_result.model_copy(update={"artifacts": registered_artifacts})
            missing_registered_keys = sorted(
                key
                for key in required_artifact_keys
                if not any(
                    artifact.artifact_key == key
                    and artifact.registration_status == RegistrationStatus.REGISTERED
                    for artifact in registered_artifacts
                )
            )
            if missing_registered_keys:
                return await self._record_result(call=execution_call, spec=spec, result=_failed_result(
                    call=execution_call,
                    spec=spec,
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                    error=CapabilityError(
                        kind=CapabilityErrorKind.CONTRACT_VIOLATION,
                        message="required capability artifacts were not registered: "
                        + ", ".join(missing_registered_keys),
                        details_ref=provider_result.raw_ref,
                    ),
                    summary="required capability artifacts were not registered",
                    raw_ref=provider_result.raw_ref,
                ))

        acceptance_report = (
            validate_json_instance(
                provider_result.contract_data
                if provider_result.contract_data is not None
                else provider_result.data,
                spec.acceptance_schema,
            )
            if (
                provider_result.status == CapabilityStatus.SUCCEEDED
                and spec.acceptance_schema
            )
            else None
        )
        if acceptance_report is not None and not acceptance_report.valid:
            return await self._record_result(call=execution_call, spec=spec, result=_failed_result(
                call=execution_call,
                spec=spec,
                started_at=started_at,
                started_monotonic=started_monotonic,
                error=CapabilityError(
                    kind=CapabilityErrorKind.UNSATISFIED_RESULT,
                    message="; ".join(
                        f"{issue.path}: {issue.message}"
                        for issue in acceptance_report.issues
                    ),
                    details_ref=provider_result.raw_ref,
                ),
                summary="capability result did not satisfy its declared acceptance contract",
                raw_ref=provider_result.raw_ref,
                capability_outcome=provider_result.capability_outcome,
                data=provider_result.data,
                contract_data=provider_result.contract_data,
                artifacts=provider_result.artifacts,
                resources=provider_result.resources,
                evidence_refs=provider_result.evidence_refs,
            ))

        # A declared source contract is part of the capability result contract,
        # so its projection and registration happen at this one runtime
        # boundary.  The Agent never schedules a second verification tool.
        citation_projection = list(provider_result.citation_projection)
        source_outcome = provider_result.source_outcome
        source_sidecar_resource = None
        source_candidates = list(provider_result.source_candidates)
        if provider_result.status == CapabilityStatus.SUCCEEDED:
            from src.capabilities.source_runtime import (
                project_and_persist_declared_source,
            )

            automatic_source = project_and_persist_declared_source(
                spec=spec,
                result=provider_result,
                arguments=dict(execution_call.arguments),
                context=context,
            )
            if automatic_source is not None:
                citation_projection = list(automatic_source.citation_projection)
                source_outcome = automatic_source.source_outcome
                source_sidecar_resource = automatic_source.source_sidecar_resource
                # Candidates are an internal hand-off detail.  Once the
                # runtime has processed the declared source they must not be
                # exposed as a second Agent action.
                source_candidates = []

        finished_at = (
            None
            if provider_result.status == CapabilityStatus.PENDING
            else max(utc_now(), started_at)
        )
        result = CapabilityResult(
            ok=provider_result.status in {CapabilityStatus.SUCCEEDED, CapabilityStatus.PENDING},
            status=provider_result.status,
            call_id=execution_call.call_id,
            capability_id=spec.capability_id,
            capability_version=spec.version,
            provider_type=spec.provider_type,
            summary=provider_result.summary,
            data=provider_result.data,
            contract_data=provider_result.contract_data,
            citation_projection=citation_projection,
            source_candidates=source_candidates,
            resources=provider_result.resources,
            complete=provider_result.complete,
            has_more=provider_result.has_more,
            cursor=provider_result.cursor,
            artifacts=provider_result.artifacts,
            evidence_refs=provider_result.evidence_refs,
            capability_outcome=provider_result.capability_outcome,
            source_outcome=source_outcome,
            source_sidecar_resource=source_sidecar_resource,
            error=provider_result.error,
            needs_input_control=needs_input_control,
            raw_ref=provider_result.raw_ref,
            started_at=started_at,
            finished_at=finished_at,
            elapsed_ms=max(0, int((time.monotonic() - started_monotonic) * 1000)),
            idempotency_reused=provider_result.idempotency_reused,
            legacy_envelope=provider_result.legacy_envelope,
        )
        return await self._record_result(
            call=execution_call,
            spec=spec,
            result=result,
        )


class ScopedCapabilityExecutor:
    """Exact allowlist executor that never resolves specs from the live registry."""

    def __init__(
        self,
        parent: CapabilityExecutor,
        snapshot: CapabilityRegistrySnapshot,
    ) -> None:
        self._snapshot = CapabilityRegistrySnapshot.model_validate(
            snapshot.model_dump(mode="json")
        )
        frozen_registry = CapabilityRegistry()
        provider_types = set()
        for item in self._snapshot.items:
            spec = item.to_spec()
            frozen_registry.register_spec(spec)
            provider_types.add(spec.provider_type)
        for provider_type in provider_types:
            try:
                provider = parent._registry.get_provider(provider_type)
            except CapabilityProviderNotFoundError:
                continue
            frozen_registry.register_provider(provider)
        self._executor = CapabilityExecutor(
            frozen_registry,
            artifact_registrar=parent._artifact_registrar,
            input_materializer=parent._input_materializer,
            execution_state_store=parent._execution_state_store,
            _bound_registry_snapshot_id=self._snapshot.snapshot_id,
            _transport_binding=parent._transport_binding,
        )

    def bind_transport_request_id(self, request_id: str) -> None:
        self._executor.bind_transport_request_id(request_id)

    @property
    def snapshot(self) -> CapabilityRegistrySnapshot:
        return self._snapshot.model_copy(deep=True)

    @staticmethod
    def _reject(
        *,
        call: CapabilityCall,
        spec: CapabilitySpec | None,
        message: str,
    ) -> CapabilityResult:
        started_at = utc_now()
        started_monotonic = time.monotonic()
        return _failed_result(
            call=call,
            spec=spec,
            started_at=started_at,
            started_monotonic=started_monotonic,
            error=CapabilityError(
                kind=CapabilityErrorKind.CONTRACT_VIOLATION,
                message=message,
            ),
            summary="capability call is outside the frozen registry snapshot",
        )

    async def invoke(
        self,
        call: CapabilityCall,
        *,
        task_phase: TaskPhase | str = TaskPhase.EXECUTE,
        agent_mode: AgentMode | str | None = None,
        provider_metadata: dict[str, Any] | None = None,
    ) -> CapabilityResult:
        spec = self._snapshot.resolve_spec(
            call.capability_id,
            call.capability_version,
        )
        if call.registry_snapshot_id != self._snapshot.snapshot_id:
            return self._reject(
                call=call,
                spec=spec,
                message="registry_snapshot_id does not match the scoped executor",
            )
        if spec is None:
            return self._reject(
                call=call,
                spec=None,
                message=(
                    "capability identity is not present in the frozen registry snapshot: "
                    f"{call.capability_id}@{call.capability_version}"
                ),
            )
        expected_idempotency_key = build_idempotency_key(
            capability_id=call.capability_id,
            capability_version=call.capability_version,
            arguments=call.arguments,
            input_refs=call.input_refs,
            project_id=call.project_id,
            conversation_id=call.conversation_id,
            context_fingerprint=call.context_fingerprint,
            registry_snapshot_id=call.registry_snapshot_id,
            task_node_id=call.task_node_id,
        )
        if call.idempotency_key != expected_idempotency_key:
            return self._reject(
                call=call,
                spec=spec,
                message="idempotency_key does not bind the registry snapshot identity",
            )
        return await self._executor.invoke(
            call,
            task_phase=task_phase,
            agent_mode=agent_mode,
            provider_metadata=provider_metadata,
        )
