"""Provider for the existing retrieval subagent StructuredTool boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from src.capabilities.models import (
    CapabilityError,
    CapabilityErrorKind,
    CapabilitySpec,
    CapabilityStatus,
    Permission,
    ProviderType,
    DiscoveryPolicy,
)
from src.capabilities.providers.base import ProviderContext, ProviderInvocationResult
from src.capabilities.providers.native_tool import capability_spec_from_structured_tool
from src.capabilities.result_mapper import map_tool_envelope_to_capability_result
from src.capabilities.providers.structured_tool import invoke_structured_tool_function


def capability_spec_from_subagent_tool(
    tool: Any,
    *,
    version: str = "1.0.0",
    discovery: DiscoveryPolicy | dict[str, Any] | None = None,
) -> CapabilitySpec:
    native_spec = capability_spec_from_structured_tool(
        tool,
        category="retrieval_subagent",
        permission=Permission.READ,
        version=version,
        discovery=discovery,
    )
    return native_spec.model_copy(
        update={
            "provider_type": ProviderType.SUBAGENT,
            "execution_policy": native_spec.execution_policy.model_copy(
                update={"timeout_seconds": 120}
            ),
            "provider_ref": {
                **native_spec.provider_ref,
                "category": "retrieval_subagent",
            },
        }
    )


@dataclass(frozen=True)
class _SubagentBinding:
    tool: Any | None
    tool_name: str
    handler: Callable[
        [dict[str, Any], list[dict[str, Any]], ProviderContext],
        Awaitable[ProviderInvocationResult],
    ] | None = None


class SubagentProvider:
    provider_type = ProviderType.SUBAGENT

    def __init__(self) -> None:
        self._bindings: dict[str, _SubagentBinding] = {}

    def register(self, spec: CapabilitySpec, tool: Any) -> None:
        if spec.provider_type != ProviderType.SUBAGENT:
            raise ValueError("SubagentProvider only accepts subagent capability specs")
        tool_name = str(spec.provider_ref.get("tool_name") or getattr(tool, "name", "") or "").strip()
        if not tool_name:
            raise ValueError("subagent capability requires tool_name")
        binding = _SubagentBinding(
            tool=getattr(tool, "_evo_raw_tool", None) or tool,
            tool_name=tool_name,
        )
        existing = self._bindings.get(spec.capability_id)
        if existing is not None and existing != binding:
            raise ValueError(f"subagent capability already bound: {spec.capability_id}")
        self._bindings[spec.capability_id] = binding

    def register_handler(
        self,
        spec: CapabilitySpec,
        *,
        tool_name: str,
        handler: Callable[
            [dict[str, Any], list[dict[str, Any]], ProviderContext],
            Awaitable[ProviderInvocationResult],
        ],
    ) -> None:
        if spec.provider_type != ProviderType.SUBAGENT:
            raise ValueError("SubagentProvider only accepts subagent capability specs")
        binding = _SubagentBinding(tool=None, tool_name=str(tool_name), handler=handler)
        existing = self._bindings.get(spec.capability_id)
        if existing is not None and existing != binding:
            raise ValueError(f"subagent capability already bound: {spec.capability_id}")
        self._bindings[spec.capability_id] = binding

    async def invoke(
        self,
        *,
        arguments: dict[str, Any],
        input_refs: list[dict[str, Any]],
        context: ProviderContext,
    ) -> ProviderInvocationResult:
        if input_refs:
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Subagent input reference validation failed",
                error=CapabilityError(
                    kind=CapabilityErrorKind.VALIDATION_ERROR,
                    message="retrieval subagent does not accept file input_refs",
                ),
            )
        binding = self._bindings.get(context.capability_id)
        if binding is None:
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Retrieval subagent is unavailable",
                error=CapabilityError(
                    kind=CapabilityErrorKind.PROVIDER_UNAVAILABLE,
                    message="retrieval subagent binding is unavailable",
                ),
            )
        if binding.handler is not None:
            return await binding.handler(arguments, input_refs, context)
        try:
            raw_result = await invoke_structured_tool_function(
                binding.tool,
                arguments,
                timeout_seconds=context.timeout_seconds,
            )
        except TimeoutError:
            raise
        except Exception as exc:  # noqa: BLE001
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Retrieval subagent execution failed",
                error=CapabilityError(
                    kind=CapabilityErrorKind.EXECUTION_ERROR,
                    message=f"retrieval subagent failed: {exc.__class__.__name__}",
                ),
            )

        from src.services.tool_result_envelope import make_tool_result_envelope
        from src.services.resource_store import resource_scope_from_provider_context
        from src.services.tool_machine_projection import normalize_machine_data

        envelope = make_tool_result_envelope(
            binding.tool_name,
            raw_result,
            category="retrieval_subagent",
            resource_scope=resource_scope_from_provider_context(context),
        )
        mapped = map_tool_envelope_to_capability_result(
            tool_name=binding.tool_name,
            category="retrieval_subagent",
            raw_result=raw_result,
            envelope=envelope,
            capability_version=context.capability_version,
            call_id_namespace="subagent-provider",
        )
        from src.subagents.source_citation_projection import (
            verified_source_refs_from_result,
        )

        citation_projection = verified_source_refs_from_result(raw_result)
        return ProviderInvocationResult(
            status=mapped.status,
            summary=mapped.summary,
            data=mapped.data,
            contract_data=normalize_machine_data(raw_result),
            citation_projection=citation_projection,
            resources=mapped.resources,
            complete=mapped.complete,
            has_more=mapped.has_more,
            cursor=mapped.cursor,
            artifacts=mapped.artifacts,
            evidence_refs=mapped.evidence_refs,
            capability_outcome=mapped.capability_outcome,
            error=mapped.error,
            raw_ref=mapped.raw_ref,
            legacy_envelope=envelope,
        )
