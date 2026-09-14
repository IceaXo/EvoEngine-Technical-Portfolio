"""Capability provider and spec extraction for existing StructuredTool objects."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Iterable

from src.capabilities.models import (
    CapabilityError,
    CapabilityErrorKind,
    CapabilitySpec,
    CapabilityStatus,
    DiscoveryPolicy,
    Permission,
    ProviderType,
)
from src.capabilities.providers.base import ProviderContext, ProviderInvocationResult
from src.capabilities.registry import CapabilityRegistry
from src.capabilities.result_mapper import (
    capability_id_for_tool,
    map_tool_envelope_to_capability_result,
)
from src.capabilities.providers.structured_tool import invoke_structured_tool_function


def permission_from_tool_catalog(value: str | Permission) -> Permission:
    if isinstance(value, Permission):
        return value
    normalized = str(value or "read_only").strip().lower()
    if normalized in {"read", "read_only", "deep_read"}:
        return Permission.READ
    if normalized == "write":
        return Permission.WRITE
    if normalized == "execute":
        return Permission.EXECUTE
    return Permission.CONTROL


def _input_schema(tool: Any) -> dict[str, Any]:
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None and hasattr(args_schema, "model_json_schema"):
        schema = args_schema.model_json_schema()
        if isinstance(schema, dict):
            return schema
    if hasattr(tool, "get_input_schema"):
        schema_model = tool.get_input_schema()
        if hasattr(schema_model, "model_json_schema"):
            schema = schema_model.model_json_schema()
            if isinstance(schema, dict):
                return schema
    return {"type": "object", "additionalProperties": True}


def _declared_output_schema(tool: Any) -> dict[str, Any] | None:
    """Read an explicit output contract carried by a StructuredTool.

    LangChain's ``StructuredTool`` exposes an argument model but no symmetric
    result model.  EvoEngine tools may attach ``_evo_output_schema`` so the
    unified capability registry can validate and show the real result contract
    instead of silently degrading every native/subagent result to ``object``.
    """

    for candidate in (tool, getattr(tool, "_evo_raw_tool", None)):
        schema = getattr(candidate, "_evo_output_schema", None)
        if isinstance(schema, dict) and schema:
            return dict(schema)
    return None


def _declared_model_result_contract(tool: Any) -> str:
    """Read the model-projection policy declared by the tool implementation."""

    for candidate in (tool, getattr(tool, "_evo_raw_tool", None)):
        policy = str(getattr(candidate, "_evo_model_result_policy", "") or "").strip()
        if not policy:
            continue
        if policy != "bounded_window":
            raise ValueError(f"unsupported native model result policy: {policy}")
        return policy
    return ""


def _declared_task_node_role(tool: Any) -> str:
    for candidate in (tool, getattr(tool, "_evo_raw_tool", None)):
        role = str(getattr(candidate, "_evo_task_node_role", "") or "").strip()
        if role:
            return role
    return ""


def capability_spec_from_structured_tool(
    tool: Any,
    *,
    category: str = "",
    permission: str | Permission = Permission.READ,
    version: str = "1.0.0",
    requires_confirmation: bool = False,
    side_effects: Iterable[str] = (),
    output_schema: dict[str, Any] | None = None,
    discovery: DiscoveryPolicy | dict[str, Any] | None = None,
) -> CapabilitySpec:
    tool_name = str(getattr(tool, "name", "") or "").strip()
    if not tool_name:
        raise ValueError("StructuredTool requires a stable name")
    normalized_permission = permission_from_tool_catalog(permission)
    normalized_side_effects = list(side_effects)
    if not normalized_side_effects:
        if normalized_permission == Permission.WRITE:
            normalized_side_effects = ["writes_state"]
        elif normalized_permission == Permission.EXECUTE:
            normalized_side_effects = ["executes_work"]
        elif normalized_permission == Permission.CONTROL:
            normalized_side_effects = ["controls_runtime"]
    description = str(getattr(tool, "description", "") or "").strip()
    model_result_policy = _declared_model_result_contract(tool)
    provider_ref: dict[str, Any] = {
        "tool_name": tool_name,
        "category": str(category or "").strip(),
    }
    if model_result_policy:
        provider_ref["model_result_policy"] = model_result_policy
    task_node_role = _declared_task_node_role(tool)
    if task_node_role:
        provider_ref["task_node_role"] = task_node_role
    return CapabilitySpec(
        capability_id=capability_id_for_tool(tool_name, category),
        version=version,
        display_name=tool_name,
        description=description,
        provider_type=ProviderType.NATIVE,
        provider_ref=provider_ref,
        input_schema=_input_schema(tool),
        input_refs_schema={"type": "array", "maxItems": 0},
        output_schema=output_schema or _declared_output_schema(tool) or {"type": "object"},
        permission=normalized_permission,
        side_effects=normalized_side_effects,
        requires_confirmation=requires_confirmation,
        discovery=(
            discovery
            if isinstance(discovery, DiscoveryPolicy)
            else DiscoveryPolicy.model_validate(discovery or {})
        ),
    )


@dataclass(frozen=True)
class _NativeToolBinding:
    tool: Any
    tool_name: str
    category: str
    model_result_policy: str = ""


class NativeToolProvider:
    provider_type = ProviderType.NATIVE

    def __init__(self) -> None:
        self._bindings: dict[str, _NativeToolBinding] = {}

    def register(self, spec: CapabilitySpec, tool: Any) -> None:
        if spec.provider_type != ProviderType.NATIVE:
            raise ValueError("NativeToolProvider only accepts native capability specs")
        tool_name = str(spec.provider_ref.get("tool_name") or getattr(tool, "name", "") or "").strip()
        if not tool_name:
            raise ValueError("native capability requires tool_name")
        raw_tool = getattr(tool, "_evo_raw_tool", None) or tool
        binding = _NativeToolBinding(
            tool=raw_tool,
            tool_name=tool_name,
            category=str(spec.provider_ref.get("category") or "").strip(),
            model_result_policy=str(
                spec.provider_ref.get("model_result_policy") or ""
            ).strip(),
        )
        existing = self._bindings.get(spec.capability_id)
        if existing is not None and existing != binding:
            raise ValueError(f"native capability already bound: {spec.capability_id}")
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
                summary="Native tool input reference contract validation failed",
                error=CapabilityError(
                    kind=CapabilityErrorKind.VALIDATION_ERROR,
                    message="native tool does not accept file input_refs",
                ),
            )
        binding = self._bindings.get(context.capability_id)
        if binding is None:
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Native tool is unavailable",
                error=CapabilityError(
                    kind=CapabilityErrorKind.PROVIDER_UNAVAILABLE,
                    message="native tool binding is unavailable",
                ),
            )
        from src.services.conversation_file_registry import (
            bind_file_registration_authority,
        )

        task_node_id = str(context.metadata.get("task_node_id") or "").strip()
        tool_run_id = (
            str(context.metadata.get("call_id") or "").strip()
            if task_node_id
            else ""
        )
        try:
            with bind_file_registration_authority(
                request_id=context.request_id,
                transport_request_id=context.transport_request_id,
                task_node_id=task_node_id,
                tool_run_id=tool_run_id,
                tool_name=binding.tool_name,
            ):
                contextual_invoker = getattr(
                    binding.tool,
                    "_evo_invoke_with_provider_context",
                    None,
                )
                if callable(contextual_invoker):
                    raw_result = await asyncio.wait_for(
                        contextual_invoker(
                            arguments=dict(arguments),
                            context=context,
                        ),
                        timeout=context.timeout_seconds,
                    )
                else:
                    raw_result = await invoke_structured_tool_function(
                        binding.tool,
                        arguments,
                        timeout_seconds=context.timeout_seconds,
                    )
        except TimeoutError:
            raise
        except Exception as exc:  # noqa: BLE001 - isolate native implementation details
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Native tool execution failed",
                error=CapabilityError(
                    kind=CapabilityErrorKind.EXECUTION_ERROR,
                    message=f"native tool failed: {exc.__class__.__name__}",
                ),
            )

        from src.services.tool_result_envelope import make_tool_result_envelope
        from src.services.resource_store import resource_scope_from_provider_context
        envelope = make_tool_result_envelope(
            binding.tool_name,
            raw_result,
            category=binding.category,
            resource_scope=resource_scope_from_provider_context(context),
            model_result_policy=binding.model_result_policy,
        )
        mapped = map_tool_envelope_to_capability_result(
            tool_name=binding.tool_name,
            category=binding.category,
            raw_result=raw_result,
            envelope=envelope,
            capability_version=context.capability_version,
            call_id_namespace="native-provider",
        )
        if (
            str(context.provider_ref.get("task_node_role") or "").strip()
            == "preparation"
            or (
                isinstance(raw_result, dict)
                and raw_result.get("not_progress_node") is True
            )
        ):
            # Workspace files are not durable ConversationFile artifacts yet.
            # A legacy metadata/control tool may also explicitly declare
            # ``not_progress_node`` after an action projection has already
            # begun. Keep descriptors in contract_data/data, but do not attach
            # an unrelated pre-existing file to that action receipt. This lets
            # the orphan begin close without weakening file lineage checks.
            mapped = mapped.model_copy(update={"artifacts": []})
        citation_projection: list[dict[str, Any]] = []
        if (
            binding.category in {"source", "knowledge_assets"}
            and isinstance(envelope.get("source_sidecar_resource"), dict)
        ):
            from src.subagents.source_citation_projection import (
                verified_source_refs_from_result,
            )

            citation_projection = verified_source_refs_from_result(envelope)
        return ProviderInvocationResult(
            status=mapped.status,
            summary=mapped.summary,
            data=mapped.data,
            contract_data=mapped.contract_data,
            citation_projection=citation_projection,
            resources=mapped.resources,
            complete=mapped.complete,
            has_more=mapped.has_more,
            cursor=mapped.cursor,
            artifacts=mapped.artifacts,
            evidence_refs=mapped.evidence_refs,
            capability_outcome=mapped.capability_outcome,
            source_outcome=mapped.source_outcome,
            error=mapped.error,
            raw_ref=mapped.raw_ref,
            legacy_envelope=envelope,
        )


def register_native_structured_tools(
    registry: CapabilityRegistry,
    provider: NativeToolProvider,
    tools: Iterable[Any],
    *,
    catalog_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[CapabilitySpec]:
    registered: list[CapabilitySpec] = []
    metadata_by_id = catalog_by_id or {}
    for tool in tools:
        tool_name = str(getattr(tool, "name", "") or "").strip()
        metadata = metadata_by_id.get(tool_name) or {}
        category = str(
            getattr(tool, "_evo_tool_category", "")
            or metadata.get("category")
            or metadata.get("pack")
            or ""
        ).strip()
        spec = capability_spec_from_structured_tool(
            tool,
            category=category,
            permission=metadata.get("permission") or Permission.READ,
            requires_confirmation=bool(metadata.get("requires_confirmation", False)),
            side_effects=metadata.get("side_effects") or (),
            output_schema=metadata.get("output_schema"),
            discovery=metadata.get("discovery"),
        )
        if spec.provider_type != ProviderType.NATIVE:
            raise ValueError(f"tool is not a native capability: {tool_name}")
        registry.register_spec(spec, aliases=[tool_name])
        provider.register(spec, tool)
        registered.append(spec)
    return registered
