"""Sandbox manifest adapter and Provider over the existing submit/result tools."""

from __future__ import annotations

import asyncio
import logging
import re
from concurrent.futures import Future
from dataclasses import dataclass
from threading import RLock
from typing import Any, Iterable

from src.capabilities.models import (
    AsyncMode,
    CapabilityError,
    CapabilityErrorKind,
    CapabilitySpec,
    CapabilityStatus,
    DiscoveryPolicy,
    ExecutionPolicy,
    InputRefHandling,
    InputSourceType,
    Permission,
    ProviderType,
)
from src.capabilities.providers.base import ProviderContext, ProviderInvocationResult
from src.capabilities.registry import CapabilityRegistry
from src.capabilities.result_mapper import map_tool_envelope_to_capability_result
from src.capabilities.providers.structured_tool import invoke_structured_tool_function
from src.capabilities.sandbox_output_contract import compile_sandbox_output_contract


logger = logging.getLogger(__name__)


def sandbox_capability_id(app_id: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", ".", str(app_id or "").strip().lower()).strip(".")
    if not normalized:
        raise ValueError("sandbox manifest requires app_id")
    return f"sandbox.{normalized}.run"


def _file_input_bindings(app: dict[str, Any]) -> dict[str, dict[str, str]]:
    from src.tools.sandbox_tools import _compact_submit_field

    bindings: dict[str, dict[str, str]] = {}
    for field in app.get("submit_fields") or []:
        if not isinstance(field, dict):
            continue
        field_type = str(field.get("type") or "").strip()
        key = str(field.get("key") or "").strip()
        if not key or field_type not in {"db_file", "query_source"}:
            continue
        compact = _compact_submit_field(field)
        accepted = compact.get("accepted_input_keys") if isinstance(compact, dict) else None
        if not isinstance(accepted, dict):
            continue
        database_key = str(accepted.get("database_asset_id") or "").strip()
        conversation_key = str(accepted.get("conversation_file_id") or "").strip()
        database_name_key = str(accepted.get("name_hint") or "").strip()
        if database_key and conversation_key:
            binding = {
                "database_asset": database_key,
                "conversation_file": conversation_key,
            }
            if database_name_key:
                binding["database_asset_name"] = database_name_key
            if field_type == "query_source":
                binding["inline"] = key
            aliases = {key, database_key, database_name_key, conversation_key}
            for alias in aliases:
                if alias:
                    bindings[alias] = binding
    return bindings


def sandbox_capability_spec_from_manifest(app: dict[str, Any]) -> CapabilitySpec:
    from src.tools.sandbox_tools import (
        _SANDBOX_CONTRACT_FINGERPRINT_SCHEMA,
        _agent_execution_policy,
        _build_submit_args_schema,
        _sandbox_contract_fingerprint,
    )

    app_id = str(app.get("app_id") or "").strip()
    if not app_id:
        raise ValueError("sandbox manifest requires app_id")
    typed_tool_id = f"sandbox_submit_{app_id}"
    raw_discovery = app.get("discovery")
    declared_discovery = dict(raw_discovery) if isinstance(raw_discovery, dict) else {}
    raw_discovery = {
        "visibility": "public",
        "match_mode": "natural",
        "aliases": [
            app_id,
            str(app.get("name") or app_id).strip(),
        ],
        "search_terms": [
            value
            for value in (
                str(app.get("description") or "").strip(),
                str(app.get("category_name") or app.get("category_id") or "").strip(),
            )
            if value
        ],
        **declared_discovery,
    }
    if str(raw_discovery.get("match_mode") or "exact_only") == "natural":
        raw_discovery.update(
            {
                "execution_entry_id": typed_tool_id,
                "load_tool_ids": [typed_tool_id],
            }
        )
    discovery = DiscoveryPolicy.model_validate(raw_discovery)
    contract_fingerprint = _sandbox_contract_fingerprint(app)
    output_contract = compile_sandbox_output_contract(app)
    args_schema = _build_submit_args_schema(app).model_json_schema()
    argument_defaults = {
        str(name): property_schema.get("default")
        for name, property_schema in (args_schema.get("properties") or {}).items()
        if isinstance(property_schema, dict)
        and "default" in property_schema
        and property_schema.get("default") is not None
    }
    # The typed tool validates its direct argument source oneOf before this
    # call. Capability callers may instead carry the same file source in the
    # separate input_refs contract, so Provider enforces these source groups
    # after merging both channels. Keep the groups machine-readable without
    # making arguments-only JSON validation reject legitimate input_refs.
    args_schema.pop("allOf", None)
    args_schema["additionalProperties"] = False
    required_arguments: list[str] = []
    for field in app.get("submit_fields") or []:
        if not isinstance(field, dict) or not bool(field.get("required")):
            continue
        key = str(field.get("key") or "").strip()
        if not key or key == "job_name" or field.get("default_value") not in (None, ""):
            continue
        if (
            str(field.get("type") or "").strip() not in {"db_file", "query_source"}
            and key in (args_schema.get("properties") or {})
        ):
            required_arguments.append(key)
    if required_arguments:
        args_schema["required"] = sorted(set(required_arguments))

    bindings = _file_input_bindings(app)
    input_ref_aliases = sorted(bindings)
    unique_binding_count = len(
        {
            tuple(sorted(value.items()))
            for value in bindings.values()
            if isinstance(value, dict)
        }
    )
    input_refs_schema: dict[str, Any] = {
        "type": "array",
        "maxItems": unique_binding_count,
    }
    if bindings:
        required_input_ref_fields = ["source_type", "source_id"]
        if unique_binding_count > 1:
            required_input_ref_fields.append("input_key")
        input_refs_schema["items"] = {
            "type": "object",
            "additionalProperties": False,
            "required": required_input_ref_fields,
            "properties": {
                "source_type": {
                    "enum": [
                        InputSourceType.CONVERSATION_FILE.value,
                        InputSourceType.DATABASE_ASSET.value,
                    ]
                },
                "source_id": {"type": "string", "minLength": 1},
                "input_key": {"enum": input_ref_aliases},
                "file_name": {"type": "string", "minLength": 1},
                "sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
            },
        }
    execution_mode, wait_seconds = _agent_execution_policy(app)
    runtime = app.get("runtime") if isinstance(app.get("runtime"), dict) else {}
    timeout_seconds = int(runtime.get("timeout_seconds") or 120)
    provider_timeout = max(30, min(86_400, timeout_seconds + max(0, wait_seconds) + 30))
    search_terms: list[str] = []
    for value in (
        app.get("category_name"),
        app.get("category"),
        app.get("tags"),
        app.get("keywords"),
    ):
        if isinstance(value, list):
            search_terms.extend(str(item or "") for item in value)
        elif value not in (None, ""):
            search_terms.append(str(value))
    introduction = app.get("introduction") if isinstance(app.get("introduction"), dict) else {}
    for value in introduction.values():
        if isinstance(value, str):
            search_terms.append(value)
        elif isinstance(value, dict):
            for nested in value.values():
                if isinstance(nested, str):
                    search_terms.append(nested)
                elif isinstance(nested, list):
                    search_terms.extend(str(item or "") for item in nested)
    for field in app.get("submit_fields") or []:
        if not isinstance(field, dict):
            continue
        search_terms.extend(
            str(field.get(key) or "")
            for key in ("label", "help_text", "description")
            if str(field.get(key) or "").strip()
        )
    return CapabilitySpec(
        capability_id=sandbox_capability_id(app_id),
        version=f"{_SANDBOX_CONTRACT_FINGERPRINT_SCHEMA}:{contract_fingerprint}",
        display_name=str(app.get("name") or app_id),
        description=str(app.get("description") or ""),
        provider_type=ProviderType.SANDBOX,
        provider_ref={
            "tool_name": typed_tool_id,
            "app_id": app_id,
            "app_name": str(app.get("name") or app_id),
            "manifest_version": str(app.get("manifest_version") or ""),
            "manifest_hash": str(app.get("manifest_hash") or ""),
            "contract_fingerprint": contract_fingerprint,
            "contract_schema": _SANDBOX_CONTRACT_FINGERPRINT_SCHEMA,
            "execution_mode": execution_mode,
            "wait_timeout_seconds": wait_seconds,
            "input_bindings": bindings,
            "field_contracts": list(args_schema.get("x-evo-field-contracts") or []),
            "argument_defaults": argument_defaults,
            "output_contract_fingerprint": (
                output_contract.fingerprint if output_contract is not None else ""
            ),
            "search_terms": list(dict.fromkeys(item.strip() for item in search_terms if item.strip())),
        },
        input_schema=args_schema,
        input_refs_schema=input_refs_schema,
        input_ref_handling=InputRefHandling.PASS_THROUGH,
        output_schema=(
            output_contract.output_schema
            if output_contract is not None
            else {"type": "object"}
        ),
        permission=Permission.EXECUTE,
        side_effects=["creates_sandbox_job", "creates_files"],
        execution_policy=ExecutionPolicy(
            timeout_seconds=provider_timeout,
            idempotent=True,
            async_mode=(
                AsyncMode.NONBLOCKING_LONG
                if execution_mode == "nonblocking_long"
                else AsyncMode.BLOCKING
            ),
        ),
        produces=(list(output_contract.produces) if output_contract is not None else []),
        discovery=discovery,
    )


def sandbox_result_capability_spec() -> CapabilitySpec:
    return CapabilitySpec(
        capability_id="sandbox.job.result",
        version="1.0.0",
        display_name="sandbox_get_result",
        description="Read or resume an existing sandbox job",
        provider_type=ProviderType.SANDBOX,
        provider_ref={
            "tool_name": "sandbox_get_result",
            "execution_mode": "nonblocking_long",
            "input_bindings": {},
            # This observes/promotes the original submit node from scheduler
            # facts; it is not an independent plan action or receipt authority.
            "task_node_role": "promotion",
        },
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["job_id"],
            "properties": {"job_id": {"type": "string", "minLength": 1}},
        },
        input_refs_schema={"type": "array", "maxItems": 0},
        input_ref_handling=InputRefHandling.PASS_THROUGH,
        output_schema={"type": "object"},
        permission=Permission.READ,
        execution_policy=ExecutionPolicy(
            timeout_seconds=120,
            # A poll is read-only but not replayable: the same job_id must be
            # observed again because running can become terminal.
            idempotent=False,
            async_mode=AsyncMode.POLL,
        ),
    )


def sandbox_utility_capability_spec(tool: Any) -> CapabilitySpec:
    """Compile stable sandbox read/control tools into the same Provider path."""

    tool_name = str(getattr(tool, "name", "") or "").strip()
    if tool_name == "sandbox_get_result":
        return sandbox_result_capability_spec()
    capability_ids = {
        "sandbox_submit": "sandbox.generic.submit",
        "sandbox_list_jobs": "sandbox.job.list",
        "sandbox_get_app_manifest": "sandbox.app.manifest.get",
        "sandbox_list_apps": "sandbox.app.list",
    }
    capability_id = capability_ids.get(tool_name)
    if capability_id is None:
        raise ValueError(f"unsupported sandbox utility tool: {tool_name}")
    args_schema = getattr(tool, "args_schema", None)
    input_schema = (
        args_schema.model_json_schema()
        if args_schema is not None and hasattr(args_schema, "model_json_schema")
        else {"type": "object", "additionalProperties": True}
    )
    output_schema = getattr(tool, "_evo_output_schema", None)
    is_submit = tool_name == "sandbox_submit"
    return CapabilitySpec(
        capability_id=capability_id,
        version="1.0.0",
        display_name=tool_name,
        description=str(getattr(tool, "description", "") or tool_name),
        provider_type=ProviderType.SANDBOX,
        provider_ref={
            "tool_name": tool_name,
            "execution_mode": "utility",
            "input_bindings": {},
            "task_node_role": "action" if is_submit else "navigation",
        },
        input_schema=input_schema,
        input_refs_schema={"type": "array", "maxItems": 0},
        input_ref_handling=InputRefHandling.PASS_THROUGH,
        output_schema=(
            dict(output_schema)
            if isinstance(output_schema, dict) and output_schema
            else {"type": "object"}
        ),
        permission=Permission.EXECUTE if is_submit else Permission.READ,
        side_effects=["creates_sandbox_job", "creates_files"] if is_submit else [],
        execution_policy=ExecutionPolicy(
            timeout_seconds=120,
            idempotent=is_submit,
            async_mode=(
                AsyncMode.NONBLOCKING_LONG if is_submit else AsyncMode.BLOCKING
            ),
        ),
    )


@dataclass(frozen=True)
class _SandboxBinding:
    submit_tool: Any
    result_tool: Any | None


class SandboxProvider:
    provider_type = ProviderType.SANDBOX

    def __init__(self) -> None:
        self._bindings: dict[str, _SandboxBinding] = {}
        self._completed: dict[str, ProviderInvocationResult] = {}
        self._inflight: dict[str, Future[ProviderInvocationResult]] = {}
        self._guard = RLock()
        self.registration_diagnostics: list[dict[str, str]] = []

    def register(self, spec: CapabilitySpec, *, submit_tool: Any, result_tool: Any | None = None) -> None:
        if spec.provider_type != ProviderType.SANDBOX:
            raise ValueError("SandboxProvider only accepts sandbox capability specs")
        binding = _SandboxBinding(
            submit_tool=getattr(submit_tool, "_evo_raw_tool", None) or submit_tool,
            result_tool=(getattr(result_tool, "_evo_raw_tool", None) or result_tool) if result_tool else None,
        )
        existing = self._bindings.get(spec.capability_id)
        if existing is not None and existing != binding:
            raise ValueError(f"sandbox capability already bound: {spec.capability_id}")
        self._bindings[spec.capability_id] = binding

    async def _invoke_tool(self, tool: Any, arguments: dict[str, Any], timeout: int) -> Any:
        return await invoke_structured_tool_function(
            tool,
            arguments,
            timeout_seconds=timeout,
        )

    @staticmethod
    def _map_result(
        tool_name: str,
        raw_result: Any,
        version: str,
        context: ProviderContext,
    ) -> ProviderInvocationResult:
        from src.services.tool_result_envelope import make_tool_result_envelope
        from src.services.resource_store import resource_scope_from_provider_context
        from src.services.tool_machine_projection import normalize_machine_data

        envelope = make_tool_result_envelope(
            tool_name,
            raw_result,
            category="sandbox",
            resource_scope=resource_scope_from_provider_context(context),
        )
        mapped = map_tool_envelope_to_capability_result(
            tool_name=tool_name,
            category="sandbox",
            raw_result=raw_result,
            envelope=envelope,
            capability_version=version,
            call_id_namespace="sandbox-provider",
        )
        # Capability output validation must see the complete bounded sandbox
        # result metadata, not the model-facing compact envelope.  The raw
        # payload contains previews only; full contents remain behind raw_ref.
        return ProviderInvocationResult(
            status=mapped.status,
            summary=mapped.summary,
            data=mapped.data,
            contract_data=normalize_machine_data(raw_result),
            resources=mapped.resources,
            complete=mapped.complete,
            has_more=mapped.has_more,
            cursor=mapped.cursor,
            artifacts=mapped.artifacts,
            evidence_refs=mapped.evidence_refs,
            error=mapped.error,
            raw_ref=mapped.raw_ref,
            legacy_envelope=envelope,
        )

    @staticmethod
    def _job_id(value: Any) -> str:
        if not isinstance(value, dict):
            return ""
        job = value.get("job") if isinstance(value.get("job"), dict) else {}
        return str(
            value.get("job_id")
            or value.get("sandbox_job_id")
            or job.get("job_id")
            or ""
        ).strip()

    async def _get_result(
        self,
        *,
        binding: _SandboxBinding,
        job_id: str,
        context: ProviderContext,
        public_tool_name: str = "sandbox_get_result",
    ) -> ProviderInvocationResult:
        if binding.result_tool is None:
            return ProviderInvocationResult(
                status=CapabilityStatus.PENDING,
                summary="Sandbox job is pending",
                data={"job_id": job_id, "status": "pending"},
            )
        task_node_id = str(context.metadata.get("task_node_id") or "").strip()
        # Every capability invocation has a call_id, but it is authoritative
        # task lineage only when the invocation is bound to a task action.
        # Treating an ordinary unbound blocking call as a half-bound action
        # makes its synchronous result path fail the node/call parity check.
        tool_run_id = (
            str(context.metadata.get("call_id") or "").strip()
            if task_node_id
            else ""
        )
        promotion_job_id = str(
            context.metadata.get("sandbox_promotion_job_id") or ""
        ).strip()
        promotion_required = bool(
            context.metadata.get("sandbox_promotion_required")
        )
        origin_tool_name = str(
            context.provider_ref.get("tool_name") or public_tool_name
        ).strip()
        from src.services.conversation_file_registry import (
            bind_file_registration_authority,
        )

        try:
            with bind_file_registration_authority(
                request_id=context.request_id,
                transport_request_id=context.transport_request_id,
                task_node_id=task_node_id,
                tool_run_id=tool_run_id,
                tool_name=origin_tool_name,
            ):
                if promotion_job_id and promotion_job_id != job_id:
                    raise ValueError(
                        "sandbox promotion job does not match the authoritative resume job"
                    )
                lineage_invoker = getattr(
                    binding.result_tool,
                    "_evo_get_result_with_lineage",
                    None,
                )
                if callable(lineage_invoker) and task_node_id and tool_run_id:
                    result_raw = await asyncio.wait_for(
                        asyncio.to_thread(
                            lineage_invoker,
                            job_id=job_id,
                            task_node_id=task_node_id,
                            tool_run_id=tool_run_id,
                            origin_tool_name=origin_tool_name,
                            lineage_request_id=context.request_id,
                        ),
                        timeout=context.timeout_seconds,
                    )
                elif promotion_required or bool(task_node_id) != bool(tool_run_id):
                    raise ValueError(
                        "sandbox result promotion requires exact task-node and begin-call lineage"
                    )
                else:
                    result_raw = await self._invoke_tool(
                        binding.result_tool,
                        {"job_id": job_id},
                        context.timeout_seconds,
                    )
        except TimeoutError:
            raise
        except Exception as exc:  # noqa: BLE001
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Sandbox result retrieval failed",
                error=CapabilityError(
                    kind=(
                        CapabilityErrorKind.CONTRACT_VIOLATION
                        if isinstance(exc, ValueError)
                        and (promotion_required or bool(promotion_job_id))
                        else CapabilityErrorKind.EXECUTION_ERROR
                    ),
                    message=f"sandbox result retrieval failed: {exc.__class__.__name__}",
                ),
            )
        if promotion_required:
            returned_job_id = self._job_id(result_raw)
            returned_task_node_id = str(
                result_raw.get("task_tree_node_id")
                if isinstance(result_raw, dict)
                else ""
            ).strip()
            if (
                not isinstance(result_raw, dict)
                or returned_job_id != job_id
                or returned_task_node_id != task_node_id
            ):
                return ProviderInvocationResult(
                    status=CapabilityStatus.FAILED,
                    summary="Sandbox result promotion lineage validation failed",
                    error=CapabilityError(
                        kind=CapabilityErrorKind.CONTRACT_VIOLATION,
                        message=(
                            "sandbox result response does not match the authoritative "
                            "job and task-node binding"
                        ),
                    ),
                )
        if isinstance(result_raw, str) and any(
            marker in result_raw.lower()
            for marker in ("仍在运行", "尚未产生", "queued", "running")
        ):
            result_raw = {
                "ok": True,
                "status": "running",
                "job_id": job_id,
                "summary": result_raw,
            }
        return self._map_result(
            public_tool_name,
            result_raw,
            context.capability_version,
            context,
        )

    @staticmethod
    def _merge_input_refs(
        arguments: dict[str, Any],
        input_refs: list[dict[str, Any]],
        context: ProviderContext,
    ) -> tuple[dict[str, Any], list[str]]:
        defaults = context.provider_ref.get("argument_defaults")
        merged = dict(defaults) if isinstance(defaults, dict) else {}
        merged.update(arguments)
        bindings = context.provider_ref.get("input_bindings")
        normalized_bindings = bindings if isinstance(bindings, dict) else {}
        unique_bindings = {
            tuple(sorted(value.items())): value
            for value in normalized_bindings.values()
            if isinstance(value, dict)
        }
        for item in input_refs:
            input_key = str(item.get("input_key") or "").strip()
            if not input_key and len(unique_bindings) == 1:
                binding = next(iter(unique_bindings.values()))
            else:
                binding = normalized_bindings.get(input_key)
            if not isinstance(binding, dict):
                raise ValueError(f"unknown sandbox input_key: {input_key or '<missing>'}")
            source_type = str(item.get("source_type") or "").strip()
            target_key = str(binding.get(source_type) or "").strip()
            if not target_key:
                raise ValueError(f"sandbox input source is not supported: {source_type}")
            try:
                source_id = int(str(item.get("source_id") or "").strip())
            except (TypeError, ValueError) as exc:
                raise ValueError("sandbox file source_id must be an integer") from exc
            existing = merged.get(target_key)
            if existing not in (None, "", source_id) and int(existing) != source_id:
                raise ValueError(f"conflicting sandbox input value: {target_key}")
            merged[target_key] = source_id
        from src.tools.sandbox_tools import _evaluate_sandbox_field_contracts

        missing, errors = _evaluate_sandbox_field_contracts(
            merged,
            list(context.provider_ref.get("field_contracts") or []),
        )
        if errors:
            raise ValueError("; ".join(errors))
        return merged, missing

    async def _invoke_uncached(
        self,
        *,
        arguments: dict[str, Any],
        input_refs: list[dict[str, Any]],
        context: ProviderContext,
    ) -> ProviderInvocationResult:
        binding = self._bindings.get(context.capability_id)
        if binding is None:
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Sandbox capability is unavailable",
                error=CapabilityError(
                    kind=CapabilityErrorKind.PROVIDER_UNAVAILABLE,
                    message="sandbox capability binding is unavailable",
                ),
            )
        resume_job_id = str(context.metadata.get("resume_job_id") or "").strip()
        if resume_job_id:
            return await self._get_result(
                binding=binding,
                job_id=resume_job_id,
                context=context,
                public_tool_name=str(
                    context.provider_ref.get("tool_name")
                    or "sandbox_get_result"
                ).strip(),
            )
        try:
            merged_arguments, missing = self._merge_input_refs(arguments, input_refs, context)
        except ValueError as exc:
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Sandbox input validation failed",
                error=CapabilityError(
                    kind=CapabilityErrorKind.VALIDATION_ERROR,
                    message=str(exc),
                ),
            )
        if missing:
            return ProviderInvocationResult(
                status=CapabilityStatus.NEEDS_INPUT,
                summary="Sandbox capability requires additional input",
                error=CapabilityError(
                    kind=CapabilityErrorKind.NEEDS_INPUT,
                    message="missing required sandbox inputs",
                    missing_fields=missing,
                ),
            )
        try:
            submit_result = await self._invoke_tool(
                binding.submit_tool,
                merged_arguments,
                context.timeout_seconds,
            )
        except TimeoutError:
            raise
        except Exception as exc:  # noqa: BLE001
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Sandbox submission failed",
                error=CapabilityError(
                    kind=CapabilityErrorKind.EXECUTION_ERROR,
                    message=f"sandbox submission failed: {exc.__class__.__name__}",
                ),
            )
        public_tool_name = str(
            context.metadata.get("public_tool_name")
            or context.provider_ref.get("tool_name")
            or "sandbox_submit"
        )
        submit_mapped = self._map_result(
            public_tool_name,
            submit_result,
            context.capability_version,
            context,
        )
        if submit_mapped.status not in {CapabilityStatus.SUCCEEDED, CapabilityStatus.PENDING}:
            return submit_mapped
        submit_payload = submit_result if isinstance(submit_result, dict) else {}
        job_id = self._job_id(submit_payload)
        execution_mode = str(context.provider_ref.get("execution_mode") or "nonblocking_long")
        if job_id and str(context.metadata.get("task_node_id") or "").strip():
            # A submit response, including an immediately terminal scheduler
            # status, proves only that the external job exists.  Persist one
            # pending receipt first so the exact job/node/begin binding becomes
            # authoritative; terminal success is promoted only after the
            # original provider path validates outputs and registers files.
            return submit_mapped.model_copy(
                update={
                    "status": CapabilityStatus.PENDING,
                    "summary": "Sandbox job accepted; awaiting authoritative result promotion",
                    "data": {
                        **submit_mapped.data,
                        "job_id": job_id,
                        "status": "pending",
                    },
                    "complete": False,
                    "has_more": True,
                    "cursor": f"sandbox-job:{job_id}",
                    "error": None,
                }
            )
        if execution_mode == "nonblocking_long" or not job_id or binding.result_tool is None:
            return submit_mapped
        return await self._get_result(
            binding=binding,
            job_id=job_id,
            context=context,
            public_tool_name=public_tool_name,
        )

    async def invoke(
        self,
        *,
        arguments: dict[str, Any],
        input_refs: list[dict[str, Any]],
        context: ProviderContext,
    ) -> ProviderInvocationResult:
        if str(context.metadata.get("resume_job_id") or "").strip():
            return await self._invoke_uncached(
                arguments=arguments,
                input_refs=input_refs,
                context=context,
            )
        idempotency_key = str(context.metadata.get("idempotency_key") or "").strip()
        if not idempotency_key:
            return await self._invoke_uncached(
                arguments=arguments,
                input_refs=input_refs,
                context=context,
            )
        owner = False
        with self._guard:
            cached = self._completed.get(idempotency_key)
            if cached is not None:
                return cached.model_copy(update={"idempotency_reused": True}, deep=True)
            future = self._inflight.get(idempotency_key)
            if future is None:
                future = Future()
                self._inflight[idempotency_key] = future
                owner = True
        if not owner:
            result = await asyncio.wrap_future(future)
            return result.model_copy(update={"idempotency_reused": True}, deep=True)
        try:
            result = await self._invoke_uncached(
                arguments=arguments,
                input_refs=input_refs,
                context=context,
            )
            if result.status in {CapabilityStatus.SUCCEEDED, CapabilityStatus.PENDING}:
                with self._guard:
                    if len(self._completed) >= 1000:
                        self._completed.pop(next(iter(self._completed)))
                    self._completed[idempotency_key] = result.model_copy(deep=True)
            future.set_result(result.model_copy(deep=True))
            return result
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            with self._guard:
                self._inflight.pop(idempotency_key, None)


def register_sandbox_manifests(
    registry: CapabilityRegistry,
    provider: SandboxProvider,
    apps: Iterable[dict[str, Any]],
    tools: Iterable[Any],
    *,
    strict_app_ids: Iterable[str] | None = None,
) -> list[CapabilitySpec]:
    """Register active manifest capabilities against existing sandbox tools."""

    tools_by_name = {
        str(getattr(tool, "name", "") or "").strip(): tool
        for tool in tools
        if str(getattr(tool, "name", "") or "").strip()
    }
    result_tool = tools_by_name.get("sandbox_get_result")
    strict_ids = {str(item or "").strip() for item in strict_app_ids or []}
    registered: list[CapabilitySpec] = []
    for app in apps:
        if not isinstance(app, dict):
            continue
        if str(app.get("status") or "active").strip().lower() != "active":
            continue
        app_id = str(app.get("app_id") or "").strip()
        if not app_id:
            continue
        submit_tool_name = f"sandbox_submit_{app_id}"
        submit_tool = tools_by_name.get(submit_tool_name)
        try:
            if submit_tool is None:
                raise ValueError(
                    f"sandbox submit tool is unavailable: {submit_tool_name}"
                )
            spec = sandbox_capability_spec_from_manifest(app)
            registry.register_spec(spec, aliases=[submit_tool_name])
            provider.register(spec, submit_tool=submit_tool, result_tool=result_tool)
            registered.append(spec)
        except Exception as exc:
            diagnostic = {
                "app_id": app_id,
                "tool_id": submit_tool_name,
                "error_type": exc.__class__.__name__,
                "message": str(exc),
            }
            provider.registration_diagnostics.append(diagnostic)
            logger.warning(
                "sandbox_capability_registration_skipped app_id=%s tool_id=%s error_type=%s",
                app_id,
                submit_tool_name,
                exc.__class__.__name__,
            )
            if app_id in strict_ids:
                raise
    return registered
