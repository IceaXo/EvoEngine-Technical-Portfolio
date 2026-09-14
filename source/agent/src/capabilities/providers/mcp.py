"""Optional, read-only MCP Provider experiment.

No MCP dependency or connection is created unless a client factory is injected
and the provider is explicitly enabled.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import re
import time
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from pydantic import Field

from src.capabilities.models import (
    CapabilityError,
    CapabilityErrorKind,
    CapabilitySpec,
    CapabilityStatus,
    Permission,
    ProviderType,
    StrictModel,
)
from src.capabilities.providers.base import ProviderContext, ProviderInvocationResult
from src.capabilities.registry import CapabilityRegistry
from src.capabilities.result_mapper import map_tool_envelope_to_capability_result
from src.capabilities.validation import check_json_schema


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _normalized_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", ".", str(value or "").strip().lower()).strip(".")


class MCPToolDescriptor(StrictModel):
    server_id: str = Field(min_length=1, max_length=256)
    tool_name: str = Field(min_length=1, max_length=256)
    description: str = Field(default="", max_length=4000)
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})
    output_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})
    read_only: bool = True
    auth_scope: str = Field(default="user", pattern=r"^(user|service)$")


@runtime_checkable
class MCPClientProtocol(Protocol):
    def list_tools(self, *, auth_context: dict[str, Any]) -> Any: ...

    def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        auth_context: dict[str, Any],
    ) -> Any: ...

    def close(self) -> Any: ...


MCPClientFactory = Callable[[str, str], MCPClientProtocol | Awaitable[MCPClientProtocol]]


def capability_spec_from_mcp_descriptor(descriptor: MCPToolDescriptor) -> CapabilitySpec:
    if not descriptor.read_only:
        raise ValueError("E9 MCP experiment only accepts read-only tools")
    for schema_name, schema in (
        ("input_schema", descriptor.input_schema),
        ("output_schema", descriptor.output_schema),
    ):
        report = check_json_schema(schema)
        if not report.valid:
            raise ValueError(f"invalid MCP {schema_name}: {report.issues[0].message}")
    server = _normalized_id(descriptor.server_id)
    tool = _normalized_id(descriptor.tool_name)
    if not server or not tool:
        raise ValueError("MCP server_id and tool_name must contain an identifier")
    return CapabilitySpec(
        capability_id=f"mcp.{server}.{tool}",
        version="1.0.0",
        display_name=descriptor.tool_name,
        description=descriptor.description,
        provider_type=ProviderType.MCP,
        provider_ref={
            "server_id": descriptor.server_id,
            "tool_name": descriptor.tool_name,
            "auth_scope": descriptor.auth_scope,
            "read_only": True,
        },
        input_schema=descriptor.input_schema,
        input_refs_schema={"type": "array", "maxItems": 0},
        output_schema=descriptor.output_schema,
        permission=Permission.READ,
        side_effects=[],
    )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class MCPProvider:
    provider_type = ProviderType.MCP

    def __init__(
        self,
        client_factory: MCPClientFactory | None = None,
        *,
        enabled: bool | None = None,
        schema_cache_ttl_seconds: int = 300,
    ) -> None:
        self._client_factory = client_factory
        self._enabled = _bool_env("EVO_ENABLE_MCP_PROVIDER", False) if enabled is None else bool(enabled)
        self._schema_cache_ttl_seconds = max(1, min(3600, int(schema_cache_ttl_seconds)))
        self._bindings: dict[str, MCPToolDescriptor] = {}
        self._clients: dict[tuple[str, str], MCPClientProtocol] = {}
        self._schema_cache: dict[tuple[str, str], tuple[float, list[MCPToolDescriptor]]] = {}
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def register(self, spec: CapabilitySpec, descriptor: MCPToolDescriptor) -> None:
        if spec.provider_type != ProviderType.MCP:
            raise ValueError("MCPProvider only accepts MCP capability specs")
        if not descriptor.read_only:
            raise ValueError("write-capable MCP tools are disabled in E9")
        existing = self._bindings.get(spec.capability_id)
        if existing is not None and existing != descriptor:
            raise ValueError(f"MCP capability already bound: {spec.capability_id}")
        self._bindings[spec.capability_id] = descriptor

    @staticmethod
    def _auth_identity(
        *,
        server_id: str,
        auth_scope: str,
        user_id: int | None,
        auth_context: dict[str, Any],
    ) -> str:
        scope_identity = "service" if auth_scope == "service" else f"user:{user_id or 'anonymous'}"
        fingerprint = hashlib.sha256(
            json.dumps(auth_context, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:20]
        return f"{server_id}:{scope_identity}:{fingerprint}"

    async def _client(
        self,
        *,
        server_id: str,
        auth_scope: str,
        user_id: int | None,
        auth_context: dict[str, Any],
    ) -> tuple[MCPClientProtocol, str]:
        if not self._enabled or self._client_factory is None:
            raise RuntimeError("MCP provider is disabled")
        auth_identity = self._auth_identity(
            server_id=server_id,
            auth_scope=auth_scope,
            user_id=user_id,
            auth_context=auth_context,
        )
        key = (server_id, auth_identity)
        async with self._lock:
            existing = self._clients.get(key)
            if existing is not None:
                return existing, auth_identity
            client = await _maybe_await(self._client_factory(server_id, auth_identity))
            if not isinstance(client, MCPClientProtocol):
                raise TypeError("MCP client does not implement the required protocol")
            self._clients[key] = client
            return client, auth_identity

    async def discover(
        self,
        *,
        server_id: str,
        user_id: int | None,
        auth_context: dict[str, Any] | None = None,
        auth_scope: str = "user",
    ) -> list[MCPToolDescriptor]:
        normalized_auth = dict(auth_context or {})
        client, auth_identity = await self._client(
            server_id=server_id,
            auth_scope=auth_scope,
            user_id=user_id,
            auth_context=normalized_auth,
        )
        cache_key = (server_id, auth_identity)
        now = time.monotonic()
        cached = self._schema_cache.get(cache_key)
        if cached is not None and now - cached[0] < self._schema_cache_ttl_seconds:
            return [item.model_copy(deep=True) for item in cached[1]]
        raw_tools = await _maybe_await(client.list_tools(auth_context=normalized_auth))
        descriptors: list[MCPToolDescriptor] = []
        for raw in raw_tools if isinstance(raw_tools, list) else []:
            if not isinstance(raw, dict):
                continue
            annotations = raw.get("annotations") if isinstance(raw.get("annotations"), dict) else {}
            read_only = bool(raw.get("read_only", annotations.get("readOnlyHint", True)))
            if not read_only:
                continue
            descriptor = MCPToolDescriptor(
                server_id=server_id,
                tool_name=str(raw.get("name") or ""),
                description=str(raw.get("description") or ""),
                input_schema=raw.get("inputSchema") if isinstance(raw.get("inputSchema"), dict) else {"type": "object"},
                output_schema=raw.get("outputSchema") if isinstance(raw.get("outputSchema"), dict) else {"type": "object"},
                read_only=True,
                auth_scope=auth_scope,
            )
            capability_spec_from_mcp_descriptor(descriptor)
            descriptors.append(descriptor)
        self._schema_cache[cache_key] = (now, descriptors)
        return [item.model_copy(deep=True) for item in descriptors]

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
                summary="MCP input reference validation failed",
                error=CapabilityError(
                    kind=CapabilityErrorKind.VALIDATION_ERROR,
                    message="MCP read tools do not accept file input_refs",
                ),
            )
        descriptor = self._bindings.get(context.capability_id)
        if descriptor is None or not self._enabled or self._client_factory is None:
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="MCP capability is unavailable",
                error=CapabilityError(
                    kind=CapabilityErrorKind.PROVIDER_UNAVAILABLE,
                    message="MCP provider is disabled or capability is not bound",
                ),
            )
        auth_context = context.metadata.get("mcp_auth")
        normalized_auth = dict(auth_context) if isinstance(auth_context, dict) else {}
        try:
            client, _auth_identity = await self._client(
                server_id=descriptor.server_id,
                auth_scope=descriptor.auth_scope,
                user_id=context.user_id,
                auth_context=normalized_auth,
            )
            raw_result = await asyncio.wait_for(
                _maybe_await(
                    client.call_tool(
                        descriptor.tool_name,
                        dict(arguments),
                        auth_context=normalized_auth,
                    )
                ),
                timeout=context.timeout_seconds,
            )
        except TimeoutError:
            raise
        except Exception as exc:  # noqa: BLE001
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="MCP capability execution failed",
                error=CapabilityError(
                    kind=CapabilityErrorKind.PROVIDER_UNAVAILABLE,
                    message=f"MCP server call failed: {exc.__class__.__name__}",
                ),
            )

        from src.services.tool_result_envelope import make_tool_result_envelope
        from src.services.resource_store import resource_scope_from_provider_context
        from src.services.tool_machine_projection import normalize_machine_data

        projected_tool_name = f"mcp_{_normalized_id(descriptor.server_id).replace('.', '_')}_{_normalized_id(descriptor.tool_name).replace('.', '_')}"
        envelope = make_tool_result_envelope(
            projected_tool_name,
            raw_result,
            category="mcp",
            resource_scope=resource_scope_from_provider_context(context),
        )
        if isinstance(raw_result, dict) and isinstance(raw_result.get("evidence_refs"), list):
            envelope["evidence_refs"] = raw_result["evidence_refs"]
        mapped = map_tool_envelope_to_capability_result(
            tool_name=projected_tool_name,
            category="mcp",
            raw_result=raw_result,
            envelope=envelope,
            capability_version=context.capability_version,
            call_id_namespace="mcp-provider",
        )
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

    async def close(self) -> None:
        async with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
            self._schema_cache.clear()
        for client in clients:
            try:
                await _maybe_await(client.close())
            except Exception:
                continue


def register_mcp_descriptors(
    registry: CapabilityRegistry,
    provider: MCPProvider,
    descriptors: list[MCPToolDescriptor],
) -> list[CapabilitySpec]:
    specs: list[CapabilitySpec] = []
    for descriptor in descriptors:
        spec = capability_spec_from_mcp_descriptor(descriptor)
        registry.register_spec(
            spec,
            aliases=[f"mcp_{_normalized_id(descriptor.server_id)}_{_normalized_id(descriptor.tool_name)}"],
        )
        provider.register(spec, descriptor)
        specs.append(spec)
    return specs
