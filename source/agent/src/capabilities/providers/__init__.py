"""Provider protocol for capability execution backends."""

from src.capabilities.providers.base import (
    CapabilityProvider,
    ProviderArtifactSource,
    ProviderContext,
    ProviderInvocationResult,
)
from src.capabilities.providers.skill_script import SkillScriptProvider
from src.capabilities.providers.native_tool import (
    NativeToolProvider,
    capability_spec_from_structured_tool,
    permission_from_tool_catalog,
    register_native_structured_tools,
)
from src.capabilities.providers.sandbox import (
    SandboxProvider,
    register_sandbox_manifests,
    sandbox_capability_spec_from_manifest,
    sandbox_result_capability_spec,
)
from src.capabilities.providers.subagent import (
    SubagentProvider,
    capability_spec_from_subagent_tool,
)
from src.capabilities.providers.internal_server import (
    InternalServerProvider,
    internal_deliverable_capability_specs,
    register_internal_deliverable_capabilities,
)
from src.capabilities.providers.mcp import (
    MCPClientProtocol,
    MCPProvider,
    MCPToolDescriptor,
    capability_spec_from_mcp_descriptor,
    register_mcp_descriptors,
)

__all__ = [
    "CapabilityProvider",
    "ProviderArtifactSource",
    "ProviderContext",
    "ProviderInvocationResult",
    "SkillScriptProvider",
    "NativeToolProvider",
    "capability_spec_from_structured_tool",
    "permission_from_tool_catalog",
    "register_native_structured_tools",
    "SandboxProvider",
    "register_sandbox_manifests",
    "sandbox_capability_spec_from_manifest",
    "sandbox_result_capability_spec",
    "SubagentProvider",
    "capability_spec_from_subagent_tool",
    "InternalServerProvider",
    "internal_deliverable_capability_specs",
    "register_internal_deliverable_capabilities",
    "MCPClientProtocol",
    "MCPProvider",
    "MCPToolDescriptor",
    "capability_spec_from_mcp_descriptor",
    "register_mcp_descriptors",
]
