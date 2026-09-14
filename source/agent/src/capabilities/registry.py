"""In-process registry for capability specs and provider implementations."""

from __future__ import annotations

from threading import RLock
from typing import Iterable

from src.capabilities.models import (
    AgentMode,
    CapabilityRegistrySnapshot,
    CapabilitySpec,
    ProviderType,
)
from src.capabilities.providers.base import CapabilityProvider


class CapabilityRegistrationError(ValueError):
    pass


class CapabilityNotFoundError(KeyError):
    pass


class CapabilityProviderNotFoundError(KeyError):
    pass


def _normalize_identifier(value: str) -> str:
    return str(value or "").strip().lower()


class CapabilityRegistry:
    """Thread-safe metadata registry.

    The registry is deliberately independent from LangChain. It can be built and
    tested without compiling an agent graph.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._specs: dict[tuple[str, str], CapabilitySpec] = {}
        self._active_versions: dict[str, str] = {}
        self._aliases: dict[str, str] = {}
        self._providers: dict[ProviderType, CapabilityProvider] = {}

    def register_provider(self, provider: CapabilityProvider, *, replace: bool = False) -> None:
        provider_type = ProviderType(provider.provider_type)
        with self._lock:
            existing = self._providers.get(provider_type)
            if existing is not None and existing is not provider and not replace:
                raise CapabilityRegistrationError(f"provider already registered: {provider_type.value}")
            self._providers[provider_type] = provider

    def get_provider(self, provider_type: ProviderType | str) -> CapabilityProvider:
        normalized = ProviderType(provider_type)
        with self._lock:
            provider = self._providers.get(normalized)
        if provider is None:
            raise CapabilityProviderNotFoundError(normalized.value)
        return provider

    def register_spec(
        self,
        spec: CapabilitySpec,
        *,
        aliases: Iterable[str] = (),
        make_active: bool = True,
    ) -> CapabilitySpec:
        stored = spec.model_copy(deep=True)
        key = (stored.capability_id, stored.version)
        normalized_aliases = [_normalize_identifier(alias) for alias in aliases if _normalize_identifier(alias)]
        with self._lock:
            existing = self._specs.get(key)
            if existing is not None and existing != stored:
                raise CapabilityRegistrationError(
                    f"different spec already registered for {stored.capability_id}@{stored.version}"
                )
            for alias in normalized_aliases:
                existing_target = self._aliases.get(alias)
                if existing_target is not None and existing_target != stored.capability_id:
                    raise CapabilityRegistrationError(
                        f"alias {alias!r} already points to {existing_target}"
                    )
            self._specs[key] = stored
            if make_active or stored.capability_id not in self._active_versions:
                self._active_versions[stored.capability_id] = stored.version
            for alias in normalized_aliases:
                self._aliases[alias] = stored.capability_id
        return stored.model_copy(deep=True)

    def resolve(self, identifier: str, *, version: str | None = None) -> CapabilitySpec:
        normalized = _normalize_identifier(identifier)
        with self._lock:
            capability_id = self._aliases.get(normalized, normalized)
            selected_version = str(version or self._active_versions.get(capability_id) or "").strip()
            spec = self._specs.get((capability_id, selected_version)) if selected_version else None
        if spec is None:
            suffix = f"@{version}" if version else ""
            raise CapabilityNotFoundError(f"{identifier}{suffix}")
        return spec.model_copy(deep=True)

    def freeze(self, allowlist: Iterable[str]) -> CapabilityRegistrySnapshot:
        """Capture an exact, immutable capability view without aliases or active-version lookup."""

        exact_identities: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for raw_identity in allowlist:
            identity = str(raw_identity or "").strip()
            if identity.count("@") != 1:
                raise CapabilityRegistrationError(
                    "snapshot allowlist entries must use exact capability_id@version identities"
                )
            capability_id, version = identity.rsplit("@", 1)
            normalized_id = _normalize_identifier(capability_id)
            normalized_version = str(version or "").strip()
            if capability_id != normalized_id or not normalized_id or not normalized_version:
                raise CapabilityRegistrationError(
                    "snapshot allowlist entries must use exact lowercase capability_id@version identities"
                )
            key = (normalized_id, normalized_version)
            if key in seen:
                raise CapabilityRegistrationError(
                    f"duplicate snapshot allowlist entry: {normalized_id}@{normalized_version}"
                )
            seen.add(key)
            exact_identities.append(key)

        if not exact_identities:
            raise CapabilityRegistrationError("snapshot allowlist cannot be empty")

        with self._lock:
            missing = [
                f"{capability_id}@{version}"
                for capability_id, version in exact_identities
                if (capability_id, version) not in self._specs
            ]
            if missing:
                raise CapabilityNotFoundError(", ".join(missing))
            specs = [
                self._specs[(capability_id, version)].model_copy(deep=True)
                for capability_id, version in exact_identities
            ]
        return CapabilityRegistrySnapshot.from_specs(specs)

    def list_specs(
        self,
        *,
        agent_mode: AgentMode | str | None = None,
        include_disabled: bool = False,
        include_unavailable: bool = False,
        active_only: bool = True,
    ) -> list[CapabilitySpec]:
        with self._lock:
            specs = list(self._specs.values())
            active_versions = dict(self._active_versions)
        filtered: list[CapabilitySpec] = []
        for spec in specs:
            if active_only and active_versions.get(spec.capability_id) != spec.version:
                continue
            if not include_disabled and not spec.enabled:
                continue
            if not include_unavailable and spec.availability == "unavailable":
                continue
            filtered.append(spec.model_copy(deep=True))
        filtered.sort(key=lambda item: (item.capability_id, item.version))
        return filtered

    def catalog_entries(self, *, agent_mode: AgentMode | str | None = None) -> list[dict[str, str]]:
        return [
            {
                "capability_id": spec.capability_id,
                "version": spec.version,
                "provider_type": spec.provider_type.value,
                "permission": spec.permission.value,
                "summary": spec.description or spec.display_name,
            }
            for spec in self.list_specs(agent_mode=agent_mode)
        ]
