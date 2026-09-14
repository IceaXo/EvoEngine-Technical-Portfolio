"""Real-time capability ledger (P0-A0/A4).

The ledger reconstructs the runtime capability surface -- catalog tools, typed
Skill contracts, guide-only Skills and sandbox exposure -- and emits one row
per declared or model-visible capability.  It is the reproducibility artifact
for the P0-A invariant:

    model-visible capability_id
      = discovery ID = load_tools contract ID
      = Executor ID = Provider execution ID
      = CapabilityResult / trace ID

Isolation rules (A4):
- The ledger NEVER touches the shared ``get_skill_registry()`` singleton, never
  resets live load events and never calls private enablement mutations on a
  live registry.  Independent generation constructs its own ``SkillRegistry``
  instance (its constructor performs discovery + enablement inside the new
  object only); callers may instead inject a frozen registry/build snapshot.
- Generation is read-only and side-effect free; callers choose the output path
  (``/tmp``), never the repository.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.capabilities.models import CapabilitySpec, ProviderType
from src.capabilities.skill_contracts import SkillTypedExposure
from src.capabilities.source_projection import declared_source_projection

# Exposure classes follow EVOENGINE_P0_..._20260805.md section 5.2.
EXPOSURE_CONTROL = "control"
EXPOSURE_PUBLIC = "public"
EXPOSURE_EXACT_ONLY = "exact_only"
EXPOSURE_INTERNAL = "internal"
EXPOSURE_COMPAT = "compat"
EXPOSURE_UNAVAILABLE = "unavailable"

# Ledger row kinds.
KIND_NATIVE_CATALOG = "native_catalog"
KIND_SKILL_TYPED = "skill_typed"
KIND_SKILL_LEGACY = "skill_legacy"
KIND_SKILL_GUIDE = "skill_guide"
KIND_SANDBOX_TYPED = "sandbox_typed"
KIND_SANDBOX_GENERIC = "sandbox_generic"
KIND_SUBAGENT = "subagent"

# Catalog tool packs the retrieval subagent may expose (mirrors
# ``retrieval_subagent._build_retrieval_tools`` knowledge asset set plus web
# search; delegated typed Skill routes are resolved separately).
RETRIEVAL_SUBAGENT_CATALOG_TOOLS = frozenset(
    {
        "web_search",
        "search_milvus_knowledge",
        "read_asset_summary",
        "read_asset_page_parsed",
        "read_asset_raw",
    }
)

PROVIDER_EXECUTOR_LABELS: dict[str, str] = {
    ProviderType.NATIVE.value: "CapabilityExecutor -> NativeToolProvider",
    ProviderType.SKILL_SCRIPT.value: "CapabilityExecutor -> SkillScriptProvider",
    ProviderType.SANDBOX.value: "CapabilityExecutor -> SandboxProvider",
    ProviderType.SUBAGENT.value: "CapabilityExecutor -> SubagentProvider",
    ProviderType.MCP.value: "CapabilityExecutor -> McpProvider",
    ProviderType.INTERNAL_SERVER.value: "CapabilityExecutor -> InternalServerProvider",
}


def _stable_json_hash(value: Any) -> str:
    """Canonical JSON digest for schema/registry fingerprints."""

    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class CapabilityLedgerEntry:
    """One capability row in the real-time ledger."""

    capability_id: str
    version: str
    kind: str
    provider_type: str
    provider_ref: dict[str, Any]
    exposure_class: str
    model_visible: bool
    discovery_path: str
    discovery_id: str
    load_tool_ids: list[str]
    executor_capability_id: str
    load_schema_hash: str
    output_schema_hash: str
    acceptance_schema_hash: str
    executor: str
    availability: str
    availability_reason: str = ""
    result_contract: dict[str, Any] = field(default_factory=dict)
    source_contract: str = ""
    callers: tuple[str, ...] = ()
    tests_evidence: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "version": self.version,
            "kind": self.kind,
            "provider_type": self.provider_type,
            "provider_ref": self.provider_ref,
            "exposure_class": self.exposure_class,
            "model_visible": self.model_visible,
            "discovery_path": self.discovery_path,
            "discovery_id": self.discovery_id,
            "load_tool_ids": list(self.load_tool_ids),
            "executor_capability_id": self.executor_capability_id,
            "load_schema_hash": self.load_schema_hash,
            "output_schema_hash": self.output_schema_hash,
            "acceptance_schema_hash": self.acceptance_schema_hash,
            "executor": self.executor,
            "availability": self.availability,
            "availability_reason": self.availability_reason,
            "result_contract": self.result_contract,
            "source_contract": self.source_contract,
            "callers": list(self.callers),
            "tests_evidence": self.tests_evidence,
        }


@dataclass(frozen=True)
class CapabilityLedger:
    """One reproducible ledger snapshot plus summary invariants."""

    entries: tuple[CapabilityLedgerEntry, ...]
    registry_version: str = ""
    catalog_version: str = ""
    schema_hash: str = ""

    def summary(self) -> dict[str, Any]:
        rows = [entry.to_row() for entry in self.entries]
        model_visible = [
            row for row in rows if row["model_visible"] and row["availability"] == "available"
        ]
        duplicates: list[str] = []
        identities: dict[str, list[str]] = {}
        for row in rows:
            key = f"{row['capability_id']}@{row['version']}"
            identities.setdefault(key, []).append(row["kind"])
        for key, kinds in sorted(identities.items()):
            if len(kinds) > 1:
                duplicates.append(key)
        unavailable_but_visible = [
            row["capability_id"]
            for row in rows
            if row["model_visible"] and row["availability"] != "available"
        ]
        return {
            "total_rows": len(rows),
            "model_visible_available": len(model_visible),
            "duplicate_identities": duplicates,
            "unavailable_but_model_visible": unavailable_but_visible,
            "identity_drift": self.identity_drift(),
            "by_kind": _count_by(rows, "kind"),
            "by_exposure_class": _count_by(rows, "exposure_class"),
        }

    def identity_drift(self) -> list[str]:
        """Rows where discovery/load/executor/trace IDs diverge.

        The business capability_id and its typed execution tool_id are two
        different identities by design; drift means the load_tool_ids do not
        contain the capability's declared execution tool, or the discovery
        execution entry differs from the loaded tool.  Guide-only entries are
        not executable capabilities and are excluded.
        """

        drift: list[str] = []
        for entry in self.entries:
            if entry.kind in {KIND_SKILL_GUIDE, KIND_SKILL_LEGACY}:
                continue
            if entry.discovery_id != entry.capability_id:
                drift.append(
                    f"{entry.capability_id}: discovery_id {entry.discovery_id} "
                    f"!= capability_id {entry.capability_id}"
                )
            declared_tool_id = str(
                entry.provider_ref.get("tool_name")
                or entry.provider_ref.get("catalog_tool_id")
                or entry.provider_ref.get("typed_tool_id")
                or ""
            ).strip()
            if declared_tool_id and entry.load_tool_ids and declared_tool_id not in {
                str(item or "").strip() for item in entry.load_tool_ids
            }:
                drift.append(
                    f"{entry.capability_id}: declared tool {declared_tool_id} "
                    f"not in load_tool_ids {entry.load_tool_ids}"
                )
        return drift

    def to_json(self) -> str:
        return json.dumps(
            {
                **self.body(),
                "schema_hash": self.schema_hash,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    def body(self) -> dict[str, Any]:
        """Ledger content excluding the self-referential hash field."""

        return {
            "schema_version": "evoengine.capability-ledger/v1",
            "registry_version": self.registry_version,
            "catalog_version": self.catalog_version,
            "entries": [entry.to_row() for entry in self.entries],
            "summary": self.summary(),
        }


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _compact_provider_ref(provider_ref: Mapping[str, Any] | None) -> dict[str, Any]:
    source = provider_ref if isinstance(provider_ref, Mapping) else {}
    allowed = {
        "tool_name",
        "catalog_tool_id",
        "skill_id",
        "script_name",
        "app_id",
        "typed_tool_id",
        "registry_version",
        "manifest_version",
        "manifest_hash",
    }
    return {
        str(key): value
        for key, value in source.items()
        if key in allowed and value not in (None, "", [])
    }


def _result_contract_summary(spec: CapabilitySpec) -> dict[str, Any]:
    """ResourceRef/artifact facts.  Presence of an evidence_policy is NOT
    resource_ref capability: only an explicit machine projection mark counts.
    """

    evidence_policy = spec.evidence_policy if isinstance(spec.evidence_policy, dict) else {}
    return {
        "output_schema_present": bool(spec.output_schema),
        "acceptance_schema_present": bool(spec.acceptance_schema),
        "artifact_specs": len(spec.produces),
        "resource_ref_declared": bool(evidence_policy.get("resource_ref_capable")),
        "evidence_policy_declared": bool(evidence_policy),
        "side_effects": list(spec.side_effects),
    }


def _source_contract_label(spec: CapabilitySpec) -> str:
    declaration = declared_source_projection(spec.evidence_policy)
    if declaration is None:
        return ""
    return str(declaration.result_type or "declared")


def _discovery_path_label(
    *,
    visibility: str,
    match_mode: str,
    exposure_class: str,
) -> str:
    if exposure_class in {EXPOSURE_COMPAT, EXPOSURE_UNAVAILABLE}:
        return "hidden"
    if visibility != "public":
        return str(visibility)
    return str(match_mode)


def build_capability_ledger(
    *,
    skills_root: str | Path | None = None,
    settings: Any | None = None,
    sandbox_exposure_context: Any | None = None,
    skill_registry: Any | None = None,
    skill_typed_schema_build: Any | None = None,
    project_id: str | None = None,
    conversation_id: str | None = None,
    user_id: int | None = None,
    request_id: str | None = None,
) -> CapabilityLedger:
    """Generate one real-time ledger.

    Isolation contract: the shared skill registry singleton is never touched.
    ``skill_registry``/``skill_typed_schema_build`` may be injected by callers;
    when omitted the function constructs its OWN ``SkillRegistry`` instance
    (constructor-side discovery/enablement only, inside the new object) and
    compiles an isolated typed build from the skills directory.
    """

    from src.agents import lead_agent
    from src.capabilities.sandbox_contracts import (
        SandboxEffectiveExposure,
        load_sandbox_capability_registry,
    )
    from src.capabilities.skill_contracts import compile_skill_typed_schema_build
    from src.capabilities.skill_runtime_availability import (
        build_skill_runtime_availability_snapshot,
    )
    from src.config.settings import get_settings
    from src.skills.registry import SkillRegistry
    from src.tools.sandbox_tools import (
        compile_sandbox_typed_schema_build,
        fetch_sandbox_exposure_context,
    )

    resolved_settings = settings or get_settings()
    resolved_skills_root = (
        str(skills_root)
        if skills_root is not None
        else str(Path(__file__).resolve().parents[2] / "skills")
    )
    resolved_registry = skill_registry or SkillRegistry(resolved_skills_root)
    resolved_sandbox_context = (
        sandbox_exposure_context
        if sandbox_exposure_context is not None
        else fetch_sandbox_exposure_context()
    )
    sandbox_typed_schema_build = compile_sandbox_typed_schema_build(
        resolved_sandbox_context
    )
    resolved_skill_build = skill_typed_schema_build or compile_skill_typed_schema_build(
        resolved_skills_root,
        reserved_tool_ids={
            *lead_agent.TOOL_CATALOG_BY_ID,
            *resolved_sandbox_context.snapshot.typed_tool_ids,
        },
        allowed_skill_ids=lead_agent._public_enabled_skill_ids(resolved_registry),
    )
    runtime_availability = build_skill_runtime_availability_snapshot(
        resolved_skill_build,
    )
    available_tool_ids = lead_agent._available_tool_ids_for_build(
        resolved_settings,
        project_id=project_id,
        conversation_id=conversation_id,
        user_id=user_id,
        sandbox_exposure_context=resolved_sandbox_context,
        sandbox_typed_schema_build=sandbox_typed_schema_build,
        skill_typed_schema_build=resolved_skill_build,
        unavailable_skill_tool_ids=runtime_availability.unavailable_tool_ids,
    )
    registry_document = load_sandbox_capability_registry()

    entries: list[CapabilityLedgerEntry] = []

    # ── 1. Native catalog tools ────────────────────────────────────────────
    for raw_entry in lead_agent._DECLARED_TOOL_CATALOG_ENTRIES:
        tool_id = str(raw_entry.get("tool_id") or "").strip()
        if not tool_id:
            continue
        spec = lead_agent._TOOL_CATALOG_REGISTRY.resolve(tool_id)
        entries.append(
            _catalog_tool_entry(
                spec,
                raw_entry=raw_entry,
                available_tool_ids=available_tool_ids,
            )
        )

    # ── 2+3. Typed + legacy Skill capabilities ─────────────────────────────
    from src.subagents.retrieval_subagent import (
        _DELEGATED_RETRIEVAL_SKILL_ROUTES,
    )

    delegated_typed_ids = {
        route[1] for route in _DELEGATED_RETRIEVAL_SKILL_ROUTES.values()
    }
    for item in resolved_skill_build.items:
        spec = item.capability_spec()
        tool_id = str(
            spec.provider_ref.get("tool_name")
            or spec.provider_ref.get("catalog_tool_id")
            or ""
        ).strip()
        is_typed = item.exposure == SkillTypedExposure.TYPED
        entry = _skill_capability_entry(
            spec,
            tool_id=tool_id,
            exposure=item.exposure,
            available_tool_ids=available_tool_ids,
            model_visible=is_typed,
            delegated_to_subagent=is_typed and (tool_id in delegated_typed_ids),
        )
        if entry is not None:
            entries.append(entry)

    # ── 4. Guide-only Skills (isolated registry view, never the singleton) ──
    typed_skill_ids = {
        str(spec.provider_ref.get("skill_id") or "").strip()
        for spec in resolved_skill_build.specs
    }
    for skill in resolved_registry.list_skills():
        skill_id = str(getattr(skill, "skill_id", "") or "").strip()
        if not skill_id or skill_id in typed_skill_ids:
            continue
        entries.append(_guide_skill_entry(skill, available_tool_ids=available_tool_ids))

    # ── 5. Sandbox apps (declared + effective exposure) ────────────────────
    registry_by_app_id = {entry.app_id: entry for entry in registry_document.apps}
    for app_id, registry_entry in sorted(registry_by_app_id.items()):
        entries.append(
            _sandbox_app_entry(
                app_id=app_id,
                registry_entry=registry_entry,
                sandbox_context=resolved_sandbox_context,
                available_tool_ids=available_tool_ids,
                sandbox_typed_schema_build=sandbox_typed_schema_build,
            )
        )

    ledger = CapabilityLedger(
        entries=tuple(sorted(entries, key=lambda item: (item.kind, item.capability_id))),
        registry_version=str(
            getattr(resolved_skill_build, "registry_version", "") or ""
        ),
        catalog_version=str(resolved_sandbox_context.catalog_version or ""),
    )
    object.__setattr__(
        ledger,
        "schema_hash",
        _stable_json_hash(
            {
                "schema_version": "evoengine.capability-ledger/v1",
                "registry_version": ledger.registry_version,
                "catalog_version": ledger.catalog_version,
                "entries": [entry.to_row() for entry in ledger.entries],
                "summary": ledger.summary(),
            }
        ),
    )
    return ledger


def _catalog_tool_entry(
    spec: CapabilitySpec,
    *,
    raw_entry: Mapping[str, Any],
    available_tool_ids: frozenset[str] | set[str],
) -> CapabilityLedgerEntry:
    from src.agents import lead_agent

    tool_id = str(spec.provider_ref.get("catalog_tool_id") or "").strip()
    discovery = spec.discovery
    visibility = str(discovery.visibility.value)
    match_mode = str(discovery.match_mode.value)
    permission = str(spec.permission.value)
    is_hidden = tool_id in lead_agent.MODEL_HIDDEN_CATALOG_TOOL_IDS
    is_retired = tool_id in lead_agent.RETIRED_SANDBOX_COMPAT_TOOL_IDS
    is_legacy = tool_id in lead_agent.HIDDEN_LEGACY_RESEARCH_TOOL_IDS
    if is_hidden or is_retired or is_legacy or visibility == "hidden":
        exposure_class = EXPOSURE_COMPAT
        model_visible = False
        availability = "available"
        availability_reason = "model_hidden_compat_entry"
    elif permission == "control":
        exposure_class = EXPOSURE_CONTROL
        model_visible = tool_id in available_tool_ids
        availability = "available"
        availability_reason = ""
    elif visibility == "internal":
        exposure_class = EXPOSURE_INTERNAL
        model_visible = tool_id in available_tool_ids
        availability = "available"
        availability_reason = ""
    elif match_mode == "natural":
        exposure_class = EXPOSURE_PUBLIC
        model_visible = tool_id in available_tool_ids
        availability = "available"
        availability_reason = ""
    else:
        exposure_class = EXPOSURE_EXACT_ONLY
        model_visible = tool_id in available_tool_ids
        availability = "available"
        availability_reason = ""
    if tool_id not in available_tool_ids and exposure_class not in {
        EXPOSURE_COMPAT,
        EXPOSURE_UNAVAILABLE,
    }:
        availability = "degraded"
        availability_reason = "not_in_available_tool_ids"
    kind = (
        KIND_SUBAGENT
        if spec.provider_type == ProviderType.SUBAGENT
        else KIND_NATIVE_CATALOG
    )
    callers = _native_callers(tool_id, model_visible=model_visible)
    return CapabilityLedgerEntry(
        capability_id=tool_id,
        version=spec.version,
        kind=kind,
        provider_type=spec.provider_type.value,
        provider_ref=_compact_provider_ref(spec.provider_ref),
        exposure_class=exposure_class,
        model_visible=model_visible,
        discovery_path=_discovery_path_label(
            visibility=visibility,
            match_mode=match_mode,
            exposure_class=exposure_class,
        ),
        discovery_id=tool_id,
        load_tool_ids=list(discovery.load_tool_ids or [tool_id]),
        executor_capability_id=tool_id,
        load_schema_hash=_stable_json_hash(spec.input_schema),
        output_schema_hash=_stable_json_hash(spec.output_schema),
        acceptance_schema_hash=_stable_json_hash(spec.acceptance_schema),
        executor=PROVIDER_EXECUTOR_LABELS.get(
            spec.provider_type.value, "CapabilityExecutor -> Provider"
        ),
        availability=availability,
        availability_reason=availability_reason,
        result_contract=_result_contract_summary(spec),
        source_contract=_source_contract_label(spec),
        callers=callers,
    )


def _native_callers(tool_id: str, *, model_visible: bool) -> tuple[str, ...]:
    callers: list[str] = []
    if model_visible:
        callers.append("lead_agent")
    if tool_id in RETRIEVAL_SUBAGENT_CATALOG_TOOLS:
        callers.append("retrieval_subagent")
    return tuple(callers)


def _skill_capability_entry(
    spec: CapabilitySpec,
    *,
    tool_id: str,
    exposure: SkillTypedExposure,
    available_tool_ids: frozenset[str] | set[str],
    model_visible: bool,
    delegated_to_subagent: bool,
) -> CapabilityLedgerEntry | None:
    if exposure == SkillTypedExposure.LEGACY:
        return CapabilityLedgerEntry(
            capability_id=spec.capability_id,
            version=spec.version,
            kind=KIND_SKILL_LEGACY,
            provider_type=spec.provider_type.value,
            provider_ref=_compact_provider_ref(spec.provider_ref),
            exposure_class=EXPOSURE_COMPAT,
            model_visible=False,
            discovery_path="hidden",
            discovery_id=spec.capability_id,
            load_tool_ids=[],
            executor_capability_id=spec.capability_id,
            load_schema_hash=_stable_json_hash(spec.input_schema),
            output_schema_hash=_stable_json_hash(spec.output_schema),
            acceptance_schema_hash=_stable_json_hash(spec.acceptance_schema),
            executor=PROVIDER_EXECUTOR_LABELS.get(
                spec.provider_type.value, "CapabilityExecutor -> Provider"
            ),
            availability=spec.availability,
            availability_reason="legacy_contract_compat_only",
            result_contract=_result_contract_summary(spec),
            source_contract=_source_contract_label(spec),
            callers=(),
        )
    discovery = spec.discovery
    visibility = str(discovery.visibility.value)
    match_mode = str(discovery.match_mode.value)
    if visibility == "internal":
        exposure_class = EXPOSURE_INTERNAL
    elif match_mode == "natural":
        exposure_class = EXPOSURE_PUBLIC
    else:
        exposure_class = EXPOSURE_EXACT_ONLY
    availability = spec.availability
    availability_reason = ""
    if tool_id not in available_tool_ids and spec.availability == "available":
        availability = "degraded"
        availability_reason = "not_in_available_tool_ids"
    elif availability == "unavailable":
        availability_reason = "runtime_dependency_missing"
    callers = ["lead_agent"] if model_visible and tool_id in available_tool_ids else []
    if delegated_to_subagent and tool_id in available_tool_ids:
        callers.append("retrieval_subagent")
    script_name = str(spec.provider_ref.get("script_name") or "").strip()
    return CapabilityLedgerEntry(
        capability_id=spec.capability_id,
        version=spec.version,
        kind=KIND_SKILL_TYPED,
        provider_type=spec.provider_type.value,
        provider_ref=_compact_provider_ref(spec.provider_ref),
        exposure_class=exposure_class,
        model_visible=model_visible and tool_id in available_tool_ids,
        discovery_path=_discovery_path_label(
            visibility=visibility,
            match_mode=match_mode,
            exposure_class=exposure_class,
        ),
        discovery_id=spec.capability_id,
        load_tool_ids=list(discovery.load_tool_ids or [tool_id]),
        executor_capability_id=spec.capability_id,
        load_schema_hash=_stable_json_hash(spec.input_schema),
        output_schema_hash=_stable_json_hash(spec.output_schema),
        acceptance_schema_hash=_stable_json_hash(spec.acceptance_schema),
        executor=(
            PROVIDER_EXECUTOR_LABELS.get(spec.provider_type.value, "CapabilityExecutor -> Provider")
            + (f" / {script_name}" if script_name else "")
        ),
        availability=availability,
        availability_reason=availability_reason,
        result_contract=_result_contract_summary(spec),
        source_contract=_source_contract_label(spec),
        callers=tuple(callers),
    )


def _guide_skill_entry(skill: Any, *, available_tool_ids: frozenset[str] | set[str]) -> CapabilityLedgerEntry:
    from src.capabilities.discovery import guide_discovery_contract

    contract = guide_discovery_contract(skill)
    match_mode = contract.match_mode
    skill_id = str(getattr(skill, "skill_id", "") or "").strip()
    visibility = str(getattr(skill, "visibility", "public") or "public").strip().lower()
    enabled = bool(getattr(skill, "enabled", True))
    if not enabled:
        exposure_class = EXPOSURE_UNAVAILABLE
        availability = "unavailable"
        availability_reason = "skill_disabled"
    elif visibility != "public":
        exposure_class = EXPOSURE_INTERNAL
        availability = "available"
        availability_reason = ""
    elif match_mode == "natural":
        exposure_class = EXPOSURE_PUBLIC
        availability = "available"
        availability_reason = ""
    else:
        exposure_class = EXPOSURE_EXACT_ONLY
        availability = "available"
        availability_reason = ""
    # A guide is never an executable model capability; model_visible tracks
    # whether discovery may surface it at all (public only).
    model_visible = enabled and visibility == "public"
    discovery_path = (
        "hidden"
        if exposure_class in {EXPOSURE_COMPAT, EXPOSURE_UNAVAILABLE}
        else "internal"
        if visibility != "public"
        else match_mode
    )
    return CapabilityLedgerEntry(
        capability_id=skill_id,
        version=str(getattr(skill, "version", "") or "1.0.0").strip(),
        kind=KIND_SKILL_GUIDE,
        provider_type=ProviderType.SKILL_SCRIPT.value,
        provider_ref={"skill_id": skill_id},
        exposure_class=exposure_class,
        model_visible=model_visible,
        discovery_path=discovery_path,
        discovery_id=skill_id,
        load_tool_ids=["activate_skill"] if model_visible else [],
        executor_capability_id=skill_id,
        load_schema_hash="",
        output_schema_hash="",
        acceptance_schema_hash="",
        executor="guide_only(activate_skill)",
        availability=availability,
        availability_reason=availability_reason,
        result_contract={"guide_only": True},
        source_contract="",
        callers=("lead_agent",) if model_visible else (),
    )


def _sandbox_app_entry(
    *,
    app_id: str,
    registry_entry: Any,
    sandbox_context: Any,
    available_tool_ids: frozenset[str] | set[str],
    sandbox_typed_schema_build: Any,
) -> CapabilityLedgerEntry:
    from src.capabilities.sandbox_contracts import SandboxEffectiveExposure

    decision = next(
        (
            item
            for item in sandbox_context.snapshot.apps
            if str(item.app_id) == app_id
        ),
        None,
    )
    requested_exposure = str(registry_entry.exposure.value) if registry_entry is not None else "generic"
    review_status = str(registry_entry.review_status.value) if registry_entry is not None else ""
    if decision is None:
        exposure_class = EXPOSURE_UNAVAILABLE
        availability = "unavailable"
        availability_reason = "server_manifest_missing"
        kind = KIND_SANDBOX_GENERIC
        typed_tool_id = ""
        model_visible = False
        discovery_path = "hidden"
    elif decision.exposure == SandboxEffectiveExposure.TYPED:
        exposure_class = EXPOSURE_PUBLIC
        availability = "available"
        availability_reason = "typed_contract_accepted"
        kind = KIND_SANDBOX_TYPED
        typed_tool_id = decision.typed_tool_id
        model_visible = typed_tool_id in available_tool_ids
        discovery_path = "natural"
    elif decision.exposure == SandboxEffectiveExposure.GENERIC:
        exposure_class = EXPOSURE_EXACT_ONLY
        availability = "available"
        availability_reason = f"generic_route({decision.reason})"
        kind = KIND_SANDBOX_GENERIC
        typed_tool_id = ""
        model_visible = "sandbox_submit" in available_tool_ids
        discovery_path = "exact_only"
    else:
        exposure_class = EXPOSURE_UNAVAILABLE
        availability = "unavailable"
        availability_reason = str(decision.reason or "blocked")
        kind = KIND_SANDBOX_GENERIC
        typed_tool_id = ""
        model_visible = False
        discovery_path = "hidden"
    capability_id = (
        f"sandbox.{app_id}.run"
        if typed_tool_id
        else f"sandbox.{app_id}.run(generic)"
    )
    callers = ("lead_agent",) if model_visible else ()
    return CapabilityLedgerEntry(
        capability_id=capability_id,
        version=(
            str(decision.manifest_version or "")
            if decision is not None
            else ""
        ),
        kind=kind,
        provider_type=ProviderType.SANDBOX.value,
        provider_ref={
            "app_id": app_id,
            "requested_exposure": requested_exposure,
            "review_status": review_status,
            "typed_tool_id": typed_tool_id,
        },
        exposure_class=exposure_class,
        model_visible=model_visible,
        discovery_path=discovery_path,
        discovery_id=capability_id,
        load_tool_ids=[typed_tool_id] if typed_tool_id and model_visible else [],
        executor_capability_id=capability_id,
        load_schema_hash="",
        output_schema_hash="",
        acceptance_schema_hash="",
        executor=(
            "CapabilityExecutor -> SandboxProvider"
            if kind == KIND_SANDBOX_TYPED
            else "sandbox_submit generic route"
        ),
        availability=availability,
        availability_reason=availability_reason,
        result_contract={"sandbox_manifest": True},
        source_contract="",
        callers=callers,
    )


def ledger_to_json(entries: Iterable[CapabilityLedgerEntry]) -> str:
    """Serialize plain rows for reporting; does not build a new snapshot."""

    return json.dumps(
        [entry.to_row() for entry in entries],
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def write_ledger(ledger: CapabilityLedger, path: str) -> None:
    """Write one ledger snapshot to ``path`` (caller-chosen, never repo data)."""

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(ledger.to_json())
