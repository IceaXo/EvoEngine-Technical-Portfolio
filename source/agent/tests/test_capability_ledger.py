"""P0-A0: real-time capability ledger generation and invariants.

The ledger must be reproducible from the same sources a live agent build
uses, and every model-visible capability row must satisfy:

    capability_id = discovery ID = load_tools ID = Executor ID = Provider ID

Unavailable or degraded capabilities must never be discoverable as
executable candidates, exact-only capabilities must not be recalled by
natural language, and typed/generic sandbox exposure stays exclusive.
"""

from __future__ import annotations

import json

import pytest

from src.capabilities.ledger import (
    CapabilityLedger,
    EXPOSURE_EXACT_ONLY,
    EXPOSURE_INTERNAL,
    EXPOSURE_PUBLIC,
    KIND_NATIVE_CATALOG,
    KIND_SANDBOX_GENERIC,
    KIND_SANDBOX_TYPED,
    KIND_SKILL_GUIDE,
    KIND_SKILL_TYPED,
    KIND_SUBAGENT,
    build_capability_ledger,
    _stable_json_hash,
)
from src.capabilities.sandbox_contracts import (
    SANDBOX_CAPABILITY_REGISTRY_SCHEMA_VERSION,
    SandboxCapabilityRegistryDocument,
    SandboxCapabilityRegistryEntry,
    SandboxRequestedExposure,
    SandboxReviewStatus,
    build_sandbox_exposure_context,
)


def _empty_sandbox_context():
    return build_sandbox_exposure_context(
        [],
        SandboxCapabilityRegistryDocument(
            schema_version=SANDBOX_CAPABILITY_REGISTRY_SCHEMA_VERSION,
            registry_version="ledger-test-empty-v1",
            apps=[],
        ),
    )


@pytest.fixture(scope="module")
def ledger() -> CapabilityLedger:
    return build_capability_ledger(sandbox_exposure_context=_empty_sandbox_context())


def _rows(ledger: CapabilityLedger) -> list[dict]:
    return [entry.to_row() for entry in ledger.entries]


def test_ledger_covers_all_capability_kinds(ledger) -> None:
    rows = _rows(ledger)
    kinds = {row["kind"] for row in rows}
    assert {
        KIND_NATIVE_CATALOG,
        KIND_SKILL_TYPED,
        KIND_SKILL_GUIDE,
        KIND_SANDBOX_GENERIC,
        KIND_SUBAGENT,
    } <= kinds
    assert any(row["kind"] == KIND_SKILL_TYPED and row["model_visible"] for row in rows)
    # Retrieval remains a task-level capability even when additional
    # task-level subagents are registered beside it.
    subagent_rows = [row for row in rows if row["kind"] == KIND_SUBAGENT]
    assert {row["capability_id"] for row in subagent_rows} >= {
        "run_retrieval_subagent"
    }


def test_ledger_identities_are_unique(ledger) -> None:
    identities = [f"{row['capability_id']}@{row['version']}" for row in _rows(ledger)]
    assert len(identities) == len(set(identities))
    assert ledger.summary()["duplicate_identities"] == []


def test_unavailable_and_degraded_capabilities_are_not_model_visible(
    ledger,
) -> None:
    hidden = [
        row["capability_id"]
        for row in _rows(ledger)
        if row["model_visible"] and row["availability"] != "available"
    ]
    assert hidden == []
    # Every model-visible row must have an executor and a discovery path.
    for row in _rows(ledger):
        if not row["model_visible"]:
            continue
        assert row["executor"], row["capability_id"]
        assert row["discovery_path"] in {
            "natural",
            "exact_only",
            "control",
            "public",
        }, row["capability_id"]


def test_ledger_json_is_stable_and_parseable(ledger) -> None:
    document = json.loads(ledger.to_json())
    assert document["schema_version"] == "evoengine.capability-ledger/v1"
    assert len(document["entries"]) == len(_rows(ledger))
    summary = document["summary"]
    assert summary["total_rows"] == len(_rows(ledger))
    assert summary["model_visible_available"] > 0
    assert summary["duplicate_identities"] == []


def test_hidden_compat_entries_never_model_visible(ledger) -> None:
    compat_rows = [
        row
        for row in _rows(ledger)
        if row["exposure_class"] == "compat"
    ]
    assert compat_rows, "expected model-hidden compatibility entries in ledger"
    for row in compat_rows:
        assert row["model_visible"] is False
        assert row["discovery_path"] == "hidden"


def test_exact_only_capabilities_are_not_natural_language_discoverable(
    ledger,
) -> None:
    from src.agents import lead_agent
    from src.capabilities.discovery import (
        CapabilityDiscoveryEntry,
        CapabilityDiscoveryIndex,
    )

    exact_ids = [
        row["capability_id"]
        for row in _rows(ledger)
        if row["exposure_class"] == "exact_only"
        and row["model_visible"]
        and row["kind"] in {KIND_NATIVE_CATALOG, KIND_SUBAGENT}
    ]
    assert exact_ids
    entries = [
        CapabilityDiscoveryEntry.from_spec(
            lead_agent._TOOL_CATALOG_REGISTRY.resolve(tool_id),
        )
        for tool_id in exact_ids
        if lead_agent._known_tool_id(tool_id)
    ]
    index = CapabilityDiscoveryIndex(entries)
    for tool_id in exact_ids:
        # Natural phrasing for an exact-only tool must not recall it...
        natural = index.search(f"请帮我执行与 {tool_id} 相关的通用任务")
        assert tool_id not in {hit.get("tool_id") for hit in natural}
        # ...while the exact stable ID resolves it.
        exact = index.search(tool_id)
        assert tool_id in {hit.get("tool_id") for hit in exact}


def test_sandbox_typed_and_generic_exposure_are_exclusive(monkeypatch) -> None:
    """One app may expose exactly one route: typed or generic, never both."""

    from src.capabilities.ledger import build_capability_ledger
    from src.capabilities.sandbox_contracts import load_sandbox_capability_registry
    from src.capabilities.skill_contracts import compile_skill_typed_schema_build
    from src.tools.sandbox_tools import compile_sandbox_typed_schema_build

    document = load_sandbox_capability_registry()
    # Empty manifests: every app must fail closed as unavailable/blocked.
    offline_context = build_sandbox_exposure_context([], document)
    offline_build = compile_sandbox_typed_schema_build(offline_context)
    assert not offline_build.tool_ids
    offline_ledger = build_capability_ledger(
        sandbox_exposure_context=offline_context
    )
    sandbox_rows = [
        entry
        for entry in offline_ledger.entries
        if entry.kind in {KIND_SANDBOX_TYPED, KIND_SANDBOX_GENERIC}
    ]
    assert sandbox_rows
    for entry in sandbox_rows:
        assert entry.model_visible is False
        assert entry.availability == "unavailable"

    # A manifest matching the accepted pin flips exactly that app to typed.
    pinned = next(entry for entry in document.apps if entry.app_id == "cdhit")
    manifest = {
        "app_id": pinned.app_id,
        "name": "CD-HIT",
        "status": "active",
        "manifest_version": pinned.accepted_manifest_version,
        "manifest_hash": pinned.accepted_manifest_hash,
        "runtime": {
            "concurrency": "heavy",
            "agent_execution_mode": "nonblocking_long",
            "timeout_seconds": 3600,
        },
        "submit_fields": [
            {
                "key": "job_name",
                "type": "text",
                "required": True,
            },
            {
                "key": "input_fasta",
                "type": "query_source",
                "required": True,
                "file_input_key": "input_asset_id",
            },
            {
                "key": "program",
                "type": "select",
                "required": True,
                "default_value": "cd-hit",
                "options": [
                    {"value": "cd-hit"},
                    {"value": "cd-hit-est"},
                ],
            },
            {
                "key": "identity",
                "type": "number",
                "required": False,
            },
        ],
    }
    typed_context = build_sandbox_exposure_context([manifest], document)
    typed_build = compile_sandbox_typed_schema_build(typed_context)
    assert typed_build.tool_ids == ("sandbox_submit_cdhit",)
    typed_ledger = build_capability_ledger(
        sandbox_exposure_context=typed_context
    )
    cdhit_rows = [
        entry
        for entry in typed_ledger.entries
        if entry.capability_id in {
            "sandbox.cdhit.run",
            "sandbox.cdhit.run(generic)",
        }
    ]
    assert len(cdhit_rows) == 1
    entry = cdhit_rows[0]
    assert entry.kind == KIND_SANDBOX_TYPED
    assert entry.model_visible is True
    assert entry.availability == "available"
    # The generic route for the same app must not appear at the same time.
    generic_ids = [
        entry.capability_id
        for entry in typed_ledger.entries
        if entry.kind == KIND_SANDBOX_GENERIC and entry.model_visible
    ]
    assert "sandbox.cdhit.run(generic)" not in generic_ids


def test_retrieval_subagent_delegated_skills_share_lead_contract(ledger) -> None:
    """Delegated retrieval routes resolve to the same typed capability rows
    the lead agent exposes, carrying the same schema and executor."""

    from src.subagents.retrieval_subagent import (
        _DELEGATED_RETRIEVAL_SKILL_ROUTES,
    )

    rows_by_id = {row["capability_id"]: row for row in _rows(ledger)}
    for wrapper, (capability_id, typed_tool_id) in sorted(
        _DELEGATED_RETRIEVAL_SKILL_ROUTES.items()
    ):
        row = rows_by_id.get(capability_id)
        assert row is not None, f"delegated capability missing: {capability_id}"
        assert row["kind"] == KIND_SKILL_TYPED
        assert row["model_visible"] is True
        assert row["availability"] == "available"
        assert "retrieval_subagent" in row["callers"], capability_id
        # The typed tool ID used by load_tools is exactly the one the wrapper
        # resolves at runtime.
        assert row["provider_ref"].get("tool_name") == typed_tool_id


def test_artifact_tools_gated_by_runtime_flag(monkeypatch) -> None:
    """Artifact execution tools must not be discoverable while disabled."""

    monkeypatch.delenv("EVO_ENABLE_ARTIFACT_TOOLS", raising=False)
    off_ledger = build_capability_ledger(
        sandbox_exposure_context=_empty_sandbox_context()
    )
    off_rows = {
        entry.capability_id: entry
        for entry in off_ledger.entries
        if entry.capability_id.startswith("artifact_")
    }
    assert off_rows
    for row in off_rows.values():
        assert row.model_visible is False

    monkeypatch.setenv("EVO_ENABLE_ARTIFACT_TOOLS", "true")
    on_ledger = build_capability_ledger(
        sandbox_exposure_context=_empty_sandbox_context()
    )
    on_ids = {
        entry.capability_id
        for entry in on_ledger.entries
        if entry.model_visible and entry.availability == "available"
    }
    assert "artifact_run_action" in on_ids


def test_ledger_generation_is_stable_across_two_runs() -> None:
    """A4: two consecutive builds must produce identical rows and hashes."""

    first = build_capability_ledger(sandbox_exposure_context=_empty_sandbox_context())
    second = build_capability_ledger(sandbox_exposure_context=_empty_sandbox_context())
    assert first.schema_hash
    assert first.schema_hash == second.schema_hash
    assert first.to_json() == second.to_json()
    assert [entry.to_row() for entry in first.entries] == [
        entry.to_row() for entry in second.entries
    ]


def test_ledger_does_not_touch_shared_skill_registry(monkeypatch) -> None:
    """A4: generation must not mutate the shared registry or its load events."""

    from src.skills.registry import get_skill_registry

    shared = get_skill_registry("skills")
    shared.reset_load_events()
    before_skills = [skill.skill_id for skill in shared.list_skills()]
    before_events = list(shared._load_events)

    ledger = build_capability_ledger(
        sandbox_exposure_context=_empty_sandbox_context()
    )

    after_skills = [skill.skill_id for skill in shared.list_skills()]
    after_events = list(shared._load_events)
    assert after_skills == before_skills
    assert after_events == before_events
    assert ledger.schema_hash


def test_ledger_rows_carry_schema_hashes_and_executor_identity() -> None:
    """A4: discovery/load/executor/trace IDs and schema hashes per row."""

    ledger = build_capability_ledger(sandbox_exposure_context=_empty_sandbox_context())
    typed = [
        entry
        for entry in ledger.entries
        if entry.kind == KIND_SKILL_TYPED and entry.model_visible
    ]
    assert typed
    for entry in typed:
        assert entry.discovery_id == entry.capability_id
        assert entry.executor_capability_id == entry.capability_id
        assert entry.load_tool_ids, entry.capability_id
        assert len(entry.load_schema_hash) == 32
        assert len(entry.output_schema_hash) == 32
        assert entry.load_schema_hash != entry.output_schema_hash or not entry.output_schema_hash
    assert ledger.identity_drift() == []


def test_ledger_evidence_policy_is_not_resource_ref_capability() -> None:
    """A4: declared evidence_policy must not be mislabeled as resource_ref."""

    from src.capabilities.models import CapabilitySpec, Permission, ProviderType

    spec = CapabilitySpec(
        capability_id="skill.demo.source",
        version="1.0.0",
        display_name="demo",
        provider_type=ProviderType.SKILL_SCRIPT,
        permission=Permission.READ,
        evidence_policy={
            "source_projection": {
                "schema_version": "evoengine.capability-source-projection/v1",
                "result_type": "paper_records/v1",
                "payload_path": "result",
            }
        },
    )
    from src.capabilities.ledger import _result_contract_summary

    summary = _result_contract_summary(spec)
    assert summary["evidence_policy_declared"] is True
    assert summary["resource_ref_declared"] is False

    spec_with_ref = spec.model_copy(
        update={"evidence_policy": {"resource_ref_capable": True}}
    )
    summary_with_ref = _result_contract_summary(spec_with_ref)
    assert summary_with_ref["resource_ref_declared"] is True


def test_ledger_internal_and_hidden_guides_are_not_model_visible() -> None:
    """A4: guide discovery visibility matches model_visible."""

    ledger = build_capability_ledger(sandbox_exposure_context=_empty_sandbox_context())
    guides = [entry for entry in ledger.entries if entry.kind == KIND_SKILL_GUIDE]
    assert guides
    for entry in guides:
        if entry.exposure_class == EXPOSURE_INTERNAL:
            assert entry.model_visible is False
            assert entry.discovery_path in {"internal", "hidden"}
        if entry.model_visible:
            # A public guide may be natural or exact-only discoverable
            # (computational references), but must load via activate_skill.
            assert entry.exposure_class in {EXPOSURE_PUBLIC, EXPOSURE_EXACT_ONLY}
            assert entry.load_tool_ids == ["activate_skill"]


def test_ledger_write_goes_to_caller_path(tmp_path) -> None:
    """A4: the generation entry point writes only to a caller-chosen path."""

    from src.capabilities.ledger import write_ledger

    ledger = build_capability_ledger(sandbox_exposure_context=_empty_sandbox_context())
    target = tmp_path / "ledger.json"
    write_ledger(ledger, str(target))
    document = json.loads(target.read_text(encoding="utf-8"))
    assert document["schema_version"] == "evoengine.capability-ledger/v1"
    assert document["summary"]["total_rows"] == len(ledger.entries)
