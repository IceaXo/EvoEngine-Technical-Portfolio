from __future__ import annotations

import asyncio
import hashlib
import json

from src.capabilities.idempotency import build_idempotency_key
from src.capabilities.models import CapabilitySpec, Permission, ProviderType
from src.capabilities.skill_runtime import ContractedSkillRuntime


def _legacy_vector() -> dict:
    return {
        "capability_id": "native.example.run",
        "capability_version": "1.0.0",
        "arguments": {"query": "TP53"},
        "input_refs": [],
        "project_id": "project-1",
        "conversation_id": "conversation-1",
        "context_fingerprint": "plan-1",
    }


def test_unbound_idempotency_key_preserves_legacy_canonical_vector() -> None:
    vector = _legacy_vector()
    expected = hashlib.sha256(
        json.dumps(
            vector,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert build_idempotency_key(**vector) == expected
    assert build_idempotency_key(**vector, task_node_id=None) == expected
    assert build_idempotency_key(**vector, task_node_id="  ") == expected


def test_bound_idempotency_key_is_stable_per_node_and_distinct_across_nodes() -> None:
    vector = _legacy_vector()

    first = build_idempotency_key(**vector, task_node_id="node-action-a")
    retry = build_idempotency_key(**vector, task_node_id=" node-action-a ")
    replacement = build_idempotency_key(
        **vector,
        task_node_id="node-action-a-replacement",
    )

    assert retry == first
    assert replacement != first
    assert first != build_idempotency_key(**vector)


class _CaptureExecutor:
    def __init__(self) -> None:
        self.calls = []

    async def invoke(self, call, **_kwargs):
        self.calls.append(call)
        return call


def test_skill_runtime_builds_key_after_extracting_exact_task_node(
    tmp_path,
    monkeypatch,
) -> None:
    capture = _CaptureExecutor()
    spec = CapabilitySpec(
        capability_id="skill.example.run",
        version="1.0.0",
        display_name="Example Skill",
        provider_type=ProviderType.SKILL_SCRIPT,
        provider_ref={"skill_id": "example", "script_name": "run.py"},
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        permission=Permission.EXECUTE,
    )
    runtime = ContractedSkillRuntime(
        tmp_path,
        specs=[spec],
        executor=capture,
    )
    monkeypatch.setattr(
        "src.services.conversation_file_registry.get_plan_revision_fingerprint",
        lambda **_kwargs: "plan-revision-1",
    )

    returned = asyncio.run(
        runtime.invoke_capability(
            capability_id=spec.capability_id,
            capability_version=spec.version,
            params={"query": "TP53"},
            input_files=[],
            request_id="request-1",
            project_id="project-1",
            conversation_id="conversation-1",
            user_id=7,
            task_node_id="node-action-1",
            task_tree_binding_required=True,
            workspace_root=tmp_path,
        )
    )

    assert returned is capture.calls[0]
    assert returned.task_node_id == "node-action-1"
    assert returned.idempotency_key == build_idempotency_key(
        capability_id=spec.capability_id,
        capability_version=spec.version,
        arguments={"query": "TP53"},
        input_refs=[],
        project_id="project-1",
        conversation_id="conversation-1",
        context_fingerprint="plan-revision-1",
        task_node_id="node-action-1",
    )
