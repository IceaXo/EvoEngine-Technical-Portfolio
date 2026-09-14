from __future__ import annotations

import ast
import inspect
import json
import textwrap
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from langchain_core.tools import StructuredTool

from src.capabilities.models import ResourceRef
from src.subagents import retrieval_subagent
from src.subagents.retrieval_types import EvidencePack


class _RecordingSkillRuntime:
    def __init__(self, responder: Callable[[dict[str, Any]], Any] | None = None, specs=None):
        self.calls: list[dict[str, Any]] = []
        self._responder = responder
        self._specs = {spec.capability_id: spec for spec in (specs or [])}

    def resolve_capability(self, capability_id: str, *, capability_version: str | None = None):
        spec = self._specs.get(capability_id)
        if spec is not None:
            return spec
        from src.capabilities.models import CapabilitySpec, Permission, ProviderType

        return CapabilitySpec(
            capability_id=capability_id,
            version=str(capability_version or "1.0.0"),
            display_name=capability_id,
            provider_type=ProviderType.SKILL_SCRIPT,
            permission=Permission.EXECUTE,
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                },
            },
        )

    async def invoke_capability(self, **kwargs):
        self.calls.append(kwargs)
        if self._responder is None:
            raise AssertionError("this test must not invoke a formal Skill")
        return self._responder(kwargs)


def _skill_result(
    capability_id: str,
    data: dict[str, Any],
    *,
    contract_data: dict[str, Any] | None = None,
    citation_projection: list[dict[str, Any]] | None = None,
    source_sidecar_resource: ResourceRef | None = None,
    ok: bool = True,
    error_kind: str | None = None,
):
    return SimpleNamespace(
        ok=ok,
        data=data,
        contract_data=contract_data,
        citation_projection=list(citation_projection or []),
        source_sidecar_resource=source_sidecar_resource,
        source_outcome=SimpleNamespace(
            model_dump=lambda **_kwargs: {
                "status": "complete" if citation_projection else "not_requested",
                "attempted": bool(citation_projection),
            }
        ),
        capability_id=capability_id,
        capability_version="1.0.0",
        call_id=f"call-{len(data)}",
        status=SimpleNamespace(value="succeeded" if ok else "failed"),
        raw_ref=None,
        error=(
            SimpleNamespace(kind=SimpleNamespace(value=error_kind))
            if error_kind
            else None
        ),
    )


def _raw_safe_tool(name: str, skill_runtime=None):
    wrapped = retrieval_subagent._build_safe_database_tools(
        skill_runtime=skill_runtime or _RecordingSkillRuntime(),
        available_tool_ids={
            typed_tool_id
            for _capability_id, typed_tool_id in (
                retrieval_subagent._DELEGATED_RETRIEVAL_SKILL_ROUTES.values()
            )
        },
    )[name]
    return getattr(wrapped, "_evo_raw_tool", wrapped).func


def test_retrieval_skill_runtime_uses_only_the_frozen_schema_build(monkeypatch) -> None:
    from src.capabilities import skill_runtime as skill_runtime_module
    from src.capabilities.skill_contracts import compile_skill_typed_schema_build

    frozen_build = compile_skill_typed_schema_build("skills")
    monkeypatch.setattr(
        skill_runtime_module,
        "scan_skill_capability_contracts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("retrieval must not rescan Skill manifests")
        ),
    )

    runtime = retrieval_subagent._retrieval_skill_runtime_from_frozen_build(
        frozen_build,
        available_tool_ids={
            typed_tool_id
            for _capability_id, typed_tool_id in (
                retrieval_subagent._DELEGATED_RETRIEVAL_SKILL_ROUTES.values()
            )
        },
    )

    assert runtime.resolve_capability("skill.pubmed.literature.query")
    assert runtime.resolve_capability("skill.uniprot.query")
    assert runtime.resolve_capability("skill.public_data.clinvar")
    assert runtime.resolve_capability("skill.public_data.geo")
    assert runtime.resolve_capability("skill.public_data.ensembl")
    assert runtime.resolve_capability("skill.open_targets.query")


def test_retrieval_runtime_rejects_legacy_or_mismatched_frozen_items() -> None:
    from src.capabilities.skill_contracts import SkillTypedExposure
    from src.capabilities.skill_runtime import ContractedSkillNotFoundError

    class _FrozenBuild:
        @staticmethod
        def item_for_capability(capability_id: str):
            return SimpleNamespace(
                exposure=SkillTypedExposure.LEGACY,
                tool_id=dict(
                    retrieval_subagent._DELEGATED_RETRIEVAL_SKILL_ROUTES.values()
                )[capability_id],
                capability_spec=lambda: (_ for _ in ()).throw(
                    AssertionError("legacy item must not enter retrieval runtime")
                ),
            )

    available = {
        typed_tool_id
        for _capability_id, typed_tool_id in (
            retrieval_subagent._DELEGATED_RETRIEVAL_SKILL_ROUTES.values()
        )
    }
    runtime = retrieval_subagent._retrieval_skill_runtime_from_frozen_build(
        _FrozenBuild(),  # type: ignore[arg-type]
        available_tool_ids=available,
    )

    for capability_id, _typed_tool_id in (
        retrieval_subagent._DELEGATED_RETRIEVAL_SKILL_ROUTES.values()
    ):
        with pytest.raises(ContractedSkillNotFoundError):
            runtime.resolve_capability(capability_id)


def test_retrieval_database_wrappers_are_formalized_and_frozen_availability_gated() -> None:
    direct_provider_whitelist = set()
    delegated_routes = retrieval_subagent._DELEGATED_RETRIEVAL_SKILL_ROUTES
    all_typed_ids = {typed_tool_id for _capability_id, typed_tool_id in delegated_routes.values()}
    all_capability_ids = {
        capability_id for capability_id, _typed_tool_id in delegated_routes.values()
    }
    all_tools = retrieval_subagent._build_safe_database_tools(
        skill_runtime=_RecordingSkillRuntime(),
        available_tool_ids=all_typed_ids,
    )

    # Every retrieval database wrapper, including CrossRef, must be backed by
    # one frozen formal Capability from the parent turn's snapshot.
    assert set(all_tools) == set(delegated_routes) | direct_provider_whitelist
    assert len(all_capability_ids) == len(delegated_routes)
    assert len(all_typed_ids) == len(delegated_routes)

    # The wrapper must disappear when its typed execution ID was unavailable
    # in the parent turn's immutable capability snapshot.
    for wrapper_name, (_capability_id, typed_tool_id) in delegated_routes.items():
        tools_without_route = retrieval_subagent._build_safe_database_tools(
            skill_runtime=_RecordingSkillRuntime(),
            available_tool_ids=all_typed_ids - {typed_tool_id},
        )
        assert wrapper_name not in tools_without_route

    # No wrapper may perform a direct safe_http call anymore: every database
    # wrapper must reach providers through the frozen ContractedSkillRuntime,
    # including the CrossRef exact-DOI identity contract.
    source = textwrap.dedent(
        inspect.getsource(retrieval_subagent._build_safe_database_tools)
    )
    tree = ast.parse(source)
    builder = tree.body[0]
    assert isinstance(builder, (ast.FunctionDef, ast.AsyncFunctionDef))
    nested_functions = {
        node.name: node
        for node in builder.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def direct_call_names(node: ast.AST) -> set[str]:
        names: set[str] = set()
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            if isinstance(child.func, ast.Name):
                names.add(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                names.add(child.func.attr)
        return names

    direct_safe_http_callers = {
        name
        for name, node in nested_functions.items()
        if any(call.startswith("safe_http_") for call in direct_call_names(node))
    }
    assert direct_safe_http_callers == direct_provider_whitelist
    assert "_invoke_retrieval_skill" in direct_call_names(
        nested_functions["_safe_query_ncbi_summary"]
    )
    for wrapper_name in delegated_routes:
        calls = direct_call_names(nested_functions[wrapper_name])
        assert calls & {"_invoke_retrieval_skill", "_safe_query_ncbi_summary"}


def test_clinvar_and_geo_adapters_use_formal_skill_without_private_ncbi(
    monkeypatch,
) -> None:
    records_by_capability = {
        "skill.public_data.clinvar": {
            "uid": "123",
            "title": "ClinVar BRCA1 record",
        },
        "skill.public_data.geo": {
            "uid": "200000001",
            "accession": "GSE1",
            "title": "GEO expression dataset",
        },
    }

    def respond(call):
        capability_id = call["capability_id"]
        record = records_by_capability[capability_id]
        assert call["params"]["limit"] == 4
        return _skill_result(
            capability_id,
            {
                "ok": True,
                "provider": "ncbi_eutils",
                "database": (
                    "ClinVar"
                    if capability_id.endswith("clinvar")
                    else "NCBI GEO"
                ),
                "operation": "query",
                "data": {
                    "search": {"esearchresult": {"idlist": [record["uid"]]}},
                    "summary": {
                        "result": {
                            "uids": [record["uid"]],
                            record["uid"]: record,
                        }
                    },
                    "ids": [record["uid"]],
                    "records": [record],
                },
            },
        )

    runtime = _RecordingSkillRuntime(respond)
    tools = retrieval_subagent._build_safe_database_tools(
        skill_runtime=runtime,
        available_tool_ids={
            "skill_public_data_clinvar",
            "skill_public_data_geo",
        },
        request_id="req-ncbi",
        project_id="project-ncbi",
        conversation_id="conv-ncbi",
        user_id=9,
    )

    for tool_name, capability_id in (
        ("safe_query_clinvar", "skill.public_data.clinvar"),
        ("safe_query_geo", "skill.public_data.geo"),
    ):
        wrapped = tools[tool_name]
        raw_tool = getattr(wrapped, "_evo_raw_tool", wrapped).func
        payload = raw_tool(query="BRCA1", max_results=4)
        assert payload["ok"] is True
        assert payload["ids"]
        assert payload["records"] == [records_by_capability[capability_id]]

    assert [call["capability_id"] for call in runtime.calls] == [
        "skill.public_data.clinvar",
        "skill.public_data.geo",
    ]
    assert all(call["request_id"] == "req-ncbi" for call in runtime.calls)
    assert all(call["project_id"] == "project-ncbi" for call in runtime.calls)
    assert all(call["conversation_id"] == "conv-ncbi" for call in runtime.calls)
    assert all(call["user_id"] == 9 for call in runtime.calls)


def test_ensembl_adapter_uses_frozen_formal_skill_lookup(monkeypatch) -> None:
    record = {
        "id": "ENSG00000141510",
        "display_name": "TP53",
        "species": "homo_sapiens",
        "biotype": "protein_coding",
        "seq_region_name": "17",
        "start": 7661779,
        "end": 7687546,
        "strand": -1,
    }

    def respond(call):
        assert call["capability_id"] == "skill.public_data.ensembl"
        assert call["params"] == {
            "operation": "lookup",
            "species": "homo_sapiens",
            "gene_symbol": "TP53",
        }
        return _skill_result(
            "skill.public_data.ensembl",
            {
                "ok": True,
                "provider": "ensembl_rest",
                "database": "Ensembl",
                "operation": "lookup",
                "data": record,
                "evidence_refs": [],
            },
        )

    runtime = _RecordingSkillRuntime(respond)
    tools = retrieval_subagent._build_safe_database_tools(
        skill_runtime=runtime,
        available_tool_ids={"skill_public_data_ensembl"},
        request_id="req-ensembl",
        project_id="project-ensembl",
        conversation_id="conv-ensembl",
        user_id=12,
    )

    wrapped = tools["safe_query_ensembl"]
    raw_tool = getattr(wrapped, "_evo_raw_tool", wrapped).func
    payload = raw_tool(query="human TP53", max_results=5)

    assert payload["ok"] is True
    assert payload["record"] == record
    assert payload["capability_calls"][0]["capability_id"] == "skill.public_data.ensembl"
    assert len(runtime.calls) == 1
    assert runtime.calls[0]["request_id"] == "req-ensembl"
    assert runtime.calls[0]["project_id"] == "project-ensembl"
    assert runtime.calls[0]["conversation_id"] == "conv-ensembl"
    assert runtime.calls[0]["user_id"] == 12


def test_ncbi_transcript_adapter_uses_exact_refseq_formal_skill(monkeypatch) -> None:
    record = {
        "accession_version": "NM_000546.6",
        "definition": "Homo sapiens tumor protein p53 (TP53), transcript variant 1, mRNA.",
        "official_symbol": "TP53",
        "organism": "Homo sapiens",
        "taxid": 9606,
        "transcript_sequence": "ACTTGTCATGGCGACTGTCCAGAA",
        "transcript_length": 24,
        "cds_start": 4,
        "cds_end": 21,
    }

    def respond(call):
        assert call["capability_id"] == "skill.public_data.ncbi_gene_cds"
        assert call["params"] == {
            "gene_symbol": "TP53",
            "organism": "Homo sapiens",
            "transcript_accession": "NM_000546.6",
        }
        return _skill_result(
            "skill.public_data.ncbi_gene_cds",
            {
                "ok": True,
                "provider": "ncbi_eutils",
                "database": "NCBI Gene / RefSeq",
                "operation": "query",
                "data": record,
                "evidence_refs": [],
            },
        )

    runtime = _RecordingSkillRuntime(respond)
    tools = retrieval_subagent._build_safe_database_tools(
        skill_runtime=runtime,
        available_tool_ids={"skill_public_data_ncbi_gene_cds"},
    )
    raw_tool = getattr(
        tools["safe_query_ncbi_transcript"],
        "_evo_raw_tool",
        tools["safe_query_ncbi_transcript"],
    ).func
    payload = raw_tool(
        query="Fetch Homo sapiens TP53 RefSeq NM_000546.6 mRNA sequence",
        max_results=1,
    )

    assert payload["ok"] is True
    assert payload["records"][0]["accession"] == "NM_000546.6"
    assert payload["provider_observation"]["requested_ids"] == ["NM_000546.6"]
    assert payload["provider_observation"]["returned_ids"] == ["NM_000546.6"]
    assert payload["provider_observation"]["coverage"] == "exhausted"
    assert len(runtime.calls) == 1


def test_retrieval_input_normalizes_model_proposed_operational_budgets():
    payload = retrieval_subagent.RetrievalSubagentInput(
        task="Collect evidence",
        max_iterations=99,
        max_evidence=25,
    )

    assert payload.max_iterations == 6
    assert payload.max_evidence == 12


def test_retrieval_input_contract_has_one_evidence_pack_output_mode():
    schema = retrieval_subagent.RetrievalSubagentInput.model_json_schema()

    assert "required_output" not in schema["properties"]
    assert schema["properties"]["output_mode"]["const"] == "evidence_pack"
    assert retrieval_subagent.RetrievalSubagentInput(task="qPCR evidence").output_mode == "evidence_pack"
    assert (
        retrieval_subagent.RetrievalSubagentInput(
            checkpoint_resource_id="resource:conversation-file:99"
        ).task
        == ""
    )


def test_retrieval_checkpoint_runs_three_batches_without_gaps_duplicates_or_messages(
    monkeypatch,
) -> None:
    invocations: list[tuple[str, str]] = []
    checkpoint_store: dict[str, dict[str, Any]] = {}

    def fake_tool(name: str) -> StructuredTool:
        def run(query: str, max_results: int = 5) -> dict[str, Any]:
            del max_results
            invocations.append((name, query))
            return {"ok": True, "provider": name, "records": []}

        return StructuredTool.from_function(
            func=run,
            name=name,
            description=f"deterministic test tool {name}",
        )

    tools = [
        fake_tool("safe_query_alpha"),
        fake_tool("safe_query_beta"),
        fake_tool("safe_query_gamma"),
    ]
    monkeypatch.setattr(
        retrieval_subagent,
        "_retrieval_skill_runtime_from_frozen_build",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        retrieval_subagent,
        "_build_retrieval_tools",
        lambda **_kwargs: tools,
    )
    monkeypatch.setattr(
        retrieval_subagent,
        "_plan_tool_queries",
        lambda **_kwargs: {
            "safe_query_alpha": ["query-alpha"],
            "safe_query_beta": ["query-beta"],
            "safe_query_gamma": ["query-gamma"],
        },
    )

    def persist_checkpoint(*, payload: dict[str, Any], **_kwargs) -> ResourceRef:
        file_id = 100 + len(checkpoint_store)
        resource_id = f"resource:conversation-file:{file_id}"
        checkpoint_store[resource_id] = deepcopy(payload)
        return ResourceRef(
            resource_id=resource_id,
            uri=f"conversation-file://{file_id}",
            kind="retrieval_checkpoint",
            storage="conversation_file",
            media_type="application/json",
            size_bytes=len(json.dumps(payload)),
            sha256=(f"{file_id:064x}"[-64:]),
            conversation_file_id=file_id,
        )

    monkeypatch.setattr(
        retrieval_subagent,
        "_persist_retrieval_checkpoint",
        persist_checkpoint,
    )
    monkeypatch.setattr(
        retrieval_subagent,
        "_load_retrieval_checkpoint",
        lambda *, resource_id, **_kwargs: deepcopy(checkpoint_store.get(str(resource_id))),
    )
    monkeypatch.setattr(retrieval_subagent, "citations_enabled", lambda: False)
    monkeypatch.setattr(
        retrieval_subagent,
        "append_reference_records",
        lambda _records: {"ok": True},
    )
    monkeypatch.setattr(
        retrieval_subagent,
        "_register_retrieval_temp_files",
        lambda **_kwargs: {"ok": True},
    )

    wrapped = retrieval_subagent.build_retrieval_subagent_tool(
        settings=SimpleNamespace(),
        skill_typed_schema_build=SimpleNamespace(),
        available_tool_ids=set(),
        project_id="project-batches",
        conversation_id="conversation-batches",
        user_id=7,
        request_id="request-batches",
    )
    run = getattr(wrapped, "_evo_raw_tool", wrapped).func

    first = run(task="three sources", sources=["web"], max_iterations=3, batch_size=1)
    second = run(
        checkpoint_resource_id=first["checkpoint_resource_id"],
    )
    third = run(
        checkpoint_resource_id=second["checkpoint_resource_id"],
    )

    assert (first["complete"], first["has_more"], first["batch_index"]) == (
        False,
        True,
        1,
    )
    assert (second["complete"], second["has_more"], second["batch_index"]) == (
        False,
        True,
        2,
    )
    assert (third["complete"], third["has_more"], third["batch_index"]) == (
        True,
        False,
        3,
    )
    assert invocations == [
        ("safe_query_alpha", "query-alpha"),
        ("safe_query_beta", "query-beta"),
        ("safe_query_gamma", "query-gamma"),
    ]
    assert first["retrieval_task_id"] == second["retrieval_task_id"] == third["retrieval_task_id"]
    assert len(
        {
            first["subagent_run_id"],
            second["subagent_run_id"],
            third["subagent_run_id"],
        }
    ) == 3
    assert [first["batch_id"], second["batch_id"], third["batch_id"]] == [
        f'{first["retrieval_task_id"]}:batch:1',
        f'{first["retrieval_task_id"]}:batch:2',
        f'{first["retrieval_task_id"]}:batch:3',
    ]
    for result in (first, second, third):
        assert result["raw_ref"]
        assert result["resources"]
        assert result["resources"][0]["resource_id"] == result["checkpoint_resource_id"]
        assert result["resources"][0]["sha256"]
    final_checkpoint = checkpoint_store[third["checkpoint_resource_id"]]
    assert len(final_checkpoint["completed_work_ids"]) == 3
    assert len({item["work_id"] for item in final_checkpoint["tool_results"]}) == 3
    assert final_checkpoint["completed_scope"] == final_checkpoint["completed_work_ids"]
    assert final_checkpoint["pending_scope"] == []
    assert final_checkpoint["capability_call_fingerprints"] == final_checkpoint["completed_work_ids"]
    assert final_checkpoint["objective_hash"]
    assert final_checkpoint["registry_snapshot_id"]
    assert final_checkpoint["message_history_included"] is False
    assert "messages" not in final_checkpoint
    assert final_checkpoint["task"] == "three sources"
    assert final_checkpoint["sources"] == ["web"]


def test_asset_search_derives_page_read_only_when_hit_needs_context() -> None:
    common = {
        "asset_id": 42,
        "asset_sha256": "sha-42",
        "parser_version": "mineru-v1",
        "parse_run_id": "run-42",
        "page_idx": 7,
        "block_idx": 19,
        "block_ref": "block:42:19",
        "document_ref": {
            "document_id": "asset:42",
            "source_kind": "project_asset",
            "source_id": "42",
            "content_sha256": "sha-42",
            "parser_version": "mineru-v1",
            "parse_run_id": "run-42",
        },
    }
    short = retrieval_subagent._derived_asset_read_work_items(
        {
            "tool": "search_milvus_knowledge",
            "work_id": "search-work",
            "query": "tail fact",
            "output": {"results": [{**common, "content": "short hit"}]},
        }
    )
    assert len(short) == 1
    assert short[0]["tool_name"] == "read_asset_page_parsed"
    assert short[0]["arguments"] == {"asset_id": 42, "page_idx": 7}
    assert short[0]["parent_work_id"] == "search-work"
    assert short[0]["document_ref"]["parse_run_id"] == "run-42"
    assert short[0]["locator"]["block_ref"] == "block:42:19"

    enough = retrieval_subagent._derived_asset_read_work_items(
        {
            "tool": "search_milvus_knowledge",
            "output": {"results": [{**common, "content": "x" * 400}]},
        }
    )
    assert enough == []


def test_retrieval_checkpoint_resumes_derived_page_read_without_replaying_search(
    monkeypatch,
) -> None:
    invocations: list[tuple[str, dict[str, Any]]] = []
    checkpoint_store: dict[str, dict[str, Any]] = {}

    def search(query: str) -> dict[str, Any]:
        invocations.append(("search_milvus_knowledge", {"query": query}))
        return {
            "ok": True,
            "results": [
                {
                    "asset_id": 42,
                    "asset_sha256": "sha-42",
                    "parser_version": "mineru-v1",
                    "parse_run_id": "run-42",
                    "file_name": "paper.pdf",
                    "page_idx": 7,
                    "block_idx": 19,
                    "block_ref": "block:42:19",
                    "page_ref": "asset:42:parse:run-42:page:7",
                    "locator_verified": True,
                    "content": "short hit",
                    "document_ref": {
                        "document_id": "asset:42",
                        "source_kind": "project_asset",
                        "source_id": "42",
                        "content_sha256": "sha-42",
                        "parser_version": "mineru-v1",
                        "parse_run_id": "run-42",
                    },
                }
            ],
        }

    def read(asset_id: int, page_idx: int) -> dict[str, Any]:
        invocations.append(
            (
                "read_asset_page_parsed",
                {"asset_id": asset_id, "page_idx": page_idx},
            )
        )
        return {
            "ok": True,
            "asset_id": asset_id,
            "asset_sha256": "sha-42",
            "parser_version": "mineru-v1",
            "parse_run_id": "run-42",
            "page_idx": page_idx,
            "complete": True,
            "has_more": False,
            "cursor": None,
            "data": [
                {
                    "page_idx": page_idx,
                    "block_idx": 19,
                    "block_ref": "block:42:19",
                    "page_ref": "asset:42:parse:run-42:page:7",
                    "type": "text",
                    "text": "Expanded local context for the short search hit.",
                }
            ],
        }

    tools = [
        StructuredTool.from_function(
            func=search,
            name="search_milvus_knowledge",
            description="search one selected knowledge asset",
        ),
        StructuredTool.from_function(
            func=read,
            name="read_asset_page_parsed",
            description="read an exact parsed page",
        ),
    ]
    monkeypatch.setattr(
        retrieval_subagent,
        "_retrieval_skill_runtime_from_frozen_build",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(retrieval_subagent, "_build_retrieval_tools", lambda **_kwargs: tools)
    monkeypatch.setattr(
        retrieval_subagent,
        "_plan_tool_queries",
        lambda **_kwargs: {"search_milvus_knowledge": ["tail fact"]},
    )

    def persist_checkpoint(*, payload: dict[str, Any], **_kwargs) -> ResourceRef:
        file_id = 920 + len(checkpoint_store)
        resource_id = f"resource:conversation-file:{file_id}"
        checkpoint_store[resource_id] = deepcopy(payload)
        return ResourceRef(
            resource_id=resource_id,
            uri=f"conversation-file://{file_id}",
            kind="retrieval_checkpoint",
            storage="conversation_file",
            media_type="application/json",
            size_bytes=len(json.dumps(payload)),
            sha256=(f"{file_id:064x}"[-64:]),
            conversation_file_id=file_id,
        )

    monkeypatch.setattr(retrieval_subagent, "_persist_retrieval_checkpoint", persist_checkpoint)
    monkeypatch.setattr(
        retrieval_subagent,
        "_load_retrieval_checkpoint",
        lambda *, resource_id, **_kwargs: deepcopy(checkpoint_store.get(str(resource_id))),
    )
    monkeypatch.setattr(retrieval_subagent, "citations_enabled", lambda: False)
    monkeypatch.setattr(retrieval_subagent, "append_reference_records", lambda _records: {"ok": True})
    monkeypatch.setattr(
        retrieval_subagent,
        "_register_retrieval_temp_files",
        lambda **_kwargs: {"ok": True},
    )

    wrapped = retrieval_subagent.build_retrieval_subagent_tool(
        settings=SimpleNamespace(),
        skill_typed_schema_build=SimpleNamespace(),
        available_tool_ids=set(),
        project_id="project-doc",
        conversation_id="conversation-doc",
        user_id=7,
        request_id="request-doc",
    )
    run = getattr(wrapped, "_evo_raw_tool", wrapped).func
    first = run(
        task="find the tail fact",
        sources=["knowledge_assets"],
        max_iterations=2,
        batch_size=1,
    )
    assert first["complete"] is False
    assert first["has_more"] is True
    first_checkpoint = checkpoint_store[first["checkpoint_resource_id"]]
    assert len(first_checkpoint["work_items"]) == 2
    derived = first_checkpoint["work_items"][1]
    assert derived["parent_work_id"] == first_checkpoint["work_items"][0]["work_id"]
    assert derived["document_ref"]["parse_run_id"] == "run-42"

    second = run(checkpoint_resource_id=first["checkpoint_resource_id"])
    assert second["complete"] is True
    assert second["has_more"] is False
    assert invocations == [
        ("search_milvus_knowledge", {"query": "tail fact"}),
        ("read_asset_page_parsed", {"asset_id": 42, "page_idx": 7}),
    ]
    final_checkpoint = checkpoint_store[second["checkpoint_resource_id"]]
    assert len(final_checkpoint["completed_work_ids"]) == 2
    assert len({item["work_id"] for item in final_checkpoint["tool_results"]}) == 2
    assert final_checkpoint["message_history_included"] is False


def test_registered_skill_sources_skip_legacy_verifiers_and_complete_acceptance(
    monkeypatch,
) -> None:
    registration = {
        "schema_version": "evoengine.source-registration/v1",
        "status": "registered",
        "authority": "capability_runtime",
        "capability_id": "skill.uniprot.query",
        "capability_version": "1.0.0",
        # A batch has one candidate identity even when it returns two records.
        "candidate_id": "source-candidate:" + "c" * 64,
        "result_resource_id": "resource:conversation-file:801",
    }
    sources = [
        {
            "type": "data",
            "source_type": "bio_database",
            "provider": "uniprot_rest",
            "provider_record_id": accession,
            "title": accession,
            "url": f"https://www.uniprot.org/uniprotkb/{accession}/entry",
            "source_registration": registration,
        }
        for accession in ("P69905", "P68871")
    ]
    sidecar = {
        "resource_id": "resource:conversation-file:802",
        "uri": "conversation-file://802",
        "kind": "source_sidecar",
        "storage": "conversation_file",
        "media_type": "application/json",
        "size_bytes": 256,
        "sha256": "d" * 64,
        "conversation_file_id": 802,
    }

    def run(query: str, max_results: int = 5) -> dict[str, Any]:
        del query, max_results
        return {
            "ok": True,
            "source_type": "bio_database",
            "database": "uniprot",
            "provider": "uniprot",
            "records": [
                {
                    "accession": accession,
                    "recommended_name": accession,
                    "function": f"Function for {accession}",
                }
                for accession in ("P69905", "P68871")
            ],
            "registered_source_ledger": sources,
            "source_ledger": sources,
            "source_sidecar_resources": [sidecar],
            "source_sidecar_refs": [sidecar["resource_id"]],
        }

    tool = StructuredTool.from_function(
        func=run,
        name="safe_query_uniprot",
        description="registered formal source test",
    )
    monkeypatch.setattr(
        retrieval_subagent,
        "_retrieval_skill_runtime_from_frozen_build",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        retrieval_subagent,
        "_build_retrieval_tools",
        lambda **_kwargs: [tool],
    )
    monkeypatch.setattr(
        retrieval_subagent,
        "_plan_tool_queries",
        lambda **_kwargs: {"safe_query_uniprot": ["P69905 P68871"]},
    )
    monkeypatch.setattr(
        retrieval_subagent,
        "_load_retrieval_checkpoint",
        lambda **_kwargs: None,
    )
    checkpoint = ResourceRef(
        resource_id="resource:conversation-file:803",
        uri="conversation-file://803",
        kind="retrieval_checkpoint",
        storage="conversation_file",
        media_type="application/json",
        size_bytes=512,
        sha256="e" * 64,
        conversation_file_id=803,
    )
    monkeypatch.setattr(
        retrieval_subagent,
        "_persist_retrieval_checkpoint",
        lambda **_kwargs: checkpoint,
    )
    monkeypatch.setattr(retrieval_subagent, "citations_enabled", lambda: True)
    monkeypatch.setattr(
        retrieval_subagent,
        "verify_evidence_pack_sources",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("registered Skill sources must not be network-verified again")
        ),
    )
    monkeypatch.setattr(
        retrieval_subagent,
        "verify_claim_evidence_links",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("registered Skill sources must not be claim-verified again")
        ),
    )
    monkeypatch.setattr(
        retrieval_subagent,
        "append_reference_records",
        lambda _records: {"ok": True, "count": 2},
    )
    monkeypatch.setattr(
        retrieval_subagent,
        "_register_retrieval_temp_files",
        lambda **_kwargs: {"ok": True},
    )

    wrapped = retrieval_subagent.build_retrieval_subagent_tool(
        settings=SimpleNamespace(),
        skill_typed_schema_build=SimpleNamespace(),
        available_tool_ids=set(),
        project_id="project-source",
        conversation_id="conversation-source",
        user_id=7,
        request_id="request-source",
    )
    result = getattr(wrapped, "_evo_raw_tool", wrapped).func(
        task="P69905 P68871",
        sources=["uniprot"],
        max_iterations=1,
        batch_size=1,
    )

    assert result["ok"] is True
    assert result["retrieval_run_completed"] is True
    assert result["evidence_pack_claims_supported"] is True
    assert result["capability_outcome"]["evidence"] == "available"
    assert result["completion_signal"]["acceptance"][
        "evidence_pack_claims_supported"
    ] is True
    assert [item["provider_record_id"] for item in result["source_ledger"]] == [
        "P69905",
        "P68871",
    ]
    assert result["source_sidecar_refs"] == [sidecar["resource_id"]]


def test_retrieval_checkpoint_failure_preserves_a_structured_error(monkeypatch) -> None:
    monkeypatch.setattr(
        retrieval_subagent,
        "_retrieval_skill_runtime_from_frozen_build",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        retrieval_subagent,
        "_build_retrieval_tools",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        retrieval_subagent,
        "_load_retrieval_checkpoint",
        lambda **_kwargs: None,
    )
    wrapped = retrieval_subagent.build_retrieval_subagent_tool(
        settings=SimpleNamespace(),
        skill_typed_schema_build=SimpleNamespace(),
        available_tool_ids=set(),
        project_id="project-batches",
        conversation_id="conversation-batches",
        user_id=7,
        request_id="request-batches",
    )
    run = getattr(wrapped, "_evo_raw_tool", wrapped).func

    result = run(checkpoint_resource_id="resource:conversation-file:404")

    assert result["ok"] is False
    assert result["error_kind"] == "validation_error"
    assert result["error_code"] == "retrieval_checkpoint_invalid"
    assert "checkpoint" in result["error"]
    assert "unknown" not in result["model_summary"].lower()


def test_bio_database_source_is_union_when_no_concrete_database_is_named():
    selected = retrieval_subagent._normalized_sources(
        ["pubmed", "bio_database"],
        "获取人源 TP53 的公共生物数据库记录。",
    )

    tools = retrieval_subagent._build_retrieval_tools(
        settings=SimpleNamespace(),
        sources=selected,
        skill_runtime=_RecordingSkillRuntime(),
        available_tool_ids={
            typed_tool_id
            for _capability_id, typed_tool_id in (
                retrieval_subagent._DELEGATED_RETRIEVAL_SKILL_ROUTES.values()
            )
        },
        project_id="project-1",
        conversation_id="conversation-1",
        user_id=7,
        request_id="req-1",
    )
    tool_names = [str(getattr(tool, "name", "")) for tool in tools]

    assert "skill_pubmed_literature_query" in tool_names
    assert "skill_public_data_ensembl" in tool_names
    assert "skill_uniprot_query" in tool_names
    assert tool_names.index("skill_public_data_ensembl") < tool_names.index(
        "skill_pubmed_literature_query"
    )


def test_explicit_ncbi_refseq_source_does_not_expand_broad_database_lane():
    selected = retrieval_subagent._normalized_sources(
        ["ncbi", "bio_database", "web"],
        "Fetch Homo sapiens TP53 RefSeq NM_000546.6 mRNA sequence.",
    )

    tools = retrieval_subagent._build_retrieval_tools(
        settings=SimpleNamespace(),
        sources=selected,
        skill_runtime=_RecordingSkillRuntime(),
        available_tool_ids={
            typed_tool_id
            for _capability_id, typed_tool_id in (
                retrieval_subagent._DELEGATED_RETRIEVAL_SKILL_ROUTES.values()
            )
        },
        project_id=None,
        conversation_id=None,
        user_id=None,
        request_id=None,
    )

    assert [str(getattr(tool, "name", "")) for tool in tools] == [
        "skill_public_data_ncbi_gene_cds",
        "web_search",
    ]


def test_pubmed_ncbi_label_does_not_select_nuccore_transcript_adapter():
    selected = retrieval_subagent._normalized_sources(
        ["pubmed"],
        "只用 PubMed/NCBI 查询 PMID 19114008 的文献记录。",
    )

    tools = retrieval_subagent._build_retrieval_tools(
        settings=SimpleNamespace(),
        sources=selected,
        skill_runtime=_RecordingSkillRuntime(),
        available_tool_ids={
            typed_tool_id
            for _capability_id, typed_tool_id in (
                retrieval_subagent._DELEGATED_RETRIEVAL_SKILL_ROUTES.values()
            )
        },
        project_id=None,
        conversation_id=None,
        user_id=None,
        request_id=None,
    )

    tool_names = [str(getattr(tool, "name", "")) for tool in tools]
    assert "skill_pubmed_literature_query" in tool_names
    assert "skill_public_data_ncbi_gene_cds" not in tool_names


def test_concrete_database_source_does_not_expand_broad_fallback_lane():
    selected = retrieval_subagent._normalized_sources(
        ["uniprot", "bio_database", "web"],
        "Look up accession P69905 in UniProt and provide evidence.",
    )

    tools = retrieval_subagent._build_retrieval_tools(
        settings=SimpleNamespace(),
        sources=selected,
        skill_runtime=_RecordingSkillRuntime(),
        available_tool_ids={
            typed_tool_id
            for _capability_id, typed_tool_id in (
                retrieval_subagent._DELEGATED_RETRIEVAL_SKILL_ROUTES.values()
            )
        },
        project_id=None,
        conversation_id=None,
        user_id=None,
        request_id=None,
    )
    tool_names = [str(getattr(tool, "name", "")) for tool in tools]

    assert tool_names == ["skill_uniprot_query", "web_search"]


def test_exact_identifiers_bypass_model_query_rewrites() -> None:
    doi_task = "Verify DOI 10.1186/1471-2105-9-559 and 10.9999/evoengine-source-test."
    pmid_task = "Fetch PMID 19114008."

    assert retrieval_subagent._tool_query_variants(
        tool_name="safe_query_crossref",
        task=doi_task,
        planned_queries={"safe_query_crossref": ["rewritten without identifiers"]},
    ) == [doi_task]
    assert retrieval_subagent._tool_query_variants(
        tool_name="safe_query_pubmed",
        task=pmid_task,
        planned_queries={"safe_query_pubmed": ["generic WGCNA paper"]},
    ) == [pmid_task]
    assert "pubmed" in retrieval_subagent._normalized_sources([], pmid_task)


def test_mixed_pubmed_stable_ids_use_independent_exact_query_scopes() -> None:
    task = "Verify PMID 19114008 and DOI 10.1186/1471-2105-9-559."

    assert retrieval_subagent._tool_query_variants(
        tool_name="safe_query_pubmed",
        task=task,
        planned_queries={"safe_query_pubmed": ["generic literature search"]},
    ) == [
        "PMID 19114008",
        "DOI 10.1186/1471-2105-9-559",
    ]


def test_doi_extraction_stops_at_chinese_punctuation_before_pmid() -> None:
    assert retrieval_subagent._doi_ids_from_text(
        "核验 DOI 10.1186/1471-2105-9-559，PMID 19114008；"
        "并检查 10.9999/evoengine-source-test。"
    ) == [
        "10.1186/1471-2105-9-559",
        "10.9999/evoengine-source-test",
    ]


def test_pubmed_explicit_pmid_uses_formal_skill_exact_fetch(monkeypatch) -> None:
    def respond(call):
        assert call["capability_id"] == "skill.pubmed.literature.query"
        assert call["params"] == {
            "operation": "fetch_pubmed_summaries",
            "pmids": ["19114008"],
        }
        return _skill_result(
            "skill.pubmed.literature.query",
            {
                "result": {
                    "provider": "ncbi_eutils",
                    "database": "pubmed",
                    "records": [
                {
                    "uid": "19114008",
                    "pmid": "19114008",
                    "title": "WGCNA: an R package for weighted correlation network analysis.",
                    "abstract": "A weighted correlation network analysis method.",
                    "articleids": [{"idtype": "pubmed", "value": "19114008"}],
                },
                {
                    "uid": "99999999",
                    "pmid": "99999999",
                    "title": "An unrelated Provider record.",
                    "abstract": "This record was not requested.",
                    "articleids": [{"idtype": "pubmed", "value": "99999999"}],
                },
                    ],
                },
                "source_ledger": [{"pmid": "19114008"}],
            },
        )

    runtime = _RecordingSkillRuntime(respond)

    payload = _raw_safe_tool("safe_query_pubmed", runtime)(
        query="Please retrieve PMID 19114008 exactly.",
        max_results=5,
    )

    assert len(runtime.calls) == 1
    assert payload["direct_lookup"] is True
    assert [item["pmid"] for item in payload["records"]] == ["19114008"]
    assert payload["requested_ids"] == ["19114008"]
    assert payload["returned_ids"] == ["19114008"]
    assert payload["provider_observation"]["returned_ids"] == ["19114008"]
    assert payload["provider_observation"]["coverage"] == "exhausted"
    assert payload["source_ledger"] == [{"pmid": "19114008"}]


def test_pubmed_exact_fetch_uses_full_contract_data_when_data_is_resource_only() -> None:
    provider_contract = {
        "ok": True,
        "operation": "fetch_pubmed_summaries",
        "result": {
            "provider": "ncbi_eutils",
            "database": "pubmed",
            "records": [
                {
                    "pmid": "19114008",
                    "title": "WGCNA: an R package for weighted correlation network analysis.",
                    "abstract": "Weighted correlation network analysis identifies modules.",
                    "doi": "10.1186/1471-2105-9-559",
                }
            ],
            "source_ledger": [{"pmid": "19114008"}],
        },
    }

    runtime = _RecordingSkillRuntime(
        lambda _call: _skill_result(
            "skill.pubmed.literature.query",
            {
                "resource": {"resource_id": "resource:conversation-file:1"},
                "message": "完整机器结果位于 resource；当前 data 仅为引用。",
            },
            contract_data=provider_contract,
        )
    )
    wrapped = retrieval_subagent._build_safe_database_tools(
        skill_runtime=runtime,
        available_tool_ids={"skill_pubmed_literature_query"},
    )["safe_query_pubmed"]

    invoked = retrieval_subagent._invoke_retrieval_tool(
        wrapped,
        task="PMID 19114008",
        max_evidence=3,
    )
    payload = invoked["output"]

    assert payload["returned_ids"] == ["19114008"]
    assert payload["not_found_ids"] == []
    assert payload["records"][0]["title"].startswith("WGCNA:")
    assert payload["provider_observation"]["coverage"] == "exhausted"
    assert "model_summary" not in payload
    assert retrieval_subagent._provider_observations_from_tool_results([invoked]) == [
        {
            "provider": "pubmed",
            "query_mode": "exact_pmid",
            "requested_ids": ["19114008"],
            "returned_ids": ["19114008"],
            "not_found_ids": [],
            "unresolved_ids": [],
            "coverage": "exhausted",
            "coverage_scope": "PubMed exact PMID record lookup for requested_ids=[19114008] only",
        }
    ]


def test_pubmed_explicit_doi_uses_exact_field_and_keeps_only_matching_record(
    monkeypatch,
) -> None:
    doi = "10.1186/1471-2105-9-559"
    def respond(call):
        assert call["capability_id"] == "skill.pubmed.literature.query"
        assert call["params"] == {
            "operation": "search_pubmed",
            "query": f"{doi}[DOI]",
            "max_records": 5,
            "sort": "relevance",
        }
        return _skill_result(
            "skill.pubmed.literature.query",
            {
                "result": {
                    "provider": "ncbi_eutils",
                    "database": "pubmed",
                    "records": [
                {
                    "uid": "19114008",
                    "pmid": "19114008",
                    "title": "WGCNA: an R package for weighted correlation network analysis.",
                    "abstract": "A weighted correlation network analysis method.",
                    "articleids": [
                        {"idtype": "doi", "value": doi.upper()},
                        {"idtype": "pubmed", "value": "19114008"},
                    ],
                }
                    ],
                },
            },
        )

    runtime = _RecordingSkillRuntime(respond)

    payload = _raw_safe_tool("safe_query_pubmed", runtime)(
        query=f"Verify DOI {doi} in PubMed.",
        max_results=5,
    )

    assert len(runtime.calls) == 1
    assert payload["returned_ids"] == [doi]
    assert payload["not_found_ids"] == []
    assert payload["unresolved_ids"] == []
    assert [item["pmid"] for item in payload["records"]] == ["19114008"]
    sources = retrieval_subagent._authoritative_sources_from_tool_results(
        [{"tool": "safe_query_pubmed", "query": f"DOI {doi}", "output": payload}]
    )
    assert [item["doi"] for item in sources] == [doi]


def test_pubmed_invalid_doi_drops_lexical_false_positive_and_has_zero_evidence(
    monkeypatch,
) -> None:
    requested_doi = "10.9999/evoengine-source-test"
    unrelated_doi = "10.2147/jir.s350109"

    runtime = _RecordingSkillRuntime(
        lambda _call: _skill_result(
            "skill.pubmed.literature.query",
            {
                "result": {
                    "provider": "ncbi_eutils",
                    "database": "pubmed",
                    "records": [
                        {
                            "uid": "35010900",
                            "pmid": "35010900",
                            "title": "An unrelated lexical search result.",
                            "abstract": "This record does not have the requested DOI.",
                            "doi": unrelated_doi,
                        }
                    ],
                },
                # Formal search projection occurs before the adapter's exact
                # DOI identity filter. This unrelated ledger must not escape.
                "source_ledger": [
                    {"pmid": "35010900", "doi": unrelated_doi}
                ],
            },
        )
    )

    payload = _raw_safe_tool("safe_query_pubmed", runtime)(
        query=f"Verify DOI {requested_doi} in PubMed.",
        max_results=5,
    )

    assert payload["records"] == []
    assert payload["returned_ids"] == []
    assert payload["not_found_ids"] == [requested_doi]
    assert payload["unresolved_ids"] == []
    assert payload["coverage"] == "exhausted"
    assert "source_ledger" not in payload
    pack, _warnings = retrieval_subagent._fallback_evidence_pack_from_tool_results(
        task=f"Verify DOI {requested_doi}",
        tool_results=[
            {
                "tool": "safe_query_pubmed",
                "query": f"Verify DOI {requested_doi}",
                "output": payload,
            }
        ],
        max_evidence=5,
    )
    assert pack.evidence == []
    assert pack.claims == []


def test_pubmed_exact_doi_transient_failure_is_unresolved_not_not_found(
    monkeypatch,
) -> None:
    doi = "10.9999/evoengine-temporary-test"
    runtime = _RecordingSkillRuntime(
        lambda _call: _skill_result(
            "skill.pubmed.literature.query",
            {},
            ok=False,
            error_kind="provider_unavailable",
        )
    )

    payload = _raw_safe_tool("safe_query_pubmed", runtime)(
        query=f"Verify DOI {doi} in PubMed.",
        max_results=1,
    )

    assert payload["records"] == []
    assert payload["returned_ids"] == []
    assert payload["not_found_ids"] == []
    assert payload["unresolved_ids"] == [doi]
    assert payload["coverage"] == "unknown"
    assert payload["coverage_scope"] == ""


def test_pubmed_exact_doi_invalid_esearch_payload_is_unresolved_not_not_found(
    monkeypatch,
) -> None:
    doi = "10.9999/evoengine-invalid-payload"
    runtime = _RecordingSkillRuntime(
        lambda _call: _skill_result(
            "skill.pubmed.literature.query",
            {},
            ok=False,
            error_kind="contract_violation",
        )
    )
    payload = _raw_safe_tool("safe_query_pubmed", runtime)(
        query=f"Verify DOI {doi} in PubMed.",
        max_results=1,
    )

    assert payload["records"] == []
    assert payload["returned_ids"] == []
    assert payload["not_found_ids"] == []
    assert payload["unresolved_ids"] == [doi]
    assert payload["coverage"] == "unknown"


def test_pubmed_authoritative_projection_rechecks_requested_doi_identity() -> None:
    requested_doi = "10.9999/evoengine-source-test"
    sources = retrieval_subagent._authoritative_sources_from_tool_results(
        [
            {
                "tool": "safe_query_pubmed",
                "query": f"Verify DOI {requested_doi}",
                "output": {
                    "requested_ids": [requested_doi],
                    "records": [
                        {
                            "uid": "35010900",
                            "title": "Unrelated PubMed record",
                            "articleids": [
                                {"idtype": "doi", "value": "10.2147/jir.s350109"}
                            ],
                        }
                    ],
                },
            }
        ]
    )

    assert sources == []


def test_pubmed_authoritative_projection_rechecks_requested_pmid_identity() -> None:
    sources = retrieval_subagent._authoritative_sources_from_tool_results(
        [
            {
                "tool": "safe_query_pubmed",
                "query": "Fetch PMID 19114008 exactly.",
                "output": {
                    "requested_ids": ["19114008"],
                    "records": [
                        {
                            "uid": "19114008",
                            "pmid": "19114008",
                            "title": "WGCNA",
                            "abstract": "Weighted correlation network analysis.",
                        },
                        {
                            "uid": "99999999",
                            "pmid": "99999999",
                            "title": "Unrequested record",
                            "abstract": "This record must not enter evidence.",
                        },
                    ],
                },
            }
        ]
    )

    assert [(item["pmid"], item["title"]) for item in sources] == [
        ("19114008", "WGCNA")
    ]


def test_crossref_exact_lookup_classifies_mixed_found_and_404(monkeypatch) -> None:
    valid_doi = "10.1186/1471-2105-9-559"
    invalid_doi = "10.9999/evoengine-source-test"
    script_payload = {
        "ok": True,
        "source_type": "paper",
        "provider": "crossref",
        "query": f"Verify {valid_doi} and {invalid_doi}.",
        "records": [
            {
                "doi": valid_doi,
                "title": "WGCNA: an R package for weighted correlation network analysis",
                "publisher": "Springer Science and Business Media LLC",
                "published": "2008-12-29",
                "official_url": f"https://doi.org/{valid_doi}",
            }
        ],
        "raw_records": [{"DOI": valid_doi}],
        "requested_ids": [valid_doi, invalid_doi],
        "returned_ids": [valid_doi],
        "not_found_ids": [invalid_doi],
        "unresolved_ids": [],
        "failures": [],
        "coverage": "exhausted",
        "coverage_scope": (
            f"CrossRef exact DOI identity lookup for requested_ids=[{valid_doi},{invalid_doi}] only"
        ),
        "recovery": "do_not_retry",
        "provider_observation": {
            "provider": "crossref",
            "query_mode": "exact_doi",
            "requested_ids": [valid_doi, invalid_doi],
            "returned_ids": [valid_doi],
            "not_found_ids": [invalid_doi],
            "unresolved_ids": [],
            "coverage": "exhausted",
            "coverage_scope": (
                f"CrossRef exact DOI identity lookup for requested_ids=[{valid_doi},{invalid_doi}] only"
            ),
        },
    }

    def responder(kwargs):
        assert kwargs["capability_id"] == "skill.crossref.doi.identity"
        return _skill_result(
            capability_id="skill.crossref.doi.identity",
            data={"result": script_payload},
            contract_data={"result": script_payload},
            ok=True,
        )

    payload = _raw_safe_tool(
        "safe_query_crossref",
        skill_runtime=_RecordingSkillRuntime(responder),
    )(
        query=f"Verify {valid_doi} and {invalid_doi}.",
        max_results=2,
    )

    assert payload["requested_ids"] == [valid_doi, invalid_doi]
    assert payload["returned_ids"] == [valid_doi]
    assert payload["not_found_ids"] == [invalid_doi]
    assert payload["unresolved_ids"] == []
    assert payload["coverage"] == "exhausted"
    assert payload["recovery"] == "do_not_retry"
    assert valid_doi in payload["coverage_scope"]
    assert invalid_doi in payload["coverage_scope"]
    assert payload["capability_calls"][0]["capability_id"] == (
        "skill.crossref.doi.identity"
    )

    pack, _warnings = retrieval_subagent._fallback_evidence_pack_from_tool_results(
        task="CrossRef exact DOI identities",
        tool_results=[
            {
                "tool": "safe_query_crossref",
                "query": "exact DOIs",
                "output": payload,
            }
        ],
        max_evidence=5,
    )
    assert [item.doi for item in pack.evidence] == [valid_doi]
    assert pack.claims[0].support_level == "primary"
    assert invalid_doi not in json.dumps(pack.model_dump(mode="json"), ensure_ascii=False)
    observations = retrieval_subagent._provider_observations_from_tool_results(
        [{"tool": "safe_query_crossref", "query": "exact DOIs", "output": payload}]
    )
    outcome = retrieval_subagent._crossref_scope_outcome_fields(observations)
    assert observations[0]["not_found_ids"] == [invalid_doi]
    assert outcome["coverage"] == "exhausted"
    assert outcome["recovery"] == "do_not_retry"


def test_exact_doi_not_found_drops_unrelated_web_fallback_from_evidence_pack() -> None:
    invalid_doi = "10.9999/evoengine-source-test"
    crossref = {
        "ok": True,
        "source_type": "paper",
        "provider": "crossref",
        "records": [],
        "requested_ids": [invalid_doi],
        "returned_ids": [],
        "not_found_ids": [invalid_doi],
        "unresolved_ids": [],
        "coverage": "exhausted",
        "provider_observation": {
            "provider": "crossref",
            "query_mode": "exact_doi",
            "requested_ids": [invalid_doi],
            "returned_ids": [],
            "not_found_ids": [invalid_doi],
            "unresolved_ids": [],
            "coverage": "exhausted",
            "coverage_scope": "CrossRef exact DOI identity lookup for the requested ID only",
        },
    }
    tool_results = [
        {"tool": "safe_query_crossref", "query": invalid_doi, "output": crossref},
        {
            "tool": "web_search",
            "query": invalid_doi,
            "output": {
                "ok": True,
                "items": [
                    {
                        "title": "Unrelated vertigo article",
                        "url": "https://example.org/vertigo",
                        "snippet": "This result does not mention the requested DOI.",
                    },
                    {
                        "title": "Unrelated battery researcher",
                        "url": "https://example.org/researcher",
                        "snippet": "This result is unrelated to DOI registration.",
                    },
                ],
            },
        },
    ]

    pack, _warnings = retrieval_subagent._fallback_evidence_pack_from_tool_results(
        task=f"Verify DOI {invalid_doi}",
        tool_results=tool_results,
        max_evidence=5,
    )
    observations = retrieval_subagent._provider_observations_from_tool_results(tool_results)

    assert pack.claims == []
    assert pack.evidence == []
    assert observations[0]["requested_ids"] == [invalid_doi]
    assert observations[0]["not_found_ids"] == [invalid_doi]
    assert observations[0]["coverage"] == "exhausted"


def test_crossref_transient_failure_is_not_not_found_or_exhausted(monkeypatch) -> None:
    doi = "10.9999/evoengine-temporary-test"
    script_payload = {
        "ok": False,
        "source_type": "paper",
        "provider": "crossref",
        "query": f"Verify DOI {doi}.",
        "records": [],
        "raw_records": [],
        "requested_ids": [doi],
        "returned_ids": [],
        "not_found_ids": [],
        "unresolved_ids": [doi],
        "failures": [{"doi": doi, "http_status": 503}],
        "coverage": "unknown",
        "coverage_scope": "",
        "recovery": "retry_same_call",
        "provider_observation": {
            "provider": "crossref",
            "query_mode": "exact_doi",
            "requested_ids": [doi],
            "returned_ids": [],
            "not_found_ids": [],
            "unresolved_ids": [doi],
            "coverage": "unknown",
            "coverage_scope": "",
        },
    }

    def responder(kwargs):
        return _skill_result(
            capability_id="skill.crossref.doi.identity",
            data={"result": script_payload},
            contract_data={"result": script_payload},
            ok=True,
        )

    payload = _raw_safe_tool(
        "safe_query_crossref",
        skill_runtime=_RecordingSkillRuntime(responder),
    )(
        query=f"Verify DOI {doi}.",
        max_results=1,
    )

    assert payload["returned_ids"] == []
    assert payload["not_found_ids"] == []
    assert payload["unresolved_ids"] == [doi]
    assert payload["coverage"] == "unknown"
    assert payload["coverage_scope"] == ""
    assert payload["recovery"] == "retry_same_call"
    observations = retrieval_subagent._provider_observations_from_tool_results(
        [{"tool": "safe_query_crossref", "query": doi, "output": payload}]
    )
    outcome = retrieval_subagent._crossref_scope_outcome_fields(observations)
    assert "coverage" not in outcome
    assert outcome["recovery"] == "retry_same_call"


def test_safe_query_uniprot_projects_required_p69905_fields(monkeypatch) -> None:
    function_text = (
        "Involved in oxygen transport from the lung to the various peripheral tissues."
    )
    formal_record = {
        "provider": "uniprot_rest",
        "database": "UniProtKB",
        "accession": "P69905",
        "primary_accession": "P69905",
        "uni_prot_kb_id": "HBA_HUMAN",
        "entry_name": "HBA_HUMAN",
        "protein_name": "Hemoglobin subunit alpha",
        "recommended_name": "Hemoglobin subunit alpha",
        "gene_name": "HBA1",
        "gene_names": ["HBA1", "HBA2"],
        "organism": "Homo sapiens",
        "function": function_text,
        "sequence_length": 142,
        "go_terms": [],
        "source_url": "https://www.uniprot.org/uniprotkb/P69905/entry",
        "official_url": "https://www.uniprot.org/uniprotkb/P69905/entry",
        "snippet": "formal provider projection",
    }
    registered_source = {
        "type": "data",
        "source_type": "bio_database",
        "provider": "uniprot_rest",
        "provider_record_id": "P69905",
        "title": "Hemoglobin subunit alpha",
        "url": "https://www.uniprot.org/uniprotkb/P69905/entry",
        "source_registration": {
            "schema_version": "evoengine.source-registration/v1",
            "status": "registered",
            "authority": "capability_runtime",
            "capability_id": "skill.uniprot.query",
            "capability_version": "1.0.0",
            "candidate_id": "source-candidate:" + "a" * 64,
            "result_resource_id": "resource:conversation-file:71",
        },
    }
    sidecar = ResourceRef(
        resource_id="resource:conversation-file:72",
        uri="conversation-file://72",
        kind="source_sidecar",
        storage="conversation_file",
        media_type="application/json",
        size_bytes=128,
        sha256="b" * 64,
        conversation_file_id=72,
    )

    def respond(call):
        assert call["capability_id"] == "skill.uniprot.query"
        assert call["params"] == {
            "accessions": ["P69905"],
            "include_sequence": False,
        }
        assert call["request_id"] == "req-source"
        assert call["project_id"] == "project-source"
        assert call["conversation_id"] == "conv-source"
        assert call["user_id"] == 17
        return _skill_result(
            "skill.uniprot.query",
            {
                "ok": True,
                "operation": "entries",
                "provider": "uniprot_rest",
                "database": "UniProtKB",
                "requested_ids": ["P69905"],
                "returned_ids": ["P69905"],
                "not_found_ids": [],
                "unresolved_ids": [],
                "records": [formal_record],
                "failures": [],
                "source_ledger": [{"provider_record_id": "P69905"}],
            },
            citation_projection=[registered_source],
            source_sidecar_resource=sidecar,
        )

    runtime = _RecordingSkillRuntime(respond)
    wrapped = retrieval_subagent._build_safe_database_tools(
        skill_runtime=runtime,
        available_tool_ids={"skill_uniprot_query"},
        request_id="req-source",
        project_id="project-source",
        conversation_id="conv-source",
        user_id=17,
    )["safe_query_uniprot"]
    raw_tool = getattr(wrapped, "_evo_raw_tool", wrapped).func

    payload = raw_tool(query="P69905", max_results=5)
    record = payload["records"][0]

    assert len(runtime.calls) == 1
    assert payload["requested_ids"] == ["P69905"]
    assert payload["returned_ids"] == ["P69905"]
    assert payload["not_found_ids"] == []
    assert payload["unresolved_ids"] == []
    assert payload["coverage"] == "exhausted"
    assert payload["provider_observation"]["query_mode"] == "exact_accession"
    assert len(payload["records"]) == 1
    assert record["accession"] == "P69905"
    assert record["entry_name"] == "HBA_HUMAN"
    assert record["recommended_name"] == "Hemoglobin subunit alpha"
    assert record["gene_names"] == ["HBA1", "HBA2"]
    assert record["organism"] == "Homo sapiens"
    assert "oxygen transport" in record["function"]
    assert record["official_url"] == "https://www.uniprot.org/uniprotkb/P69905/entry"
    assert "FUNCTION=Involved in oxygen transport" in record["snippet"]
    assert payload["registered_source_ledger"] == [registered_source]
    assert payload["source_ledger"] == [registered_source]
    assert payload["source_sidecar_refs"] == [sidecar.resource_id]
    assert payload["source_sidecar_resources"] == [
        sidecar.model_dump(mode="json", exclude_none=True)
    ]
    pack, _warnings = retrieval_subagent._fallback_evidence_pack_from_tool_results(
        task="UniProt P69905",
        tool_results=[
            {
                "tool": "safe_query_uniprot",
                "query": "P69905",
                "output": payload,
            }
        ],
        max_evidence=1,
    )
    evidence = pack.evidence[0]
    assert evidence.title == "Hemoglobin subunit alpha"
    assert "gene_names=HBA1, HBA2" in evidence.snippet
    assert "oxygen transport" in evidence.snippet
    assert evidence.url == "https://www.uniprot.org/uniprotkb/P69905/entry"
    assert evidence.metadata["record_fields"]["entry_name"] == "HBA_HUMAN"


def test_safe_query_opentarget_delegates_and_preserves_all_entity_hits(
    monkeypatch,
) -> None:
    records = [
        {
            "id": "ENSG00000146648",
            "entity": "target",
            "name": "EGFR",
            "description": "Epidermal growth factor receptor",
            "official_url": (
                "https://platform.opentargets.org/target/ENSG00000146648"
            ),
        },
        {
            "id": "EFO_0000270",
            "entity": "disease",
            "name": "asthma",
            "description": "A chronic inflammatory airway disease",
            "official_url": (
                "https://platform.opentargets.org/disease/EFO_0000270"
            ),
        },
    ]

    def respond(call):
        assert call["capability_id"] == "skill.open_targets.query"
        assert call["params"] == {
            "operation": "search_open_targets",
            "query": "EGFR asthma",
            "entity_type": "all",
            "max_records": 5,
        }
        return _skill_result(
            "skill.open_targets.query",
            {
                "ok": True,
                "operation": "search_open_targets",
                "result": {
                    "provider": "open_targets_graphql",
                    "database": "Open Targets",
                    "query": "EGFR asthma",
                    "entity_type": "all",
                    "records": records,
                    "returned_count": 2,
                    "capability_outcome": {
                        "coverage": "unknown",
                        "coverage_scope": "",
                    },
                    "source_ledger": [
                        {"provider_record_id": "ENSG00000146648"},
                        {"provider_record_id": "EFO_0000270"},
                    ],
                },
            },
        )

    runtime = _RecordingSkillRuntime(respond)
    monkeypatch.setattr(
        retrieval_subagent,
        "safe_http_post_provider",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("retrieval adapter must not call Open Targets directly")
        ),
        raising=False,
    )
    tools = retrieval_subagent._build_safe_database_tools(
        skill_runtime=runtime,
        available_tool_ids={"skill_open_targets_query"},
        request_id="req-open-targets",
        project_id="project-open-targets",
        conversation_id="conv-open-targets",
        user_id=17,
    )

    assert "safe_query_opentarget" in tools
    raw_tool = getattr(
        tools["safe_query_opentarget"],
        "_evo_raw_tool",
        tools["safe_query_opentarget"],
    ).func
    payload = raw_tool(query="EGFR asthma", max_results=5)

    assert len(runtime.calls) == 1
    assert payload["ok"] is True
    assert payload["entity_type"] == "all"
    assert payload["records"] == records
    assert [item["entity"] for item in payload["records"]] == [
        "target",
        "disease",
    ]
    assert payload["provider_observation"]["returned_ids"] == [
        "ENSG00000146648",
        "EFO_0000270",
    ]
    assert payload["source_ledger"] == [
        {"provider_record_id": "ENSG00000146648"},
        {"provider_record_id": "EFO_0000270"},
    ]


def test_safe_query_uniprot_404_is_not_a_temporary_failure(monkeypatch) -> None:
    runtime = _RecordingSkillRuntime(
        lambda _call: _skill_result(
            "skill.uniprot.query",
            {
                "ok": True,
                "operation": "entries",
                "provider": "uniprot_rest",
                "database": "UniProtKB",
                "requested_ids": ["P69905"],
                "returned_ids": [],
                "not_found_ids": ["P69905"],
                "unresolved_ids": [],
                "records": [],
                "failures": [],
            },
        )
    )

    payload = _raw_safe_tool("safe_query_uniprot", runtime)(
        query="P69905", max_results=5
    )

    assert payload["ok"] is True
    assert payload["records"] == []
    assert payload["returned_ids"] == []
    assert payload["not_found_ids"] == ["P69905"]
    assert payload["unresolved_ids"] == []
    assert payload["coverage"] == "exhausted"


def test_safe_query_uniprot_temporary_failure_is_unresolved(monkeypatch) -> None:
    runtime = _RecordingSkillRuntime(
        lambda _call: _skill_result(
            "skill.uniprot.query",
            {
                "ok": False,
                "operation": "entries",
                "provider": "uniprot_rest",
                "database": "UniProtKB",
                "requested_ids": ["P69905"],
                "returned_ids": [],
                "not_found_ids": [],
                "unresolved_ids": ["P69905"],
                "records": [],
                "failures": [
                    {"accession": "P69905", "error_type": "TimeoutError"}
                ],
            },
            ok=False,
            error_kind="provider_unavailable",
        )
    )

    payload = _raw_safe_tool("safe_query_uniprot", runtime)(
        query="P69905", max_results=5
    )

    assert payload["ok"] is False
    assert payload["records"] == []
    assert payload["not_found_ids"] == []
    assert payload["unresolved_ids"] == ["P69905"]
    assert payload["coverage"] == "unknown"


def test_register_retrieval_temp_files_uses_temporary_drawer_section(monkeypatch):
    captured = {}

    def fake_batch(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "registered_count": len(kwargs["files"]), "results": []}

    monkeypatch.setattr(retrieval_subagent.conversation_file_registry, "INTERNAL_TOKEN", "token")
    monkeypatch.setattr(retrieval_subagent.conversation_file_registry, "register_conversation_file_batch", fake_batch)
    pack = EvidencePack(
        query="qPCR",
        claims=[],
        evidence=[
            {
                "evidence_id": "E1",
                "source_type": "pubmed",
                "title": "Paper",
                "snippet": "snippet",
            }
        ],
        limitations=[],
    )

    payload = retrieval_subagent._register_retrieval_temp_files(
        project_id="project-1",
        conversation_id="conversation-1",
        user_id=7,
        request_id="req-1",
        trace_ref="trace-1",
        trace_payload={"task": "qPCR", "warnings": []},
        evidence_pack=pack,
        reference_records=[{"ref_id": "E1", "title": "Paper"}],
        tool_results=[
            {
                "tool": "safe_query_pubmed",
                "query": "qPCR",
                "output": {"raw_ref": "raw-1", "records": [{"title": "Paper"}]},
            }
        ],
    )

    assert payload["ok"] is True
    assert captured["project_id"] == "project-1"
    assert captured["conversation_id"] == "conversation-1"
    assert captured["request_id"] == "req-1"
    assert len(captured["files"]) == 4
    assert {item["source_type"] for item in captured["files"]} == {
        "retrieval_trace",
        "retrieval_reference",
        "tool_raw_result",
    }
    assert all(item["drawer_section"] == "temporary_output" for item in captured["files"])
    assert all(item["trace_ref"] == "trace-1" for item in captured["files"])
    assert all(item["tool_name"] == "run_retrieval_subagent" for item in captured["files"])


def test_fallback_evidence_pack_from_pubmed_tool_results():
    provider_output = {
            "ok": True,
            "records": [
                {
                    "uid": "19246619",
                    "title": "The MIQE guidelines: minimum information for publication of quantitative real-time PCR experiments.",
                    "source": "Clin Chem",
                    "fulljournalname": "Clinical chemistry",
                    "pubdate": "2009 Apr",
                    "articleids": [
                        {"idtype": "pubmed", "value": "19246619"},
                        {"idtype": "doi", "value": "10.1373/clinchem.2008.112797"},
                    ],
                }
            ],
        }

    pack, warnings = retrieval_subagent._fallback_evidence_pack_from_tool_results(
        task="MIQE guideline qPCR",
        tool_results=[
            {
                "tool": "safe_query_pubmed",
                "query": "MIQE guideline qPCR",
                "output": {"raw_ref": "raw-1", **provider_output},
            }
        ],
        max_evidence=2,
    )

    assert warnings == ["fallback_evidence_pack_from_tool_results"]
    assert pack.claims
    assert len(pack.evidence) == 1
    assert pack.evidence[0].pmid == "19246619"
    assert pack.evidence[0].doi == "10.1373/clinchem.2008.112797"
    assert pack.evidence[0].url == "https://pubmed.ncbi.nlm.nih.gov/19246619/"
    assert pack.claims[0].support_level == "primary"
    identity_claim = next(
        item for item in pack.claims if item.claim_id == "C-ID-1"
    )
    assert identity_claim.claim == (
        "PMID=19246619; DOI=10.1373/clinchem.2008.112797; "
        "title=The MIQE guidelines: minimum information for publication of "
        "quantitative real-time PCR experiments."
    )
    assert identity_claim.claim in pack.evidence[0].snippet


def test_fallback_evidence_pack_preserves_structured_uniprot_record():
    provider_output = {
                "ok": True,
                "source_type": "bio_database",
                "database": "uniprot",
                "records": [
                    {
                        "primaryAccession": "P69905",
                        "proteinDescription": {
                            "recommendedName": {
                                "fullName": {"value": "Hemoglobin subunit alpha"}
                            }
                        },
                        "genes": [{"geneName": {"value": "HBA1"}}],
                        "comments": [
                            {
                                "commentType": "FUNCTION",
                                "texts": [{"value": "Involved in oxygen transport"}],
                            }
                        ],
                    }
                ],
            }

    pack, warnings = retrieval_subagent._fallback_evidence_pack_from_tool_results(
        task="UniProt P69905",
        tool_results=[
            {
                "tool": "safe_query_uniprot",
                "query": "P69905",
                "output": {"raw_ref": "raw-1", **provider_output},
            }
        ],
        max_evidence=2,
    )

    assert warnings == ["fallback_evidence_pack_from_tool_results"]
    assert len(pack.evidence) == 1
    evidence = pack.evidence[0]
    assert evidence.provider == "uniprot"
    assert evidence.provider_record_id == "P69905"
    assert evidence.title == "Hemoglobin subunit alpha"
    assert "HBA1" in evidence.snippet
    assert "oxygen transport" in evidence.snippet
    assert pack.claims[0].support_level == "database"
    assert pack.claims[0].claim in evidence.snippet


def test_web_fallback_does_not_claim_query_is_satisfied() -> None:
    pack, warnings = retrieval_subagent._fallback_evidence_pack_from_tool_results(
        task="invalid DOI",
        tool_results=[
            {
                "tool": "web_search",
                "query": "invalid DOI",
                "output": {
                    "results": [
                        {
                            "title": "An unrelated reachable page",
                            "url": "https://example.org/unrelated",
                            "snippet": "This page is reachable but does not establish the requested DOI.",
                        }
                    ]
                },
            }
        ],
        max_evidence=2,
    )

    assert warnings == ["fallback_evidence_pack_from_tool_results"]
    assert pack.evidence
    assert pack.claims[0].support_level == "source_fallback"
    assert retrieval_subagent._evidence_pack_is_sufficient(pack) is False


def test_fallback_evidence_pack_keeps_milvus_grounded_content() -> None:
    provider_output = {
                "results": [
                    {
                        "asset_id": 42,
                        "file_name": "paper.pdf",
                        "page_idx": 3,
                        "block_idx": 17,
                        "chunk_id": "42:17",
                        "content": "A grounded method span.",
                    }
                ]
            }

    pack, warnings = retrieval_subagent._fallback_evidence_pack_from_tool_results(
        task="method",
        tool_results=[
            {
                "tool": "search_milvus_knowledge",
                "query": "method",
                "output": {"raw_ref": "raw-1", **provider_output},
            }
        ],
        max_evidence=2,
    )

    assert warnings == ["fallback_evidence_pack_from_tool_results"]
    assert pack.evidence[0].source_type == "knowledge_asset"
    assert pack.evidence[0].title == "paper.pdf"
    assert pack.evidence[0].snippet == "A grounded method span."
    assert pack.evidence[0].metadata["chunk_id"] == "42:17"


def test_evidence_pack_sufficiency_requires_every_claim_to_reference_returned_evidence():
    supported = EvidencePack(
        query="two targets",
        claims=[
            {
                "claim_id": "C1",
                "claim": "Target one is covered.",
                "support_level": "database",
                "evidence_ids": ["E1"],
            },
            {
                "claim_id": "C2",
                "claim": "Target two is covered.",
                "support_level": "database",
                "evidence_ids": ["E2"],
            },
        ],
        evidence=[
            {"evidence_id": "E1", "title": "Source one"},
            {"evidence_id": "E2", "title": "Source two"},
        ],
    )
    incomplete = EvidencePack(
        query="two targets",
        claims=[
            {
                "claim_id": "C1",
                "claim": "Target one is covered.",
                "support_level": "database",
                "evidence_ids": ["E1"],
            },
            {
                "claim_id": "C2",
                "claim": "Target two was not found.",
                "support_level": "database",
                "evidence_ids": [],
            },
        ],
        evidence=[{"evidence_id": "E1", "title": "Source one"}],
    )

    assert retrieval_subagent._evidence_pack_is_sufficient(supported) is True
    assert retrieval_subagent._evidence_pack_is_sufficient(incomplete) is False


def test_evidence_pack_sufficiency_rejects_fallback_and_explicit_inability():
    fallback = EvidencePack(
        query="target evidence",
        claims=[
            {
                "claim_id": "C1",
                "claim": "Sources were found.",
                "support_level": "source_fallback",
                "evidence_ids": ["E1"],
            }
        ],
        evidence=[{"evidence_id": "E1", "title": "Candidate source"}],
        limitations=["Evidence Pack 由工具结果兜底构造。"],
    )
    unable = EvidencePack(
        query="target evidence",
        claims=[
            {
                "claim_id": "C1",
                "claim": "The record exists.",
                "support_level": "database",
                "evidence_ids": ["E1"],
            }
        ],
        evidence=[{"evidence_id": "E1", "title": "Candidate source"}],
        limitations=["The requested evidence was not retrieved, so the question cannot answer from this record."],
    )

    assert retrieval_subagent._evidence_pack_is_sufficient(fallback) is False
    assert retrieval_subagent._evidence_pack_is_sufficient(unable) is False


def test_requested_minimum_evidence_count_understands_per_item_source_requirement():
    task = "概括两个主要限制，每条限制给出一个可核验文献来源。"

    assert retrieval_subagent._requested_minimum_evidence_count(task) == 2
    assert retrieval_subagent._requested_minimum_evidence_count("Give three sources about qPCR") == 3


def test_evidence_pack_sufficiency_honors_minimum_distinct_source_count():
    pack = EvidencePack(
        query="two sources",
        claims=[
            {
                "claim_id": "C1",
                "claim": "Claim one.",
                "support_level": "review",
                "evidence_ids": ["E1"],
            },
            {
                "claim_id": "C2",
                "claim": "Claim two.",
                "support_level": "review",
                "evidence_ids": ["E1"],
            },
        ],
        evidence=[{"evidence_id": "E1", "title": "One source"}],
    )

    assert retrieval_subagent._evidence_pack_is_sufficient(pack, minimum_evidence_count=1) is True
    assert retrieval_subagent._evidence_pack_is_sufficient(pack, minimum_evidence_count=2) is False


def test_formal_pubmed_efetch_returns_abstract_records(monkeypatch):
    import importlib.util
    from pathlib import Path

    script_path = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "pubmed-literature-retrieval"
        / "scripts"
        / "query_pubmed_literature.py"
    )
    spec = importlib.util.spec_from_file_location("formal_pubmed_query_test", script_path)
    assert spec is not None and spec.loader is not None
    pubmed_query = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pubmed_query)

    xml = """<?xml version='1.0'?>
    <PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>12345678</PMID><Article>
      <ArticleTitle>Prime editing review.</ArticleTitle>
      <Abstract><AbstractText>Delivery remains challenging.</AbstractText></Abstract>
      <Journal><JournalIssue><PubDate><Year>2025</Year></PubDate></JournalIssue><Title>Example Journal</Title></Journal>
      <AuthorList><Author><LastName>Chen</LastName><ForeName>Lin</ForeName></Author></AuthorList>
      <PublicationTypeList><PublicationType>Review</PublicationType></PublicationTypeList>
    </Article></MedlineCitation><PubmedData><ArticleIdList>
      <ArticleId IdType='pubmed'>12345678</ArticleId><ArticleId IdType='doi'>10.1000/example</ArticleId>
    </ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>"""

    class Response:
        ok = True
        status_code = 200
        redacted_url = "https://eutils.ncbi.nlm.nih.gov/efetch"
        text = xml

        @staticmethod
        def to_trace_metadata():
            return {"status_code": 200}

    monkeypatch.setattr(pubmed_query, "safe_http_get_provider", lambda *_args, **_kwargs: Response())

    records = pubmed_query._fetch_pubmed_details(["12345678"])

    assert records[0]["pmid"] == "12345678"
    assert records[0]["abstract"] == "Delivery remains challenging."
    assert records[0]["doi"] == "10.1000/example"


def test_evidence_pack_sources_are_reconciled_from_provider_results():
    provider_output = {
                "records": [
                    {
                        "uid": "12345678",
                        "title": "Authoritative title.",
                        "abstract": "Authoritative abstract.",
                        "fulljournalname": "Example Journal",
                        "pubdate": "2026 Jan",
                        "articleids": [
                            {"idtype": "pubmed", "value": "12345678"},
                            {"idtype": "doi", "value": "10.1000/example"},
                        ],
                    }
                ]
            }
    pack = EvidencePack(
        query="target",
        claims=[
            {
                "claim_id": "C1",
                "claim": "Supported claim.",
                "support_level": "review",
                "evidence_ids": ["E1", "E2"],
            }
        ],
        evidence=[
            {
                "evidence_id": "E1",
                "source_type": "pubmed",
                "title": "Authoritative title.",
                "pmid": "12345678",
                "year": "2025",
                "snippet": "Model-written snippet.",
            },
            {
                "evidence_id": "E2",
                "source_type": "pubmed",
                "title": "Invented source",
                "pmid": "99999999",
            },
        ],
    )

    reconciled, warnings = retrieval_subagent._reconcile_evidence_pack_sources(
        pack,
        [
            {
                "tool": "safe_query_pubmed",
                "query": "target",
                "output": {"raw_ref": "raw-1", **provider_output},
            }
        ],
    )

    assert warnings == ["unverified_evidence_removed:1"]
    assert len(reconciled.evidence) == 1
    assert reconciled.evidence[0].year == "2026"
    assert reconciled.evidence[0].doi == "10.1000/example"
    assert reconciled.evidence[0].provider == "pubmed"
    assert reconciled.evidence[0].provider_record_id == "12345678"
    assert reconciled.evidence[0].snippet == (
        "Authoritative abstract.\n\n"
        "PMID=12345678; DOI=10.1000/example; title=Authoritative title."
    )
    assert reconciled.evidence[0].metadata["retrieval_reconciled"] is True
    assert "source_verified" not in reconciled.evidence[0].metadata
    assert reconciled.claims[0].evidence_ids == ["E1"]


def test_authoritative_database_sources_keep_stable_record_ids() -> None:
    sources = retrieval_subagent._authoritative_sources_from_tool_results(
        [
            {
                "tool": "safe_query_uniprot",
                "query": "TP53",
                "output": {
                    "source_type": "bio_database",
                    "database": "uniprot",
                    "records": [
                        {
                            "primaryAccession": "P04637",
                            "proteinDescription": {
                                "recommendedName": {"fullName": {"value": "Cellular tumor antigen p53"}}
                            },
                        }
                    ],
                },
            },
            {
                "tool": "safe_query_ensembl",
                "query": "TP53",
                "output": {
                    "source_type": "bio_database",
                    "database": "ensembl",
                    "record": {
                        "id": "ENSG00000141510",
                        "display_name": "TP53",
                        "description": "tumor protein p53",
                    },
                },
            },
        ]
    )

    by_provider = {item["provider"]: item for item in sources}
    assert by_provider["uniprot"]["provider_record_id"] == "P04637"
    assert by_provider["uniprot"]["url"].endswith("/P04637/entry")
    assert by_provider["ensembl"]["provider_record_id"] == "ENSG00000141510"
    assert "ENSG00000141510" in by_provider["ensembl"]["url"]


def test_authoritative_knowledge_source_keeps_asset_locator() -> None:
    sources = retrieval_subagent._authoritative_sources_from_tool_results(
        [
            {
                "tool": "search_milvus_knowledge",
                "query": "method",
                "output": {
                    "results": [
                        {
                            "asset_id": 42,
                            "asset_sha256": "abc",
                            "file_name": "paper.pdf",
                            "page_idx": 3,
                            "block_idx": 17,
                            "chunk_id": "42:17",
                            "content": "A grounded method span.",
                        }
                    ]
                },
            }
        ]
    )

    assert len(sources) == 1
    assert sources[0]["source_type"] == "knowledge_asset"
    assert sources[0]["provider_record_id"] == "42"
    assert sources[0]["metadata"]["asset_id"] == 42
    assert sources[0]["metadata"]["page_idx"] == 3
    assert sources[0]["metadata"]["block_idx"] == 17
    assert sources[0]["metadata"]["chunk_id"] == "42:17"
    assert sources[0]["snippet"] == "A grounded method span."


def test_authoritative_knowledge_content_is_preserved_for_machine_projection() -> None:
    sources = retrieval_subagent._authoritative_sources_from_tool_results(
        [
            {
                "tool": "search_milvus_knowledge",
                "query": "method",
                "output": {
                    "results": [
                        {
                            "asset_id": 42,
                            "file_name": "paper.pdf",
                            "page_idx": 3,
                            "block_idx": 17,
                            "chunk_id": "42:17",
                            "content": "x" * 5000,
                        }
                    ]
                },
            }
        ]
    )

    assert len(sources[0]["snippet"]) == 5000
    assert sources[0]["snippet"] == "x" * 5000


def test_reconciliation_replaces_model_locator_metadata_with_authoritative_values() -> None:
    provider_output = {
                "results": [
                    {
                        "asset_id": 42,
                        "file_name": "paper.pdf",
                        "page_idx": 3,
                        "block_idx": 17,
                        "chunk_id": "42:17",
                        "content": "Grounded span.",
                    }
                ]
            }
    pack = EvidencePack(
        query="method",
        evidence=[
            {
                "evidence_id": "E1",
                "source_type": "knowledge_asset",
                "title": "paper.pdf",
                "url": "https://evil.example/forged",
                "doi": "10.9999/forged",
                "pmid": "99999999",
                "raw_ref": "forged-raw",
                "provider": "search_milvus_knowledge",
                "provider_record_id": "42",
                "metadata": {
                    "asset_id": 999,
                    "page_idx": 88,
                    "block_idx": 77,
                    "chunk_id": "forged",
                    "descriptive_note": "keep",
                },
            }
        ],
    )

    reconciled, warnings = retrieval_subagent._reconcile_evidence_pack_sources(
        pack,
        [
            {
                "tool": "search_milvus_knowledge",
                "query": "method",
                "output": {"raw_ref": "raw-1", **provider_output},
            }
        ],
    )

    assert warnings == []
    assert reconciled.evidence[0].metadata["asset_id"] == 42
    assert reconciled.evidence[0].metadata["page_idx"] == 3
    assert reconciled.evidence[0].metadata["block_idx"] == 17
    assert reconciled.evidence[0].metadata["chunk_id"] == "42:17"
    assert "descriptive_note" not in reconciled.evidence[0].metadata
    assert reconciled.evidence[0].url is None
    assert reconciled.evidence[0].doi is None
    assert reconciled.evidence[0].pmid is None
    assert reconciled.evidence[0].raw_ref == "raw-1"
    assert reconciled.evidence[0].provider == "search_milvus_knowledge"
    assert reconciled.evidence[0].provider_record_id == "42"
    assert reconciled.evidence[0].title == "paper.pdf"
    assert reconciled.evidence[0].snippet == "Grounded span."
