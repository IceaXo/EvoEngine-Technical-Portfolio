from __future__ import annotations

import json
import hashlib
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator, model_validator

from src.capabilities.skill_contracts import SkillTypedExposure, SkillTypedSchemaBuild
from src.capabilities.skill_runtime import ContractedSkillRuntime, run_coroutine_sync
from src.config.settings import Settings
from src.agents.citation_settings import citations_enabled
from src.services import conversation_file_registry
from src.services.tool_result_envelope import wrap_structured_tool
from src.subagents.claim_evidence_verifier import verify_claim_evidence_links
from src.subagents.evidence_pack import evidence_pack_to_plain_dict, normalize_evidence_pack
from src.subagents.reference_store import append_reference_records, evidence_pack_to_reference_records
from src.subagents.retrieval_types import (
    EvidencePack,
    RetrievalSubagentResult,
    SourceVerification,
)
from src.subagents.source_evidence_contracts import (
    extract_dois_from_text,
    materialize_source_records,
    normalize_doi,
    normalize_pmid,
)
from src.subagents.source_verifier import verify_evidence_pack_sources
from src.subagents.source_ledger import (
    build_source_ledger_from_evidence_pack,
    normalize_source_record,
)
from src.support.chat_deepseek_safe import ChatDeepSeekThinkingSafe
from src.tools.enterprise_knowledge import build_enterprise_tools
from src.tools.web_search import build_web_search_tool


DEFAULT_DB_TOOLS: frozenset[str] = frozenset(
    {
        "safe_query_pubmed",
        "safe_query_uniprot",
        "safe_query_ensembl",
        "safe_query_clinvar",
        "safe_query_geo",
        "safe_query_opentarget",
    }
)
BROAD_DATABASE_SOURCES: frozenset[str] = frozenset({"bio_database", "database"})

SOURCE_TOOL_HINTS: dict[str, set[str]] = {
    "pubmed": {"safe_query_pubmed"},
    "literature": {"safe_query_pubmed"},
    "paper": {"safe_query_pubmed"},
    "crossref": {"safe_query_crossref"},
    "doi": {"safe_query_crossref"},
    "uniprot": {"safe_query_uniprot"},
    "ensembl": {"safe_query_ensembl"},
    # ``NCBI`` is an umbrella organization, not a Nuccore/transcript source.
    # PubMed requests commonly say "PubMed/NCBI" and must not therefore fan
    # out into the RefSeq transcript adapter.  The concrete Nuccore aliases
    # below still select that adapter deterministically.
    "nuccore": {"safe_query_ncbi_transcript"},
    "nucleotide": {"safe_query_ncbi_transcript"},
    "refseq": {"safe_query_ncbi_transcript"},
    "clinvar": {"safe_query_clinvar"},
    "geo": {"safe_query_geo"},
    "opentarget": {"safe_query_opentarget"},
    "open_targets": {"safe_query_opentarget"},
    "kegg": {"web_search"},
    "string": {"web_search"},
    "bio_database": set(DEFAULT_DB_TOOLS),
    "database": set(DEFAULT_DB_TOOLS),
}

_PMID_TASK_RE = re.compile(r"\bPMID\s*[:：]?\s*(\d{6,9})\b", re.IGNORECASE)
_UNIPROT_ACCESSION_RE = re.compile(
    r"(?<![A-Z0-9])(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}){1,2}[0-9])(?![A-Z0-9])",
    re.IGNORECASE,
)
_REFSEQ_TRANSCRIPT_RE = re.compile(
    r"(?<![A-Z0-9])(?:NM|NR|XM|XR)_[0-9]+(?:\.[0-9]+)?(?![A-Z0-9])",
    re.IGNORECASE,
)
_DELEGATED_RETRIEVAL_SKILL_ROUTES: dict[str, tuple[str, str]] = {
    "safe_query_clinvar": (
        "skill.public_data.clinvar",
        "skill_public_data_clinvar",
    ),
    "safe_query_crossref": (
        "skill.crossref.doi.identity",
        "skill_crossref_doi_identity",
    ),
    "safe_query_geo": (
        "skill.public_data.geo",
        "skill_public_data_geo",
    ),
    "safe_query_pubmed": (
        "skill.pubmed.literature.query",
        "skill_pubmed_literature_query",
    ),
    "safe_query_uniprot": (
        "skill.uniprot.query",
        "skill_uniprot_query",
    ),
    "safe_query_ensembl": (
        "skill.public_data.ensembl",
        "skill_public_data_ensembl",
    ),
    "safe_query_ncbi_transcript": (
        "skill.public_data.ncbi_gene_cds",
        "skill_public_data_ncbi_gene_cds",
    ),
    "safe_query_opentarget": (
        "skill.open_targets.query",
        "skill_open_targets_query",
    ),
}

# Reverse map: the formal typed tool ID (the model-visible capability entry
# shared with the lead agent) back to the internal result adapter name.
_TYPED_TO_RETRIEVAL_WRAPPER: dict[str, str] = {
    typed_tool_id: wrapper_name
    for wrapper_name, (_capability_id, typed_tool_id) in _DELEGATED_RETRIEVAL_SKILL_ROUTES.items()
}


def _retrieval_wrapper_name(tool_name: str) -> str:
    """Normalize a model-visible typed tool ID to the internal adapter name."""

    normalized = str(tool_name or "").strip()
    return _TYPED_TO_RETRIEVAL_WRAPPER.get(normalized, normalized)


def _retrieval_typed_tool_id(wrapper_name: str) -> str:
    """Normalize an internal adapter name to the model-visible typed tool ID."""

    route = _DELEGATED_RETRIEVAL_SKILL_ROUTES.get(str(wrapper_name or "").strip())
    return route[1] if route is not None else str(wrapper_name or "").strip()


def _doi_ids_from_text(value: str) -> list[str]:
    return extract_dois_from_text(value)


def _pmid_ids_from_text(value: str) -> list[str]:
    pmids: list[str] = []
    for raw in _PMID_TASK_RE.findall(str(value or "")):
        pmid = str(raw or "").strip()
        if pmid and pmid not in pmids:
            pmids.append(pmid)
    return pmids


def _uniprot_accessions_from_text(value: str) -> list[str]:
    accessions: list[str] = []
    for raw in _UNIPROT_ACCESSION_RE.findall(str(value or "")):
        accession = str(raw or "").strip().upper()
        if accession and accession not in accessions:
            accessions.append(accession)
    return accessions


def _first_list_text(value: Any) -> str:
    if isinstance(value, list):
        return next((str(item or "").strip() for item in value if str(item or "").strip()), "")
    return str(value or "").strip() if isinstance(value, (str, int, float)) else ""


def _project_uniprot_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """Project the formal UniProt Skill result into the retrieval adapter shape."""

    accession = str(
        record.get("accession")
        or record.get("primary_accession")
        or record.get("primaryAccession")
        or ""
    ).strip().upper()
    if not accession:
        return None
    entry_name = str(
        record.get("entry_name")
        or record.get("uni_prot_kb_id")
        or record.get("uniProtkbId")
        or ""
    ).strip()
    recommended_name = str(
        record.get("recommended_name") or record.get("protein_name") or ""
    ).strip()
    raw_gene_names = record.get("gene_names")
    gene_names = list(
        dict.fromkeys(
            str(item or "").strip()
            for item in (
                raw_gene_names
                if isinstance(raw_gene_names, list)
                else [record.get("gene_name")]
            )
            if str(item or "").strip()
        )
    )
    organism = str(record.get("organism") or "").strip()
    function_text = str(record.get("function") or "").strip()
    official_url = str(record.get("official_url") or "").strip()
    if not official_url:
        official_url = f"https://www.uniprot.org/uniprotkb/{quote(accession)}/entry"
    facts = [f"accession={accession}"]
    if entry_name:
        facts.append(f"entry_name={entry_name}")
    if recommended_name:
        facts.append(f"recommended_name={recommended_name}")
    if gene_names:
        facts.append(f"gene_names={', '.join(gene_names)}")
    if organism:
        facts.append(f"organism={organism}")
    if function_text:
        facts.append(f"FUNCTION={function_text}")
    facts.append(f"official_url={official_url}")
    return {
        "accession": accession,
        "primaryAccession": accession,
        "entry_name": entry_name,
        "recommended_name": recommended_name,
        "gene_names": gene_names,
        "organism": organism,
        "function": function_text,
        "official_url": official_url,
        "url": official_url,
        "title": recommended_name or entry_name or accession,
        "snippet": "; ".join(facts),
    }


class RetrievalSubagentInput(BaseModel):
    task: str = Field(
        default="",
        description="首次运行时提供检索任务；携带 checkpoint_resource_id 续跑时可留空。",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="允许使用的来源类型或工具线索，如 pubmed、web、knowledge_assets、bio_database、uniprot、clinvar、geo。",
    )
    max_iterations: int = Field(default=6, ge=1, le=6, description="子代理最多工具/模型迭代轮数。")
    max_evidence: int = Field(default=5, ge=1, le=12, description="Evidence Pack 最多返回多少条证据。")
    checkpoint_resource_id: str | None = Field(
        default=None,
        description="上一批返回的 checkpoint_resource_id；首次运行留空。",
    )
    batch_size: int = Field(
        default=6,
        ge=1,
        le=6,
        description="本次最多执行多少个尚未完成的检索工作项。",
    )
    output_mode: Literal["evidence_pack"] = Field(
        default="evidence_pack",
        description=(
            "固定输出模式。检索子代理只返回紧凑 Evidence Pack，不返回完整原始序列、"
            "引物、分析文件、沙盒结果或业务报告。"
        ),
    )

    @field_validator("max_iterations", mode="before")
    @classmethod
    def normalize_iteration_budget(cls, value: Any) -> Any:
        """Keep one evidence pass inside the parent turn's execution budget."""
        try:
            return max(1, min(int(value), 6))
        except (TypeError, ValueError):
            return value

    @field_validator("max_evidence", mode="before")
    @classmethod
    def normalize_evidence_budget(cls, value: Any) -> Any:
        try:
            return max(1, min(int(value), 12))
        except (TypeError, ValueError):
            return value

    @model_validator(mode="after")
    def require_task_or_checkpoint(self) -> "RetrievalSubagentInput":
        if not self.task.strip() and not str(self.checkpoint_resource_id or "").strip():
            raise ValueError("首次检索必须提供 task；续跑可只提供 checkpoint_resource_id")
        return self


class SafeQueryInput(BaseModel):
    query: str = Field(description="短查询词，例如 TP53 human Ensembl gene ID。")
    max_results: int = Field(default=5, ge=1, le=10, description="最多返回多少条记录。")


def _gene_symbol_from_query(query: str) -> str:
    text = str(query or "")
    blocked = {
        "A",
        "AN",
        "AND",
        "API",
        "DATABASE",
        "CDS",
        "DANIO",
        "DOWNLOAD",
        "ENSEMBL",
        "FASTA",
        "FETCH",
        "FIND",
        "GENE",
        "GET",
        "HOMO",
        "HUMAN",
        "ID",
        "INFO",
        "INFORMATION",
        "MRNA",
        "MUS",
        "MUSCULUS",
        "NCBI",
        "NORVEGICUS",
        "NUCLEOTIDE",
        "QUERY",
        "REFSEQ",
        "RATTUS",
        "RERIO",
        "RETRIEVE",
        "SAPIENS",
        "SEQUENCE",
        "THE",
        "TRANSCRIPT",
    }
    candidates = re.findall(
        r"(?<![A-Z0-9-])[A-Z][A-Z0-9-]{1,14}(?![A-Z0-9-])",
        text.upper(),
    )
    for candidate in candidates:
        if candidate not in blocked and not candidate.startswith("ENS"):
            return candidate
    stripped = re.sub(r"[^A-Za-z0-9-]+", " ", text).strip().split()
    for token in stripped:
        if 2 <= len(token) <= 15 and token.upper() not in blocked:
            return token
    return text.strip()[:40] or "TP53"


def _species_from_query(query: str) -> str:
    text = str(query or "").lower()
    if any(term in text for term in ("mouse", "mus musculus", "小鼠")):
        return "mus_musculus"
    if any(term in text for term in ("rat", "rattus", "大鼠")):
        return "rattus_norvegicus"
    if any(term in text for term in ("zebrafish", "danio", "斑马鱼")):
        return "danio_rerio"
    return "homo_sapiens"


def _refseq_transcript_accessions_from_text(value: str) -> list[str]:
    return list(
        dict.fromkeys(
            match.group(0).upper()
            for match in _REFSEQ_TRANSCRIPT_RE.finditer(str(value or ""))
        )
    )


def _retrieval_skill_runtime_from_frozen_build(
    skill_typed_schema_build: SkillTypedSchemaBuild,
    *,
    available_tool_ids: set[str] | frozenset[str],
) -> ContractedSkillRuntime:
    """Create an executor over only the exact specs frozen for this Agent build."""

    if skill_typed_schema_build is None:
        raise ValueError("retrieval requires the Agent build's frozen Skill schema")
    available = {
        str(tool_id or "").strip()
        for tool_id in available_tool_ids
        if str(tool_id or "").strip()
    }
    specs = []
    for capability_id, typed_tool_id in _DELEGATED_RETRIEVAL_SKILL_ROUTES.values():
        if typed_tool_id not in available:
            continue
        try:
            item = skill_typed_schema_build.item_for_capability(capability_id)
        except Exception:  # noqa: BLE001 - malformed frozen mapping fails closed
            continue
        if (
            item is None
            or item.exposure != SkillTypedExposure.TYPED
            or item.tool_id != typed_tool_id
        ):
            continue
        specs.append(item.capability_spec())
    skills_root = Path(__file__).resolve().parents[2] / "skills"
    # Passing ``specs`` is deliberate: ContractedSkillRuntime must not rescan
    # manifests and accidentally execute a contract other than this turn's
    # immutable Agent build.
    return ContractedSkillRuntime(skills_root, specs=tuple(specs))


def _skill_provider_payload(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    # ``data`` is allowed to be a resource-only machine projection for large
    # results. Providers inside the retrieval compositor need the complete
    # typed contract, which CapabilityResult retains separately.
    contract_data = getattr(result, "contract_data", None)
    raw_outer = (
        contract_data
        if isinstance(contract_data, dict)
        else result.data
        if isinstance(getattr(result, "data", None), dict)
        else {}
    )
    outer = dict(raw_outer)
    nested = raw_outer.get("result")
    provider_payload = dict(nested) if isinstance(nested, dict) else dict(raw_outer)

    # A formal Skill has already crossed CapabilityExecutor's one automatic
    # source boundary.  Carry only its compact registered Citation handles and
    # durable sidecar reference into the retrieval compositor.  Do not copy the
    # full child Evidence Pack and do not ask retrieval to verify it again.
    citations = [
        dict(item)
        for item in (getattr(result, "citation_projection", None) or [])
        if isinstance(item, dict)
        and isinstance(item.get("source_registration"), dict)
        and item["source_registration"].get("status") == "registered"
        and item["source_registration"].get("authority") == "capability_runtime"
    ]
    source_sidecar = getattr(result, "source_sidecar_resource", None)
    if hasattr(source_sidecar, "model_dump"):
        source_sidecar = source_sidecar.model_dump(mode="json", exclude_none=True)
    elif isinstance(source_sidecar, dict):
        source_sidecar = dict(source_sidecar)
    else:
        source_sidecar = None
    source_outcome = getattr(result, "source_outcome", None)
    if hasattr(source_outcome, "model_dump"):
        source_outcome = source_outcome.model_dump(mode="json", exclude_none=True)
    elif isinstance(source_outcome, dict):
        source_outcome = dict(source_outcome)
    else:
        source_outcome = None
    source_bridge = {
        **({"source_ledger": citations, "registered_source_ledger": citations} if citations else {}),
        **(
            {
                "source_sidecar_resources": [source_sidecar],
                "source_sidecar_refs": [source_sidecar["resource_id"]],
            }
            if isinstance(source_sidecar, dict)
            and str(source_sidecar.get("resource_id") or "").strip()
            else {}
        ),
        **({"source_outcome": source_outcome} if source_outcome else {}),
    }
    if source_bridge:
        outer.update(source_bridge)
        provider_payload.update(source_bridge)
    return provider_payload, outer


def _skill_failure_type(result: Any) -> str:
    error = getattr(result, "error", None)
    kind = getattr(error, "kind", None)
    value = getattr(kind, "value", kind)
    return str(value or getattr(result, "status", None) or "capability_execution_failed")


def _skill_call_fact(result: Any) -> dict[str, Any]:
    status = getattr(result, "status", "")
    return {
        "capability_id": str(getattr(result, "capability_id", "") or ""),
        "capability_version": str(getattr(result, "capability_version", "") or ""),
        "call_id": str(getattr(result, "call_id", "") or ""),
        "status": str(getattr(status, "value", status) or ""),
        "raw_ref": str(getattr(result, "raw_ref", "") or "") or None,
    }


def _source_contract_fields(
    provider_payload: dict[str, Any],
    outer_payload: dict[str, Any],
) -> dict[str, Any]:
    """Keep formal source facts without copying the bulky raw capability result."""

    projected: dict[str, Any] = {}
    registered = provider_payload.get(
        "registered_source_ledger",
        outer_payload.get("registered_source_ledger"),
    )
    if isinstance(registered, list) and registered:
        # The Runtime projection is authoritative.  Keeping a child
        # evidence_pack/source_projection beside it would duplicate source
        # construction and can re-introduce the old verifier path.
        projected["source_ledger"] = list(registered)
        projected["registered_source_ledger"] = list(registered)
        for key in (
            "source_sidecar_resources",
            "source_sidecar_refs",
            "source_outcome",
        ):
            value = provider_payload.get(key, outer_payload.get(key))
            if value is not None:
                projected[key] = value
        return projected
    for key in (
        "source_ledger",
        "evidence_pack",
        "source_projection",
        "source_sidecar_resources",
        "source_sidecar_refs",
        "source_outcome",
    ):
        value = provider_payload.get(key, outer_payload.get(key))
        if value is not None:
            projected[key] = value
    return projected


def _merge_source_contract_fields(
    payload_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    if len(payload_pairs) == 1:
        return _source_contract_fields(*payload_pairs[0])
    ledger: list[dict[str, Any]] = []
    registered_ledger: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_registered: set[str] = set()
    sidecars: list[dict[str, Any]] = []
    seen_sidecars: set[str] = set()
    projections: list[dict[str, Any]] = []
    for provider_payload, outer_payload in payload_pairs:
        fields = _source_contract_fields(provider_payload, outer_payload)
        for item in fields.get("source_ledger") or []:
            if not isinstance(item, dict):
                continue
            key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            ledger.append(item)
        for item in fields.get("registered_source_ledger") or []:
            if not isinstance(item, dict):
                continue
            key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            if key in seen_registered:
                continue
            seen_registered.add(key)
            registered_ledger.append(item)
        for item in fields.get("source_sidecar_resources") or []:
            if not isinstance(item, dict):
                continue
            resource_id = str(item.get("resource_id") or "").strip()
            if not resource_id or resource_id in seen_sidecars:
                continue
            seen_sidecars.add(resource_id)
            sidecars.append(item)
        projection = fields.get("source_projection")
        if isinstance(projection, dict):
            projections.append(projection)
    return {
        **({"source_ledger": ledger} if ledger else {}),
        **({"registered_source_ledger": registered_ledger} if registered_ledger else {}),
        **({"source_sidecar_resources": sidecars} if sidecars else {}),
        **(
            {"source_sidecar_refs": [item["resource_id"] for item in sidecars]}
            if sidecars
            else {}
        ),
        **({"source_projections": projections} if projections else {}),
    }


def _filter_source_contract_pair_for_pubmed_records(
    provider_payload: dict[str, Any],
    outer_payload: dict[str, Any],
    *,
    dois: set[str],
    pmids: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep source handles only for PubMed records accepted by exact lookup."""

    def matches(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        doi = normalize_doi(item.get("doi"))
        pmid = normalize_pmid(
            item.get("pmid")
            or item.get("provider_record_id")
            or item.get("record_id")
        )
        return bool((doi and doi in dois) or (pmid and pmid in pmids))

    def filtered(payload: dict[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        for key in ("source_ledger", "registered_source_ledger"):
            if isinstance(result.get(key), list):
                accepted = [item for item in result[key] if matches(item)]
                if accepted:
                    result[key] = accepted
                else:
                    result.pop(key, None)
        return result

    return filtered(provider_payload), filtered(outer_payload)


def _invoke_retrieval_skill(
    runtime: ContractedSkillRuntime,
    *,
    capability_id: str,
    params: dict[str, Any],
    query: str,
    request_id: str | None,
    project_id: str | None,
    conversation_id: str | None,
    user_id: int | None,
) -> Any:
    return run_coroutine_sync(
        runtime.invoke_capability(
            capability_id=capability_id,
            params=params,
            input_files=[],
            request_id=str(request_id or "").strip()
            or f"req_retrieval_skill_{uuid.uuid4().hex}",
            project_id=project_id,
            conversation_id=conversation_id,
            user_id=user_id,
            current_user_message=query,
        )
    )


def _build_safe_database_tools(
    *,
    skill_runtime: ContractedSkillRuntime,
    available_tool_ids: set[str] | frozenset[str],
    project_id: str | None = None,
    conversation_id: str | None = None,
    user_id: int | None = None,
    request_id: str | None = None,
) -> dict[str, StructuredTool]:
    resource_scope = _retrieval_resource_scope(
        project_id=project_id,
        conversation_id=conversation_id,
        user_id=user_id,
        request_id=request_id,
    )
    def safe_query_pubmed(query: str, max_results: int = 5) -> dict[str, Any]:
        limit = max(1, min(20, int(max_results)))
        requested_dois = _doi_ids_from_text(query)
        if requested_dois:
            records: list[dict[str, Any]] = []
            returned_dois: list[str] = []
            not_found_dois: list[str] = []
            unresolved_dois: list[str] = []
            failures: list[dict[str, Any]] = []
            pmids: list[str] = []
            capability_calls: list[dict[str, Any]] = []
            source_payloads: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for doi in requested_dois:
                try:
                    result = _invoke_retrieval_skill(
                        skill_runtime,
                        capability_id="skill.pubmed.literature.query",
                        params={
                            "operation": "search_pubmed",
                            "query": f"{doi}[DOI]",
                            "max_records": limit,
                            "sort": "relevance",
                        },
                        query=query,
                        request_id=request_id,
                        project_id=project_id,
                        conversation_id=conversation_id,
                        user_id=user_id,
                    )
                except Exception as exc:
                    unresolved_dois.append(doi)
                    failures.append({"doi": doi, "error_type": type(exc).__name__})
                    continue
                capability_calls.append(_skill_call_fact(result))
                provider_payload, outer_payload = _skill_provider_payload(result)
                if not bool(getattr(result, "ok", False)):
                    unresolved_dois.append(doi)
                    failures.append(
                        {"doi": doi, "error_type": _skill_failure_type(result)}
                    )
                    continue
                matching_records = [
                    item
                    for item in provider_payload.get("records") or []
                    if isinstance(item, dict)
                    and normalize_doi(_article_id(item, "doi") or item.get("doi")) == doi
                ]
                if not matching_records:
                    not_found_dois.append(doi)
                    continue
                records.extend(matching_records)
                returned_dois.append(doi)
                matching_pmids: set[str] = set()
                for item in matching_records:
                    pmid = str(item.get("pmid") or item.get("uid") or "").strip()
                    if pmid and pmid not in pmids:
                        pmids.append(pmid)
                    normalized_pmid = normalize_pmid(pmid)
                    if normalized_pmid:
                        matching_pmids.add(normalized_pmid)
                source_payloads.append(
                    _filter_source_contract_pair_for_pubmed_records(
                        provider_payload,
                        outer_payload,
                        dois={doi},
                        pmids=matching_pmids,
                    )
                )

            exact_scope_complete = not unresolved_dois
            coverage_scope = (
                "PubMed exact DOI field lookup for requested_ids=["
                + ",".join(requested_dois)
                + "] only"
                if exact_scope_complete
                else ""
            )
            return {
                "ok": exact_scope_complete,
                "source_type": "pubmed",
                "provider": "pubmed",
                "query": query,
                "ids": pmids,
                "records": records,
                "direct_lookup": True,
                "requested_ids": requested_dois,
                "returned_ids": returned_dois,
                "not_found_ids": not_found_dois,
                "unresolved_ids": unresolved_dois,
                "failures": failures,
                "capability_calls": capability_calls,
                "coverage": "exhausted" if exact_scope_complete else "unknown",
                "coverage_scope": coverage_scope,
                **_merge_source_contract_fields(source_payloads),
                "provider_observation": {
                    "provider": "pubmed",
                    "query_mode": "exact_doi",
                    "requested_ids": requested_dois,
                    "returned_ids": returned_dois,
                    "not_found_ids": not_found_dois,
                    "unresolved_ids": unresolved_dois,
                    "coverage": "exhausted" if exact_scope_complete else "unknown",
                    "coverage_scope": coverage_scope,
                },
                **(
                    {"error": "PubMed exact DOI lookup has unresolved provider responses."}
                    if unresolved_dois
                    else {}
                ),
            }
        requested_pmids = _pmid_ids_from_text(query)
        if requested_pmids:
            try:
                result = _invoke_retrieval_skill(
                    skill_runtime,
                    capability_id="skill.pubmed.literature.query",
                    params={
                        "operation": "fetch_pubmed_summaries",
                        "pmids": requested_pmids,
                    },
                    query=query,
                    request_id=request_id,
                    project_id=project_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
            except Exception as exc:
                return {
                    "ok": False,
                    "source_type": "pubmed",
                    "provider": "pubmed",
                    "query": query,
                    "ids": requested_pmids,
                    "records": [],
                    "requested_ids": requested_pmids,
                    "returned_ids": [],
                    "not_found_ids": [],
                    "unresolved_ids": requested_pmids,
                    "coverage": "unknown",
                    "coverage_scope": "",
                    "error": f"PubMed exact PMID lookup failed: {type(exc).__name__}",
                    "provider_observation": {
                        "provider": "pubmed",
                        "query_mode": "exact_pmid",
                        "requested_ids": requested_pmids,
                        "returned_ids": [],
                        "not_found_ids": [],
                        "unresolved_ids": requested_pmids,
                        "coverage": "unknown",
                        "coverage_scope": "",
                    },
                }
            provider_payload, outer_payload = _skill_provider_payload(result)
            if not bool(getattr(result, "ok", False)):
                return {
                    "ok": False,
                    "source_type": "pubmed",
                    "provider": "pubmed",
                    "query": query,
                    "ids": requested_pmids,
                    "records": [],
                    "requested_ids": requested_pmids,
                    "returned_ids": [],
                    "not_found_ids": [],
                    "unresolved_ids": requested_pmids,
                    "coverage": "unknown",
                    "coverage_scope": "",
                    "error": (
                        "PubMed exact PMID capability failed: "
                        + _skill_failure_type(result)
                    ),
                    "capability_calls": [_skill_call_fact(result)],
                    "provider_observation": {
                        "provider": "pubmed",
                        "query_mode": "exact_pmid",
                        "requested_ids": requested_pmids,
                        "returned_ids": [],
                        "not_found_ids": [],
                        "unresolved_ids": requested_pmids,
                        "coverage": "unknown",
                        "coverage_scope": "",
                    },
                }
            records = [
                item
                for item in provider_payload.get("records") or []
                if isinstance(item, dict)
                and (
                    normalize_pmid(
                        item.get("pmid")
                        or item.get("uid")
                        or _article_id(item, "pubmed")
                    )
                    in requested_pmids
                )
            ]
            returned_pmids = list(
                dict.fromkeys(
                    pmid
                    for item in records
                    if (
                        pmid := normalize_pmid(
                            item.get("pmid")
                            or item.get("uid")
                            or _article_id(item, "pubmed")
                        )
                    )
                )
            )
            not_found_pmids = [pmid for pmid in requested_pmids if pmid not in returned_pmids]
            coverage_scope = (
                "PubMed exact PMID record lookup for requested_ids=["
                + ",".join(requested_pmids)
                + "] only"
            )
            return {
                "ok": True,
                "source_type": "pubmed",
                "provider": "pubmed",
                "query": query,
                "ids": requested_pmids,
                "records": records,
                "direct_lookup": True,
                "requested_ids": requested_pmids,
                "returned_ids": returned_pmids,
                "not_found_ids": not_found_pmids,
                "unresolved_ids": [],
                "coverage": "exhausted",
                "coverage_scope": coverage_scope,
                "capability_calls": [_skill_call_fact(result)],
                **_source_contract_fields(provider_payload, outer_payload),
                "provider_observation": {
                    "provider": "pubmed",
                    "query_mode": "exact_pmid",
                    "requested_ids": requested_pmids,
                    "returned_ids": returned_pmids,
                    "not_found_ids": not_found_pmids,
                    "unresolved_ids": [],
                    "coverage": "exhausted",
                    "coverage_scope": coverage_scope,
                },
            }
        try:
            result = _invoke_retrieval_skill(
                skill_runtime,
                capability_id="skill.pubmed.literature.query",
                params={
                    "operation": "search_pubmed",
                    "query": query,
                    "max_records": limit,
                    "sort": "relevance",
                },
                query=query,
                request_id=request_id,
                project_id=project_id,
                conversation_id=conversation_id,
                user_id=user_id,
            )
        except Exception as exc:
            return {
                "ok": False,
                "source_type": "pubmed",
                "provider": "pubmed",
                "query": query,
                "ids": [],
                "records": [],
                "error": f"PubMed capability invocation failed: {type(exc).__name__}",
                "coverage": "unknown",
            }
        provider_payload, outer_payload = _skill_provider_payload(result)
        if not bool(getattr(result, "ok", False)):
            return {
                "ok": False,
                "source_type": "pubmed",
                "provider": "pubmed",
                "query": query,
                "ids": [],
                "records": [],
                "capability_calls": [_skill_call_fact(result)],
                "coverage": "unknown",
                "error": "PubMed capability failed: " + _skill_failure_type(result),
                "provider_observation": {
                    "provider": "pubmed",
                    "query_mode": "search",
                    "requested_ids": [],
                    "returned_ids": [],
                    "not_found_ids": [],
                    "unresolved_ids": [],
                    "coverage": "unknown",
                    "coverage_scope": "",
                },
            }
        records = [
            item
            for item in provider_payload.get("records") or []
            if isinstance(item, dict)
        ]
        ids = list(
            dict.fromkeys(
                pmid
                for item in records
                if (pmid := normalize_pmid(item.get("pmid") or item.get("uid")))
            )
        )
        return {
            "ok": True,
            "source_type": "pubmed",
            "provider": "pubmed",
            "query": query,
            "ids": ids,
            "records": records,
            "capability_calls": [_skill_call_fact(result)],
            **_source_contract_fields(provider_payload, outer_payload),
            "provider_observation": {
                "provider": "pubmed",
                "query_mode": "search",
                "requested_ids": [],
                "returned_ids": ids,
                "not_found_ids": [],
                "unresolved_ids": [],
                "coverage": "unknown",
                "coverage_scope": "",
            },
        }

    def safe_query_crossref(query: str, max_results: int = 5) -> dict[str, Any]:
        try:
            result = _invoke_retrieval_skill(
                skill_runtime,
                capability_id="skill.crossref.doi.identity",
                params={
                    "query": query,
                    "max_results": max(1, min(8, int(max_results))),
                },
                query=query,
                request_id=request_id,
                project_id=project_id,
                conversation_id=conversation_id,
                user_id=user_id,
            )
        except Exception as exc:
            return {
                "ok": False,
                "source_type": "paper",
                "provider": "crossref",
                "query": query,
                "records": [],
                "raw_records": [],
                "requested_ids": [],
                "returned_ids": [],
                "not_found_ids": [],
                "unresolved_ids": [],
                "failures": [],
                "coverage": "unknown",
                "error": f"CrossRef capability invocation failed: {type(exc).__name__}",
            }
        provider_payload, outer_payload = _skill_provider_payload(result)
        call_fact = _skill_call_fact(result)
        data = provider_payload
        if not bool(getattr(result, "ok", False)) or not isinstance(data, dict):
            return {
                "ok": False,
                "source_type": "paper",
                "provider": "crossref",
                "query": query,
                "records": [],
                "requested_ids": [],
                "returned_ids": [],
                "not_found_ids": [],
                "unresolved_ids": [],
                "failures": [],
                "capability_calls": [call_fact],
                "coverage": "unknown",
                "error": f"CrossRef capability failed: {_skill_failure_type(result)}",
            }
        return {
            "ok": bool(data.get("ok", False)),
            "source_type": "paper",
            "provider": "crossref",
            "query": query,
            "records": [
                item
                for item in data.get("records") or []
                if isinstance(item, dict)
            ],
            "raw_records": [
                item
                for item in data.get("raw_records") or []
                if isinstance(item, dict)
            ],
            "requested_ids": [
                str(item or "").strip()
                for item in data.get("requested_ids") or []
                if str(item or "").strip()
            ],
            "returned_ids": [
                str(item or "").strip()
                for item in data.get("returned_ids") or []
                if str(item or "").strip()
            ],
            "not_found_ids": [
                str(item or "").strip()
                for item in data.get("not_found_ids") or []
                if str(item or "").strip()
            ],
            "unresolved_ids": [
                str(item or "").strip()
                for item in data.get("unresolved_ids") or []
                if str(item or "").strip()
            ],
            "failures": [
                item
                for item in data.get("failures") or []
                if isinstance(item, dict)
            ],
            "coverage": str(data.get("coverage") or "unknown"),
            "coverage_scope": str(data.get("coverage_scope") or ""),
            "recovery": str(data.get("recovery") or "unknown"),
            "provider_observation": (
                data.get("provider_observation")
                if isinstance(data.get("provider_observation"), dict)
                else {}
            ),
            "capability_calls": [call_fact],
            **_source_contract_fields(provider_payload, outer_payload),
        }

    def _safe_query_ncbi_summary(
        *,
        capability_id: str,
        database: str,
        query: str,
        max_results: int,
    ) -> dict[str, Any]:
        try:
            result = _invoke_retrieval_skill(
                skill_runtime,
                capability_id=capability_id,
                params={
                    "query": query,
                    "limit": max(1, min(50, int(max_results))),
                },
                query=query,
                request_id=request_id,
                project_id=project_id,
                conversation_id=conversation_id,
                user_id=user_id,
            )
        except Exception as exc:
            return {
                "ok": False,
                "source_type": "bio_database",
                "database": database,
                "provider": "ncbi_eutils",
                "query": query,
                "ids": [],
                "records": [],
                "coverage": "unknown",
                "error": f"{database} capability invocation failed: {type(exc).__name__}",
            }
        provider_payload, outer_payload = _skill_provider_payload(result)
        call_fact = _skill_call_fact(result)
        data = provider_payload.get("data")
        if not bool(getattr(result, "ok", False)) or not isinstance(data, dict):
            return {
                "ok": False,
                "source_type": "bio_database",
                "database": database,
                "provider": "ncbi_eutils",
                "query": query,
                "ids": [],
                "records": [],
                "capability_calls": [call_fact],
                "coverage": "unknown",
                "error": f"{database} capability failed: {_skill_failure_type(result)}",
            }
        ids = [
            str(item or "").strip()
            for item in data.get("ids") or []
            if str(item or "").strip()
        ]
        records = [
            item for item in data.get("records") or [] if isinstance(item, dict)
        ]
        return {
            "ok": True,
            "source_type": "bio_database",
            "database": database,
            "provider": "ncbi_eutils",
            "query": query,
            "ids": ids,
            "records": records,
            "summary": data.get("summary")
            if isinstance(data.get("summary"), dict)
            else {},
            "search_meta": {
                "search": data.get("search")
                if isinstance(data.get("search"), dict)
                else {},
                "capability_call": call_fact,
            },
            "capability_calls": [call_fact],
            **_source_contract_fields(provider_payload, outer_payload),
        }

    def safe_query_clinvar(query: str, max_results: int = 5) -> dict[str, Any]:
        return _safe_query_ncbi_summary(
            capability_id="skill.public_data.clinvar",
            database="clinvar",
            query=query,
            max_results=max_results,
        )

    def safe_query_geo(query: str, max_results: int = 5) -> dict[str, Any]:
        return _safe_query_ncbi_summary(
            capability_id="skill.public_data.geo",
            database="geo",
            query=query,
            max_results=max_results,
        )

    def safe_query_uniprot(query: str, max_results: int = 5) -> dict[str, Any]:
        requested_accessions = _uniprot_accessions_from_text(query)
        if requested_accessions:
            records: list[dict[str, Any]] = []
            returned_accessions: list[str] = []
            not_found_accessions: list[str] = []
            unresolved_accessions: list[str] = []
            failures: list[dict[str, Any]] = []
            capability_calls: list[dict[str, Any]] = []
            source_payloads: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for offset in range(0, len(requested_accessions), 8):
                batch = requested_accessions[offset : offset + 8]
                try:
                    result = _invoke_retrieval_skill(
                        skill_runtime,
                        capability_id="skill.uniprot.query",
                        params={
                            "accessions": batch,
                            "include_sequence": False,
                        },
                        query=query,
                        request_id=request_id,
                        project_id=project_id,
                        conversation_id=conversation_id,
                        user_id=user_id,
                    )
                except Exception as exc:
                    unresolved_accessions.extend(batch)
                    failures.append(
                        {"accessions": batch, "error_type": type(exc).__name__}
                    )
                    continue
                capability_calls.append(_skill_call_fact(result))
                provider_payload, outer_payload = _skill_provider_payload(result)
                if (
                    not bool(getattr(result, "ok", False))
                    and provider_payload.get("ok") is not False
                ):
                    unresolved_accessions.extend(batch)
                    failures.append(
                        {
                            "accessions": batch,
                            "error_type": _skill_failure_type(result),
                        }
                    )
                    continue
                source_payloads.append((provider_payload, outer_payload))
                batch_records = [
                    projected
                    for item in provider_payload.get("records") or []
                    if isinstance(item, dict)
                    and (projected := _project_uniprot_record(item)) is not None
                ]
                records.extend(batch_records)
                returned_accessions.extend(
                    accession
                    for item in provider_payload.get("returned_ids") or []
                    if (accession := str(item or "").strip().upper())
                    and accession not in returned_accessions
                )
                not_found_accessions.extend(
                    accession
                    for item in provider_payload.get("not_found_ids") or []
                    if (accession := str(item or "").strip().upper())
                    and accession not in not_found_accessions
                )
                unresolved_accessions.extend(
                    accession
                    for item in provider_payload.get("unresolved_ids") or []
                    if (accession := str(item or "").strip().upper())
                    and accession not in unresolved_accessions
                )
                failures.extend(
                    item
                    for item in provider_payload.get("failures") or []
                    if isinstance(item, dict)
                )

            exact_scope_complete = not unresolved_accessions
            coverage_scope = (
                "UniProt exact accession record lookup for requested_ids=["
                + ",".join(requested_accessions)
                + "] only"
                if exact_scope_complete
                else ""
            )
            return {
                "ok": exact_scope_complete,
                "source_type": "bio_database",
                "database": "uniprot",
                "provider": "uniprot",
                "query": query,
                "records": records,
                "direct_lookup": True,
                "requested_ids": requested_accessions,
                "returned_ids": returned_accessions,
                "not_found_ids": not_found_accessions,
                "unresolved_ids": unresolved_accessions,
                "failures": failures,
                "capability_calls": capability_calls,
                "coverage": "exhausted" if exact_scope_complete else "unknown",
                "coverage_scope": coverage_scope,
                **_merge_source_contract_fields(source_payloads),
                "provider_observation": {
                    "provider": "uniprot",
                    "query_mode": "exact_accession",
                    "requested_ids": requested_accessions,
                    "returned_ids": returned_accessions,
                    "not_found_ids": not_found_accessions,
                    "unresolved_ids": unresolved_accessions,
                    "coverage": "exhausted" if exact_scope_complete else "unknown",
                    "coverage_scope": coverage_scope,
                },
                **(
                    {"error": "UniProt exact accession lookup has unresolved Provider responses."}
                    if unresolved_accessions
                    else {}
                ),
            }
        try:
            result = _invoke_retrieval_skill(
                skill_runtime,
                capability_id="skill.uniprot.query",
                params={
                    "query": query,
                    "include_sequence": False,
                    "size": max(1, min(20, int(max_results))),
                },
                query=query,
                request_id=request_id,
                project_id=project_id,
                conversation_id=conversation_id,
                user_id=user_id,
            )
        except Exception as exc:
            return {
                "ok": False,
                "source_type": "bio_database",
                "database": "uniprot",
                "provider": "uniprot",
                "query": query,
                "records": [],
                "coverage": "unknown",
                "error": f"UniProt capability invocation failed: {type(exc).__name__}",
            }
        provider_payload, outer_payload = _skill_provider_payload(result)
        if not bool(getattr(result, "ok", False)):
            return {
                "ok": False,
                "source_type": "bio_database",
                "database": "uniprot",
                "provider": "uniprot",
                "query": query,
                "records": [],
                "capability_calls": [_skill_call_fact(result)],
                "coverage": "unknown",
                "error": "UniProt capability failed: " + _skill_failure_type(result),
                "provider_observation": {
                    "provider": "uniprot",
                    "query_mode": "search",
                    "requested_ids": [],
                    "returned_ids": [],
                    "not_found_ids": [],
                    "unresolved_ids": [],
                    "coverage": "unknown",
                    "coverage_scope": "",
                },
            }
        records = [
            projected
            for item in provider_payload.get("records") or []
            if isinstance(item, dict)
            and (projected := _project_uniprot_record(item)) is not None
        ]
        returned_ids = [
            str(item.get("accession") or "").strip().upper()
            for item in records
            if str(item.get("accession") or "").strip()
        ]
        return {
            "ok": True,
            "source_type": "bio_database",
            "database": "uniprot",
            "provider": "uniprot",
            "query": query,
            "records": records,
            "capability_calls": [_skill_call_fact(result)],
            **_source_contract_fields(provider_payload, outer_payload),
            "provider_observation": {
                "provider": "uniprot",
                "query_mode": "search",
                "requested_ids": [],
                "returned_ids": returned_ids,
                "not_found_ids": [],
                "unresolved_ids": [],
                "coverage": "unknown",
                "coverage_scope": "",
            },
        }

    def safe_query_ncbi_transcript(query: str, max_results: int = 5) -> dict[str, Any]:
        accessions = _refseq_transcript_accessions_from_text(query)
        symbol = _gene_symbol_from_query(query)
        species = _species_from_query(query)
        organism = {
            "homo_sapiens": "Homo sapiens",
            "mus_musculus": "Mus musculus",
            "rattus_norvegicus": "Rattus norvegicus",
            "danio_rerio": "Danio rerio",
        }.get(species, species.replace("_", " "))
        if not accessions:
            return {
                "ok": False,
                "source_type": "bio_database",
                "database": "ncbi_nuccore",
                "provider": "ncbi_eutils",
                "query": query,
                "records": [],
                "error": "An explicit RefSeq transcript accession is required",
                "provider_observation": {
                    "provider": "ncbi_nuccore",
                    "query_mode": "exact_refseq_transcript",
                    "requested_ids": [],
                    "returned_ids": [],
                    "not_found_ids": [],
                    "unresolved_ids": [],
                    "coverage": "not_applicable",
                    "coverage_scope": "",
                },
            }
        accession = accessions[0]
        scope = f"NCBI Nuccore exact RefSeq transcript {accession}"
        try:
            result = _invoke_retrieval_skill(
                skill_runtime,
                capability_id="skill.public_data.ncbi_gene_cds",
                params={
                    "gene_symbol": symbol,
                    "organism": organism,
                    "transcript_accession": accession,
                },
                query=query,
                request_id=request_id,
                project_id=project_id,
                conversation_id=conversation_id,
                user_id=user_id,
            )
        except Exception as exc:
            return {
                "ok": False,
                "source_type": "bio_database",
                "database": "ncbi_nuccore",
                "provider": "ncbi_eutils",
                "query": query,
                "records": [],
                "error": f"NCBI transcript capability invocation failed: {type(exc).__name__}",
                "provider_observation": {
                    "provider": "ncbi_nuccore",
                    "query_mode": "exact_refseq_transcript",
                    "requested_ids": [accession],
                    "returned_ids": [],
                    "not_found_ids": [],
                    "unresolved_ids": [accession],
                    "coverage": "unknown",
                    "coverage_scope": scope,
                },
            }
        provider_payload, outer_payload = _skill_provider_payload(result)
        capability_call = _skill_call_fact(result)
        data = provider_payload.get("data")
        data = data if isinstance(data, dict) else {}
        returned_accession = str(data.get("accession_version") or "").strip()
        if not bool(getattr(result, "ok", False)):
            error_kind = _skill_failure_type(result)
            not_found = error_kind == "not_found"
            return {
                "ok": False,
                "source_type": "bio_database",
                "database": "ncbi_nuccore",
                "provider": "ncbi_eutils",
                "query": query,
                "records": [],
                "capability_calls": [capability_call],
                "error": "NCBI transcript capability failed: " + error_kind,
                "provider_observation": {
                    "provider": "ncbi_nuccore",
                    "query_mode": "exact_refseq_transcript",
                    "requested_ids": [accession],
                    "returned_ids": [],
                    "not_found_ids": [accession] if not_found else [],
                    "unresolved_ids": [] if not_found else [accession],
                    "coverage": "exhausted" if not_found else "unknown",
                    "coverage_scope": scope,
                },
            }
        resolved_accession = returned_accession or accession
        record_url = (
            "https://www.ncbi.nlm.nih.gov/nuccore/"
            f"{quote(resolved_accession)}"
        )
        record = {
            **data,
            "accession": resolved_accession,
            "title": str(
                data.get("definition")
                or f"NCBI RefSeq transcript {resolved_accession}"
            ),
            "url": record_url,
            "snippet": (
                f"NCBI Nuccore {resolved_accession}; "
                f"gene={data.get('official_symbol')}; "
                f"organism={data.get('organism')}; "
                f"transcript_length={data.get('transcript_length')} bp; "
                f"CDS={data.get('cds_start')}..{data.get('cds_end')}."
            ),
        }
        return {
            "ok": True,
            "source_type": "bio_database",
            "database": "ncbi_nuccore",
            "provider": "ncbi_eutils",
            "query": query,
            "records": [record],
            "capability_calls": [capability_call],
            **_source_contract_fields(provider_payload, outer_payload),
            "provider_observation": {
                "provider": "ncbi_nuccore",
                "query_mode": "exact_refseq_transcript",
                "requested_ids": [accession],
                "returned_ids": [resolved_accession],
                "not_found_ids": [],
                "unresolved_ids": [],
                "coverage": "exhausted",
                "coverage_scope": scope,
            },
        }

    def safe_query_ensembl(query: str, max_results: int = 5) -> dict[str, Any]:
        symbol = _gene_symbol_from_query(query)
        species = _species_from_query(query)
        try:
            result = _invoke_retrieval_skill(
                skill_runtime,
                capability_id="skill.public_data.ensembl",
                params={
                    "operation": "lookup",
                    "species": species,
                    "gene_symbol": symbol,
                },
                query=query,
                request_id=request_id,
                project_id=project_id,
                conversation_id=conversation_id,
                user_id=user_id,
            )
        except Exception as exc:
            return {
                "ok": False,
                "source_type": "bio_database",
                "database": "ensembl",
                "provider": "ensembl_rest",
                "query": query,
                "species": species,
                "symbol": symbol,
                "record": None,
                "error": f"Ensembl capability invocation failed: {type(exc).__name__}",
            }
        provider_payload, outer_payload = _skill_provider_payload(result)
        capability_call = _skill_call_fact(result)
        if not bool(getattr(result, "ok", False)):
            return {
                "ok": False,
                "source_type": "bio_database",
                "database": "ensembl",
                "provider": "ensembl_rest",
                "query": query,
                "species": species,
                "symbol": symbol,
                "record": None,
                "capability_calls": [capability_call],
                "error": "Ensembl capability failed: " + _skill_failure_type(result),
            }
        return {
            "ok": True,
            "source_type": "bio_database",
            "database": "ensembl",
            "provider": "ensembl_rest",
            "query": query,
            "species": species,
            "symbol": symbol,
            "record": provider_payload.get("data"),
            "capability_calls": [capability_call],
            **_source_contract_fields(provider_payload, outer_payload),
        }

    def safe_query_opentarget(query: str, max_results: int = 5) -> dict[str, Any]:
        try:
            limit = max(1, min(10, int(max_results)))
        except (TypeError, ValueError):
            limit = 5
        try:
            result = _invoke_retrieval_skill(
                skill_runtime,
                capability_id="skill.open_targets.query",
                params={
                    "operation": "search_open_targets",
                    "query": str(query or "").strip(),
                    "entity_type": "all",
                    "max_records": limit,
                },
                query=query,
                request_id=request_id,
                project_id=project_id,
                conversation_id=conversation_id,
                user_id=user_id,
            )
        except Exception as exc:
            return {
                "ok": False,
                "source_type": "bio_database",
                "database": "open_targets",
                "query": query,
                "entity_type": "all",
                "records": [],
                "failures": [{"error_type": type(exc).__name__}],
            }
        capability_call = _skill_call_fact(result)
        provider_payload, outer_payload = _skill_provider_payload(result)
        if not bool(getattr(result, "ok", False)):
            return {
                "ok": False,
                "source_type": "bio_database",
                "database": "open_targets",
                "query": query,
                "entity_type": "all",
                "records": [],
                "failures": [{"error_type": _skill_failure_type(result)}],
                "capability_calls": [capability_call],
            }
        records = [
            dict(item)
            for item in provider_payload.get("records") or []
            if isinstance(item, dict)
        ]
        returned_ids = list(
            dict.fromkeys(
                str(item.get("id") or "").strip()
                for item in records
                if str(item.get("id") or "").strip()
            )
        )
        outcome = (
            provider_payload.get("capability_outcome")
            if isinstance(provider_payload.get("capability_outcome"), dict)
            else {}
        )
        return {
            "ok": True,
            "source_type": "bio_database",
            "database": "open_targets",
            "query": query,
            "entity_type": "all",
            "records": records,
            "returned_count": len(records),
            "capability_calls": [capability_call],
            "provider_observation": {
                "provider": "open_targets",
                "query_mode": "entity_search",
                "requested_ids": [],
                "returned_ids": returned_ids,
                "not_found_ids": [],
                "unresolved_ids": [],
                "coverage": str(outcome.get("coverage") or "unknown"),
                "coverage_scope": str(outcome.get("coverage_scope") or ""),
            },
            **_source_contract_fields(provider_payload, outer_payload),
        }

    tool_defs = {
        "safe_query_pubmed": safe_query_pubmed,
        "safe_query_crossref": safe_query_crossref,
        "safe_query_uniprot": safe_query_uniprot,
        "safe_query_ncbi_transcript": safe_query_ncbi_transcript,
        "safe_query_ensembl": safe_query_ensembl,
        "safe_query_clinvar": safe_query_clinvar,
        "safe_query_geo": safe_query_geo,
        "safe_query_opentarget": safe_query_opentarget,
    }
    available = {
        str(tool_id or "").strip()
        for tool_id in available_tool_ids
        if str(tool_id or "").strip()
    }
    built: dict[str, StructuredTool] = {}
    for wrapper_name, func in tool_defs.items():
        route = _DELEGATED_RETRIEVAL_SKILL_ROUTES.get(wrapper_name)
        if route is not None:
            capability_id, typed_tool_id = route
            if typed_tool_id not in available:
                continue
            # A5: the retrieval planner sees the same formal typed tool ID as
            # the lead Agent.  This StructuredTool is only an internal work-
            # item adapter: it accepts the planner's short query and invokes
            # the frozen Capability through ``_invoke_retrieval_skill``.  It
            # must not pretend to expose the formal input schema and then
            # silently discard formal fields.
            spec = skill_runtime.resolve_capability(capability_id)
            description = (
                str(spec.description or "")
                + " （检索子代理内部经冻结的正式 Skill Capability 执行）"
            ).strip()
            tool = StructuredTool.from_function(
                func=func,
                name=typed_tool_id,
                description=description,
                args_schema=SafeQueryInput,
            )
        else:
            tool = StructuredTool.from_function(
                func=func,
                name=wrapper_name,
                description=wrapper_name,
                args_schema=SafeQueryInput,
            )
        built[wrapper_name] = wrap_structured_tool(
            tool,
            category="retrieval_subagent",
            resource_scope=resource_scope,
        )
    return built


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(mode="json"))
        except Exception:
            pass
    return repr(value)


def _store_subagent_trace(payload: dict[str, Any]) -> str:
    """Return a deterministic logical ID; durable bytes live in ResourceRefs."""
    digest = hashlib.sha256(
        json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"retrieval-trace:{digest[:32]}"


def _normalized_sources(sources: list[str] | None, task: str) -> set[str]:
    selected = {str(item or "").strip().lower() for item in (sources or []) if str(item or "").strip()}
    text = str(task or "").lower()
    if any(term in text for term in ("pubmed", "文献", "论文")):
        selected.add("pubmed")
    if _pmid_ids_from_text(text):
        selected.add("pubmed")
    if _doi_ids_from_text(text):
        selected.add("crossref")
    if any(term in text for term in ("最新", "网页", "官网", "web", "policy", "guideline")):
        selected.add("web")
    if any(term in text for term in ("知识库", "项目资料", "上传", "附件", "asset")):
        selected.add("knowledge_assets")
    if any(
        term in text
        for term in (
            "refseq",
            "cds",
            "exon",
            "外显子",
            "转录本",
            "transcript",
            "mrna",
            "sequence",
            "accession",
            "ensembl",
            "uniprot",
            "clinvar",
            "opentarget",
            "open targets",
        )
    ):
        selected.add("bio_database")
    for key in SOURCE_TOOL_HINTS:
        if key in text:
            selected.add(key)
    if not selected:
        selected.add("pubmed")
    return selected


def _build_retrieval_tools(
    *,
    settings: Settings,
    sources: set[str],
    skill_runtime: ContractedSkillRuntime,
    available_tool_ids: set[str] | frozenset[str],
    project_id: str | None,
    conversation_id: str | None,
    user_id: int | None,
    request_id: str | None,
) -> list[StructuredTool]:
    tools: list[StructuredTool] = []
    safe_database_tools = _build_safe_database_tools(
        skill_runtime=skill_runtime,
        available_tool_ids=available_tool_ids,
        project_id=project_id,
        conversation_id=conversation_id,
        user_id=user_id,
        request_id=request_id,
    )

    database_allowlist: set[str] = set()
    # ``bio_database`` is a broad fallback lane.  Once the caller has named a
    # concrete database, expanding that broad lane to every Provider wastes
    # time and can bury an already successful authoritative record under
    # unrelated empty/error results.  Keep the broad union only when no
    # concrete non-literature database was selected.
    specific_database_sources = {
        source
        for source in sources
        if source
        in {
            "uniprot",
            "ensembl",
            "ncbi",
            "nuccore",
            "nucleotide",
            "refseq",
            "clinvar",
            "geo",
            "opentarget",
            "open_targets",
            "crossref",
            "doi",
        }
        or source.startswith("safe_query_")
    }
    for source in sources:
        if source in BROAD_DATABASE_SOURCES and specific_database_sources:
            continue
        hints = SOURCE_TOOL_HINTS.get(source, set())
        database_allowlist.update(hints)
        if source.startswith("safe_query_"):
            database_allowlist.add(source)
    database_tool_order = {
        "safe_query_ncbi_transcript": 5,
        "safe_query_ensembl": 10,
        "safe_query_uniprot": 20,
        "safe_query_pubmed": 30,
        "safe_query_crossref": 35,
        "safe_query_opentarget": 40,
        "safe_query_clinvar": 50,
        "safe_query_geo": 60,
    }
    for tool_name in sorted(database_allowlist, key=lambda item: (database_tool_order.get(item, 100), item)):
        tool = safe_database_tools.get(tool_name)
        if tool is not None:
            tools.append(tool)

    if "web_search" in database_allowlist or "web" in sources or "web_search" in sources:
        tools.append(build_web_search_tool(settings))

    if {"knowledge_assets", "knowledge", "assets", "project"} & sources and project_id and user_id is not None:
        knowledge_names = {
            "search_milvus_knowledge",
            "read_asset_summary",
            "read_asset_page_parsed",
            "read_asset_raw",
        }
        tools.extend(
            tool
            for tool in build_enterprise_tools(
                project_id=project_id,
                user_id=user_id,
                conversation_id=conversation_id,
                request_id=request_id,
                usage_caller="retrieval",
            )
            if str(getattr(tool, "name", "")) in knowledge_names
        )

    return tools


def _build_subagent_llm(settings: Settings) -> ChatDeepSeekThinkingSafe:
    kwargs: dict[str, Any] = {
        "model": os.getenv("EVO_RETRIEVAL_SUBAGENT_LLM") or settings.llm_model,
        "api_key": settings.deepseek_api_key,
        "temperature": 0,
        "max_tokens": _int_env("EVO_RETRIEVAL_SUBAGENT_MAX_TOKENS", 2048),
        "timeout": _float_env("EVO_RETRIEVAL_SUBAGENT_TIMEOUT_SEC", 90.0),
        "max_retries": _int_env("EVO_RETRIEVAL_SUBAGENT_MAX_RETRIES", 1),
    }
    if os.getenv("EVO_RETRIEVAL_SUBAGENT_THINKING_MODE", "disabled").strip().lower() == "disabled":
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return ChatDeepSeekThinkingSafe(**kwargs)


def _tool_args_for_task(tool_name: str, task: str, *, max_evidence: int) -> dict[str, Any]:
    if _retrieval_wrapper_name(tool_name).startswith("safe_query_"):
        return {"query": task, "max_results": max(3, min(10, max_evidence * 2))}
    if tool_name in {"web_search", "search_milvus_knowledge"}:
        return {"query": task}
    if tool_name == "read_asset_page_parsed":
        return {}
    return {"query": task}


def _parse_json_object(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    if not value:
        return {}
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(value[start : end + 1])
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _plan_tool_queries(
    *,
    settings: Settings,
    task: str,
    tool_names: list[str],
) -> dict[str, list[str]]:
    if not tool_names:
        return {}
    prompt = (
        "把用户检索任务改写为每个工具最合适的短查询词。只返回 JSON object，键为工具名，值为查询词字符串或最多 2 个查询词数组。\n"
        "要求：skill_pubmed_literature_query 用英文实体词/主题词，不要带“总结/证据/作者/年份”等指令词；"
        "任务中出现明确技术、药物、基因或疾病短语时必须完整保留，并用 PubMed 字段标签和 AND/OR 连接核心限定词；"
        "不要把具体方法退化为 CRISPR、gene editing、cancer 等上位概念；"
        "skill_public_data_ncbi_gene_cds 必须原样保留 RefSeq accession、物种和基因；"
        "skill_public_data_ensembl/skill_uniprot_query 只保留物种和基因/蛋白实体；web_search 可包含中文或英文；知识库检索保留项目关键词。\n\n"
        f"用户任务：{task}\n"
        f"工具名：{tool_names}\n"
        "示例输出：{\"skill_pubmed_literature_query\":[\"TP53 tumor suppressor cancer\", \"TP53 mutation cancer prognosis\"], \"skill_public_data_ensembl\":\"human TP53\", \"web_search\":\"TP53 tumor suppressor cancer review\"}"
    )
    try:
        response = _build_subagent_llm(settings).invoke(
            [
                SystemMessage(content="你只负责生成检索 query JSON，不解释。"),
                HumanMessage(content=prompt),
            ]
        )
        content = getattr(response, "content", "")
        parsed = _parse_json_object(content if isinstance(content, str) else str(content or ""))
        planned: dict[str, list[str]] = {}
        for key, value in parsed.items():
            tool_name = str(key)
            if tool_name not in tool_names:
                continue
            values = value if isinstance(value, list) else [value]
            cleaned = []
            for item in values:
                text = str(item or "").strip()
                if text and text not in cleaned:
                    cleaned.append(text)
            if cleaned:
                planned[tool_name] = cleaned[:2]
        return planned
    except Exception:
        return {}


def _tool_query_variants(
    *,
    tool_name: str,
    task: str,
    planned_queries: dict[str, list[str]],
) -> list[str]:
    # Exact identifiers are control data, not natural-language query material.
    # Keep the original task so an LLM rewrite cannot drop or mutate an ID, and
    # execute the complete identifier set in one Provider call.
    wrapper_name = _retrieval_wrapper_name(tool_name)
    if wrapper_name == "safe_query_crossref" and _doi_ids_from_text(task):
        return [task]
    if wrapper_name == "safe_query_pubmed":
        pmids = _pmid_ids_from_text(task)
        dois = _doi_ids_from_text(task)
        if pmids and dois:
            # Keep independent stable-ID scopes independent.  This prevents a
            # DOI lookup from shadowing a PMID lookup (or vice versa) while
            # still bypassing model-authored query rewrites.
            return [
                "PMID " + " PMID ".join(pmids),
                "DOI " + " DOI ".join(dois),
            ]
        if pmids or dois:
            return [task]
    if wrapper_name == "safe_query_uniprot" and _uniprot_accessions_from_text(task):
        return [task]
    if wrapper_name == "safe_query_ncbi_transcript" and _refseq_transcript_accessions_from_text(task):
        return [task]
    return planned_queries.get(tool_name) or planned_queries.get(wrapper_name) or [task]


def _invoke_retrieval_tool(
    tool: StructuredTool,
    *,
    task: str,
    max_evidence: int,
    query_task: str | None = None,
    explicit_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tool_name = str(getattr(tool, "name", "") or tool.__class__.__name__)
    if tool_name.startswith("read_asset") and not explicit_args:
        return {
            "tool": tool_name,
            "skipped": True,
            "reason": "read_asset tools need asset_id from a prior search result and are not used in the first isolated runner pass.",
        }
    effective_query = str(query_task or task)
    args = (
        dict(explicit_args)
        if explicit_args is not None
        else _tool_args_for_task(tool_name, effective_query, max_evidence=max_evidence)
    )
    started = time.monotonic()
    try:
        # A retrieval subagent is itself the compositor. It must consume the
        # nested tool's machine result, not the bounded model/UI envelope that
        # will be built once for the subagent's final Evidence Pack.
        machine_tool = getattr(tool, "_evo_raw_tool", None)
        raw_func = getattr(machine_tool, "func", None) or getattr(tool, "func", None)
        if callable(raw_func):
            output = raw_func(**args)
        else:
            output = tool.invoke(args)
        return {
            "tool": tool_name,
            "query": effective_query,
            "args": args,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "output": output,
        }
    except Exception as exc:
        return {
            "tool": tool_name,
            "query": effective_query,
            "args": args,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "output": {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            },
        }


def _derived_asset_read_work_items(
    tool_result: dict[str, Any],
    *,
    max_items: int = 3,
) -> list[dict[str, Any]]:
    """Schedule page expansion only when a Milvus hit is not self-sufficient.

    The search hit already contains one authoritative block. A page read is
    useful only for a missing/very short span or a table/figure whose local
    context is required; blindly reading every hit would recreate the context
    growth this subagent exists to prevent.
    """

    if str(tool_result.get("tool") or "") != "search_milvus_knowledge":
        return []
    output = tool_result.get("output")
    rows = output.get("results") if isinstance(output, dict) else None
    if not isinstance(rows, list):
        return []
    derived: list[dict[str, Any]] = []
    seen_pages: set[tuple[int, int, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            asset_id = int(row.get("asset_id"))
            page_idx = int(row.get("page_idx"))
        except (TypeError, ValueError):
            continue
        text = " ".join(str(row.get("content") or "").split())
        block_type = str(row.get("block_type") or "text").strip().lower()
        needs_context = (
            len(text) < 160
            or "table" in block_type
            or "image" in block_type
            or "figure" in block_type
        )
        if not needs_context:
            continue
        parse_run_id = str(
            row.get("parse_run_id")
            or (row.get("document_ref") or {}).get("parse_run_id")
            or ""
        )
        key = (asset_id, page_idx, parse_run_id)
        if key in seen_pages:
            continue
        seen_pages.add(key)
        arguments = {"asset_id": asset_id, "page_idx": page_idx}
        work_id = hashlib.sha256(
            (
                "read_asset_page_parsed\x1f"
                + json.dumps(arguments, sort_keys=True)
                + "\x1f"
                + parse_run_id
            ).encode("utf-8")
        ).hexdigest()[:24]
        derived.append(
            {
                "work_id": work_id,
                "tool_name": "read_asset_page_parsed",
                "query": str(tool_result.get("query") or ""),
                "arguments": arguments,
                "parent_work_id": str(tool_result.get("work_id") or ""),
                "document_ref": (
                    dict(row.get("document_ref"))
                    if isinstance(row.get("document_ref"), dict)
                    else None
                ),
                "locator": {
                    "asset_id": asset_id,
                    "page_idx": page_idx,
                    "block_idx": row.get("block_idx"),
                    "block_ref": row.get("block_ref"),
                },
                "reason": "search_hit_requires_local_page_context",
            }
        )
        if len(derived) >= max(1, int(max_items)):
            break
    return derived


def _article_id(record: dict[str, Any], id_type: str) -> str:
    for item in record.get("articleids") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("idtype") or "").strip().lower() == id_type:
            value = str(item.get("value") or "").strip()
            if value:
                return value
    return ""


def _year_from_record(record: dict[str, Any]) -> str | None:
    text = str(record.get("pubdate") or record.get("sortpubdate") or "").strip()
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return match.group(0) if match else None


def _raw_payload_from_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    output = result.get("output") if isinstance(result, dict) else None
    if isinstance(output, dict):
        nested = output.get("result")
        if isinstance(nested, dict):
            return nested
        return output
    return {}


def _registered_skill_sources_from_tool_results(
    tool_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collect only sources registered by a delegated Capability Runtime."""

    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in tool_results:
        payload = _raw_payload_from_tool_result(result)
        for item in payload.get("registered_source_ledger") or []:
            if not isinstance(item, dict):
                continue
            registration = item.get("source_registration")
            if (
                not isinstance(registration, dict)
                or registration.get("status") != "registered"
                or registration.get("authority") != "capability_runtime"
            ):
                continue
            # One batch Capability call has one candidate_id but may contain
            # many independently citable records. Deduplicate by the source
            # record's stable identity, never by its batch hand-off identity.
            provider = str(item.get("provider") or "").strip().lower()
            provider_record_id = str(
                item.get("provider_record_id") or item.get("record_id") or ""
            ).strip().lower()
            identity = next(
                (
                    value
                    for value in (
                        f"source:{str(item.get('source_id') or '').strip().lower()}"
                        if str(item.get("source_id") or "").strip()
                        else "",
                        f"provider:{provider}:{provider_record_id}"
                        if provider_record_id
                        else "",
                        f"doi:{normalize_doi(item.get('doi'))}"
                        if normalize_doi(item.get("doi"))
                        else "",
                        f"pmid:{normalize_pmid(item.get('pmid'))}"
                        if normalize_pmid(item.get("pmid"))
                        else "",
                        f"chunk:{str(item.get('chunk_id') or '').strip()}"
                        if str(item.get("chunk_id") or "").strip()
                        else "",
                        f"url:{str(item.get('url') or item.get('canonical_url') or '').strip().lower()}"
                        if str(item.get("url") or item.get("canonical_url") or "").strip()
                        else "",
                    )
                    if value
                ),
                json.dumps(item, ensure_ascii=False, sort_keys=True, default=str),
            )
            if identity in seen:
                continue
            seen.add(identity)
            sources.append(dict(item))
    normalized: list[dict[str, Any]] = []
    for item in sources:
        entry = normalize_source_record(
            item,
            fallback_index=len(normalized) + 1,
        )
        if entry is None:
            continue
        entry["display_index"] = len(normalized) + 1
        entry["ref_id"] = f"R{len(normalized) + 1}"
        entry["citation_keys"] = list(
            dict.fromkeys(
                [
                    entry["ref_id"],
                    *[
                        str(value)
                        for value in entry.get("citation_keys") or []
                        if str(value).strip()
                        and not re.fullmatch(r"R[0-9]+", str(value).strip())
                    ],
                ]
            )
        )
        normalized.append(entry)
    return normalized


def _delegated_source_sidecars_from_tool_results(
    tool_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sidecars: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in tool_results:
        payload = _raw_payload_from_tool_result(result)
        for item in payload.get("source_sidecar_resources") or []:
            if not isinstance(item, dict):
                continue
            resource_id = str(item.get("resource_id") or "").strip()
            if not resource_id or resource_id in seen:
                continue
            seen.add(resource_id)
            sidecars.append(dict(item))
    return sidecars


def _provider_observations_from_tool_results(
    tool_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    for result in tool_results:
        payload = _raw_payload_from_tool_result(result)
        raw_items = payload.get("provider_observations")
        if not isinstance(raw_items, list):
            raw_item = payload.get("provider_observation")
            raw_items = [raw_item] if isinstance(raw_item, dict) else []
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            provider = str(raw.get("provider") or "").strip().lower()
            requested_ids = [
                str(item or "").strip()
                for item in raw.get("requested_ids") or []
                if str(item or "").strip()
            ]
            if not provider or not requested_ids:
                continue
            observation = {
                "provider": provider,
                "query_mode": str(raw.get("query_mode") or "").strip(),
                "requested_ids": requested_ids,
                "returned_ids": [
                    str(item or "").strip()
                    for item in raw.get("returned_ids") or []
                    if str(item or "").strip()
                ],
                "not_found_ids": [
                    str(item or "").strip()
                    for item in raw.get("not_found_ids") or []
                    if str(item or "").strip()
                ],
                "unresolved_ids": [
                    str(item or "").strip()
                    for item in raw.get("unresolved_ids") or []
                    if str(item or "").strip()
                ],
                "coverage": str(raw.get("coverage") or "unknown").strip().lower(),
                "coverage_scope": str(raw.get("coverage_scope") or "").strip(),
            }
            key = (
                provider,
                tuple(requested_ids),
                str(observation.get("query_mode") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            observations.append(observation)
    return observations


def _crossref_scope_outcome_fields(
    provider_observations: list[dict[str, Any]],
) -> dict[str, str]:
    crossref = [
        item
        for item in provider_observations
        if item.get("provider") == "crossref" and item.get("query_mode") == "exact_doi"
    ]
    if not crossref:
        return {}
    requested_ids = {
        str(item)
        for observation in crossref
        for item in observation.get("requested_ids") or []
        if str(item)
    }
    unresolved_ids = {
        str(item)
        for observation in crossref
        for item in observation.get("unresolved_ids") or []
        if str(item)
    }
    if unresolved_ids:
        return {
            "recovery": "retry_same_call",
            "recovery_basis": (
                "CrossRef exact DOI identity lookup has temporary or invalid Provider responses for: "
                + ", ".join(sorted(unresolved_ids))
            ),
        }
    resolved_ids = {
        str(item)
        for observation in crossref
        for key in ("returned_ids", "not_found_ids")
        for item in observation.get(key) or []
        if str(item)
    }
    if not requested_ids or resolved_ids != requested_ids:
        return {}
    scopes = [
        str(item.get("coverage_scope") or "").strip()
        for item in crossref
        if str(item.get("coverage_scope") or "").strip()
    ]
    if not scopes:
        return {}
    return {
        "coverage": "exhausted",
        "coverage_scope": " | ".join(dict.fromkeys(scopes)),
        "coverage_basis": (
            "Every requested DOI returned either an exact CrossRef record or an explicit HTTP 404; "
            "this exhausts only the stated CrossRef exact-identity scope."
        ),
        "recovery": "do_not_retry",
        "recovery_basis": (
            "The same DOI set is fully classified within the stated CrossRef exact-identity scope."
        ),
    }


def _fallback_evidence_pack_from_tool_results(
    *,
    task: str,
    tool_results: list[dict[str, Any]],
    max_evidence: int,
) -> tuple[EvidencePack, list[str]]:
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    limit = max(1, int(max_evidence or 5))

    def add_evidence(item: dict[str, Any]) -> None:
        if len(evidence) >= limit:
            return
        key = (
            str(item.get("pmid") or "").strip().lower()
            or str(item.get("doi") or "").strip().lower()
            or str(item.get("url") or "").strip().lower()
            or str(item.get("title") or "").strip().lower()
        )
        if not key or key in seen:
            return
        seen.add(key)
        item["evidence_id"] = f"E{len(evidence) + 1}"
        evidence.append(item)

    # Provider and asset results already have a deterministic provenance
    # projection.  Reuse it here instead of asking another model call to
    # reconstruct identifiers, locators or record fields from prose.
    for source in _authoritative_sources_from_tool_results(tool_results):
        add_evidence(dict(source))

    claims: list[dict[str, Any]] = []
    for index, item in enumerate(evidence, start=1):
        grounded_text = str(item.get("snippet") or item.get("title") or "").strip()
        if not grounded_text:
            continue
        source_type = str(item.get("source_type") or "").strip().lower()
        support_level = (
            "database"
            if source_type in {"bio_database", "database", "dataset", "data"}
            else "primary"
            if source_type
            in {
                "knowledge_asset",
                "knowledge",
                "kb",
                "pubmed",
                "paper",
                "literature",
            }
            else "source_fallback"
        )
        claims.append(
            {
                "claim_id": f"C{index}",
                # Using the grounded span itself makes the deterministic
                # Claim-Evidence verifier sufficient; semantic interpretation
                # remains the main Agent's job.
                "claim": _first_semantic_sentence(grounded_text),
                "support_level": support_level,
                "evidence_ids": [item["evidence_id"]],
            }
        )
        pmid = normalize_pmid(item.get("pmid"))
        doi = normalize_doi(item.get("doi"))
        if pmid and doi:
            identity_facts = [f"PMID={pmid}", f"DOI={doi}"]
            title = str(item.get("title") or "").strip()
            if title:
                identity_facts.append(f"title={title}")
            claims.append(
                {
                    "claim_id": f"C-ID-{index}",
                    "claim": "; ".join(identity_facts),
                    "support_level": support_level,
                    "evidence_ids": [item["evidence_id"]],
                }
            )
    warnings = ["fallback_evidence_pack_from_tool_results"] if evidence else ["fallback_evidence_pack_empty"]
    return EvidencePack.model_validate(
        {
            "query": task,
            "claims": claims,
            "evidence": evidence,
            "limitations": [
                "Evidence Pack 由权威工具原始结果确定性构造；事实范围仅限 evidence 字段。"
            ]
            if evidence
            else ["检索工具结果不足，无法构造 Evidence Pack。"],
        }
    ), warnings


def _source_identity_keys(item: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    pmid = re.sub(r"\D+", "", str(item.get("pmid") or item.get("uid") or ""))
    if pmid:
        keys.append(f"pmid:{pmid}")
    doi = str(item.get("doi") or "").strip().lower()
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi)
    if doi:
        keys.append(f"doi:{doi}")
    provider = str(item.get("provider") or (item.get("metadata") or {}).get("source_tool") or "").strip().lower()
    provider_record_id = str(
        item.get("provider_record_id")
        or (item.get("metadata") or {}).get("provider_record_id")
        or (item.get("metadata") or {}).get("record_id")
        or ""
    ).strip()
    if provider and provider_record_id:
        keys.append(f"provider:{provider}:{provider_record_id}")
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    asset_id = str(metadata.get("asset_id") or item.get("asset_id") or "").strip()
    if asset_id:
        keys.append(f"asset:{asset_id}")
    dataset_id = str(metadata.get("dataset_id") or item.get("dataset_id") or "").strip()
    dataset_record_id = str(metadata.get("record_id") or item.get("record_id") or "").strip()
    if dataset_id and dataset_record_id:
        keys.append(f"dataset:{dataset_id}:{dataset_record_id}")
    url = str(item.get("url") or item.get("link") or item.get("href") or "").strip().lower()
    if url:
        keys.append(f"url:{url}")
    title = re.sub(r"\W+", " ", str(item.get("title") or "").lower(), flags=re.UNICODE).strip()
    if title:
        keys.append(f"title:{title}")
    return keys


def _first_semantic_sentence(text: Any) -> str:
    """Select one complete leading sentence without a character cutoff."""

    value = str(text or "").strip()
    for match in re.finditer(r"(?:[。！？!?]|\.(?:\s|$))", value):
        end = match.end()
        if value[:end].strip():
            return value[:end].strip()
    first_paragraph = value.split("\n\n", 1)[0].strip()
    return first_paragraph or value


def _first_record_value(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, (str, int, float)) and str(value).strip():
            return str(value).strip()
    return ""


def _nested_record_text(record: dict[str, Any], *path: str) -> str:
    value: Any = record
    for key in path:
        if not isinstance(value, dict):
            return ""
        value = value.get(key)
    return str(value or "").strip() if isinstance(value, (str, int, float)) else ""


def _database_record_url(database: str, record_id: str) -> str:
    db = str(database or "").strip().lower()
    value = str(record_id or "").strip()
    if not value:
        return ""
    if db == "uniprot":
        return f"https://www.uniprot.org/uniprotkb/{quote(value)}/entry"
    if db == "ensembl" and value.startswith("ENS"):
        return f"https://www.ensembl.org/Homo_sapiens/Gene/Summary?g={quote(value)}"
    if db == "geo":
        return f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={quote(value)}"
    if db == "clinvar" and value.isdigit():
        return f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{quote(value)}/"
    if db == "open_targets":
        if value.startswith("ENSG"):
            return f"https://platform.opentargets.org/target/{quote(value)}"
        if value.startswith("EFO_") or value.startswith("MONDO_"):
            return f"https://platform.opentargets.org/disease/{quote(value)}"
    return ""


def _authoritative_candidate(
    *,
    tool_name: str,
    payload: dict[str, Any],
    item: dict[str, Any],
    raw_ref: str | None,
    query: Any,
) -> dict[str, Any] | None:
    database = str(payload.get("database") or "").strip().lower()
    record_id = _first_record_value(
        item,
        "primaryAccession",
        "accession",
        "accessionversion",
        "uid",
        "pmid",
        "id",
        "identifier",
    )
    title = _first_record_value(item, "title", "name", "label", "display_name")
    if not title:
        title = _nested_record_text(item, "proteinDescription", "recommendedName", "fullName", "value")
    # Providers/readers own paging. Evidence Pack consumes one complete
    # grounded span and never applies a second character slice.
    snippet = _first_record_value(
        item,
        "content",
        "snippet",
        "summary",
        "description",
        "function",
    )
    if not snippet:
        # Structured Providers such as UniProt keep useful facts in nested
        # records. Preserve the complete projected record; the much larger
        # machine payload already lives behind raw_ref/ResourceRef.
        snippet = json.dumps(
            _json_safe(item), ensure_ascii=False, separators=(",", ":")
        )
    url = _first_record_value(item, "url", "link", "href") or _database_record_url(database, record_id)
    asset_id = _first_record_value(item, "asset_id")
    source_type = (
        "web"
        if tool_name == "web_search"
        else "knowledge_asset"
        if tool_name == "search_milvus_knowledge" or asset_id
        else str(payload.get("source_type") or "bio_database")
    )
    if not any((title, url, snippet, record_id, asset_id)):
        return None
    if not title:
        title = _first_record_value(item, "file_name") or f"{database or tool_name} {record_id or asset_id}".strip()
        if not title:
            title = _first_semantic_sentence(snippet)
    metadata: dict[str, Any] = {
        "source_tool": tool_name,
        "query": query,
        "retrieval_reconciled": True,
    }
    if database:
        metadata["database"] = database
    if record_id:
        metadata["record_id"] = record_id
        metadata["provider_record_id"] = record_id
    if database == "uniprot":
        metadata["record_fields"] = {
            key: item.get(key)
            for key in (
                "accession",
                "entry_name",
                "recommended_name",
                "gene_names",
                "organism",
                "function",
                "official_url",
            )
            if item.get(key) not in (None, "", [])
        }
    for key in (
        "asset_id",
        "asset_version",
        "asset_sha256",
        "file_name",
        "page_idx",
        "block_idx",
        "bbox",
        "chunk_id",
        "dataset_id",
        "dataset_version",
        "record_id",
    ):
        if item.get(key) not in (None, "", []):
            metadata[key] = item.get(key)
    return {
        "source_type": source_type,
        "title": title or url or _first_semantic_sentence(snippet),
        "url": url or None,
        "snippet": snippet or title,
        "raw_ref": raw_ref,
        "provider": database or tool_name,
        "provider_record_id": record_id or (asset_id if source_type == "knowledge_asset" else None),
        "canonical_url": url or None,
        "metadata": metadata,
    }


def _authoritative_sources_from_tool_results(tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    classified_exact_ids: set[str] = set()
    for result in tool_results:
        if not isinstance(result, dict):
            continue
        payload = _raw_payload_from_tool_result(result)
        observations = payload.get("provider_observations")
        if not isinstance(observations, list):
            observation = payload.get("provider_observation")
            observations = [observation] if isinstance(observation, dict) else []
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            query_mode = str(observation.get("query_mode") or "").strip().lower()
            coverage = str(observation.get("coverage") or "").strip().lower()
            unresolved = [
                str(item or "").strip()
                for item in observation.get("unresolved_ids") or []
                if str(item or "").strip()
            ]
            requested = {
                str(item or "").strip().lower()
                for item in observation.get("requested_ids") or []
                if str(item or "").strip()
            }
            resolved = {
                str(item or "").strip().lower()
                for key in ("returned_ids", "not_found_ids")
                for item in observation.get(key) or []
                if str(item or "").strip()
            }
            if (
                query_mode.startswith("exact_")
                and coverage == "exhausted"
                and not unresolved
                and requested
                and requested.issubset(resolved)
            ):
                classified_exact_ids.update(requested)
    for result in tool_results:
        if not isinstance(result, dict):
            continue
        tool_name = str(result.get("tool") or "").strip()
        payload = _raw_payload_from_tool_result(result)
        output = result.get("output") if isinstance(result.get("output"), dict) else {}
        raw_ref = str(output.get("raw_ref") or "").strip() or None
        requested_dois = {
            doi
            for value in (
                list(payload.get("requested_ids") or [])
                + _doi_ids_from_text(str(result.get("query") or ""))
            )
            if (doi := normalize_doi(value))
        }
        requested_pmids = {
            pmid
            for value in (
                list(payload.get("requested_ids") or [])
                + _pmid_ids_from_text(str(result.get("query") or ""))
            )
            if (pmid := normalize_pmid(value))
        }
        candidates: list[dict[str, Any]] = []
        if _retrieval_wrapper_name(tool_name) == "safe_query_pubmed":
            for record in payload.get("records") or []:
                if not isinstance(record, dict):
                    continue
                pmid = str(record.get("pmid") or record.get("uid") or _article_id(record, "pubmed")).strip()
                doi = normalize_doi(_article_id(record, "doi") or record.get("doi"))
                if requested_dois and doi not in requested_dois:
                    continue
                if requested_pmids and normalize_pmid(pmid) not in requested_pmids:
                    continue
                title = str(record.get("title") or "").strip()
                content_snippet = str(record.get("abstract") or title).strip()
                identity_facts = [
                    value
                    for value in (
                        f"PMID={pmid}" if pmid else "",
                        f"DOI={doi}" if doi else "",
                        f"title={title}" if title else "",
                    )
                    if value
                ]
                identity_text = "; ".join(identity_facts)
                snippet = (
                    f"{content_snippet}\n\n{identity_text}"
                    if content_snippet and identity_text
                    else content_snippet or identity_text
                )
                candidates.append(
                    {
                        "source_type": "pubmed",
                        "title": title,
                        "pmid": pmid or None,
                        "doi": doi or None,
                        "journal": str(record.get("fulljournalname") or record.get("source") or "").strip() or None,
                        "year": _year_from_record(record),
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                        "provider": "pubmed",
                        "provider_record_id": pmid or None,
                        "canonical_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                        "snippet": snippet,
                        "raw_ref": raw_ref,
                        "metadata": {
                            "source_tool": tool_name,
                            "query": result.get("query"),
                            "retrieval_reconciled": True,
                            "abstract_available": bool(str(record.get("abstract") or "").strip()),
                        },
                    }
                )
        elif _retrieval_wrapper_name(tool_name) == "safe_query_crossref":
            for record in payload.get("records") or []:
                if not isinstance(record, dict):
                    continue
                doi = normalize_doi(record.get("doi"))
                if not doi or (requested_dois and doi not in requested_dois):
                    continue
                url = f"https://doi.org/{doi}"
                candidates.append(
                    {
                        "source_type": "paper",
                        "title": str(record.get("title") or f"CrossRef record {doi}").strip(),
                        "doi": doi,
                        "url": url,
                        "provider": "crossref",
                        "provider_record_id": doi,
                        "canonical_url": url,
                        "snippet": str(
                            record.get("snippet") or record.get("title") or ""
                        ).strip(),
                        "raw_ref": raw_ref,
                        "metadata": {
                            "source_tool": tool_name,
                            "query": result.get("query"),
                            "retrieval_reconciled": True,
                            "record_type": record.get("record_type"),
                            "publisher": record.get("publisher"),
                            "published": record.get("published"),
                        },
                    }
                )
        else:
            raw_candidates: list[Any] = []
            for key in ("items", "results", "records", "articles"):
                value = payload.get(key)
                if isinstance(value, list):
                    raw_candidates.extend(value)
            record = payload.get("record")
            if isinstance(record, dict):
                raw_candidates.append(record)
            for item in raw_candidates:
                if not isinstance(item, dict):
                    continue
                if tool_name == "web_search" and classified_exact_ids:
                    serialized_item = json.dumps(
                        _json_safe(item),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).lower()
                    if not any(identifier in serialized_item for identifier in classified_exact_ids):
                        continue
                candidate = _authoritative_candidate(
                    tool_name=tool_name,
                    payload=payload,
                    item=item,
                    raw_ref=raw_ref,
                    query=result.get("query"),
                )
                if candidate:
                    candidates.append(candidate)
        for candidate in candidates:
            keys = _source_identity_keys(candidate)
            canonical = keys[0] if keys else ""
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            sources.append(candidate)
    return sources


def _reconcile_evidence_pack_sources(
    pack: EvidencePack,
    tool_results: list[dict[str, Any]],
) -> tuple[EvidencePack, list[str]]:
    authoritative = _authoritative_sources_from_tool_results(tool_results)
    by_key: dict[str, dict[str, Any]] = {}
    for source in authoritative:
        for key in _source_identity_keys(source):
            by_key.setdefault(key, source)
    if not by_key:
        return pack, ["authoritative_source_index_empty"]

    verified_evidence = []
    dropped_ids: set[str] = set()
    for evidence in pack.evidence:
        source = next(
            (by_key[key] for key in _source_identity_keys(evidence.model_dump(mode="json")) if key in by_key),
            None,
        )
        if source is None:
            dropped_ids.add(str(evidence.evidence_id or "").strip())
            continue
        # Reconciliation is the provenance trust boundary.  Clear every
        # model-authored identity, locator and verification field before
        # restoring values from the matched raw tool result.  In particular,
        # a KB hit may match by its real title without retaining a model-added
        # DOI, PMID, URL or raw trace reference that the hit never returned.
        authoritative_fields: dict[str, Any] = {
            "source_type": "unknown",
            "title": "",
            "pmid": None,
            "doi": None,
            "journal": None,
            "year": None,
            "url": None,
            "snippet": "",
            "raw_ref": None,
            "provider": None,
            "provider_record_id": None,
            "canonical_url": None,
        }
        for field, default in authoritative_fields.items():
            setattr(evidence, field, source.get(field, default))
        evidence.source_id = None
        evidence.content_hash = None
        evidence.span_id = None
        evidence.context_before = ""
        evidence.context_after = ""
        evidence.verification = SourceVerification()
        # Reconciliation is the trust boundary: model-authored metadata may
        # carry descriptive hints only, while every identity/provenance field
        # must come from the authoritative raw tool result.
        evidence.metadata = dict(source.get("metadata") or {})
        evidence.metadata.pop("source_verified", None)
        evidence.metadata["retrieval_reconciled"] = True
        verified_evidence.append(evidence)
    pack.evidence = verified_evidence
    if dropped_ids:
        for claim in pack.claims:
            claim.evidence_ids = [item for item in claim.evidence_ids if str(item).strip() not in dropped_ids]
        pack.limitations.append(
            f"Removed {len(dropped_ids)} evidence record(s) that could not be matched to raw retrieval results."
        )
    return pack, ([f"unverified_evidence_removed:{len(dropped_ids)}"] if dropped_ids else [])


def _fatal_retrieval_warnings(warnings: list[str]) -> list[str]:
    fatal_prefixes = ("subagent_final_not_json", "evidence_pack_validation_failed")
    return [item for item in warnings if any(str(item).startswith(prefix) for prefix in fatal_prefixes)]


def _requested_minimum_evidence_count(task: str) -> int:
    text = str(task or "")
    number_words = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
    }

    def _number(value: str) -> int:
        clean = str(value or "").strip().lower()
        if clean.isdigit():
            return max(1, min(12, int(clean)))
        return number_words.get(clean, 1)

    minimum = 1
    source_match = re.search(
        r"(\d+|[一二两三四五]|one|two|three|four|five)\s*(?:条|个|篇|项)?\s*(?:可核验|独立|不同)?\s*(?:来源|文献|证据|references?|sources?|citations?)",
        text,
        flags=re.IGNORECASE,
    )
    if source_match:
        minimum = max(minimum, _number(source_match.group(1)))
    if re.search(r"每(?:条|个|项|篇|个结论).{0,18}(?:来源|文献|证据|reference|source|citation)", text, flags=re.IGNORECASE):
        item_match = re.search(
            r"(\d+|[一二两三四五]|one|two|three|four|five)\s*(?:条|个|项)?\s*"
            r"(?:主要|独立|不同|关键|核心|main|key|distinct|independent)?\s*"
            r"(?:限制|结论|要点|问题|目标|limitations?|claims?|points?)",
            text,
            flags=re.IGNORECASE,
        )
        if item_match:
            minimum = max(minimum, _number(item_match.group(1)))
    return minimum


def _evidence_pack_is_sufficient(pack: EvidencePack, *, minimum_evidence_count: int = 1) -> bool:
    evidence_ids = {
        str(item.evidence_id or "").strip()
        for item in pack.evidence
        if str(item.evidence_id or "").strip()
    }
    if not pack.claims or len(evidence_ids) < max(1, int(minimum_evidence_count or 1)):
        return False
    limitation_text = " ".join(str(item or "").strip().lower() for item in pack.limitations)
    insufficient_markers = (
        "fallback",
        "兜底",
        "不足",
        "无法回答",
        "无法确定",
        "cannot answer",
        "could not answer",
        "not retrieved",
        "insufficient evidence",
    )
    if any(marker in limitation_text for marker in insufficient_markers):
        return False
    rejected_support_levels = {"source_fallback", "weak", "unknown"}
    if any(str(claim.support_level or "").strip().lower() in rejected_support_levels for claim in pack.claims):
        return False
    completed_links = [
        link for link in pack.claim_evidence_links if link.verification_state == "completed"
    ]
    if completed_links:
        acceptable_verdicts = {"supported", "partially_supported"}
        if not all(
            any(
                link.claim_id == claim.claim_id and link.verdict in acceptable_verdicts
                for link in completed_links
            )
            for claim in pack.claims
        ):
            return False
    return all(
        any(str(evidence_id or "").strip() in evidence_ids for evidence_id in claim.evidence_ids)
        for claim in pack.claims
    )


def _register_retrieval_temp_files(
    *,
    project_id: str | None,
    conversation_id: str | None,
    user_id: int | None,
    request_id: str | None,
    trace_ref: str,
    trace_payload: dict[str, Any],
    evidence_pack: EvidencePack,
    reference_records: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if user_id is None or not project_id or not conversation_id or not conversation_file_registry.INTERNAL_TOKEN:
        return {"ok": False, "skipped": True, "reason": "missing_conversation_file_context"}
    raw_tool_records: list[dict[str, Any]] = []
    for result in tool_results:
        output = result.get("output") if isinstance(result, dict) else None
        raw_ref = output.get("raw_ref") if isinstance(output, dict) else None
        raw_tool_records.append(
            {
                "tool": result.get("tool") if isinstance(result, dict) else "",
                "query": result.get("query") if isinstance(result, dict) else "",
                "raw_ref": raw_ref,
                "resources": (
                    output.get("resources")
                    if isinstance(output, dict) and isinstance(output.get("resources"), list)
                    else []
                ),
                "output": output,
            }
        )

    def _json_file(file_name: str, payload: Any, source_type: str) -> dict[str, Any]:
        text = json.dumps(_json_safe(payload), ensure_ascii=False, indent=2)
        return {
            "file_name": file_name,
            "content_bytes": text.encode("utf-8"),
            "content_text": text,
            "mime_type": "application/json; charset=utf-8",
            "source_type": source_type,
            "source_ref_id": f"retrieval:{trace_ref}:{file_name}",
            "display_path": f"retrieval/{trace_ref}/{file_name}",
            "drawer_section": "temporary_output",
            "archive_status": "pending",
            "tool_name": "run_retrieval_subagent",
            "trace_ref": trace_ref,
            "retention_policy": "ephemeral",
            "is_visible": False,
            "artifact_metadata": {
                "path": f"retrieval/{trace_ref}/{file_name}",
                "kind": "trace",
                "created_by": "retrieval_subagent",
                "trace_ref": trace_ref,
            },
        }

    files = [
        _json_file(f"{trace_ref}_trace.json", {"trace_ref": trace_ref, **trace_payload}, "retrieval_trace"),
        _json_file(f"{trace_ref}_evidence_pack.json", evidence_pack_to_plain_dict(evidence_pack), "retrieval_trace"),
        _json_file(f"{trace_ref}_raw_tools.json", {"trace_ref": trace_ref, "tool_results": raw_tool_records}, "tool_raw_result"),
    ]
    if reference_records:
        references_text = "\n".join(json.dumps(_json_safe(item), ensure_ascii=False) for item in reference_records) + "\n"
        files.append(
            {
                "file_name": f"{trace_ref}_references.jsonl",
                "content_bytes": references_text.encode("utf-8"),
                "content_text": references_text,
                "mime_type": "application/jsonl; charset=utf-8",
                "source_type": "retrieval_reference",
                "source_ref_id": f"retrieval:{trace_ref}:references",
                "display_path": f"retrieval/{trace_ref}/{trace_ref}_references.jsonl",
                "drawer_section": "temporary_output",
                "archive_status": "pending",
                "tool_name": "run_retrieval_subagent",
                "trace_ref": trace_ref,
                "retention_policy": "ephemeral",
                "is_visible": False,
                "artifact_metadata": {
                    "path": f"retrieval/{trace_ref}/{trace_ref}_references.jsonl",
                    "kind": "trace",
                    "created_by": "retrieval_subagent",
                    "trace_ref": trace_ref,
                },
            }
        )
    payload = conversation_file_registry.register_conversation_file_batch(
        user_id=int(user_id),
        project_id=str(project_id),
        conversation_id=str(conversation_id),
        request_id=str(request_id or "").strip() or None,
        files=files,
    )
    return payload if isinstance(payload, dict) else {"ok": False, "skipped": True, "reason": "register_failed"}


def _retrieval_resource_scope(
    *,
    project_id: str | None,
    conversation_id: str | None,
    user_id: int | None,
    request_id: str | None,
    call_id: str | None = None,
):
    from src.services.resource_store import ResourceScope

    return ResourceScope(
        user_id=user_id,
        project_id=project_id,
        conversation_id=conversation_id,
        request_id=request_id,
        call_id=call_id,
    )


def _retrieval_objective_hash(task: str, sources: set[str] | list[str]) -> str:
    payload = {
        "task": str(task or ""),
        "sources": sorted(str(item or "").strip().lower() for item in sources),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _retrieval_registry_snapshot_id(available_tool_ids: set[str] | frozenset[str]) -> str:
    return "registry:" + hashlib.sha256(
        json.dumps(sorted(str(item) for item in available_tool_ids)).encode("utf-8")
    ).hexdigest()[:24]


def _resource_ref_dict(resource_ref: Any) -> dict[str, Any] | None:
    if resource_ref is None:
        return None
    if hasattr(resource_ref, "model_dump"):
        payload = resource_ref.model_dump(mode="json")
        return payload if isinstance(payload, dict) else None
    return dict(resource_ref) if isinstance(resource_ref, dict) else None


def _persist_retrieval_checkpoint(
    *,
    project_id: str | None,
    conversation_id: str | None,
    user_id: int | None,
    request_id: str | None,
    payload: dict[str, Any],
):
    from src.services.resource_store import persist_resource_bytes, serialize_resource_payload

    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return persist_resource_bytes(
        scope=_retrieval_resource_scope(
            project_id=project_id,
            conversation_id=conversation_id,
            user_id=user_id,
            request_id=request_id,
            call_id=f"retrieval-checkpoint:{digest[:20]}",
        ),
        content_bytes=serialize_resource_payload(payload),
        kind="retrieval_checkpoint",
        source_type="retrieval_checkpoint",
        file_name=f"retrieval_checkpoint_{digest[:20]}.json",
        media_type="application/vnd.evoengine.retrieval-checkpoint+json",
        tool_name="run_retrieval_subagent",
        metadata={"checkpoint_digest": digest},
    )


def _load_retrieval_checkpoint(
    *,
    project_id: str | None,
    conversation_id: str | None,
    user_id: int | None,
    request_id: str | None,
    resource_id: str | None,
) -> dict[str, Any] | None:
    if not str(resource_id or "").strip():
        return None
    from src.services.resource_store import read_resource_json

    payload = read_resource_json(
        scope=_retrieval_resource_scope(
            project_id=project_id,
            conversation_id=conversation_id,
            user_id=user_id,
            request_id=request_id,
        ),
        resource_id=str(resource_id),
    )
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != "evoengine.retrieval-checkpoint/v1":
        return None
    task = str(payload.get("task") or "")
    sources = [str(item or "") for item in payload.get("sources") or []]
    objective_hash = str(payload.get("objective_hash") or "")
    if not task or objective_hash != _retrieval_objective_hash(task, sources):
        return None
    work_items = [
        item for item in payload.get("work_items") or [] if isinstance(item, dict)
    ]
    work_ids = [str(item.get("work_id") or "") for item in work_items]
    if not work_ids or any(not item for item in work_ids) or len(work_ids) != len(set(work_ids)):
        return None
    completed = {
        str(item or "") for item in payload.get("completed_work_ids") or []
    }
    if not completed.issubset(set(work_ids)):
        return None
    result_work_ids = [
        str(item.get("work_id") or "")
        for item in payload.get("tool_results") or []
        if isinstance(item, dict)
    ]
    if len(result_work_ids) != len(set(result_work_ids)):
        return None
    if any(item not in completed for item in result_work_ids):
        return None
    if payload.get("message_history_included") is not False:
        return None
    return payload


def build_retrieval_subagent_tool(
    *,
    settings: Settings,
    skill_typed_schema_build: SkillTypedSchemaBuild,
    available_tool_ids: set[str] | frozenset[str],
    project_id: str | None = None,
    conversation_id: str | None = None,
    user_id: int | None = None,
    request_id: str | None = None,
) -> StructuredTool:
    skill_runtime = _retrieval_skill_runtime_from_frozen_build(
        skill_typed_schema_build,
        available_tool_ids=available_tool_ids,
    )

    def _run_retrieval_subagent(
        task: str = "",
        sources: list[str] | None = None,
        max_iterations: int = 6,
        max_evidence: int = 5,
        checkpoint_resource_id: str | None = None,
        batch_size: int = 6,
        output_mode: Literal["evidence_pack"] = "evidence_pack",
    ) -> dict[str, Any]:
        started = time.monotonic()
        checkpoint = _load_retrieval_checkpoint(
            project_id=project_id,
            conversation_id=conversation_id,
            user_id=user_id,
            request_id=request_id,
            resource_id=checkpoint_resource_id,
        )
        if checkpoint is not None:
            # A durable checkpoint is the continuation authority.  The caller
            # must not reconstruct or restate its natural-language task,
            # sources, work list or budgets from compacted model history.
            task = str(checkpoint.get("task") or "")
            selected_sources = {
                str(item or "").strip().lower()
                for item in (checkpoint.get("sources") or [])
                if str(item or "").strip()
            }
            max_iterations = max(
                1, min(6, int(checkpoint.get("max_iterations") or max_iterations))
            )
            max_evidence = max(
                1, min(12, int(checkpoint.get("max_evidence") or max_evidence))
            )
            batch_size = max(
                1, min(6, int(checkpoint.get("batch_size") or batch_size))
            )
            output_mode = "evidence_pack"
        else:
            selected_sources = _normalized_sources(sources, task)
        objective_hash = _retrieval_objective_hash(task, selected_sources)
        retrieval_task_id = str(
            (checkpoint or {}).get("retrieval_task_id")
            or (
                "retrieval-task:"
                + hashlib.sha256(
                    f"{project_id}|{conversation_id}|{objective_hash}".encode("utf-8")
                ).hexdigest()[:24]
            )
        )
        registry_snapshot_id = str(
            (checkpoint or {}).get("registry_snapshot_id")
            or _retrieval_registry_snapshot_id(available_tool_ids)
        )
        tools = _build_retrieval_tools(
            settings=settings,
            sources=selected_sources,
            skill_runtime=skill_runtime,
            available_tool_ids=available_tool_ids,
            project_id=project_id,
            conversation_id=conversation_id,
            user_id=user_id,
            request_id=request_id,
        )
        tool_names = [str(getattr(tool, "name", "")) for tool in tools]
        if not tools and not checkpoint_resource_id:
            pack = EvidencePack(
                query=task,
                claims=[],
                evidence=[],
                limitations=["没有可用于该检索任务的子代理工具；请改用主 agent 已装载工具或补充来源。"],
            )
            trace_ref = _store_subagent_trace(
                {"task": task, "sources": sorted(selected_sources), "tool_names": [], "error": "no_tools_available"}
            )
            return {
                "ok": False,
                "output_mode": output_mode,
                "model_summary": "检索子代理未找到可用检索工具。",
                "retrieval_run_completed": False,
                "evidence_available": False,
                "evidence_pack_claims_supported": False,
                "provider_observations": [],
                "capability_outcome": {
                    "schema_version": "evoengine.capability-outcome/v1",
                    "applicability": "applicable",
                    "applicability_basis": "检索子代理接受了 Evidence Pack 任务，但没有可调用的来源工具。",
                    "evidence": "none",
                    "evidence_basis": "本次运行没有返回 Evidence 记录。",
                    "coverage": "unknown",
                    "recovery": "switch_capability",
                },
                "completion_signal": {
                    "schema_version": "tool_completion_signal_v1",
                    "status": "failed",
                    "should_stop_tool_loop": False,
                    "next_action": "main_agent_evaluate_goal_coverage",
                    "reason": "Evidence Pack 未形成；主 Agent 根据根目标选择其他已声明能力或说明限制。",
                },
                "recommended_action": "main_agent_evaluate_goal_coverage",
                "evidence_pack": evidence_pack_to_plain_dict(pack),
                "source_ledger": [],
                "trace_ref": trace_ref,
                "diagnostics": {
                    "elapsed_ms": int((time.monotonic() - started) * 1000),
                    "sources": sorted(selected_sources),
                    "tool_names": [],
                    "warnings": ["no_tools_available"],
                },
            }

        warnings: list[str] = []
        try:
            runnable_tools = tools[: max(1, min(len(tools), int(max_iterations)))]
            if checkpoint_resource_id and checkpoint is None:
                raise ValueError("retrieval checkpoint 不存在、无权访问或合同无效")
            if checkpoint is not None:
                planned_queries = dict(checkpoint.get("planned_queries") or {})
                work_items = [
                    dict(item)
                    for item in (checkpoint.get("work_items") or [])
                    if isinstance(item, dict)
                ]
                tool_results = [
                    dict(item)
                    for item in (checkpoint.get("tool_results") or [])
                    if isinstance(item, dict)
                ]
                completed_work_ids = {
                    str(item or "")
                    for item in (checkpoint.get("completed_work_ids") or [])
                    if str(item or "").strip()
                }
                batch_index = max(1, int(checkpoint.get("batch_index") or 0) + 1)
            else:
                planner_tool_names = [
                    str(getattr(tool, "name", ""))
                    for tool in runnable_tools
                    if not str(getattr(tool, "name", "")).startswith("read_asset")
                    if not (
                        str(getattr(tool, "name", "")) == "skill_crossref_doi_identity"
                        and _doi_ids_from_text(task)
                    )
                    and not (
                        str(getattr(tool, "name", "")) == "skill_pubmed_literature_query"
                        and (_pmid_ids_from_text(task) or _doi_ids_from_text(task))
                    )
                ]
                planned_queries = _plan_tool_queries(
                    settings=settings,
                    task=task,
                    tool_names=planner_tool_names,
                )
                work_items: list[dict[str, Any]] = []
                for tool in runnable_tools:
                    tool_name = str(getattr(tool, "name", ""))
                    if tool_name.startswith("read_asset"):
                        continue
                    for query_variant in _tool_query_variants(
                        tool_name=tool_name,
                        task=task,
                        planned_queries=planned_queries,
                    )[:2]:
                        work_id = hashlib.sha256(
                            f"{tool_name}\x1f{query_variant}".encode("utf-8")
                        ).hexdigest()[:24]
                        if any(item.get("work_id") == work_id for item in work_items):
                            continue
                        work_items.append(
                            {
                                "work_id": work_id,
                                "tool_name": tool_name,
                                "query": query_variant,
                            }
                        )
                        if len(work_items) >= max(1, int(max_iterations)):
                            break
                    if len(work_items) >= max(1, int(max_iterations)):
                        break
                tool_results = []
                completed_work_ids: set[str] = set()
                batch_index = 1

            subagent_run_id = "retrieval-run:" + hashlib.sha256(
                (
                    retrieval_task_id
                    + "|"
                    + str(batch_index)
                    + "|"
                    + json.dumps(sorted(completed_work_ids))
                ).encode("utf-8")
            ).hexdigest()[:24]
            batch_id = f"{retrieval_task_id}:batch:{batch_index}"

            tools_by_name = {
                str(getattr(tool, "name", "")): tool for tool in runnable_tools
            }
            # Compatibility for checkpoints written before the A5 identity
            # convergence.  New checkpoints persist formal typed IDs; old
            # internal adapter names remain readable but are never advertised.
            for typed_tool_id, wrapper_name in _TYPED_TO_RETRIEVAL_WRAPPER.items():
                if typed_tool_id in tools_by_name:
                    tools_by_name.setdefault(wrapper_name, tools_by_name[typed_tool_id])
            structured_provider_evidence_found = bool(
                _authoritative_sources_from_tool_results(tool_results)
            )
            pending_work = [
                item
                for item in work_items
                if str(item.get("work_id") or "") not in completed_work_ids
            ]
            for work_item in pending_work[: max(1, min(int(batch_size), 6))]:
                work_id = str(work_item.get("work_id") or "")
                tool_name = str(work_item.get("tool_name") or "")
                tool = tools_by_name.get(tool_name)
                if tool is None:
                    raise ValueError(f"retrieval checkpoint tool unavailable: {tool_name}")
                if tool_name == "web_search" and structured_provider_evidence_found:
                    completed_work_ids.add(work_id)
                    continue
                tool_result = _invoke_retrieval_tool(
                    tool,
                    task=task,
                    max_evidence=max_evidence,
                    query_task=str(work_item.get("query") or task),
                    explicit_args=(
                        dict(work_item.get("arguments") or {})
                        if isinstance(work_item.get("arguments"), dict)
                        else None
                    ),
                )
                tool_result["work_id"] = work_id
                tool_results.append(tool_result)
                completed_work_ids.add(work_id)
                for derived_item in _derived_asset_read_work_items(tool_result):
                    derived_work_id = str(derived_item.get("work_id") or "")
                    if not derived_work_id or any(
                        str(existing.get("work_id") or "") == derived_work_id
                        for existing in work_items
                    ):
                        continue
                    if len(work_items) >= max(1, int(max_iterations)):
                        break
                    work_items.append(derived_item)
                if _retrieval_wrapper_name(tool_name).startswith("safe_query_"):
                    provider_sources = [
                        item
                        for item in _authoritative_sources_from_tool_results([tool_result])
                        if str(item.get("source_type") or "").strip().lower()
                        in {"bio_database", "database", "dataset", "data"}
                        and str(item.get("provider_record_id") or "").strip()
                    ]
                    if provider_sources:
                        structured_provider_evidence_found = True

            remaining_work = [
                item
                for item in work_items
                if str(item.get("work_id") or "") not in completed_work_ids
            ]
            checkpoint_payload = {
                "schema_version": "evoengine.retrieval-checkpoint/v1",
                "retrieval_task_id": retrieval_task_id,
                "subagent_run_id": subagent_run_id,
                "batch_id": batch_id,
                "objective_hash": objective_hash,
                "registry_snapshot_id": registry_snapshot_id,
                "parent_request_id": str(request_id or ""),
                "previous_checkpoint_resource_id": str(
                    checkpoint_resource_id or ""
                )
                or None,
                "task": task,
                "sources": sorted(selected_sources),
                "planned_queries": planned_queries,
                "work_items": work_items,
                "completed_work_ids": sorted(completed_work_ids),
                "completed_scope": sorted(completed_work_ids),
                "pending_scope": [
                    str(item.get("work_id") or "") for item in remaining_work
                ],
                "capability_call_fingerprints": sorted(completed_work_ids),
                "tool_results": tool_results,
                "batch_index": batch_index,
                "max_iterations": max_iterations,
                "max_evidence": max_evidence,
                "batch_size": batch_size,
                "output_mode": output_mode,
                "complete": not remaining_work,
                "has_more": bool(remaining_work),
                "message_history_included": False,
            }
            checkpoint_ref = _persist_retrieval_checkpoint(
                project_id=project_id,
                conversation_id=conversation_id,
                user_id=user_id,
                request_id=request_id,
                payload=checkpoint_payload,
            )
            delegated_source_sidecars = _delegated_source_sidecars_from_tool_results(
                tool_results
            )
            registered_skill_sources = (
                _registered_skill_sources_from_tool_results(tool_results)
                if citations_enabled()
                else []
            )
            if remaining_work:
                if checkpoint_ref is None:
                    raise RuntimeError("retrieval checkpoint persistence failed")
                partial_pack, partial_warnings = _fallback_evidence_pack_from_tool_results(
                    task=task,
                    tool_results=tool_results,
                    max_evidence=max_evidence,
                )
                checkpoint_resource = _resource_ref_dict(checkpoint_ref)
                return {
                    "ok": True,
                    "output_mode": output_mode,
                    "model_summary": f"检索子代理已完成第 {batch_index} 批，尚有 {len(remaining_work)} 个工作项；使用 checkpoint 继续。",
                    "retrieval_run_completed": False,
                    "complete": False,
                    "has_more": True,
                    "cursor": checkpoint_ref.resource_id,
                    "checkpoint_resource_id": checkpoint_ref.resource_id,
                    "retrieval_task_id": retrieval_task_id,
                    "subagent_run_id": subagent_run_id,
                    "batch_id": batch_id,
                    "raw_ref": checkpoint_ref.resource_id,
                    "resources": [
                        item
                        for item in [checkpoint_resource, *delegated_source_sidecars]
                        if item is not None
                    ],
                    "source_sidecar_resources": delegated_source_sidecars,
                    "source_sidecar_refs": [
                        item["resource_id"] for item in delegated_source_sidecars
                    ],
                    "batch_index": batch_index,
                    "evidence_available": bool(partial_pack.evidence),
                    "evidence_pack_claims_supported": False,
                    "provider_observations": _provider_observations_from_tool_results(tool_results),
                    "capability_outcome": {
                        "schema_version": "evoengine.capability-outcome/v1",
                        "applicability": "applicable",
                        "applicability_basis": "检索任务已进入确定性分批执行。",
                        "evidence": "partial" if partial_pack.evidence else "none",
                        "evidence_basis": "当前仅完成部分已声明工作项。",
                        "coverage": "more_available",
                        "coverage_scope": "当前检索 checkpoint 中的已规划工作项",
                        "coverage_basis": f"仍有 {len(remaining_work)} 个工作项未执行。",
                        "recovery": "retry_same_call",
                        "recovery_basis": "携带 checkpoint_resource_id 继续下一批。",
                    },
                    "completion_signal": {
                        "schema_version": "tool_completion_signal_v1",
                        "status": "partial",
                        "should_stop_tool_loop": False,
                        "next_action": "continue_retrieval_checkpoint",
                        "reason": "同一 checkpoint 尚有未执行工作项。",
                    },
                    "recommended_action": "continue_retrieval_checkpoint",
                    "evidence_pack": evidence_pack_to_plain_dict(partial_pack),
                    "source_ledger": registered_skill_sources,
                    "trace_ref": checkpoint_ref.resource_id,
                    "diagnostics": {
                        "elapsed_ms": int((time.monotonic() - started) * 1000),
                        "warnings": partial_warnings,
                        "completed_work_ids": sorted(completed_work_ids),
                        "remaining_work_count": len(remaining_work),
                        "message_history_included": False,
                    },
                }

            provider_observations = _provider_observations_from_tool_results(tool_results)
            provider_scope_outcome = _crossref_scope_outcome_fields(provider_observations)
            deterministic_pack, deterministic_warnings = _fallback_evidence_pack_from_tool_results(
                task=task,
                tool_results=tool_results,
                max_evidence=max_evidence,
            )
            # Every retrieval tool already returns a structured, persisted raw
            # result.  Evidence identity and provenance are therefore a
            # deterministic projection problem, not another generation task.
            # The main Agent interprets the compact evidence for the user's
            # question; web/search snippets deliberately remain
            # ``source_fallback`` until that judgment is made.
            pack = deterministic_pack
            warnings.extend(deterministic_warnings)
            warnings.append("deterministic_tool_evidence_pack")
            pack, reconciliation_warnings = _reconcile_evidence_pack_sources(pack, tool_results)
            warnings.extend(reconciliation_warnings)
            pack, contract_warnings = materialize_source_records(pack)
            warnings.extend(contract_warnings)
            # Formal Skill results have already been projected, registered and
            # persisted by CapabilityExecutor. Reusing their Citation handles
            # is the only valid path: a second network/claim verification pass
            # would create another, potentially conflicting source truth.
            # The old verifier remains only for retrieval sources that have no
            # declared Capability source contract (web/asset compatibility).
            if citations_enabled() and not registered_skill_sources:
                pack, source_verification_warnings = verify_evidence_pack_sources(pack)
                warnings.extend(source_verification_warnings)
                pack, claim_verification_warnings = verify_claim_evidence_links(pack)
                warnings.extend(claim_verification_warnings)
            trace_payload = {
                "task": task,
                "sources": sorted(selected_sources),
                "tool_names": tool_names,
                "planned_queries": planned_queries,
                "tool_results": tool_results,
                "provider_observations": provider_observations,
                "warnings": warnings,
            }
            trace_ref = (
                checkpoint_ref.resource_id
                if checkpoint_ref is not None
                else _store_subagent_trace(trace_payload)
            )
            for evidence in pack.evidence:
                if not evidence.raw_ref:
                    evidence.raw_ref = trace_ref
            source_ledger = (
                registered_skill_sources
                if registered_skill_sources
                else build_source_ledger_from_evidence_pack(pack, trace_ref=trace_ref)
            )
            reference_records = evidence_pack_to_reference_records(
                pack,
                task=task,
                trace_ref=trace_ref,
                request_id=request_id,
                project_id=project_id,
                conversation_id=conversation_id,
                source_tools=tool_names,
            )
            reference_store = append_reference_records(reference_records)
            drawer_store = _register_retrieval_temp_files(
                project_id=project_id,
                conversation_id=conversation_id,
                user_id=user_id,
                request_id=request_id,
                trace_ref=trace_ref,
                trace_payload=trace_payload,
                evidence_pack=pack,
                reference_records=reference_records,
                tool_results=tool_results,
            )
            elapsed_ms = int((time.monotonic() - started) * 1000)
            fatal_warnings = _fatal_retrieval_warnings(warnings)
            evidence_available = bool(pack.claims or pack.evidence)
            minimum_evidence_count = _requested_minimum_evidence_count(task)
            evidence_pack_claims_supported = bool(
                registered_skill_sources
                or (
                    evidence_available
                    and not fatal_warnings
                    and _evidence_pack_is_sufficient(
                        pack,
                        minimum_evidence_count=minimum_evidence_count,
                    )
                )
            )
            completion_next_action = "main_agent_evaluate_goal_coverage"
            completion_reason = (
                "正式能力结果已有 Runtime 登记来源；这只描述来源可发布状态，"
                "不表示用户根目标完成。主 agent 仍需按原始目标决定交付、补充检索或继续下游。"
                if registered_skill_sources
                else "Evidence Pack 内已有可发布的 claim-evidence 支持关系；这只描述包内证据，"
                "不表示用户根目标完成。主 agent 仍需按原始目标决定交付、补充检索或继续下游。"
                if evidence_pack_claims_supported
                else "Evidence Pack 尚未形成可发布的 claim-evidence 支持关系；这只描述本次检索结果，"
                "主 agent 仍需按原始目标决定说明限制、定向恢复或选择其他能力。"
            )
            evidence_state = (
                "available"
                if evidence_pack_claims_supported
                else "partial"
                if evidence_available
                else "none"
            )
            return {
                "ok": not fatal_warnings,
                "output_mode": output_mode,
                "model_summary": (
                    f"检索子代理返回 {len(pack.claims)} 条结论、{len(pack.evidence)} 条证据；"
                    "主 agent 应结合用户目标、limitations 和当前任务树判断是否足以完成当前任务。"
                    if pack.claims or pack.evidence
                    else "检索子代理未形成可用证据。"
                ),
                "retrieval_run_completed": True,
                "complete": True,
                "has_more": False,
                "cursor": None,
                "checkpoint_resource_id": (
                    checkpoint_ref.resource_id if checkpoint_ref is not None else None
                ),
                "retrieval_task_id": retrieval_task_id,
                "subagent_run_id": subagent_run_id,
                "batch_id": batch_id,
                "raw_ref": trace_ref,
                "resources": [
                    resource
                    for resource in [
                        _resource_ref_dict(checkpoint_ref),
                        *delegated_source_sidecars,
                    ]
                    if resource is not None
                ],
                "source_sidecar_resources": delegated_source_sidecars,
                "source_sidecar_refs": [
                    item["resource_id"] for item in delegated_source_sidecars
                ],
                "batch_index": batch_index,
                "evidence_available": evidence_available,
                "evidence_pack_claims_supported": evidence_pack_claims_supported,
                "provider_observations": provider_observations,
                "capability_outcome": {
                    "schema_version": "evoengine.capability-outcome/v1",
                    "applicability": "applicable",
                    "applicability_basis": "检索子代理已接受任务并完成本次已选择来源的调用。",
                    "evidence": evidence_state,
                    "evidence_basis": (
                        "正式能力返回的来源已由 Capability Runtime 登记并提供 Citation。"
                        if registered_skill_sources
                        else "Evidence Pack 内主张均有已校验的 claim-evidence 支持关系。"
                        if evidence_pack_claims_supported
                        else "本次返回了证据，但尚未形成完整可发布的 claim-evidence 支持关系。"
                        if evidence_available
                        else "本次调用未返回可用 evidence 记录。"
                    ),
                    "coverage": "unknown",
                    "recovery": "unknown",
                    **provider_scope_outcome,
                },
                "completion_signal": {
                    "schema_version": "tool_completion_signal_v1",
                    "status": "retrieval_run_completed",
                    "should_stop_tool_loop": False,
                    "next_action": completion_next_action,
                    "reason": completion_reason,
                    "acceptance": {
                        "evidence_pack_claims_supported": evidence_pack_claims_supported,
                        "claim_count": len(pack.claims),
                        "evidence_count": len(pack.evidence),
                        "minimum_evidence_count": minimum_evidence_count,
                        "source_ledger_count": len(source_ledger),
                    },
                },
                "recommended_action": completion_next_action,
                "evidence_pack": evidence_pack_to_plain_dict(pack),
                "source_ledger": source_ledger,
                "trace_ref": trace_ref,
                "diagnostics": {
                    "elapsed_ms": elapsed_ms,
                    "sources": sorted(selected_sources),
                    "tool_names": tool_names,
                    "warnings": warnings,
                    "claim_count": len(pack.claims),
                    "evidence_count": len(pack.evidence),
                    "minimum_evidence_count": minimum_evidence_count,
                    "source_ledger_count": len(source_ledger),
                    "reference_count": int(reference_store.get("count") or 0),
                    "references_path": reference_store.get("path"),
                    "references_ok": bool(reference_store.get("ok")),
                    "fatal_warnings": fatal_warnings,
                    "drawer_temp_files_ok": bool(drawer_store.get("ok")),
                    "drawer_temp_registered_count": int(drawer_store.get("registered_count") or 0),
                },
            }
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            error_message = str(exc).strip() or type(exc).__name__
            error_code = (
                "retrieval_checkpoint_invalid"
                if "checkpoint" in error_message.lower()
                else "retrieval_execution_failed"
            )
            trace_ref = _store_subagent_trace(
                {
                    "task": task,
                    "sources": sorted(selected_sources),
                    "tool_names": tool_names,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            pack = EvidencePack(
                query=task,
                claims=[],
                evidence=[],
                limitations=["检索子代理执行失败；主 agent 未接收原始检索 payload。"],
            )
            return {
                "ok": False,
                "error_kind": "validation_error" if error_code == "retrieval_checkpoint_invalid" else "execution_error",
                "error_code": error_code,
                "error": error_message,
                "output_mode": output_mode,
                "model_summary": f"检索子代理执行失败：{error_message}",
                "retrieval_run_completed": False,
                "evidence_available": False,
                "evidence_pack_claims_supported": False,
                "provider_observations": [],
                "capability_outcome": {
                    "schema_version": "evoengine.capability-outcome/v1",
                    "applicability": "applicable",
                    "applicability_basis": "检索子代理已接受 Evidence Pack 任务。",
                    "evidence": "none",
                    "evidence_basis": "本次运行失败，没有形成可交付 Evidence Pack 证据。",
                    "coverage": "unknown",
                    "recovery": "unknown",
                },
                "completion_signal": {
                    "schema_version": "tool_completion_signal_v1",
                    "status": "failed",
                    "should_stop_tool_loop": False,
                    "next_action": "main_agent_evaluate_goal_coverage",
                    "reason": "检索运行失败；主 Agent 根据根目标判断定向恢复、切换能力或说明限制。",
                },
                "recommended_action": "main_agent_evaluate_goal_coverage",
                "evidence_pack": evidence_pack_to_plain_dict(pack),
                "source_ledger": [],
                "trace_ref": trace_ref,
                "diagnostics": {
                    "elapsed_ms": elapsed_ms,
                    "sources": sorted(selected_sources),
                    "tool_names": tool_names,
                    "error_type": type(exc).__name__,
                    "error_code": error_code,
                },
            }

    raw_tool = StructuredTool.from_function(
        func=_run_retrieval_subagent,
        name="run_retrieval_subagent",
        description=(
            "启动隔离检索子代理完成文献、网页、知识库或公开数据库检索，并且只交付紧凑 Evidence Pack；"
            "它不交付完整原始序列、引物、分析文件、沙盒结果、业务报告，也不替主 Agent 判断根任务完成。"
            "适合多源证据总结、来源核对和引用补充，避免把大段工具结果注入主上下文。"
            "跨 DOI/PMID Provider 的论文身份核验、CrossRef 精确 DOI 查询或确定性 not_found 范围应使用本工具；"
            "web_search 只能补充网页线索，不能替代这些精确 Provider 查询。"
            "provider_observations.coverage_scope 与 negative_conclusion_contract 是否定结论的范围边界；"
            "不得据此推断未查询 Provider 的结果或全局不存在，需要更强结论时应查询对应 Provider。"
            "不要用于已知 ID 文件下载、沙盒计算、沙盒结果读取、报告生成或可视化收尾。"
        ),
        args_schema=RetrievalSubagentInput,
    )
    setattr(raw_tool, "_evo_output_schema", RetrievalSubagentResult.model_json_schema())
    return wrap_structured_tool(
        raw_tool,
        category="retrieval_subagent",
        resource_scope=_retrieval_resource_scope(
            project_id=project_id,
            conversation_id=conversation_id,
            user_id=user_id,
            request_id=request_id,
        ),
    )
