from __future__ import annotations

import json

from src.subagents.retrieval_types import EvidencePack
from src.subagents.evidence_pack import normalize_evidence_pack
from src.subagents.source_evidence_contracts import (
    materialize_source_records,
    normalize_doi,
    normalize_http_url,
    normalize_pmid,
)


def test_identifier_normalizers_are_strict() -> None:
    assert normalize_doi("https://doi.org/10.1093/clinchem/hvp165") == "10.1093/clinchem/hvp165"
    assert normalize_doi("doi: 10.1000/ABC") == "10.1000/abc"
    assert normalize_doi("10.1000/bad doi") == ""
    assert normalize_pmid("PMID: 12345678") == "12345678"
    assert normalize_pmid("PMID 12345678 extra") == ""
    assert normalize_pmid("abc123") == ""


def test_http_url_normalizer_rejects_credentials_and_unsafe_schemes() -> None:
    assert normalize_http_url("HTTPS://Example.org/a#fragment") == "https://example.org/a"
    assert normalize_http_url("https://redacted:redacted@example.invalid/a") == ""
    assert normalize_http_url("javascript:alert(1)") == ""


def test_materialize_sources_dedupes_identity_but_keeps_evidence_spans() -> None:
    pack = EvidencePack(
        query="qPCR",
        claims=[{"claim_id": "C1", "claim": "MIQE defines reporting guidance", "evidence_ids": ["E1", "E2"]}],
        evidence=[
            {
                "evidence_id": "E1",
                "source_type": "pubmed",
                "title": "MIQE guidelines",
                "doi": "10.1093/clinchem/hvp165",
                "pmid": "19246619",
                "url": "https://publisher.example/paper",
                "snippet": "First span",
            },
            {
                "evidence_id": "E2",
                "source_type": "pubmed",
                "title": "MIQE guidelines",
                "doi": "https://doi.org/10.1093/clinchem/hvp165",
                "snippet": "Second span",
            },
        ],
    )

    normalized, warnings = materialize_source_records(pack)

    assert warnings == []
    assert len(normalized.sources) == 1
    assert normalized.evidence[0].source_id == normalized.evidence[1].source_id
    assert normalized.sources[0].canonical_url == "https://doi.org/10.1093/clinchem/hvp165"
    assert {(item.claim_id, item.evidence_id) for item in normalized.claim_evidence_links} == {
        ("C1", "E1"),
        ("C1", "E2"),
    }
    assert all(item.verification_state == "not_run" for item in normalized.claim_evidence_links)


def test_same_title_with_distinct_doi_is_not_merged() -> None:
    pack = EvidencePack(
        query="same title",
        evidence=[
            {"evidence_id": "E1", "title": "Shared title", "doi": "10.1000/a", "snippet": "A"},
            {"evidence_id": "E2", "title": "Shared title", "doi": "10.1000/b", "snippet": "B"},
        ],
    )

    normalized, _warnings = materialize_source_records(pack)

    assert len(normalized.sources) == 2
    assert normalized.evidence[0].source_id != normalized.evidence[1].source_id


def test_invalid_and_duplicate_links_are_removed_without_breaking_old_payloads() -> None:
    pack = EvidencePack(
        query="legacy",
        claims=[{"claim_id": "C1", "claim": "A", "evidence_ids": ["E1"]}],
        evidence=[{"evidence_id": "E1", "title": "A", "url": "https://example.org/a"}],
        claim_evidence_links=[
            {"claim_id": "C1", "evidence_id": "E1"},
            {"claim_id": "C1", "evidence_id": "E1"},
            {"claim_id": "C1", "evidence_id": "missing"},
        ],
    )

    normalized, warnings = materialize_source_records(pack)

    assert len(normalized.claim_evidence_links) == 1
    assert any(item.startswith("duplicate_claim_evidence_link") for item in warnings)
    assert any(item.startswith("invalid_claim_evidence_link") for item in warnings)
    assert normalized.schema_version == "evoengine.evidence-pack/v2"


def test_stable_aliases_union_doi_and_pmid_records() -> None:
    pack = EvidencePack(
        query="aliases",
        evidence=[
            {"evidence_id": "E1", "title": "Paper", "doi": "10.1000/shared", "pmid": "12345"},
            {"evidence_id": "E2", "title": "Paper (indexed)", "pmid": "12345"},
        ],
    )

    normalized, warnings = materialize_source_records(pack)

    assert warnings == []
    assert len(normalized.sources) == 1
    assert normalized.evidence[0].source_id == normalized.evidence[1].source_id


def test_shared_doi_with_conflicting_stable_identifier_is_marked_conflicted() -> None:
    pack = EvidencePack(
        query="conflict",
        evidence=[
            {"evidence_id": "E1", "title": "Paper A", "doi": "10.1000/shared", "pmid": "12345"},
            {"evidence_id": "E2", "title": "Paper B", "doi": "10.1000/shared", "pmid": "67890"},
        ],
    )

    normalized, warnings = materialize_source_records(pack)

    assert len(normalized.sources) == 1
    assert normalized.sources[0].verification.identity_status == "conflicted"
    assert all(item.verification.identity_status == "conflicted" for item in normalized.evidence)
    assert warnings == [f"source_identity_conflict:{normalized.sources[0].source_id}"]


def test_shared_dataset_landing_page_does_not_merge_distinct_provider_records() -> None:
    pack = EvidencePack(
        query="Orphadata classifications",
        evidence=[
            {
                "evidence_id": "E1",
                "source_type": "database_record",
                "title": "Orphadata classification component 156",
                "provider": "orphadata",
                "provider_record_id": "product3_156:2025-07-01:sha256-a",
                "url": "https://sciences.orphadata.com/classifications/",
                "snippet": "Component 156",
            },
            {
                "evidence_id": "E2",
                "source_type": "database_record",
                "title": "Orphadata classification component 205",
                "provider": "orphadata",
                "provider_record_id": "product3_205:2025-07-01:sha256-b",
                "url": "https://sciences.orphadata.com/classifications/",
                "snippet": "Component 205",
            },
        ],
    )

    normalized, warnings = materialize_source_records(pack)

    assert warnings == []
    assert len(normalized.sources) == 2
    assert normalized.evidence[0].source_id != normalized.evidence[1].source_id


def test_model_submitted_pack_cannot_self_assert_verification() -> None:
    raw = """{
      "query": "q",
      "sources": [{"source_id": "forged", "identity_status": "verified"}],
      "claims": [{"claim_id": "C1", "claim": "A", "evidence_ids": ["E1"]}],
      "evidence": [{
        "evidence_id": "E1",
        "title": "A",
        "url": "https://example.org/a",
        "source_id": "forged",
        "provider": "forged",
        "provider_record_id": "forged",
        "verification": {"identity_status": "verified", "link_status": "reachable"},
        "metadata": {
          "source_verified": true,
          "retrieval_reconciled": true,
          "source_tool": "search_milvus_knowledge",
          "provider_record_id": "forged",
          "asset_id": 999,
          "page_idx": 88,
          "block_idx": 77,
          "bbox": [0, 0, 1, 1],
          "chunk_id": "forged:chunk",
          "dataset_id": "forged-dataset",
          "dataset_version": "forged-version",
          "record_id": "forged-record",
          "query_time": "2099-01-01T00:00:00Z",
          "descriptive_note": "model-authored note"
        }
      }],
      "claim_evidence_links": [{
        "claim_id": "C1",
        "evidence_id": "E1",
        "verification_state": "completed",
        "verdict": "supported"
      }]
    }"""

    pack, warnings = normalize_evidence_pack(raw, fallback_query="q", max_evidence=5)

    assert warnings == []
    assert pack.sources == []
    assert pack.claim_evidence_links == []
    assert pack.evidence[0].source_id is None
    assert pack.evidence[0].provider is None
    assert pack.evidence[0].provider_record_id is None
    assert pack.evidence[0].verification.identity_status == "unverified"
    assert pack.evidence[0].verification.link_status == "unknown"
    assert "source_verified" not in pack.evidence[0].metadata
    assert "retrieval_reconciled" not in pack.evidence[0].metadata
    for key in (
        "source_tool",
        "provider_record_id",
        "asset_id",
        "page_idx",
        "block_idx",
        "bbox",
        "chunk_id",
        "dataset_id",
        "dataset_version",
        "record_id",
        "query_time",
    ):
        assert key not in pack.evidence[0].metadata
    assert "descriptive_note" not in pack.evidence[0].metadata


def test_evidence_pack_projection_limit_is_explicit_and_machine_trace_remains_addressable() -> None:
    raw = json.dumps(
        {
            "query": "q",
            "claims": [
                {"claim_id": f"C{index}", "claim": f"claim {index}", "evidence_ids": [f"E{index}"]}
                for index in range(1, 4)
            ],
            "evidence": [
                {
                    "evidence_id": f"E{index}",
                    "title": f"Source {index}",
                    "url": f"https://example.org/{index}",
                    "snippet": f"evidence {index}",
                    "raw_ref": f"resource:provider:{index}",
                }
                for index in range(1, 4)
            ],
        }
    )

    pack, warnings = normalize_evidence_pack(raw, fallback_query="q", max_evidence=2)

    assert len(pack.evidence) == 2
    assert len(pack.claims) == 2
    assert "evidence_pack_projection_limit_applied" in warnings
    assert any("ResourceRef/trace" in item for item in pack.limitations)
