from __future__ import annotations

import hashlib
import re
from typing import Any
from urllib.parse import urlparse, urlunparse

from src.subagents.retrieval_types import (
    ClaimEvidenceLink,
    EvidencePack,
    RetrievalEvidence,
    SourceRecord,
    SourceVerification,
)


_DOI_RE = re.compile(
    r"^10\.\d{4,9}/[^\s\x00-\x1f\x7f\"'<>\\\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+$",
    re.IGNORECASE,
)
_PMID_RE = re.compile(r"^[1-9]\d{0,11}$")

# Trailing punctuation that terminates a DOI inside natural language.  A DOI
# character class must never absorb sentence punctuation; otherwise
# ``10.1186/1471-2105-9-559.`` becomes a different (invalid) identity and the
# record is wrongly rejected as unrequested.
_DOI_TRAILING_PUNCT_RE = re.compile(r"[.,;，。；？?！!）)\]]+$")

# Candidate DOI pattern inside running text.  ``.`` stays inside the class
# (DOIs legitimately contain it, e.g. ``10.1000/foo.bar``) and any trailing
# punctuation is stripped by ``normalize_doi``; comma/semicolon/question mark
# (English and full-width) and CJK text terminate the candidate.
_DOI_CANDIDATE_RE = re.compile(
    r"\b10\.\d{4,9}/[^\s,;?，。；？!！）)\]\u3000\x00-\x1f\x7f\"'<>\\\u4e00-\u9fff\uff00-\uffef]+",
    re.IGNORECASE,
)


def normalize_doi(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"^doi:\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.IGNORECASE).strip()
    text = _DOI_TRAILING_PUNCT_RE.sub("", text).strip()
    return text.lower() if _DOI_RE.fullmatch(text) else ""


def extract_dois_from_text(value: Any) -> list[str]:
    """Deterministically extract and normalize every DOI in running text.

    This is the single shared DOI extraction implementation for source
    contracts: CrossRef projection, the CrossRef typed Skill and retrieval
    adapters must all consume it instead of maintaining private regexes.
    Handles doi.org URLs, ``doi:`` prefixes, multiple DOIs in one sentence and
    English/Chinese sentence punctuation.
    """

    text = str(value or "")
    if not text.strip():
        return []
    candidates = _DOI_CANDIDATE_RE.findall(text)
    seen: set[str] = set()
    extracted: list[str] = []
    for candidate in candidates:
        doi = normalize_doi(candidate)
        if not doi or doi in seen:
            continue
        seen.add(doi)
        extracted.append(doi)
    return extracted


def normalize_pmid(value: Any) -> str:
    """Return a PMID only when the complete input is a valid numeric identifier."""

    text = str(value or "").strip()
    text = re.sub(r"^pmid:\s*", "", text, flags=re.IGNORECASE).strip()
    return text if _PMID_RE.fullmatch(text) else ""


def normalize_http_url(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text)
    except Exception:
        return ""
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return ""
    if parsed.username or parsed.password:
        return ""
    host = parsed.hostname.lower().rstrip(".")
    if not host:
        return ""
    port = parsed.port
    netloc = host if port is None else f"{host}:{port}"
    return urlunparse(
        parsed._replace(
            scheme=parsed.scheme.lower(),
            netloc=netloc,
            fragment="",
        )
    )


def canonical_url_for_evidence(evidence: RetrievalEvidence) -> str:
    doi = normalize_doi(evidence.doi)
    if doi:
        return f"https://doi.org/{doi}"
    pmid = normalize_pmid(evidence.pmid)
    if pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    return normalize_http_url(evidence.canonical_url) or normalize_http_url(evidence.url)


def source_identity_aliases(evidence: RetrievalEvidence) -> set[str]:
    aliases: set[str] = set()
    doi = normalize_doi(evidence.doi)
    if doi:
        aliases.add(f"doi:{doi}")
    pmid = normalize_pmid(evidence.pmid)
    if pmid:
        aliases.add(f"pmid:{pmid}")
    provider = str(evidence.provider or evidence.metadata.get("source_tool") or "").strip().lower()
    record_id = str(
        evidence.provider_record_id
        or evidence.metadata.get("provider_record_id")
        or evidence.metadata.get("record_id")
        or ""
    ).strip()
    if provider and record_id:
        aliases.add(f"provider:{provider}:{record_id}")
    dataset_id = str(evidence.metadata.get("dataset_id") or "").strip()
    dataset_version = str(evidence.metadata.get("dataset_version") or "").strip()
    dataset_record = str(evidence.metadata.get("record_id") or "").strip()
    if dataset_id and dataset_record:
        aliases.add(f"dataset:{dataset_id}:{dataset_version}:{dataset_record}")
    asset_id = str(evidence.metadata.get("asset_id") or "").strip()
    asset_version = str(
        evidence.metadata.get("asset_version") or evidence.metadata.get("asset_sha256") or ""
    ).strip()
    if asset_id:
        aliases.add(f"asset:{asset_id}:{asset_version}")
    url = canonical_url_for_evidence(evidence)
    # A URL is a locator, not necessarily a record identity. Dataset landing
    # pages commonly describe several independently versioned components, so
    # allowing a shared URL to union records that already have stable IDs can
    # silently collapse distinct sources. Use URL identity only as the
    # fallback for evidence without a stronger identifier.
    if url and not aliases:
        aliases.add(f"url:{url.lower()}")
    return aliases


def source_identity_key(evidence: RetrievalEvidence) -> str:
    aliases = source_identity_aliases(evidence)
    if not aliases:
        return f"evidence:{evidence.evidence_id}"
    priority = ("doi:", "pmid:", "provider:", "dataset:", "asset:", "url:")
    for prefix in priority:
        match = next((item for item in sorted(aliases) if item.startswith(prefix)), None)
        if match:
            return match
    return sorted(aliases)[0]


def stable_source_id(identity_key: str) -> str:
    digest = hashlib.sha256(str(identity_key or "").encode("utf-8")).hexdigest()[:20]
    return f"SRC-{digest}"


def _identity_conflicts(items: list[RetrievalEvidence]) -> bool:
    def values(field: str) -> set[str]:
        return {
            str(getattr(item, field, None) or "").strip().lower()
            for item in items
            if str(getattr(item, field, None) or "").strip()
        }

    normalized_dois = {normalize_doi(item.doi) for item in items if normalize_doi(item.doi)}
    normalized_pmids = {normalize_pmid(item.pmid) for item in items if normalize_pmid(item.pmid)}
    # Title punctuation, translation and truncation are common across providers;
    # a title difference alone is not an identity conflict. Stable identifiers are.
    return len(normalized_dois) > 1 or len(normalized_pmids) > 1 or len(values("provider_record_id")) > 1


def materialize_source_records(pack: EvidencePack) -> tuple[EvidencePack, list[str]]:
    """Assign stable source IDs, de-duplicate identities and validate links.

    This is a contract operation only. It does not claim that a source has
    been reached or that its content supports a Claim.
    """

    warnings: list[str] = []
    evidence_count = len(pack.evidence)
    parents = list(range(evidence_count))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    alias_owner: dict[str, int] = {}
    for index, evidence in enumerate(pack.evidence):
        for alias in source_identity_aliases(evidence):
            owner = alias_owner.setdefault(alias, index)
            union(index, owner)

    grouped_by_root: dict[int, list[RetrievalEvidence]] = {}
    for index, evidence in enumerate(pack.evidence):
        grouped_by_root.setdefault(find(index), []).append(evidence)

    grouped: dict[str, list[RetrievalEvidence]] = {}
    for root, items in grouped_by_root.items():
        aliases = set().union(*(source_identity_aliases(item) for item in items))
        identity_key = source_identity_key(items[0]) if not aliases else sorted(aliases)[0]
        for prefix in ("doi:", "pmid:", "provider:", "dataset:", "asset:", "url:"):
            preferred = next((item for item in sorted(aliases) if item.startswith(prefix)), None)
            if preferred:
                identity_key = preferred
                break
        if not aliases:
            identity_key = f"evidence:{root}:{items[0].evidence_id}"
        grouped[identity_key] = items

    sources: list[SourceRecord] = []
    source_ids: set[str] = set()
    for identity_key, evidence_items in grouped.items():
        source_id = stable_source_id(identity_key)
        source_ids.add(source_id)
        first = evidence_items[0]
        conflicted = _identity_conflicts(evidence_items)
        for evidence in evidence_items:
            evidence.source_id = source_id
            if evidence.doi:
                evidence.doi = normalize_doi(evidence.doi) or evidence.doi
            if evidence.pmid:
                evidence.pmid = normalize_pmid(evidence.pmid) or evidence.pmid
            if not evidence.canonical_url:
                evidence.canonical_url = canonical_url_for_evidence(evidence) or None
            if conflicted:
                evidence.verification.identity_status = "conflicted"
        verification = SourceVerification.model_validate(first.verification.model_dump(mode="json"))
        if conflicted:
            verification.identity_status = "conflicted"
            warnings.append(f"source_identity_conflict:{source_id}")
        sources.append(
            SourceRecord(
                source_id=source_id,
                source_type=first.source_type,
                provider=(first.provider or str(first.metadata.get("source_tool") or "").strip() or None),
                provider_record_id=(
                    first.provider_record_id
                    or str(first.metadata.get("provider_record_id") or "").strip()
                    or None
                ),
                doi=normalize_doi(first.doi) or None,
                pmid=normalize_pmid(first.pmid) or None,
                canonical_url=canonical_url_for_evidence(first) or None,
                title=first.title,
                trace_ref=first.raw_ref,
                verification=verification,
            )
        )

    pack.sources = sources
    claim_ids = {str(item.claim_id or "").strip() for item in pack.claims if str(item.claim_id or "").strip()}
    evidence_ids = {
        str(item.evidence_id or "").strip() for item in pack.evidence if str(item.evidence_id or "").strip()
    }
    if len(claim_ids) != len(pack.claims):
        warnings.append("duplicate_or_empty_claim_id")
    if len(evidence_ids) != len(pack.evidence):
        warnings.append("duplicate_or_empty_evidence_id")

    normalized_links: list[ClaimEvidenceLink] = []
    seen_links: set[tuple[str, str]] = set()
    for link in pack.claim_evidence_links:
        key = (str(link.claim_id or "").strip(), str(link.evidence_id or "").strip())
        if key[0] not in claim_ids or key[1] not in evidence_ids:
            warnings.append(f"invalid_claim_evidence_link:{key[0]}:{key[1]}")
            continue
        if key in seen_links:
            warnings.append(f"duplicate_claim_evidence_link:{key[0]}:{key[1]}")
            continue
        seen_links.add(key)
        normalized_links.append(link)

    for claim in pack.claims:
        for evidence_id in claim.evidence_ids:
            key = (str(claim.claim_id or "").strip(), str(evidence_id or "").strip())
            if key[0] not in claim_ids or key[1] not in evidence_ids or key in seen_links:
                continue
            seen_links.add(key)
            normalized_links.append(ClaimEvidenceLink(claim_id=key[0], evidence_id=key[1]))
    pack.claim_evidence_links = normalized_links
    return pack, warnings
