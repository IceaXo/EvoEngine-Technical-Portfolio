from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from src.subagents.retrieval_types import EvidencePack


def _extract_json_object(text: str) -> dict | None:
    value = str(text or "").strip()
    if not value:
        return None
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"\s*```$", "", value).strip()
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(value[start : end + 1])
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item or "").strip()]
        except Exception:
            pass
        return [part.strip() for part in re.split(r"[,，;；]", text) if part.strip()]
    return []


def _coerce_metadata(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {"raw": text}
    return {}


def _coerce_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        text = str(value).strip()
        return text if text else None
    return str(value)


def _coerce_string(value: Any) -> str:
    coerced = _coerce_optional_string(value)
    return coerced or ""


_SYSTEM_OWNED_EVIDENCE_FIELDS = {
    "source_id",
    "provider",
    "provider_record_id",
    "canonical_url",
    "content_hash",
    "span_id",
    "context_before",
    "context_after",
    "verification",
}
def _safe_model_metadata(value: Any, *, trust_system_fields: bool) -> dict:
    metadata = _coerce_metadata(value)
    if trust_system_fields:
        return metadata
    # Metadata is an open-ended mapping consumed by multiple downstream
    # renderers.  A denylist cannot safely anticipate future locator aliases,
    # so untrusted synthesis output contributes no metadata at all.  The raw
    # tool result restores the authoritative mapping during reconciliation.
    return {}


def _normalize_raw_payload(parsed: dict, *, trust_system_fields: bool = False) -> dict:
    normalized = dict(parsed)
    if not trust_system_fields:
        # The synthesis model may propose claims and candidate evidence only.
        # Source identity and verification verdicts are owned by deterministic
        # reconciliation/verifier code and must never be accepted from it.
        normalized.pop("sources", None)
        normalized.pop("claim_evidence_links", None)
    claims = normalized.get("claims")
    if isinstance(claims, list):
        normalized_claims = []
        for idx, claim in enumerate(claims, 1):
            if isinstance(claim, str):
                try:
                    parsed_claim = json.loads(claim)
                except Exception:
                    parsed_claim = None
                if isinstance(parsed_claim, dict):
                    claim = parsed_claim
                else:
                    claim = {
                        "claim_id": f"C{idx}",
                        "claim": claim,
                        "support_level": "unknown",
                        "evidence_ids": [],
                    }
            if isinstance(claim, dict):
                normalized_claims.append(
                    {
                        **claim,
                        "evidence_ids": _coerce_string_list(claim.get("evidence_ids")),
                    }
                )
        normalized["claims"] = normalized_claims
    evidence = normalized.get("evidence")
    if isinstance(evidence, list):
        normalized_evidence = []
        for item in evidence:
            if isinstance(item, str):
                try:
                    parsed_item = json.loads(item)
                except Exception:
                    parsed_item = None
                item = parsed_item if isinstance(parsed_item, dict) else None
            if isinstance(item, dict):
                safe_item = (
                    dict(item)
                    if trust_system_fields
                    else {key: value for key, value in item.items() if key not in _SYSTEM_OWNED_EVIDENCE_FIELDS}
                )
                normalized_evidence.append(
                    {
                        **safe_item,
                        "evidence_id": _coerce_string(item.get("evidence_id")),
                        "source_type": _coerce_string(item.get("source_type")) or "unknown",
                        "title": _coerce_string(item.get("title")),
                        "pmid": _coerce_optional_string(item.get("pmid")),
                        "doi": _coerce_optional_string(item.get("doi")),
                        "journal": _coerce_optional_string(item.get("journal")),
                        "year": _coerce_optional_string(item.get("year")),
                        "url": _coerce_optional_string(item.get("url")),
                        "snippet": _coerce_string(item.get("snippet")),
                        "raw_ref": _coerce_optional_string(item.get("raw_ref")),
                        "metadata": _safe_model_metadata(
                            item.get("metadata"), trust_system_fields=trust_system_fields
                        ),
                    }
                )
        normalized["evidence"] = normalized_evidence
    return normalized


def normalize_evidence_pack(raw_text: str, *, fallback_query: str, max_evidence: int) -> tuple[EvidencePack, list[str]]:
    warnings: list[str] = []
    parsed = _extract_json_object(raw_text)
    if parsed is None:
        warnings.append("subagent_final_not_json")
        return EvidencePack(
            query=fallback_query,
            claims=[],
            evidence=[],
            limitations=["检索子代理未返回标准 Evidence Pack；本轮未把原始检索正文注入主上下文。"],
        ), warnings

    parsed.setdefault("query", fallback_query)
    parsed.setdefault("claims", [])
    parsed.setdefault("evidence", [])
    parsed.setdefault("limitations", [])
    parsed = _normalize_raw_payload(parsed)
    try:
        pack = EvidencePack.model_validate(parsed)
    except ValidationError as exc:
        warnings.append(f"evidence_pack_validation_failed:{exc.errors()[0].get('type', 'unknown')}")
        return EvidencePack(
            query=fallback_query,
            claims=[],
            evidence=[],
            limitations=["检索子代理返回的 Evidence Pack 结构校验失败；已保留 trace_ref 供排查。"],
        ), warnings

    limit = max(1, max_evidence)
    deduped_evidence = []
    seen_keys: set[str] = set()
    for evidence in pack.evidence:
        key = (
            str(evidence.pmid or "").strip().lower()
            or str(evidence.doi or "").strip().lower()
            or str(evidence.url or "").strip().lower()
            or str(evidence.title or "").strip().lower()
        )
        if key and key in seen_keys:
            continue
        if key:
            seen_keys.add(key)
        deduped_evidence.append(evidence)
    if len(deduped_evidence) < len(pack.evidence):
        warnings.append("duplicate_evidence_removed")
    omitted_evidence_count = max(0, len(deduped_evidence) - limit)
    pack.evidence = deduped_evidence[:limit]
    for idx, evidence in enumerate(pack.evidence, 1):
        if not evidence.evidence_id:
            evidence.evidence_id = f"E{idx}"
    valid_ids = {item.evidence_id for item in pack.evidence}
    omitted_claim_count = max(0, len(pack.claims) - limit)
    for idx, claim in enumerate(pack.claims[:limit], 1):
        if not claim.claim_id:
            claim.claim_id = f"C{idx}"
        claim.evidence_ids = [eid for eid in claim.evidence_ids if eid in valid_ids]
    pack.claims = pack.claims[:limit]
    if omitted_evidence_count or omitted_claim_count:
        warnings.append("evidence_pack_projection_limit_applied")
        pack.limitations.append(
            "Evidence Pack 是按调用参数 max_evidence 选择的语义投影："
            f"本次未投影 evidence={omitted_evidence_count}、claims={omitted_claim_count}；"
            "完整 Provider 机器结果保留在本次检索的 ResourceRef/trace 中。"
        )
    if not pack.claims and pack.evidence:
        warnings.append("evidence_without_claims")
    return pack, warnings


def evidence_pack_to_plain_dict(pack: EvidencePack) -> dict:
    """Return a JSON-safe Evidence Pack dict with common model JSON mistakes fixed."""
    data = pack.model_dump(mode="json")
    return _normalize_raw_payload(data, trust_system_fields=True)
