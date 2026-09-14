"""Deterministic discovery over executable capabilities and guide-only Skills.

Capability discovery is intentionally side-effect free.  It returns ranked
candidates, while the lead agent remains responsible for choosing a candidate,
loading the corresponding tool schema, and invoking it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from src.capabilities.models import CapabilitySpec, ProviderType


_LATIN_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
_SEPARATOR_RE = re.compile(r"[^a-z0-9\u4e00-\u9fff]+")
_LATIN_DISCOVERY_STOPWORDS = {
    "about",
    "according",
    "always",
    "and",
    "answer",
    "are",
    "between",
    "biology",
    "choice",
    "correct",
    "following",
    "for",
    "format",
    "from",
    "include",
    "into",
    "letter",
    "likely",
    "most",
    "multiple",
    "must",
    "option",
    "options",
    "question",
    "respond",
    "return",
    "selected",
    "that",
    "the",
    "their",
    "this",
    "use",
    "using",
    "which",
    "with",
    "within",
    "your",
}

_GUIDE_DISCOVERY_ROLES = frozenset(
    {
        "procedure",
        "knowledge",
        "computational_reference",
    }
)
_CAPABILITY_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]*$")


@dataclass(frozen=True)
class GuideDiscoveryContract:
    """Declarative discovery policy for a guide-only Skill.

    Existing Skills default to ``procedure`` so laboratory SOPs and other
    genuinely actionable guides retain their current natural-discovery
    behavior.  ``computational_reference`` is deliberately exact-ID-only: it
    may document a calculation, but it is not itself an executable provider.
    An optional stable capability ID records the canonical typed route without
    turning the reference guide into a second execution candidate.
    """

    role: str
    delegates_to: tuple[str, ...] = ()

    @property
    def match_mode(self) -> str:
        return (
            "exact_only"
            if self.role == "computational_reference"
            else "natural"
        )


def guide_discovery_contract(skill: Any) -> GuideDiscoveryContract:
    """Read and fail-close one guide discovery declaration.

    The declaration lives under ``routing.discovery`` in ``SKILL.md``.  An
    unknown role cannot safely claim natural discoverability, so it is treated
    as a computational reference.  Delegates are stable capability IDs, never
    tool names or natural-language phrases.
    """

    routing = getattr(skill, "routing", None)
    routing = routing if isinstance(routing, dict) else {}
    raw_discovery = routing.get("discovery")
    discovery_is_malformed = (
        "discovery" in routing and not isinstance(raw_discovery, dict)
    )
    discovery = raw_discovery if isinstance(raw_discovery, dict) else {}
    declared_role = str(
        discovery.get("role")
        or (
            "computational_reference"
            if discovery_is_malformed
            else "procedure"
        )
    ).strip().lower()
    role = (
        declared_role
        if declared_role in _GUIDE_DISCOVERY_ROLES
        else "computational_reference"
    )
    raw_delegates = discovery.get("delegates_to")
    values = (
        [raw_delegates]
        if isinstance(raw_delegates, str)
        else raw_delegates
        if isinstance(raw_delegates, list)
        else []
    )
    delegates_to = tuple(
        dict.fromkeys(
            value
            for raw_value in values
            if (value := str(raw_value or "").strip())
            and _CAPABILITY_ID_RE.fullmatch(value)
        )
    )
    return GuideDiscoveryContract(role=role, delegates_to=delegates_to)


def _direct_identifier_query(entry: "CapabilityDiscoveryEntry", query: str) -> bool:
    """Return whether the model explicitly supplied this entry's identifier.

    Control-plane capabilities stay out of fuzzy natural-language discovery,
    but the unified matcher must still resolve every catalog tool when the
    agent supplies its exact ``tool_id`` or ``capability_id``.  The routing
    query may contain the original user request after the explicit model query;
    only the explicit segment participates in this gate.
    """

    explicit_query = str(query or "").split(
        "\n\n[ORIGINAL_USER_REQUEST]\n",
        1,
    )[0].strip().casefold()
    return bool(explicit_query) and explicit_query in {
        str(entry.tool_id or "").strip().casefold(),
        str(entry.capability_id or "").strip().casefold(),
    }


def _explicit_identifier_query(query: str) -> str:
    return str(query or "").split(
        "\n\n[ORIGINAL_USER_REQUEST]\n",
        1,
    )[0].strip().casefold()


def _exact_identifier_entries(
    entries: tuple["CapabilityDiscoveryEntry", ...],
    query: str,
) -> list["CapabilityDiscoveryEntry"]:
    """Resolve an explicit stable ID before fuzzy ranking.

    A catalog Tool is the canonical entry when executable Skill contracts share
    its generic invoker ID (for example ``execute_skill_capability``). Exact Skill
    IDs may intentionally expand to several distinct script capabilities.
    """

    explicit = _explicit_identifier_query(query)
    if not explicit:
        return []
    capability_matches = [
        entry
        for entry in entries
        if str(entry.capability_id or "").strip().casefold() == explicit
    ]
    if capability_matches:
        return capability_matches
    tool_matches = [
        entry
        for entry in entries
        if str(entry.tool_id or "").strip().casefold() == explicit
    ]
    if tool_matches:
        canonical = [entry for entry in tool_matches if not entry.skill_id]
        return canonical or tool_matches
    skill_matches = [
        entry
        for entry in entries
        if str(entry.skill_id or "").strip().casefold() == explicit
    ]
    if skill_matches:
        guide_matches = [entry for entry in skill_matches if entry.kind == "guide"]
        return guide_matches or skill_matches
    alias_matches = [
        entry
        for entry in entries
        if explicit in {
            str(alias or "").strip().casefold()
            for alias in entry.aliases
            if str(alias or "").strip()
        }
    ]
    if alias_matches:
        return alias_matches
    return [
        entry
        for entry in entries
        if entry.provider_type == ProviderType.SANDBOX.value
        and str(entry.metadata.get("app_id") or "").strip().casefold() == explicit
    ]


def _compact_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _schema_contract_summary(schema: Any) -> dict[str, Any]:
    """Project a JSON Schema into a compact card, not an execution schema."""

    if not isinstance(schema, dict):
        return {"required": [], "fields": []}
    properties = schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    required = [
        str(item)
        for item in (schema.get("required") or [])
        if str(item or "").strip()
    ]
    fields: list[dict[str, Any]] = []
    for name, raw_field in properties.items():
        field_schema = raw_field if isinstance(raw_field, dict) else {}
        field_type = field_schema.get("type")
        if not field_type and isinstance(field_schema.get("anyOf"), list):
            field_type = [
                item.get("type")
                for item in field_schema["anyOf"]
                if isinstance(item, dict) and item.get("type") != "null"
            ]
        field = {
            "name": str(name),
            "type": field_type or "unspecified",
            "required": str(name) in required,
        }
        description = _compact_text(field_schema.get("description"))
        if description:
            field["description"] = description
        fields.append(field)

    required_alternatives: list[dict[str, Any]] = []

    def _branch_required_sets(branch: Any) -> list[list[str]]:
        if not isinstance(branch, dict):
            return []
        names = [
            str(item)
            for item in (branch.get("required") or [])
            if str(item or "").strip()
        ]
        if names:
            return [names]
        nested: list[list[str]] = []
        for operator in ("oneOf", "anyOf"):
            for child in branch.get(operator) or []:
                nested.extend(_branch_required_sets(child))
        return nested

    def _collect_required_alternatives(node: Any) -> None:
        if not isinstance(node, dict):
            return
        for operator in ("oneOf", "anyOf"):
            branches = node.get(operator)
            if not isinstance(branches, list):
                continue
            alternatives: list[list[str]] = []
            for branch in branches:
                alternatives.extend(_branch_required_sets(branch))
            if alternatives:
                required_alternatives.append(
                    {"operator": operator, "alternatives": alternatives}
                )
        # Root-level composition is part of the execution contract.  Do not
        # descend into property schemas, whose ``anyOf`` usually only encodes
        # nullable/union field types rather than alternative input sources.
        for child in node.get("allOf") or []:
            _collect_required_alternatives(child)

    _collect_required_alternatives(schema)
    for group in schema.get("x-evo-input-source-groups") or []:
        if not isinstance(group, dict):
            continue
        if not bool(group.get("required")) or bool(
            group.get("default_satisfies_required")
        ):
            continue
        alternatives: list[list[str]] = []
        for source in group.get("source_alternatives") or []:
            if not isinstance(source, dict):
                continue
            # Every field within one source kind is itself an accepted source
            # reference (for example asset ID or registered asset name).
            alternatives.extend(
                [str(field_name)]
                for field_name in (source.get("fields") or [])
                if str(field_name or "").strip()
            )
        if alternatives:
            required_alternatives.append(
                {
                    "logical_key": str(group.get("logical_key") or "").strip(),
                    "operator": "oneOf",
                    "alternatives": alternatives,
                }
            )

    deduplicated_alternatives: list[dict[str, Any]] = []
    seen_alternatives: set[tuple[tuple[str, ...], ...]] = set()
    for item in required_alternatives:
        identity = tuple(
            tuple(str(name) for name in names)
            for names in (item.get("alternatives") or [])
        )
        if not identity or identity in seen_alternatives:
            continue
        seen_alternatives.add(identity)
        deduplicated_alternatives.append(item)
    required_alternatives = deduplicated_alternatives
    summary = {
        "required": required,
        "fields": fields,
        "allows_additional_fields": schema.get("additionalProperties", True) is not False,
    }
    if required_alternatives:
        summary["required_alternatives"] = required_alternatives
    return summary


def _normalized_phrase(value: Any) -> str:
    return _SEPARATOR_RE.sub(" ", _compact_text(value).casefold()).strip()


def _identifier_pattern(value: str) -> re.Pattern[str] | None:
    """Build a boundary-aware identifier matcher.

    Separators in an identifier are interchangeable, so ``CD-HIT`` also
    matches ``cd hit`` and ``cdhit``.  Latin identifiers never match inside a
    larger Latin word, preventing aliases such as ``ice`` from matching
    ``service`` or ``practice``.
    """

    normalized_value = str(value or "").casefold()
    # CJK phrases are matched as CJK text below.  Building a Latin identifier
    # from only their embedded option/chain letter (for example ``提取 A 链``)
    # would make every ``A.`` answer choice look like an exact capability hit.
    if _CJK_RUN_RE.search(normalized_value):
        return None
    parts = _LATIN_TOKEN_RE.findall(normalized_value)
    if not parts:
        return None
    body = r"[\W_]*".join(re.escape(part) for part in parts)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])", re.IGNORECASE)


def _search_tokens(value: Any) -> set[str]:
    text = _compact_text(value).casefold()
    tokens = {
        token
        for token in _LATIN_TOKEN_RE.findall(text)
        if len(token) >= 3 and token not in _LATIN_DISCOVERY_STOPWORDS
    }
    for run in _CJK_RUN_RE.findall(text):
        if len(run) <= 4:
            tokens.add(run)
        for width in (2, 3, 4):
            if len(run) < width:
                continue
            tokens.update(run[index : index + width] for index in range(len(run) - width + 1))
    return tokens


def capability_tool_id(spec: CapabilitySpec) -> str:
    provider_ref = spec.provider_ref if isinstance(spec.provider_ref, dict) else {}
    explicit = str(
        provider_ref.get("tool_name")
        or provider_ref.get("catalog_tool_id")
        or ""
    ).strip()
    if explicit:
        return explicit
    if spec.provider_type == ProviderType.SANDBOX:
        app_id = str(provider_ref.get("app_id") or "").strip()
        return f"sandbox_submit_{app_id}" if app_id else ""
    if spec.provider_type == ProviderType.SKILL_SCRIPT:
        return "execute_skill_capability"
    return ""


def skill_positive_search_terms(skill: Any) -> tuple[str, ...]:
    """Return only the positive discovery vocabulary declared by a Skill.

    ``negative_conditions`` are routing gates owned by :class:`SkillRouter`,
    never synonyms for discovery.  Keeping this projection deliberately small
    prevents the cross-provider index from turning phrases such as "only asks
    for background" into positive evidence for executing a Skill.
    """

    routing = getattr(skill, "routing", None)
    routing = routing if isinstance(routing, dict) else {}
    terms: list[str] = [
        str(item or "").strip()
        for item in (getattr(skill, "triggers", None) or [])
        if str(item or "").strip()
    ]
    for key in ("strong_triggers", "weak_triggers"):
        value = routing.get(key)
        if isinstance(value, list):
            terms.extend(
                str(item or "").strip()
                for item in value
                if str(item or "").strip()
            )
        elif isinstance(value, str) and value.strip():
            terms.append(value.strip())
    return tuple(dict.fromkeys(terms))


@dataclass(frozen=True)
class CapabilityDiscoveryEntry:
    kind: str
    display_name: str
    description: str = ""
    aliases: tuple[str, ...] = ()
    strong_aliases: tuple[str, ...] = ()
    search_terms: tuple[str, ...] = ()
    capability_id: str = ""
    tool_id: str = ""
    provider_type: str = ""
    skill_id: str = ""
    script_name: str = ""
    visibility: str = "public"
    match_mode: str = "exact_only"
    operations: tuple[str, ...] = ()
    source_types: tuple[str, ...] = ()
    object_types: tuple[str, ...] = ()
    input_types: tuple[str, ...] = ()
    execution_entry_id: str = ""
    load_tool_ids: tuple[str, ...] = ()
    fallback_for: tuple[str, ...] = ()
    fallback_condition: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_spec(
        cls,
        spec: CapabilitySpec,
        *,
        aliases: Iterable[str] = (),
        search_terms: Iterable[str] = (),
    ) -> "CapabilityDiscoveryEntry":
        provider_ref = spec.provider_ref if isinstance(spec.provider_ref, dict) else {}
        policy = spec.discovery
        tool_id = capability_tool_id(spec)
        is_skill_capability = spec.provider_type == ProviderType.SKILL_SCRIPT
        is_typed_skill_capability = (
            is_skill_capability
            and str(provider_ref.get("typed_exposure") or "").strip().lower()
            == "typed"
            and tool_id != "execute_skill_capability"
        )
        strong_aliases = [
            spec.capability_id,
            tool_id,
            str(provider_ref.get("app_id") or ""),
            str(provider_ref.get("app_name") or ""),
            # A typed Skill capability is already an independently selectable
            # business contract.  The parent Skill ID belongs to the family,
            # not to every sibling: treating it as a strong alias makes a
            # query such as "Open Targets" exact-match every typed operation
            # in that Skill.  Guide/legacy entries may still use Skill identity
            # because they genuinely represent the whole Skill.
            *(
                []
                if is_typed_skill_capability
                else [str(provider_ref.get("skill_id") or "")]
            ),
            *(
                []
                if is_skill_capability
                else [str(provider_ref.get("script_name") or "")]
            ),
        ]
        generated_aliases = [
            *strong_aliases,
            spec.display_name,
            *policy.aliases,
            *[str(item or "") for item in aliases],
        ]
        generated_terms = [
            *policy.search_terms,
            *[str(item or "") for item in search_terms],
        ]
        return cls(
            kind="executable",
            capability_id=spec.capability_id,
            tool_id=tool_id,
            provider_type=spec.provider_type.value,
            display_name=spec.display_name,
            description=spec.description,
            aliases=tuple(dict.fromkeys(item.strip() for item in generated_aliases if item.strip())),
            strong_aliases=tuple(dict.fromkeys(item.strip() for item in strong_aliases if item.strip())),
            search_terms=tuple(dict.fromkeys(item.strip() for item in generated_terms if item.strip())),
            skill_id=str(provider_ref.get("skill_id") or "").strip(),
            # Script identity is provider-internal.  The Agent selects the
            # business capability and then loads its public machine contract.
            script_name=(
                ""
                if is_skill_capability
                else str(provider_ref.get("script_name") or "").strip()
            ),
            visibility=policy.visibility.value,
            match_mode=policy.match_mode.value,
            operations=tuple(policy.operations),
            source_types=tuple(policy.source_types),
            object_types=tuple(policy.object_types),
            input_types=tuple(policy.input_types),
            execution_entry_id=(
                str(policy.execution_entry_id or tool_id).strip()
                if is_typed_skill_capability
                else "execute_skill_capability"
                if is_skill_capability
                else str(policy.execution_entry_id or "").strip()
            ),
            load_tool_ids=(
                tuple(policy.load_tool_ids or ([tool_id] if tool_id else []))
                if is_typed_skill_capability
                else ("load_skill_capability", "execute_skill_capability")
                if is_skill_capability
                else tuple(policy.load_tool_ids)
            ),
            fallback_for=(
                tuple(policy.fallback.for_capability_ids)
                if policy.fallback is not None
                else ()
            ),
            fallback_condition=(
                policy.fallback.condition.value
                if policy.fallback is not None
                else ""
            ),
            metadata={
                "version": spec.version,
                "permission": spec.permission.value,
                "availability": spec.availability,
                "contract": {
                    "input": _schema_contract_summary(spec.input_schema),
                    "input_refs": _schema_contract_summary(spec.input_refs_schema),
                    "output": _schema_contract_summary(spec.output_schema),
                    "acceptance": _schema_contract_summary(spec.acceptance_schema),
                    "artifacts": [
                        {
                            "artifact_key": artifact.artifact_key,
                            "required": artifact.required,
                            "file_name": artifact.file_name,
                            "file_types": list(artifact.file_types),
                            "drawer_section": artifact.drawer_section.value,
                        }
                        for artifact in spec.produces
                    ],
                },
                "limitations": {
                    "availability": spec.availability,
                    "permission": spec.permission.value,
                    "side_effects": list(spec.side_effects),
                    "requires_confirmation": bool(spec.requires_confirmation),
                    "async_mode": spec.execution_policy.async_mode.value,
                    "timeout_seconds": spec.execution_policy.timeout_seconds,
                    "idempotent": bool(spec.execution_policy.idempotent),
                },
                **(
                    {"app_id": str(provider_ref.get("app_id") or "").strip()}
                    if str(provider_ref.get("app_id") or "").strip()
                    else {}
                ),
            },
        )

    @classmethod
    def from_skill(cls, skill: Any) -> "CapabilityDiscoveryEntry":
        skill_id = str(getattr(skill, "skill_id", "") or "").strip()
        display_name = str(getattr(skill, "name", "") or skill_id).strip()
        visibility = str(getattr(skill, "visibility", "public") or "public").strip().lower()
        guide_discovery = guide_discovery_contract(skill)
        routing = getattr(skill, "routing", None)
        routing = routing if isinstance(routing, dict) else {}
        discovery = routing.get("discovery")
        discovery = discovery if isinstance(discovery, dict) else {}

        def _values(key: str) -> tuple[str, ...]:
            raw = discovery.get(key)
            raw = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
            return tuple(
                dict.fromkeys(
                    str(item or "").strip().lower()
                    for item in raw
                    if str(item or "").strip()
                )
            )

        guide_input = getattr(skill, "inputs", {}) or {}
        if not guide_input:
            input_contract = getattr(skill, "input_contract", {}) or {}
            guide_input = input_contract if isinstance(input_contract, dict) else {}

        return cls(
            kind="guide",
            skill_id=skill_id,
            display_name=display_name,
            description=str(getattr(skill, "description", "") or "").strip(),
            aliases=tuple(
                dict.fromkeys(
                    item.strip()
                    for item in (
                        skill_id,
                        *(
                            ()
                            if guide_discovery.role == "computational_reference"
                            else (display_name,)
                        ),
                    )
                    if item.strip()
                )
            ),
            strong_aliases=(skill_id,),
            search_terms=tuple(
                item.strip()
                for item in (
                    *skill_positive_search_terms(skill),
                )
                if item.strip()
            ),
            visibility=visibility,
            match_mode=(
                guide_discovery.match_mode
                if visibility == "public"
                else "hidden"
            ),
            operations=_values("operations"),
            source_types=_values("source_types"),
            object_types=_values("object_types"),
            input_types=_values("input_types"),
            execution_entry_id="activate_skill" if visibility == "public" else "",
            load_tool_ids=("activate_skill",) if visibility == "public" else (),
            metadata={
                "category": str(getattr(skill, "category", "") or "").strip(),
                "guide_discovery_role": guide_discovery.role,
                "delegates_to": list(guide_discovery.delegates_to),
                "requires_confirmation": bool(getattr(skill, "requires_confirmation", False)),
                "contract": {
                    "guide_only": True,
                    "input": guide_input,
                    "output": getattr(skill, "produces", []) or [],
                    "artifacts": [],
                },
                "limitations": {
                    "guide_only": True,
                    "requires_confirmation": bool(
                        getattr(skill, "requires_confirmation", False)
                    ),
                },
            },
        )


class CapabilityDiscoveryIndex:
    """A small deterministic index suitable for a resident agent tool."""

    def __init__(self, entries: Iterable[CapabilityDiscoveryEntry]) -> None:
        identities: set[tuple[str, str, str]] = set()
        normalized: list[CapabilityDiscoveryEntry] = []
        direct_only: list[CapabilityDiscoveryEntry] = []
        for entry in entries:
            identity = (entry.kind, entry.capability_id, entry.skill_id)
            if identity in identities:
                continue
            identities.add(identity)
            if entry.kind == "executable" and not entry.tool_id:
                continue
            if entry.visibility != "public" or entry.match_mode == "hidden":
                continue
            if entry.match_mode == "exact_only":
                direct_only.append(entry)
                continue
            if entry.match_mode != "natural":
                continue
            normalized.append(entry)
        self._entries = tuple(normalized)
        self._direct_entries = tuple(direct_only)
        public_entries = (*self._entries, *self._direct_entries)
        self._available_capability_ids = frozenset(
            entry.capability_id
            for entry in public_entries
            if entry.kind == "executable" and entry.capability_id
        )
        fallback_entries_by_primary: dict[str, list[CapabilityDiscoveryEntry]] = {}
        for entry in public_entries:
            for primary_capability_id in entry.fallback_for:
                fallback_entries_by_primary.setdefault(
                    primary_capability_id,
                    [],
                ).append(entry)
        self._fallback_entries_by_primary = {
            capability_id: tuple(items)
            for capability_id, items in fallback_entries_by_primary.items()
        }

    @property
    def entries(self) -> tuple[CapabilityDiscoveryEntry, ...]:
        return self._entries

    @staticmethod
    def _score(
        entry: CapabilityDiscoveryEntry,
        query: str,
        *,
        operations: Iterable[str] = (),
        source_types: Iterable[str] = (),
        object_types: Iterable[str] = (),
        input_types: Iterable[str] = (),
    ) -> tuple[float, list[str]]:
        normalized_query = _normalized_phrase(query)
        query_tokens = _search_tokens(query)
        if not normalized_query:
            return 0.0, []

        score = 0.0
        reasons: list[str] = []
        exact_hits: list[str] = []
        strong_aliases = {item.casefold() for item in entry.strong_aliases}
        for alias in entry.aliases:
            normalized_alias = _normalized_phrase(alias)
            if not normalized_alias:
                continue
            pattern = _identifier_pattern(alias)
            if pattern is not None and pattern.search(query):
                exact_hits.append(alias)
                base = 100.0 if alias.casefold() in strong_aliases else 36.0
                score = max(score, base + min(20.0, len(normalized_alias)))
                continue
            if any("\u4e00" <= char <= "\u9fff" for char in alias) and normalized_alias in normalized_query:
                exact_hits.append(alias)
                base = 100.0 if alias.casefold() in strong_aliases else 36.0
                score = max(score, base + min(20.0, len(normalized_alias)))
        if exact_hits:
            reasons.append("exact identifier/name: " + ", ".join(exact_hits[:3]))

        document = " ".join(
            [entry.display_name, entry.description, *entry.aliases, *entry.search_terms]
        )
        document_tokens = _search_tokens(document)
        overlap = sorted(query_tokens & document_tokens, key=lambda item: (-len(item), item))
        if overlap:
            token_score = min(48.0, sum(2.0 + min(4.0, len(item) / 2.0) for item in overlap))
            score += token_score
            reasons.append("shared terms: " + ", ".join(overlap[:6]))

        phrase_hits: list[str] = []
        for term in entry.search_terms:
            normalized_term = _normalized_phrase(term)
            term_tokens = _search_tokens(term)
            contains_cjk = bool(_CJK_RUN_RE.search(str(term or "")))
            if (
                len(normalized_term) < 3
                or not term_tokens
                or (not contains_cjk and len(term_tokens) < 2)
            ):
                continue
            pattern = _identifier_pattern(term)
            matched = bool(pattern and pattern.search(query))
            if not matched and any("\u4e00" <= char <= "\u9fff" for char in term):
                matched = normalized_term in normalized_query
            if matched:
                phrase_hits.append(term)
        if phrase_hits:
            score += min(72.0, 18.0 * len(dict.fromkeys(phrase_hits)))
            reasons.append("metadata phrases: " + ", ".join(dict.fromkeys(phrase_hits[:4])))

        # The Agent may describe the requested operation and data shape using
        # the same stable taxonomy declared by capability contracts.  These
        # facets are ranking evidence only: a mismatch never filters a lexical
        # candidate, selects a winner, or executes anything on the Agent's
        # behalf.  This gives semantically related capabilities a deterministic
        # path into the candidate window without maintaining phrase-to-tool
        # intent tables.
        facet_groups = (
            ("operation", operations, entry.operations, 28.0),
            ("source", source_types, entry.source_types, 22.0),
            ("object", object_types, entry.object_types, 16.0),
            ("input", input_types, entry.input_types, 10.0),
        )
        for label, requested_values, declared_values, weight in facet_groups:
            requested = {
                str(item or "").strip().casefold()
                for item in requested_values
                if str(item or "").strip()
                and str(item or "").strip().casefold() != "any"
            }
            declared = {
                str(item or "").strip().casefold()
                for item in declared_values
                if str(item or "").strip()
                and str(item or "").strip().casefold() != "any"
            }
            overlap = sorted(requested & declared)
            if not overlap:
                continue
            score += weight + min(8.0, 2.0 * (len(overlap) - 1))
            reasons.append(f"{label} facet: " + ", ".join(overlap[:3]))

        normalized_display = _normalized_phrase(entry.display_name)
        if normalized_display and normalized_display == normalized_query:
            score += 40.0
        if score > 0 and entry.kind == "executable":
            score += 0.25
        return score, reasons

    def search(
        self,
        query: str,
        *,
        limit: int = 3,
        operations: Iterable[str] = (),
        source_types: Iterable[str] = (),
        object_types: Iterable[str] = (),
        input_types: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(10, int(limit or 3)))
        all_entries = (*self._entries, *self._direct_entries)
        exact_entries = _exact_identifier_entries(all_entries, query)
        exact_lookup = bool(exact_entries)
        entries_to_score = tuple(exact_entries) if exact_entries else self._entries
        scored: list[tuple[float, CapabilityDiscoveryEntry, list[str]]] = []
        for entry in entries_to_score:
            score, reasons = self._score(
                entry,
                query,
                operations=operations,
                source_types=source_types,
                object_types=object_types,
                input_types=input_types,
            )
            if score <= 0:
                continue
            if (
                not exact_lookup
                and entry.fallback_for
                and self._available_capability_ids.intersection(entry.fallback_for)
            ):
                # A fallback for the same business contract never competes
                # with an available primary in natural discovery.  Exact-ID
                # lookup remains possible after a structured primary failure.
                continue
            scored.append((score, entry, reasons))
        scored.sort(
            key=lambda item: (
                -item[0],
                0 if item[1].kind == "executable" else 1,
                item[1].capability_id or item[1].skill_id,
            )
        )
        results: list[dict[str, Any]] = []
        for score, entry, reasons in scored[:bounded_limit]:
            payload: dict[str, Any] = {
                "kind": entry.kind,
                "display_name": entry.display_name,
                "description": entry.description,
                "score": round(score, 3),
                "reasons": [str(reason) for reason in reasons],
            }
            if entry.kind == "executable":
                payload.update(
                    {
                        "capability_id": entry.capability_id,
                        "tool_id": entry.tool_id,
                        "provider_type": entry.provider_type,
                        "execution_entry_id": entry.execution_entry_id or entry.tool_id,
                        "load_tool_ids": list(entry.load_tool_ids),
                    }
                )
                if entry.skill_id:
                    payload["skill_id"] = entry.skill_id
                if entry.script_name:
                    payload["script_name"] = entry.script_name
            else:
                payload["skill_id"] = entry.skill_id
                payload["execution_entry_id"] = entry.execution_entry_id
                payload["load_tool_ids"] = list(entry.load_tool_ids)
            discovery = {
                key: [str(item) for item in values]
                for key, values in (
                    ("operations", list(entry.operations)),
                    ("source_types", list(entry.source_types)),
                    ("object_types", list(entry.object_types)),
                    ("input_types", list(entry.input_types)),
                )
                if values
            }
            if discovery:
                payload["discovery"] = discovery
            if entry.fallback_for:
                available_primary_ids = sorted(
                    self._available_capability_ids.intersection(entry.fallback_for)
                )
                payload["provider_route"] = {
                    "role": "fallback",
                    "for_capability_ids": list(entry.fallback_for),
                    "condition": entry.fallback_condition,
                    "eligibility_state": (
                        "primary_unavailable"
                        if not available_primary_ids
                        else "requires_primary_failure_outcome"
                    ),
                    "available_primary_capability_ids": available_primary_ids,
                }
            elif entry.capability_id in self._fallback_entries_by_primary:
                fallback_entries = self._fallback_entries_by_primary[
                    entry.capability_id
                ]
                payload["provider_route"] = {
                    "role": "primary",
                    "fallback_capability_ids": [
                        item.capability_id for item in fallback_entries
                    ],
                    "fallback_load_tool_ids": list(
                        dict.fromkeys(
                            tool_id
                            for item in fallback_entries
                            for tool_id in item.load_tool_ids
                            if tool_id
                        )
                    ),
                    "fallback_eligible_on": [
                        "provider_unavailable",
                        "not_applicable",
                    ],
                }
            # Discovery only identifies a candidate.  Full schemas, artifact
            # contracts, limitations, manifest hashes and execution policy are
            # returned by the subsequent load step, not duplicated here.
            for key in ("app_id", "exposure"):
                value = entry.metadata.get(key)
                if value not in (None, ""):
                    payload[key] = value
            results.append(payload)
        return results
