from __future__ import annotations

from src.capabilities.discovery import (
    CapabilityDiscoveryEntry,
    CapabilityDiscoveryIndex,
    _schema_contract_summary,
)
from src.capabilities.models import CapabilitySpec, DiscoveryPolicy, Permission, ProviderType


def _spec(
    capability_id: str,
    *,
    tool_id: str,
    display_name: str,
    description: str,
    provider_type: ProviderType = ProviderType.NATIVE,
    provider_ref: dict | None = None,
    natural: bool = True,
) -> CapabilitySpec:
    return CapabilitySpec(
        capability_id=capability_id,
        version="1.0.0",
        display_name=display_name,
        description=description,
        provider_type=provider_type,
        provider_ref={"tool_name": tool_id, **(provider_ref or {})},
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        permission=Permission.EXECUTE,
        discovery=(
            DiscoveryPolicy(
                visibility="public",
                match_mode="natural",
                operations=["analyze"],
                source_types=["any"],
                object_types=["any"],
                input_types=["any"],
                execution_entry_id=tool_id,
                load_tool_ids=[tool_id],
            )
            if natural
            else DiscoveryPolicy()
        ),
    )


def test_discovery_returns_ranked_candidates_without_loading_or_executing() -> None:
    index = CapabilityDiscoveryIndex(
        [
            CapabilityDiscoveryEntry.from_spec(
                _spec(
                    "sandbox.cdhit.run",
                    tool_id="sandbox_submit_cdhit",
                    display_name="CD-HIT",
                    description="Protein sequence clustering and redundancy removal",
                    provider_type=ProviderType.SANDBOX,
                    provider_ref={"app_id": "cdhit", "app_name": "CD-HIT"},
                )
            ),
            CapabilityDiscoveryEntry.from_spec(
                _spec(
                    "native.file.read",
                    tool_id="conversation_file_read",
                    display_name="conversation_file_read",
                    description="Read an uploaded conversation file",
                )
            ),
        ]
    )

    candidates = index.search("请用 CD-HIT 对蛋白 FASTA 做去冗余聚类")

    assert candidates[0]["capability_id"] == "sandbox.cdhit.run"
    assert candidates[0]["tool_id"] == "sandbox_submit_cdhit"
    assert "evo_control" not in candidates[0]
    assert "loaded_tool_ids" not in candidates[0]


def test_short_alias_does_not_match_inside_larger_latin_word() -> None:
    index = CapabilityDiscoveryIndex(
        [
            CapabilityDiscoveryEntry.from_spec(
                _spec(
                    "sandbox.ice.run",
                    tool_id="sandbox_submit_ice",
                    display_name="ICE",
                    description="Genome editing inference",
                    provider_type=ProviderType.SANDBOX,
                    provider_ref={"app_id": "ice", "app_name": "ICE"},
                )
            )
        ]
    )

    assert index.search("Please compare service practice choices") == []
    assert index.search("Run ICE for this editing result")[0]["tool_id"] == "sandbox_submit_ice"


def test_typed_skill_capability_does_not_inherit_parent_skill_id_alias() -> None:
    entry = CapabilityDiscoveryEntry.from_spec(
        _spec(
            "skill.open_targets.gwas_locus.rank",
            tool_id="skill_open_targets_gwas_locus_rank",
            display_name="Rank GWAS locus candidates",
            description="Rank a fixed candidate-gene list for one GWAS locus.",
            provider_type=ProviderType.SKILL_SCRIPT,
            provider_ref={
                "skill_id": "open-targets",
                "typed_exposure": "typed",
            },
        )
    )

    assert "open-targets" not in entry.strong_aliases
    assert "open-targets" not in entry.aliases
    assert "skill.open_targets.gwas_locus.rank" in entry.strong_aliases
    assert "skill_open_targets_gwas_locus_rank" in entry.strong_aliases


def test_cjk_phrase_with_embedded_letter_does_not_match_answer_option_label() -> None:
    index = CapabilityDiscoveryIndex(
        [
            CapabilityDiscoveryEntry(
                kind="guide",
                skill_id="chain-extractor",
                display_name="Chain extractor",
                search_terms=("提取 A 链",),
                visibility="public",
                match_mode="natural",
            )
        ]
    )

    assert index.search("Options:\nA.MSRRFTVTSLPPAGP\nB.MAAAA") == []


def test_generic_english_words_do_not_outrank_domain_contract_terms() -> None:
    index = CapabilityDiscoveryIndex(
        [
            CapabilityDiscoveryEntry(
                kind="guide",
                skill_id="generic-variant-guide",
                display_name="Generic variant guide",
                description="Variant annotation with external inputs",
                search_terms=("with", "for"),
                visibility="public",
                match_mode="natural",
            ),
            CapabilityDiscoveryEntry(
                kind="executable",
                capability_id="skill.clinvar.sequence.resolve",
                tool_id="skill_clinvar_sequence_resolve",
                provider_type="skill_script",
                display_name="ClinVar sequence resolver",
                description="Resolve pathogenic ClinVar protein sequence candidates",
                search_terms=("ClinVar pathogenic protein sequence",),
                visibility="public",
                match_mode="natural",
            ),
        ]
    )

    candidates = index.search(
        "According to ClinVar, which protein sequence contains a pathogenic variant?"
    )

    assert candidates[0]["capability_id"] == "skill.clinvar.sequence.resolve"
    assert all(
        "with" not in " ".join(candidate["reasons"])
        and "for" not in " ".join(candidate["reasons"])
        for candidate in candidates
    )


def test_agent_supplied_contract_facets_improve_recall_without_selecting_or_filtering() -> None:
    retrieval_spec = _spec(
        "subagent.retrieval.run",
        tool_id="run_retrieval_subagent",
        display_name="Research retrieval",
        description="Retrieve external evidence",
        provider_type=ProviderType.SUBAGENT,
    )
    retrieval_spec = retrieval_spec.model_copy(
        update={
            "discovery": DiscoveryPolicy(
                visibility="public",
                match_mode="natural",
                operations=["search", "read"],
                source_types=["public_web", "public_database"],
                object_types=["evidence", "document"],
                input_types=["any"],
                execution_entry_id="run_retrieval_subagent",
                load_tool_ids=["run_retrieval_subagent"],
            )
        }
    )
    lexical_only_spec = _spec(
        "sandbox.align.run",
        tool_id="sandbox_submit_align",
        display_name="Norm update helper",
        description="处理近期规范更新",
        provider_type=ProviderType.SANDBOX,
    )
    index = CapabilityDiscoveryIndex(
        [
            CapabilityDiscoveryEntry.from_spec(retrieval_spec),
            CapabilityDiscoveryEntry.from_spec(lexical_only_spec),
        ]
    )

    candidates = index.search(
        "帮我处理一下近期规范更新",
        operations=["search"],
        source_types=["public_web"],
        object_types=["evidence"],
    )

    assert candidates[0]["tool_id"] == "run_retrieval_subagent"
    assert any(item["tool_id"] == "sandbox_submit_align" for item in candidates)
    assert all("selected" not in item for item in candidates)


def test_empty_unknown_and_any_facets_do_not_change_lexical_recall() -> None:
    index = CapabilityDiscoveryIndex(
        [
            CapabilityDiscoveryEntry.from_spec(
                _spec(
                    "native.web.search",
                    tool_id="web_search",
                    display_name="Web search",
                    description="Search public web evidence",
                )
            ),
            CapabilityDiscoveryEntry.from_spec(
                _spec(
                    "native.file.read",
                    tool_id="conversation_file_read",
                    display_name="Conversation file reader",
                    description="Read a conversation file",
                )
            ),
        ]
    )
    query = "Search public web evidence"
    baseline = index.search(query, limit=10)

    assert index.search(
        query,
        limit=10,
        operations=[],
        source_types=[],
        object_types=[],
        input_types=[],
    ) == baseline
    assert index.search(
        query,
        limit=10,
        operations=["unknown_operation"],
        source_types=["unknown_source"],
        object_types=["unknown_object"],
        input_types=["unknown_input"],
    ) == baseline
    assert index.search(query, limit=10, input_types=["any"]) == baseline


def test_fallback_route_never_competes_with_an_available_primary() -> None:
    primary = _spec(
        "skill.uniprot.annotation_expand",
        tool_id="skill_uniprot_annotation_expand",
        display_name="Swiss-Prot annotation",
        description="Read protein annotation and GO terms",
    )
    fallback = _spec(
        "skill.uniprot.annotation_live_fallback",
        tool_id="skill_uniprot_annotation_live_fallback",
        display_name="Live UniProt annotation fallback",
        description="Read protein annotation and GO terms",
    )
    fallback_payload = fallback.model_dump(mode="json")
    fallback_payload["discovery"]["fallback"] = {
        "for_capability_ids": ["skill.uniprot.annotation_expand"],
        "condition": "primary_unavailable_or_unsupported",
    }
    fallback = CapabilitySpec.model_validate(fallback_payload)
    primary_entry = CapabilityDiscoveryEntry.from_spec(primary)
    fallback_entry = CapabilityDiscoveryEntry.from_spec(fallback)

    available_index = CapabilityDiscoveryIndex([primary_entry, fallback_entry])
    natural = available_index.search("protein annotation and GO terms", limit=10)
    assert [item["capability_id"] for item in natural] == [
        "skill.uniprot.annotation_expand"
    ]
    assert natural[0]["provider_route"] == {
        "role": "primary",
        "fallback_capability_ids": [
            "skill.uniprot.annotation_live_fallback"
        ],
        "fallback_load_tool_ids": [
            "skill_uniprot_annotation_live_fallback"
        ],
        "fallback_eligible_on": ["provider_unavailable", "not_applicable"],
    }

    explicit = available_index.search(
        "skill_uniprot_annotation_live_fallback",
        limit=10,
    )
    assert explicit[0]["provider_route"]["eligibility_state"] == (
        "requires_primary_failure_outcome"
    )

    unavailable_index = CapabilityDiscoveryIndex([fallback_entry])
    degraded = unavailable_index.search("protein annotation and GO terms", limit=10)
    assert [item["capability_id"] for item in degraded] == [
        "skill.uniprot.annotation_live_fallback"
    ]
    assert degraded[0]["provider_route"]["eligibility_state"] == (
        "primary_unavailable"
    )


def test_exact_identifier_resolution_is_not_changed_by_conflicting_facets() -> None:
    index = CapabilityDiscoveryIndex(
        [
            CapabilityDiscoveryEntry.from_spec(
                _spec(
                    "native.web.search",
                    tool_id="web_search",
                    display_name="Web search",
                    description="Search public web evidence",
                )
            ),
            CapabilityDiscoveryEntry.from_spec(
                _spec(
                    "native.file.read",
                    tool_id="conversation_file_read",
                    display_name="Conversation file reader",
                    description="Read a conversation file",
                )
            ),
        ]
    )

    candidates = index.search(
        "web_search",
        limit=10,
        operations=["delete"],
        source_types=["conversation_file"],
        object_types=["file"],
    )

    assert [candidate["tool_id"] for candidate in candidates] == ["web_search"]


def test_compact_contract_preserves_field_meaning_and_alternative_required_sources() -> None:
    summary = _schema_contract_summary(
        {
            "type": "object",
            "properties": {
                "structure_asset_id": {
                    "type": "string",
                    "description": "Existing database asset ID.",
                },
                "structure_conversation_file_id": {
                    "type": "string",
                    "description": "Existing conversation file ID.",
                },
            },
            "allOf": [
                {
                    "oneOf": [
                        {"required": ["structure_asset_id"]},
                        {"required": ["structure_conversation_file_id"]},
                    ]
                }
            ],
            "additionalProperties": False,
        }
    )

    assert summary["fields"][0]["description"] == "Existing database asset ID."
    assert summary["required_alternatives"] == [
        {
            "operator": "oneOf",
            "alternatives": [
                ["structure_asset_id"],
                ["structure_conversation_file_id"],
            ],
        }
    ]


def test_guide_card_falls_back_to_declared_input_contract() -> None:
    class _Guide:
        skill_id = "download-guide"
        name = "Download guide"
        description = "Download a structure by ID"
        triggers = ["download"]
        routing = {}
        category = "structure"
        requires_confirmation = False
        inputs = {}
        input_contract = {"sources": {"pdb_ids": "current_user_message"}}
        produces = ["structure_file"]

    entry = CapabilityDiscoveryEntry.from_skill(_Guide())

    assert entry.metadata["contract"]["input"] == _Guide.input_contract


def test_discovery_covers_native_skill_sandbox_subagent_and_guide() -> None:
    class _Guide:
        skill_id = "sequence-workflow"
        name = "Sequence workflow"
        description = "Guide for sequence analysis"
        triggers = ["序列分析"]
        routing = {}
        category = "genomics"
        requires_confirmation = False

    entries = [
        CapabilityDiscoveryEntry.from_spec(
            _spec(
                "native.file.read",
                tool_id="conversation_file_read",
                display_name="Read conversation file",
                description="读取对话上传文件内容",
            )
        ),
        CapabilityDiscoveryEntry.from_spec(
            _spec(
                "skill.sequence.calculate",
                tool_id="execute_skill_script",
                display_name="Sequence calculator",
                description="确定性序列计算",
                provider_type=ProviderType.SKILL_SCRIPT,
                provider_ref={"skill_id": "sequence-tools", "script_name": "calculate.py"},
            )
        ),
        CapabilityDiscoveryEntry.from_spec(
            _spec(
                "sandbox.mafft.run",
                tool_id="sandbox_submit_mafft",
                display_name="MAFFT",
                description="多序列比对",
                provider_type=ProviderType.SANDBOX,
                provider_ref={"app_id": "mafft"},
            )
        ),
        CapabilityDiscoveryEntry.from_spec(
            _spec(
                "subagent.retrieval.run",
                tool_id="run_retrieval_subagent",
                display_name="Research retrieval",
                description="检索外部文献和网页证据",
                provider_type=ProviderType.SUBAGENT,
            )
        ),
        CapabilityDiscoveryEntry.from_skill(_Guide()),
    ]
    index = CapabilityDiscoveryIndex(entries)

    assert index.search("读取对话文件")[0]["provider_type"] == "native"
    assert index.search("确定性序列计算")[0]["provider_type"] == "skill_script"
    assert index.search("用 MAFFT 做多序列比对")[0]["provider_type"] == "sandbox"
    assert index.search("检索外部文献证据")[0]["provider_type"] == "subagent"
    assert any(item["kind"] == "guide" for item in index.search("序列分析指南"))


def test_generic_dispatch_and_control_tools_are_not_discovery_candidates() -> None:
    entries = [
        CapabilityDiscoveryEntry.from_spec(
            _spec(
                "skill.dispatch.execute",
                tool_id="execute_skill_script",
                display_name="execute_skill_script",
                description="generic skill dispatcher",
                provider_type=ProviderType.SKILL_SCRIPT,
                natural=False,
            )
        ),
        CapabilityDiscoveryEntry.from_spec(
            _spec(
                "native.load.tools",
                tool_id="load_tools",
                display_name="load_tools",
                description="load schemas",
                natural=False,
            )
        ),
    ]

    assert CapabilityDiscoveryIndex(entries).entries == ()
