"""P0-C: Document Read contract and adapters (C0/C1).

The same parsed document must be readable through the shared
DocumentReadService contract from both the project-asset side and the
conversation-file side, with stable versioned locators, explicit
complete/has_more/cursor semantics and no vector recall for direct page
reads.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import (  # noqa: E402
    AssetScope,
    AssetStatus,
    ConversationFile,
    FileParsingRun,
    ParseStatus,
    ProjectAsset,
)
from services.document_read_service import (  # noqa: E402
    DocumentAccessDeniedError,
    DocumentNotFoundError,
    DocumentReadService,
    DocumentVersionUnavailableError,
    build_asset_document_ref,
    build_block_ref,
    build_conversation_file_document_ref,
)


def _engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in (
        ProjectAsset.__table__,
        FileParsingRun.__table__,
        ConversationFile.__table__,
    ):
        table.create(engine)
    return engine


def _content_list() -> list[dict]:
    return [
        {
            "page_idx": 0,
            "type": "text",
            "text": "Weighted gene co-expression network analysis (WGCNA) overview.",
            "bbox": [10, 20, 100, 40],
        },
        {
            "page_idx": 0,
            "type": "text",
            "text": "Correlation networks identify modules of co-expressed genes.",
            "bbox": [10, 50, 100, 70],
        },
        {
            "page_idx": 1,
            "type": "table",
            "table_caption": "Module-trait relationships",
            "table_body": "module|trait|p-value\nblue|disease|0.001",
        },
        {
            "page_idx": 2,
            "type": "text",
            "text": "Discussion and limitations of the method are described in the final section.",
            "bbox": [10, 90, 100, 110],
        },
    ]


def _make_asset(**overrides) -> ProjectAsset:
    payload = {
        "user_id": 7,
        "scope": AssetScope.KNOWLEDGE,
        "file_name": "paper.pdf",
        "file_ext": "pdf",
        "mime_type": "application/pdf",
        "size_bytes": 1024,
        "sha256": "sha-paper-1",
        "cos_key": "cos/paper.pdf",
        "file_url": "https://cos.local/paper.pdf",
        "status": AssetStatus.ACTIVE,
        "parse_status": ParseStatus.COMPLETED,
        "parsed_json": {"data": {"content_list": _content_list()}},
        "parsed_text": "parsed flat text",
        "parser_version": "mineru-v1",
    }
    payload.update(overrides)
    return ProjectAsset(**payload)


def _make_conversation_file(**overrides) -> ConversationFile:
    payload = {
        "user_id": 7,
        "project_id": "project_doc",
        "conversation_id": "conv_doc",
        "file_name": "attachment.pdf",
        "mime_type": "application/pdf",
        "size_bytes": 512,
        "sha256": "sha-attach-1",
        "content_text": "flat text",
        "parsed_json": {"data": {"content_list": _content_list()}, "meta": {"route": "chat_document"}},
    }
    payload.update(overrides)
    return ConversationFile(**payload)


def test_asset_adapter_reads_same_page_blocks_as_shared_contract() -> None:
    engine = _engine()
    with Session(engine) as db:
        asset = _make_asset(id=1)
        db.add(asset)
        db.add(
            FileParsingRun(
                id=11,
                asset_id=1,
                user_id=7,
                status="completed",
                created_at=datetime.utcnow(),
            )
        )
        db.commit()

        service = DocumentReadService(db)
        ref = build_asset_document_ref(1, user_id=7, project_id="project_doc")
        result = service.read_page(ref, 0, user_id=7)

    assert result.document_ref.parse_run_id == "11"
    assert result.evidence == "available"
    assert result.coverage == "exhausted"
    assert result.complete is True
    assert result.has_more is False
    assert len(result.items) == 2
    first = result.items[0]
    assert first["page_idx"] == 0
    assert first["block_idx"] == 0
    assert first["type"] == "text"
    assert "WGCNA" in first["text"]
    assert first["block_ref"].startswith("block:")
    assert first["content_hash"]
    # Stable ref: same document/version/parse/content yields the same ref.
    assert (
        build_block_ref(
            document_id="asset:1",
            parser_version="mineru-v1",
            parse_run_id="11",
            page_idx=0,
            block_idx=0,
            block_type="text",
            content_hash=first["content_hash"],
        )
        == first["block_ref"]
    )


def test_asset_read_page_does_not_invoke_vector_recall() -> None:
    engine = _engine()
    with Session(engine) as db:
        db.add(_make_asset(id=2, user_id=7))
        db.commit()
        service = DocumentReadService(db)
        ref = build_asset_document_ref(2, user_id=7)
        result = service.read_page(ref, 1, user_id=7)

    # Direct page read stays on parsed content; no Milvus/embedding path exists
    # in the service at all.
    assert result.retrieval_mode == "direct"
    assert len(result.items) == 1
    assert result.items[0]["type"] == "table"
    assert "Module-trait" in result.items[0]["table_caption"]


def test_asset_search_locates_unknown_position_without_full_injection() -> None:
    engine = _engine()
    with Session(engine) as db:
        db.add(_make_asset(id=3, user_id=7))
        db.commit()
        service = DocumentReadService(db)
        ref = build_asset_document_ref(3, user_id=7)
        result = service.search(ref, "limitations", user_id=7, limit=10)

    assert result.retrieval_mode == "lexical"
    assert result.evidence == "available"
    assert len(result.items) == 1
    hit = result.items[0]
    assert hit["page_idx"] == 2
    assert "final section" in hit["text"]
    assert hit["page_ref"].endswith("page:2")


def test_asset_search_empty_query_and_miss_are_none_not_fake_exhausted() -> None:
    engine = _engine()
    with Session(engine) as db:
        db.add(_make_asset(id=4, user_id=7))
        db.commit()
        service = DocumentReadService(db)
        ref = build_asset_document_ref(4, user_id=7)
        empty = service.search(ref, "", user_id=7)
        miss = service.search(ref, "nonexistent-token-xyz", user_id=7)

    assert empty.items == []
    assert miss.items == []
    # No evidence in this document is not "the paper does not contain it".
    assert miss.evidence == "none"
    assert miss.coverage == "exhausted"


def test_conversation_file_adapter_reports_lexical_mode_and_no_semantic_claims() -> None:
    engine = _engine()
    with Session(engine) as db:
        db.add(_make_conversation_file(id=21, user_id=7))
        db.commit()
        service = DocumentReadService(db)
        ref = build_conversation_file_document_ref(
            21, user_id=7, project_id="project_doc"
        )
        page = service.read_page(ref, 0, user_id=7)
        search = service.search(ref, "co-expressed", user_id=7)

    assert page.retrieval_mode == "direct"
    assert page.limitations == [
        "conversation_file has no semantic index; page/lexical mode only"
    ]
    assert page.items[0]["text"].startswith("Weighted gene")
    assert search.retrieval_mode == "lexical"
    assert search.items[0]["page_idx"] == 0
    assert "co-expressed" in search.items[0]["text"]


def test_conversation_file_search_returns_complete_cursor_semantics() -> None:
    engine = _engine()
    with Session(engine) as db:
        db.add(_make_conversation_file(id=22, user_id=7))
        db.commit()
        service = DocumentReadService(db)
        ref = build_conversation_file_document_ref(22, user_id=7)
        result = service.search(ref, "module", user_id=7, limit=1)

    assert result.has_more is True
    assert result.complete is False
    assert result.cursor

    with Session(engine) as db:
        next_result = DocumentReadService(db).search(
            ref,
            "module",
            user_id=7,
            limit=1,
            cursor=result.cursor,
        )
    assert next_result.has_more is False
    assert next_result.complete is True
    assert next_result.cursor is None
    assert {
        item["block_ref"] for item in result.items
    }.isdisjoint(item["block_ref"] for item in next_result.items)


def test_conversation_file_inspect_has_immutable_snapshot_identity() -> None:
    engine = _engine()
    with Session(engine) as db:
        db.add(_make_conversation_file(id=23, user_id=7))
        db.commit()
        result = DocumentReadService(db).inspect(
            build_conversation_file_document_ref(23, user_id=7),
            user_id=7,
        )

    assert result.items[0]["block_count"] == 4
    assert result.document_ref.parse_run_id.startswith("snapshot-")
    assert result.document_ref.parse_run_id != "current"
    assert result.document_ref.parser_version == "chat_document"


def test_page_search_and_block_read_share_one_global_block_identity() -> None:
    engine = _engine()
    with Session(engine) as db:
        db.add(_make_asset(id=24, user_id=7))
        db.commit()
        service = DocumentReadService(db)
        ref = build_asset_document_ref(24, user_id=7)
        page_hit = service.read_page(ref, 1, user_id=7).items[0]
        search_hit = service.search(ref, "Module-trait", user_id=7).items[0]
        block_hit = service.read_block(
            ref,
            page_hit["block_idx"],
            user_id=7,
            block_ref=page_hit["block_ref"],
        ).items[0]

    assert page_hit["block_idx"] == 2
    assert search_hit["block_idx"] == 2
    assert page_hit["block_ref"] == search_hit["block_ref"] == block_hit["block_ref"]


def test_page_cursor_reads_remaining_blocks_without_duplicates() -> None:
    engine = _engine()
    with Session(engine) as db:
        db.add(_make_asset(id=25, user_id=7))
        db.commit()
        service = DocumentReadService(db)
        ref = build_asset_document_ref(25, user_id=7)
        first = service.read_page(ref, 0, user_id=7, limit=1)
        second = service.read_page(ref, 0, user_id=7, limit=1, cursor=first.cursor)

    assert first.has_more is True
    assert first.complete is False
    assert second.has_more is False
    assert second.complete is True
    assert first.items[0]["block_idx"] == 0
    assert second.items[0]["block_idx"] == 1


def test_read_span_is_anchored_to_versioned_parent_block() -> None:
    engine = _engine()
    with Session(engine) as db:
        db.add(_make_asset(id=26, user_id=7))
        db.commit()
        service = DocumentReadService(db)
        ref = build_asset_document_ref(26, user_id=7)
        block = service.read_block(ref, 0, user_id=7).items[0]
        span = service.read_span(
            ref,
            0,
            0,
            8,
            user_id=7,
            block_ref=block["block_ref"],
        ).items[0]

    assert span["text"] == "Weighted"
    assert span["parent_block_ref"] == block["block_ref"]
    assert span["span_ref"].startswith("span:")


def test_natural_multi_token_query_uses_deterministic_lexical_fallback() -> None:
    engine = _engine()
    with Session(engine) as db:
        db.add(_make_asset(id=27, user_id=7))
        db.commit()
        result = DocumentReadService(db).search(
            build_asset_document_ref(27, user_id=7),
            "gene modules",
            user_id=7,
        )

    assert [item["block_idx"] for item in result.items][:2] == [1, 0]
    assert result.retrieval_mode == "lexical"


def test_access_scope_rejects_other_users() -> None:
    engine = _engine()
    with Session(engine) as db:
        db.add(_make_asset(id=5, user_id=7))
        db.commit()
        service = DocumentReadService(db)
        ref = build_asset_document_ref(5, user_id=7)

        try:
            service.read_page(ref, 0, user_id=999)
        except DocumentAccessDeniedError:
            pass
        else:
            raise AssertionError("expected DocumentAccessDeniedError")


def test_missing_document_fails_closed() -> None:
    engine = _engine()
    with Session(engine) as db:
        service = DocumentReadService(db)
        ref = build_asset_document_ref(999999, user_id=7)
        try:
            service.inspect(ref, user_id=7)
        except DocumentNotFoundError:
            pass
        else:
            raise AssertionError("expected DocumentNotFoundError")


def test_inspect_reports_version_identity_and_coverage() -> None:
    engine = _engine()
    with Session(engine) as db:
        db.add(_make_asset(id=6, user_id=7))
        db.add(
            FileParsingRun(
                id=31,
                asset_id=6,
                user_id=7,
                status="completed",
                created_at=datetime.utcnow(),
            )
        )
        db.commit()
        service = DocumentReadService(db)
        ref = build_asset_document_ref(6, user_id=7)
        result = service.inspect(ref, user_id=7)

    item = result.items[0]
    assert item["page_count"] == 3
    assert item["block_count"] == 4
    assert item["parse_run_id"] == "31"
    assert item["parser_version"] == "mineru-v1"
    assert item["content_sha256"] == "sha-paper-1"
    assert result.evidence == "available"
    assert result.coverage == "exhausted"


def test_reparse_changes_parse_run_id_and_block_refs() -> None:
    engine = _engine()
    with Session(engine) as db:
        db.add(_make_asset(id=7, user_id=7))
        db.add(
            FileParsingRun(
                id=41,
                asset_id=7,
                user_id=7,
                status="completed",
                created_at=datetime.utcnow(),
            )
        )
        db.add(
            FileParsingRun(
                id=42,
                asset_id=7,
                user_id=7,
                status="completed",
                created_at=datetime.utcnow(),
            )
        )
        db.commit()
        service = DocumentReadService(db)
        ref = build_asset_document_ref(7, user_id=7)
        result = service.read_page(ref, 0, user_id=7)

    # The unversioned builder resolves the newest completed parsing run.
    assert result.document_ref.parse_run_id == "42"
    old_ref = build_block_ref(
        document_id="asset:7",
        parser_version="mineru-v1",
        parse_run_id="41",
        page_idx=0,
        block_idx=0,
        block_type="text",
        content_hash=result.items[0]["content_hash"],
    )
    assert old_ref != result.items[0]["block_ref"]

    # FileParsingRun does not retain parsed_json snapshots today.  A caller
    # holding the old immutable version gets an explicit unavailable error,
    # never content from run 42 under the run-41 identity.
    old_document_ref = result.document_ref.__class__(
        **{**result.document_ref.to_dict(), "parse_run_id": "41"}
    )
    with Session(engine) as db:
        try:
            DocumentReadService(db).read_page(old_document_ref, 0, user_id=7)
        except DocumentVersionUnavailableError:
            pass
        else:
            raise AssertionError("expected DocumentVersionUnavailableError")
