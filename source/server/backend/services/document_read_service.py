"""Authoritative logical reads over parsed EvoEngine documents.

The service deliberately unifies *reading semantics*, not storage. Project
assets and conversation files keep their existing rows/COS objects, while all
consumers receive the same immutable document identity, stable block locators,
pagination semantics and access checks.

Milvus is a navigation index and ResourceRef is an oversized machine-result
handle. Neither replaces the authoritative parsed blocks exposed here.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

BLOCK_REF_SCHEMA_VERSION = "evoengine.document-block-ref/v1"
_CURSOR_RE = re.compile(r"^(page|search):(.*):offset:(\d+)$")


def _compact_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _block_text(block: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "text",
        "content",
        "table_caption",
        "table_body",
        "table_footnote",
        "img_caption",
        "image_caption",
        "caption",
        "img_footnote",
    ):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
        elif isinstance(value, list):
            joined = " ".join(str(item) for item in value if str(item or "").strip())
            if joined:
                parts.append(joined)
    return "\n".join(dict.fromkeys(parts))


def _block_type(block: Mapping[str, Any]) -> str:
    return str(block.get("type") or "text").strip() or "text"


def _canonical_hash(value: Any, *, length: int = 16) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def _content_list_from_parsed_json(parsed_json: Any) -> list[dict[str, Any]]:
    if not isinstance(parsed_json, dict):
        return []
    data = parsed_json.get("data")
    if not isinstance(data, dict):
        return []
    content_list = data.get("content_list")
    if not isinstance(content_list, list):
        return []
    return [dict(item) for item in content_list if isinstance(item, dict)]


def _normalized_content_list(parsed_json: Any) -> list[dict[str, Any]]:
    """Assign one global block index used by every read path.

    Parser-provided indexes are retained as ``source_block_idx`` for
    diagnostics, but never become the stable locator because parsers disagree
    on whether their index is page-local or document-global.
    """

    normalized: list[dict[str, Any]] = []
    for global_idx, raw in enumerate(_content_list_from_parsed_json(parsed_json)):
        block = dict(raw)
        if block.get("block_idx") is not None:
            block["source_block_idx"] = block.get("block_idx")
        block["block_idx"] = global_idx
        try:
            block["page_idx"] = max(0, int(block.get("page_idx") or 0))
        except (TypeError, ValueError):
            block["page_idx"] = 0
        normalized.append(block)
    return normalized


def _page_count(content_list: list[dict[str, Any]]) -> int:
    return max((int(block.get("page_idx") or 0) for block in content_list), default=-1) + 1


def _block_content_hash(block: Mapping[str, Any]) -> str:
    identity_content = {
        "type": _block_type(block),
        "text": _block_text(block),
        "bbox": block.get("bbox"),
        "image_cos_key": block.get("image_cos_key"),
    }
    return _canonical_hash(identity_content)


def build_block_ref(
    *,
    document_id: str,
    parser_version: str,
    parse_run_id: str,
    page_idx: int,
    block_idx: int,
    block_type: str,
    content_hash: str,
) -> str:
    identity = "|".join(
        (
            BLOCK_REF_SCHEMA_VERSION,
            str(document_id or ""),
            str(parser_version or ""),
            str(parse_run_id or ""),
            str(page_idx),
            str(block_idx),
            str(block_type or ""),
            str(content_hash or ""),
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16].upper()
    return f"block:{digest}:{page_idx}:{block_idx}"


@dataclass(frozen=True)
class DocumentRef:
    """One logical document at one immutable parsed version."""

    document_id: str
    source_kind: str
    source_id: str
    literature_id: int | None = None
    content_sha256: str | None = None
    parser_version: str | None = None
    parse_run_id: str | None = None
    access_scope: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "literature_id": self.literature_id,
            "content_sha256": self.content_sha256,
            "parser_version": self.parser_version,
            "parse_run_id": self.parse_run_id,
            "access_scope": dict(self.access_scope),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DocumentRef":
        return cls(
            document_id=str(value.get("document_id") or ""),
            source_kind=str(value.get("source_kind") or ""),
            source_id=str(value.get("source_id") or ""),
            literature_id=(
                int(value["literature_id"])
                if value.get("literature_id") is not None
                else None
            ),
            content_sha256=str(value.get("content_sha256") or "") or None,
            parser_version=str(value.get("parser_version") or "") or None,
            parse_run_id=str(value.get("parse_run_id") or "") or None,
            access_scope=dict(value.get("access_scope") or {}),
        )


@dataclass(frozen=True)
class ReadResult:
    document_ref: DocumentRef
    requested_scope: dict[str, Any]
    returned_scope: dict[str, Any]
    evidence: str
    coverage: str
    complete: bool
    has_more: bool
    cursor: str | None
    items: list[dict[str, Any]]
    resource_ref: str | None = None
    limitations: list[str] = field(default_factory=list)
    retrieval_mode: str = "direct"

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_ref": self.document_ref.to_dict(),
            "requested_scope": dict(self.requested_scope),
            "returned_scope": dict(self.returned_scope),
            "evidence": self.evidence,
            "coverage": self.coverage,
            "complete": self.complete,
            "has_more": self.has_more,
            "cursor": self.cursor,
            "items": self.items,
            "resource_ref": self.resource_ref,
            "limitations": list(self.limitations),
            "retrieval_mode": self.retrieval_mode,
        }


class DocumentAdapterError(ValueError):
    pass


class DocumentNotFoundError(DocumentAdapterError):
    pass


class DocumentAccessDeniedError(DocumentAdapterError):
    pass


class DocumentVersionUnavailableError(DocumentAdapterError):
    """The requested immutable parse snapshot is not stored anymore."""


def _cursor_offset(cursor: str | None, *, kind: str, scope_key: str) -> int:
    if not cursor:
        return 0
    match = _CURSOR_RE.fullmatch(str(cursor))
    if not match or match.group(1) != kind or match.group(2) != scope_key:
        raise DocumentAdapterError("cursor does not belong to the requested document scope")
    return int(match.group(3))


def _next_cursor(*, kind: str, scope_key: str, offset: int, has_more: bool) -> str | None:
    return f"{kind}:{scope_key}:offset:{offset}" if has_more else None


def _item_with_ref(
    raw: Mapping[str, Any],
    *,
    document_ref: DocumentRef,
    include_content: bool,
) -> dict[str, Any]:
    page_idx = int(raw.get("page_idx") or 0)
    block_idx = int(raw.get("block_idx") or 0)
    content_hash = _block_content_hash(raw)
    item: dict[str, Any] = dict(raw) if include_content else {}
    item.update({
        "page_ref": (
            f"{document_ref.document_id}:parse:{document_ref.parse_run_id}:page:{page_idx}"
        ),
        "block_ref": build_block_ref(
            document_id=document_ref.document_id,
            parser_version=document_ref.parser_version or "",
            parse_run_id=document_ref.parse_run_id or "",
            page_idx=page_idx,
            block_idx=block_idx,
            block_type=_block_type(raw),
            content_hash=content_hash,
        ),
        "page_idx": page_idx,
        "block_idx": block_idx,
        "type": _block_type(raw),
        "bbox": raw.get("bbox"),
        "content_hash": content_hash,
        "complete": True,
        "has_more": False,
    })
    if raw.get("source_block_idx") is not None:
        item["source_block_idx"] = raw.get("source_block_idx")
    if include_content:
        item["text"] = _block_text(raw)
    return item


def _tokenize_query(value: str) -> list[str]:
    return [
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*|[\u4e00-\u9fff]", value)
        if token.strip()
    ]


def _rank_lexical_blocks(
    content_list: list[dict[str, Any]], query: str
) -> list[dict[str, Any]]:
    needle = _compact_text(query).casefold()
    if not needle:
        return []
    tokens = list(dict.fromkeys(_tokenize_query(needle)))
    ranked: list[tuple[float, int, dict[str, Any]]] = []
    for block in content_list:
        haystack = _compact_text(_block_text(block)).casefold()
        if not haystack:
            continue
        exact = needle in haystack
        token_hits = sum(1 for token in tokens if token in haystack)
        if not exact and token_hits == 0:
            continue
        score = (10.0 if exact else 0.0) + token_hits / max(1, len(tokens))
        enriched = dict(block)
        enriched["match_score"] = round(score, 6)
        ranked.append((score, int(block["block_idx"]), enriched))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [row[2] for row in ranked]


class _ParsedDocumentAdapter:
    limitations: list[str] = []

    def document_ref(self, project_id: str | None = None) -> DocumentRef:
        raise NotImplementedError

    def content_list(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def parse_status(self) -> str:
        return ""

    def inspect(self, *, project_id: str | None = None, **_: Any) -> ReadResult:
        content_list = self.content_list()
        document_ref = self.document_ref(project_id=project_id)
        item = {
            "document_id": document_ref.document_id,
            "page_count": _page_count(content_list),
            "block_count": len(content_list),
            "parse_status": self.parse_status(),
            "parser_version": document_ref.parser_version,
            "parse_run_id": document_ref.parse_run_id,
            "content_sha256": document_ref.content_sha256,
        }
        return ReadResult(
            document_ref=document_ref,
            requested_scope={"inspect": True},
            returned_scope={
                "page_count": item["page_count"],
                "block_count": item["block_count"],
                "parse_status": item["parse_status"],
            },
            evidence="available" if content_list else "none",
            coverage="exhausted" if content_list else "not_applicable",
            complete=True,
            has_more=False,
            cursor=None,
            items=[item],
            limitations=list(self.limitations),
        )

    def read_page(
        self,
        page_idx: int,
        *,
        project_id: str | None = None,
        include_content: bool = True,
        limit: int = 200,
        cursor: str | None = None,
        **_: Any,
    ) -> ReadResult:
        requested_page = max(0, int(page_idx or 0))
        content_list = self.content_list()
        page_blocks = [
            block for block in content_list if int(block["page_idx"]) == requested_page
        ]
        offset = _cursor_offset(cursor, kind="page", scope_key=str(requested_page))
        safe_limit = max(1, min(1000, int(limit or 200)))
        selected = page_blocks[offset : offset + safe_limit]
        next_offset = offset + len(selected)
        has_more = next_offset < len(page_blocks)
        document_ref = self.document_ref(project_id=project_id)
        items = [
            _item_with_ref(
                block,
                document_ref=document_ref,
                include_content=include_content,
            )
            for block in selected
        ]
        return ReadResult(
            document_ref=document_ref,
            requested_scope={
                "page_idx": requested_page,
                "offset": offset,
                "limit": safe_limit,
            },
            returned_scope={
                "page_idx": requested_page,
                "page_count": _page_count(content_list),
                "blocks_returned": len(items),
                "blocks_total": len(page_blocks),
                "offset": offset,
            },
            evidence="available" if items else "none",
            coverage="more_available" if has_more else "exhausted",
            complete=not has_more,
            has_more=has_more,
            cursor=_next_cursor(
                kind="page",
                scope_key=str(requested_page),
                offset=next_offset,
                has_more=has_more,
            ),
            items=items,
            limitations=list(self.limitations),
        )

    def read_block(
        self,
        block_idx: int,
        *,
        project_id: str | None = None,
        block_ref: str | None = None,
        include_content: bool = True,
        **_: Any,
    ) -> ReadResult:
        content_list = self.content_list()
        requested_idx = int(block_idx)
        block = next(
            (item for item in content_list if int(item["block_idx"]) == requested_idx),
            None,
        )
        document_ref = self.document_ref(project_id=project_id)
        items = (
            [_item_with_ref(block, document_ref=document_ref, include_content=include_content)]
            if block is not None
            else []
        )
        if block_ref and (not items or items[0]["block_ref"] != block_ref):
            raise DocumentVersionUnavailableError(
                "block_ref does not match the current immutable document version"
            )
        return ReadResult(
            document_ref=document_ref,
            requested_scope={"block_idx": requested_idx, "block_ref": block_ref},
            returned_scope={"blocks_returned": len(items)},
            evidence="available" if items else "none",
            coverage="exhausted",
            complete=True,
            has_more=False,
            cursor=None,
            items=items,
            limitations=list(self.limitations),
        )

    def read_span(
        self,
        block_idx: int,
        char_start: int,
        char_end: int,
        *,
        project_id: str | None = None,
        block_ref: str | None = None,
        **kwargs: Any,
    ) -> ReadResult:
        block_result = self.read_block(
            block_idx,
            project_id=project_id,
            block_ref=block_ref,
            include_content=True,
            **kwargs,
        )
        start = max(0, int(char_start or 0))
        end = max(start, int(char_end or start))
        items: list[dict[str, Any]] = []
        if block_result.items:
            parent = block_result.items[0]
            text = str(parent.get("text") or "")
            actual_end = min(len(text), end)
            span_text = text[start:actual_end]
            span_hash = hashlib.sha256(
                f"{parent['block_ref']}|{start}|{actual_end}|{span_text}".encode("utf-8")
            ).hexdigest()[:16].upper()
            items = [
                {
                    **parent,
                    "span_ref": f"span:{span_hash}:{start}:{actual_end}",
                    "parent_block_ref": parent["block_ref"],
                    "char_start": start,
                    "char_end": actual_end,
                    "text": span_text,
                }
            ]
        return ReadResult(
            document_ref=block_result.document_ref,
            requested_scope={
                "block_idx": int(block_idx),
                "block_ref": block_ref,
                "char_start": start,
                "char_end": end,
            },
            returned_scope={
                "spans_returned": len(items),
                "char_start": items[0]["char_start"] if items else start,
                "char_end": items[0]["char_end"] if items else start,
            },
            evidence="available" if items and items[0]["text"] else "none",
            coverage="exhausted",
            complete=True,
            has_more=False,
            cursor=None,
            items=items,
            limitations=list(self.limitations),
        )

    def search(
        self,
        query: str,
        *,
        project_id: str | None = None,
        limit: int = 10,
        cursor: str | None = None,
        **_: Any,
    ) -> ReadResult:
        normalized_query = _compact_text(query)
        query_key = hashlib.sha256(normalized_query.casefold().encode("utf-8")).hexdigest()[:12]
        offset = _cursor_offset(cursor, kind="search", scope_key=query_key)
        safe_limit = max(1, min(50, int(limit or 10)))
        matches = _rank_lexical_blocks(self.content_list(), normalized_query)
        selected = matches[offset : offset + safe_limit]
        next_offset = offset + len(selected)
        has_more = next_offset < len(matches)
        document_ref = self.document_ref(project_id=project_id)
        items = [
            {
                **_item_with_ref(block, document_ref=document_ref, include_content=True),
                "match_score": block.get("match_score"),
            }
            for block in selected
        ]
        return ReadResult(
            document_ref=document_ref,
            requested_scope={
                "query": normalized_query,
                "mode": "lexical",
                "offset": offset,
                "limit": safe_limit,
            },
            returned_scope={
                "hit_count": len(items),
                "hit_total": len(matches),
                "retrieval_mode": "lexical",
                "offset": offset,
            },
            evidence="available" if items else "none",
            coverage="more_available" if has_more else "exhausted",
            complete=not has_more,
            has_more=has_more,
            cursor=_next_cursor(
                kind="search",
                scope_key=query_key,
                offset=next_offset,
                has_more=has_more,
            ),
            items=items,
            limitations=list(self.limitations),
            retrieval_mode="lexical",
        )


class ProjectAssetDocumentAdapter(_ParsedDocumentAdapter):
    source_kind = "project_asset"

    def __init__(self, *, asset: Any, parsing_run: Any | None, literature_id: int | None) -> None:
        self.asset = asset
        self.parsing_run = parsing_run
        self.literature_id = literature_id

    @property
    def document_id(self) -> str:
        return f"asset:{int(self.asset.id)}"

    def content_list(self) -> list[dict[str, Any]]:
        return _normalized_content_list(getattr(self.asset, "parsed_json", None))

    def parse_status(self) -> str:
        value = getattr(self.asset, "parse_status", "")
        return str(value.value if hasattr(value, "value") else value or "").lower()

    def _parse_run_id(self) -> str:
        if self.parsing_run is not None:
            return str(self.parsing_run.id)
        identity = {
            "sha256": str(getattr(self.asset, "sha256", "") or ""),
            "parser_version": str(getattr(self.asset, "parser_version", "") or ""),
            "content": self.content_list(),
        }
        return f"legacy-{_canonical_hash(identity)}"

    def document_ref(self, project_id: str | None = None) -> DocumentRef:
        return DocumentRef(
            document_id=self.document_id,
            source_kind=self.source_kind,
            source_id=str(self.asset.id),
            literature_id=self.literature_id,
            content_sha256=str(getattr(self.asset, "sha256", "") or "").strip() or None,
            parser_version=str(getattr(self.asset, "parser_version", "") or "").strip() or None,
            parse_run_id=self._parse_run_id(),
            access_scope={
                "user_id": getattr(self.asset, "user_id", None),
                "project_id": str(project_id or ""),
            },
        )


class ConversationFileDocumentAdapter(_ParsedDocumentAdapter):
    source_kind = "conversation_file"
    limitations = ["conversation_file has no semantic index; page/lexical mode only"]

    def __init__(self, *, conversation_file: Any, attachment: Any | None = None) -> None:
        self.conversation_file = conversation_file
        self.attachment = attachment

    @property
    def document_id(self) -> str:
        return f"conversation_file:{int(self.conversation_file.id)}"

    def content_list(self) -> list[dict[str, Any]]:
        parsed_json = (
            getattr(self.attachment, "parsed_json", None)
            if self.attachment is not None
            else getattr(self.conversation_file, "parsed_json", None)
        )
        return _normalized_content_list(parsed_json)

    def parse_status(self) -> str:
        source = self.attachment if self.attachment is not None else self.conversation_file
        value = getattr(source, "parse_status", "")
        return str(value.value if hasattr(value, "value") else value or "").lower()

    def _version_identity(self) -> tuple[str | None, str]:
        source = self.attachment if self.attachment is not None else self.conversation_file
        parsed_json = (
            source.parsed_json
            if isinstance(source.parsed_json, dict)
            else {}
        )
        meta = parsed_json.get("meta") if isinstance(parsed_json.get("meta"), dict) else {}
        parser_version = str(
            meta.get("parser_version")
            or meta.get("route")
            or getattr(self.conversation_file, "source_type", "")
            or "conversation-file-v1"
        )
        identity = {
            "sha256": str(getattr(source, "sha256", "") or ""),
            "parser_version": parser_version,
            "parsed_json": parsed_json,
        }
        return parser_version, f"snapshot-{_canonical_hash(identity)}"

    def document_ref(self, project_id: str | None = None) -> DocumentRef:
        parser_version, parse_run_id = self._version_identity()
        return DocumentRef(
            document_id=self.document_id,
            source_kind=self.source_kind,
            source_id=str(self.conversation_file.id),
            content_sha256=str(
                getattr(
                    self.attachment if self.attachment is not None else self.conversation_file,
                    "sha256",
                    "",
                )
                or ""
            ).strip()
            or None,
            parser_version=parser_version,
            parse_run_id=parse_run_id,
            access_scope={
                "user_id": getattr(self.conversation_file, "user_id", None),
                "project_id": str(
                    project_id
                    if project_id is not None
                    else getattr(self.conversation_file, "project_id", "")
                    or ""
                ),
            },
        )


def _assert_requested_version(requested: DocumentRef, actual: DocumentRef) -> None:
    comparisons = (
        ("content_sha256", requested.content_sha256, actual.content_sha256),
        ("parser_version", requested.parser_version, actual.parser_version),
        ("parse_run_id", requested.parse_run_id, actual.parse_run_id),
    )
    mismatches = [
        name
        for name, expected, current in comparisons
        if expected is not None and str(expected) != str(current or "")
    ]
    if mismatches:
        raise DocumentVersionUnavailableError(
            "requested document version is unavailable; mismatched " + ", ".join(mismatches)
        )


class DocumentReadService:
    def __init__(self, db: Any) -> None:
        self.db = db

    def resolve(self, document_ref: DocumentRef) -> _ParsedDocumentAdapter:
        if document_ref.source_kind == "project_asset":
            from database import AssetStatus, FileParsingRun, ParseStatus, ProjectAsset
            from sqlmodel import select

            asset = self.db.get(ProjectAsset, int(document_ref.source_id))
            if asset is None or asset.status != AssetStatus.ACTIVE:
                raise DocumentNotFoundError(f"asset not found: {document_ref.source_id}")
            parsing_run = self.db.exec(
                select(FileParsingRun)
                .where(
                    FileParsingRun.asset_id == int(asset.id),
                    FileParsingRun.status == ParseStatus.COMPLETED,
                )
                .order_by(FileParsingRun.id.desc())
            ).first()
            adapter: _ParsedDocumentAdapter = ProjectAssetDocumentAdapter(
                asset=asset,
                parsing_run=parsing_run,
                literature_id=document_ref.literature_id,
            )
        elif document_ref.source_kind == "conversation_file":
            from database import ChatAttachment, ConversationFile

            conversation_file = self.db.get(ConversationFile, int(document_ref.source_id))
            if conversation_file is None or bool(conversation_file.is_deleted):
                raise DocumentNotFoundError(
                    f"conversation_file not found: {document_ref.source_id}"
                )
            attachment = (
                self.db.get(ChatAttachment, int(conversation_file.chat_attachment_id))
                if conversation_file.chat_attachment_id is not None
                else None
            )
            adapter = ConversationFileDocumentAdapter(
                conversation_file=conversation_file,
                attachment=attachment,
            )
        else:
            raise DocumentAdapterError(f"unsupported source_kind: {document_ref.source_kind}")

        actual_ref = adapter.document_ref(
            project_id=str(document_ref.access_scope.get("project_id") or "")
        )
        requested_project_id = str(document_ref.access_scope.get("project_id") or "")
        if (
            document_ref.source_kind == "conversation_file"
            and requested_project_id
            and requested_project_id
            != str(getattr(adapter.conversation_file, "project_id", "") or "")
        ):
            raise DocumentAccessDeniedError(
                "conversation_file is not part of the requested project"
            )
        _assert_requested_version(document_ref, actual_ref)
        return adapter

    @staticmethod
    def authorize(adapter: _ParsedDocumentAdapter, *, user_id: int | None = None) -> None:
        owner_id = adapter.document_ref().access_scope.get("user_id")
        if user_id is not None and owner_id is not None and int(user_id) != int(owner_id):
            raise DocumentAccessDeniedError(
                f"user {user_id} cannot access document owned by {owner_id}"
            )

    def _call(
        self,
        method: str,
        document_ref: DocumentRef,
        *,
        user_id: int | None,
        **kwargs: Any,
    ) -> ReadResult:
        adapter = self.resolve(document_ref)
        self.authorize(adapter, user_id=user_id)
        return getattr(adapter, method)(
            user_id=user_id,
            project_id=document_ref.access_scope.get("project_id"),
            **kwargs,
        )

    def inspect(self, document_ref: DocumentRef, *, user_id: int | None = None) -> ReadResult:
        return self._call("inspect", document_ref, user_id=user_id)

    def read_page(
        self,
        document_ref: DocumentRef,
        page_idx: int,
        *,
        user_id: int | None = None,
        include_content: bool = True,
        limit: int = 200,
        cursor: str | None = None,
    ) -> ReadResult:
        return self._call(
            "read_page",
            document_ref,
            user_id=user_id,
            page_idx=page_idx,
            include_content=include_content,
            limit=limit,
            cursor=cursor,
        )

    def read_block(
        self,
        document_ref: DocumentRef,
        block_idx: int,
        *,
        user_id: int | None = None,
        block_ref: str | None = None,
        include_content: bool = True,
    ) -> ReadResult:
        return self._call(
            "read_block",
            document_ref,
            user_id=user_id,
            block_idx=block_idx,
            block_ref=block_ref,
            include_content=include_content,
        )

    def read_span(
        self,
        document_ref: DocumentRef,
        block_idx: int,
        char_start: int,
        char_end: int,
        *,
        user_id: int | None = None,
        block_ref: str | None = None,
    ) -> ReadResult:
        return self._call(
            "read_span",
            document_ref,
            user_id=user_id,
            block_idx=block_idx,
            block_ref=block_ref,
            char_start=char_start,
            char_end=char_end,
        )

    def search(
        self,
        document_ref: DocumentRef,
        query: str,
        *,
        user_id: int | None = None,
        limit: int = 10,
        cursor: str | None = None,
    ) -> ReadResult:
        return self._call(
            "search",
            document_ref,
            user_id=user_id,
            query=query,
            limit=limit,
            cursor=cursor,
        )


def build_asset_document_ref(
    asset_id: int,
    *,
    user_id: int | None = None,
    project_id: str | None = None,
    literature_id: int | None = None,
) -> DocumentRef:
    return DocumentRef(
        document_id=f"asset:{int(asset_id)}",
        source_kind="project_asset",
        source_id=str(int(asset_id)),
        literature_id=literature_id,
        access_scope={"user_id": user_id, "project_id": str(project_id or "")},
    )


def build_conversation_file_document_ref(
    conversation_file_id: int,
    *,
    user_id: int | None = None,
    project_id: str | None = None,
) -> DocumentRef:
    return DocumentRef(
        document_id=f"conversation_file:{int(conversation_file_id)}",
        source_kind="conversation_file",
        source_id=str(int(conversation_file_id)),
        access_scope={"user_id": user_id, "project_id": str(project_id or "")},
    )
