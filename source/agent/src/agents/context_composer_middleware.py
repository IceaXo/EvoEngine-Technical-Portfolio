"""Apply the single Context Composer before every model invocation."""

from __future__ import annotations

import os
import json
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import SystemMessage

from src.runtime.worker_diagnostics import emit_worker_checkpoint
from src.services.context_composer import (
    MINIMUM_CONTEXT_WINDOW_TOKENS,
    compose_model_messages,
)
from src.services.context_ledger import (
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    estimate_tokens,
    serialize_tools_for_ledger,
)
from src.services.resource_access_ledger import (
    empty_resource_access_ledger,
    merge_resource_access_ledger,
    resource_access_prompt_payload,
    update_resource_access_ledger_from_messages,
)
from src.services.session_compaction import (
    SESSION_COMPACT_SCHEMA,
    SessionCompactionError,
    SessionCompactionNotApplicableError,
    SessionCompactor,
    active_messages_with_checkpoint,
    session_compaction_input_signature,
)


DEFAULT_PROMPT_TOKEN_BUDGET = max(
    MINIMUM_CONTEXT_WINDOW_TOKENS,
    int(DEFAULT_CONTEXT_WINDOW_TOKENS or 0),
)
_RESOURCE_ACCESS_OPEN = "[RESOURCE_ACCESS_LEDGER]"
_RESOURCE_ACCESS_CLOSE = "[/RESOURCE_ACCESS_LEDGER]"
_RESOURCE_ACCESS_BLOCK_RE = re.compile(
    re.escape(_RESOURCE_ACCESS_OPEN)
    + r".*?"
    + re.escape(_RESOURCE_ACCESS_CLOSE)
    + r"\s*",
    flags=re.DOTALL,
)
_COMPACTION_RETRY_STATE_KEY = "_runtime_compaction_retry_v1"
_COMPACTION_RETRY_SCHEMA = "evoengine.session-compaction-retry/v1"
_COMPACTION_OUTPUT_LIMIT_ERROR = "SessionCompactionOutputLimitError"
_COMPACTION_RETRY_BASE_CALLS = 4
_COMPACTION_OUTPUT_LIMIT_BASE_CALLS = 8
_COMPACTION_RETRY_MAX_CALLS = 64
_COMPACTION_SIGNIFICANT_GROWTH_TOKENS = 32_768
_COMPACTION_SIGNIFICANT_GROWTH_MESSAGES = 8


def _prompt_token_budget() -> int:
    """Use the single declared model window from ``EVO_CONTEXT_WINDOW_TOKENS``."""

    return DEFAULT_PROMPT_TOKEN_BUDGET


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        value = int(default)
    return value if value > 0 else int(default)


def _ratio_env(name: str, default: float) -> float:
    try:
        value = float(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        value = default
    if not 0.0 < value < 1.0:
        return default
    return value


class ContextComposerMiddleware(AgentMiddleware):
    """Keep context lossless and compact only through a semantic checkpoint."""

    def __init__(
        self,
        *,
        request_id: str = "",
        prompt_token_budget: int | None = None,
        resource_access_ledger: dict[str, Any] | None = None,
        summary_model: Any | None = None,
        session_checkpoint_state: dict[str, Any] | None = None,
        session_compaction_protocol: Any | None = None,
    ) -> None:
        self._request_id = str(request_id or "")
        self._prompt_token_budget = prompt_token_budget
        self._call_index = 0
        self._cached_system_content = None
        self._cached_system_tokens = 0
        self._cached_tool_names: tuple[str, ...] | None = None
        self._cached_tool_tokens = 0
        self._resource_access_ledger = (
            resource_access_ledger
            if isinstance(resource_access_ledger, dict)
            else empty_resource_access_ledger()
        )
        self._summary_model = summary_model
        self._session_compaction_protocol = session_compaction_protocol
        self._session_checkpoint_state = (
            session_checkpoint_state
            if isinstance(session_checkpoint_state, dict)
            else {}
        )
        merge_resource_access_ledger(
            self._resource_access_ledger,
            self._resource_access_ledger,
        )
        self.last_composition_stats: dict[str, Any] = {}

    def _system_message_with_resource_state(
        self,
        request: ModelRequest[Any],
    ) -> SystemMessage | None:
        update_resource_access_ledger_from_messages(
            self._resource_access_ledger,
            request.messages,
        )
        current = request.system_message
        current_content = (
            str(getattr(current, "content", "") or "") if current is not None else ""
        )
        base_content = _RESOURCE_ACCESS_BLOCK_RE.sub("", current_content).rstrip()
        payload = resource_access_prompt_payload(self._resource_access_ledger)
        resources = payload.get("resources") or []
        if not resources:
            return current
        payload["reading_contract"] = {
            "already_read": (
                "read_ranges are observational runtime facts; an explicit "
                "resource_read cursor is still honored exactly"
            ),
            "continuation": (
                "use the returned cursor for sequential continuation; use a "
                "search match.read_cursor for exact local context"
            ),
            "navigation": (
                "use resource_search for a known term and resource_read only for "
                "needed sequential context"
            ),
        }
        block = (
            _RESOURCE_ACCESS_OPEN
            + "\n"
            + json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
            + "\n"
            + _RESOURCE_ACCESS_CLOSE
        )
        content = f"{base_content}\n\n{block}" if base_content else block
        if current is None:
            return SystemMessage(content=content)
        return current.model_copy(update={"content": content})

    def _budget_state(
        self,
        request: ModelRequest[Any],
    ) -> tuple[int, int, int, int, SystemMessage | None]:
        total_budget = max(
            MINIMUM_CONTEXT_WINDOW_TOKENS,
            int(
                self._prompt_token_budget
                if self._prompt_token_budget is not None
                else _prompt_token_budget()
            ),
        )
        system_message = self._system_message_with_resource_state(request)
        system_content = (
            getattr(system_message, "content", "")
            if system_message is not None
            else ""
        )
        if system_content != self._cached_system_content:
            self._cached_system_content = system_content
            self._cached_system_tokens = estimate_tokens(system_content)
        system_tokens = self._cached_system_tokens
        tool_names = tuple(
            str(getattr(tool, "name", "") or tool.__class__.__name__)
            for tool in (request.tools or [])
        )
        if tool_names != self._cached_tool_names:
            self._cached_tool_names = tool_names
            self._cached_tool_tokens = estimate_tokens(
                serialize_tools_for_ledger(list(request.tools or []))
            )
        tool_tokens = self._cached_tool_tokens
        reserved_output_tokens = _positive_int_env(
            "EVO_CONTEXT_RESERVED_OUTPUT_TOKENS",
            16_384,
        )
        input_capacity = total_budget - reserved_output_tokens
        if input_capacity <= 0 or system_tokens + tool_tokens >= input_capacity:
            raise SessionCompactionError(
                "system/tool contracts leave no model input capacity inside the declared context window"
            )
        message_capacity = input_capacity - system_tokens - tool_tokens
        return (
            total_budget,
            reserved_output_tokens,
            input_capacity,
            message_capacity,
            system_message,
        )

    def _checkpoint(self) -> dict[str, Any] | None:
        if (
            self._session_checkpoint_state.get("schema_version")
            != SESSION_COMPACT_SCHEMA
        ):
            return None
        checkpoint = dict(self._session_checkpoint_state)
        checkpoint.pop(_COMPACTION_RETRY_STATE_KEY, None)
        return checkpoint

    def _advance_compaction_observation(self) -> tuple[dict[str, Any], int]:
        raw = self._session_checkpoint_state.get(_COMPACTION_RETRY_STATE_KEY)
        retry = (
            dict(raw)
            if isinstance(raw, dict)
            and raw.get("schema_version") == _COMPACTION_RETRY_SCHEMA
            else {"schema_version": _COMPACTION_RETRY_SCHEMA}
        )
        observation = max(0, int(retry.get("observation") or 0)) + 1
        retry["observation"] = observation
        self._session_checkpoint_state[_COMPACTION_RETRY_STATE_KEY] = retry
        return retry, observation

    @staticmethod
    def _compaction_retry_deferred(
        retry: dict[str, Any],
        signature: dict[str, Any],
        *,
        observation: int,
    ) -> bool:
        if max(0, int(retry.get("failure_count") or 0)) <= 0:
            return False
        if observation >= max(0, int(retry.get("retry_after_observation") or 0)):
            return False
        # More input cannot cure a provider output-limit response.  For other
        # transient/format failures, a materially different closed prefix may
        # justify an early retry; ordinary one-turn growth may not.
        if str(retry.get("error_kind") or "") == _COMPACTION_OUTPUT_LIMIT_ERROR:
            return True
        if str(signature.get("fingerprint") or "") == str(
            retry.get("failed_fingerprint") or ""
        ):
            return True
        prior_tokens = max(0, int(retry.get("failed_compacted_tokens") or 0))
        prior_messages = max(0, int(retry.get("failed_compacted_messages") or 0))
        token_growth = max(
            0,
            int(signature.get("compacted_prefix_tokens_estimated") or 0)
            - prior_tokens,
        )
        message_growth = max(
            0,
            int(signature.get("compacted_message_count") or 0) - prior_messages,
        )
        significant_tokens = max(
            _COMPACTION_SIGNIFICANT_GROWTH_TOKENS,
            prior_tokens // 4,
        )
        return not (
            token_growth >= significant_tokens
            or message_growth >= _COMPACTION_SIGNIFICANT_GROWTH_MESSAGES
        )

    def _record_compaction_failure(
        self,
        retry: dict[str, Any],
        signature: dict[str, Any],
        exc: SessionCompactionError,
        *,
        observation: int,
    ) -> None:
        failures = max(0, int(retry.get("failure_count") or 0)) + 1
        error_kind = exc.__class__.__name__
        base = (
            _COMPACTION_OUTPUT_LIMIT_BASE_CALLS
            if error_kind == _COMPACTION_OUTPUT_LIMIT_ERROR
            else _COMPACTION_RETRY_BASE_CALLS
        )
        delay = min(_COMPACTION_RETRY_MAX_CALLS, base * (2 ** min(4, failures - 1)))
        retry.update(
            {
                "schema_version": _COMPACTION_RETRY_SCHEMA,
                "failure_count": failures,
                "error_kind": error_kind,
                "failed_fingerprint": str(signature.get("fingerprint") or ""),
                "failed_compacted_tokens": max(
                    0,
                    int(signature.get("compacted_prefix_tokens_estimated") or 0),
                ),
                "failed_compacted_messages": max(
                    0,
                    int(signature.get("compacted_message_count") or 0),
                ),
                "retry_after_observation": observation + delay,
            }
        )
        self._session_checkpoint_state[_COMPACTION_RETRY_STATE_KEY] = retry

    def _clear_compaction_retry(self) -> None:
        self._session_checkpoint_state.pop(_COMPACTION_RETRY_STATE_KEY, None)

    def _compaction_signature_or_lossless(
        self,
        request: ModelRequest[Any],
        common: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, ModelRequest[Any] | None]:
        try:
            signature = session_compaction_input_signature(
                request.messages,
                checkpoint=self._checkpoint(),
                request_id=self._request_id,
                retain_recent_tokens=common["retain_recent_tokens"],
            )
        except SessionCompactionNotApplicableError as exc:
            if common["prompt_tokens"] > common["input_capacity"]:
                raise SessionCompactionError(
                    "lossless prompt exceeds input capacity and has no safely compactable prefix"
                ) from exc
            return None, self._finish_common(
                request,
                common,
                common["visible"],
                {
                    "compaction_triggered": False,
                    "compaction_not_applicable": True,
                    "compaction_trigger_tokens": common["trigger_tokens"],
                    "compaction_trigger_ratio": common["trigger_ratio"],
                },
            )
        return signature, None

    def _visible_messages(self, request: ModelRequest[Any]) -> tuple[Any, ...]:
        return active_messages_with_checkpoint(
            request.messages,
            checkpoint=self._checkpoint(),
            request_id=self._request_id,
        )

    def _finalize(
        self,
        request: ModelRequest[Any],
        *,
        messages: list[Any],
        system_message: SystemMessage | None,
        total_budget: int,
        reserved_output_tokens: int,
        input_capacity: int,
        message_capacity: int,
        system_tokens: int,
        tool_tokens: int,
        extra_stats: dict[str, Any] | None = None,
    ) -> ModelRequest[Any]:
        composition = compose_model_messages(
            messages,
            message_token_budget=total_budget,
        )
        self._call_index += 1
        stats = {
            **composition.stats,
            "call_index": self._call_index,
            "prompt_token_budget": total_budget,
            "declared_context_window_tokens": total_budget,
            "reserved_output_tokens": reserved_output_tokens,
            "input_capacity_tokens": input_capacity,
            "message_token_budget_available": message_capacity,
            "system_tokens_estimated": system_tokens,
            "tool_tokens_estimated": tool_tokens,
            "prompt_tokens_estimated": (
                system_tokens
                + tool_tokens
                + int(composition.stats.get("composed_tokens_estimated") or 0)
            ),
            "resource_access_resource_count": len(
                self._resource_access_ledger.get("resources") or {}
            ),
            **(extra_stats or {}),
        }
        self.last_composition_stats = stats
        emit_worker_checkpoint(
            "native_context_composed",
            request_id=self._request_id,
            **stats,
        )
        return request.override(
            messages=list(messages),
            system_message=system_message,
        )

    def _common(self, request: ModelRequest[Any]) -> dict[str, Any]:
        (
            total_budget,
            reserved_output_tokens,
            input_capacity,
            message_capacity,
            system_message,
        ) = self._budget_state(request)
        system_content = (
            getattr(system_message, "content", "")
            if system_message is not None
            else ""
        )
        system_tokens = estimate_tokens(system_content)
        tool_tokens = self._cached_tool_tokens
        visible = list(self._visible_messages(request))
        visible_tokens = compose_model_messages(visible).stats[
            "composed_tokens_estimated"
        ]
        prompt_tokens = system_tokens + tool_tokens + int(visible_tokens)
        # Compaction itself must still receive the closed prefix plus its
        # strict checkpoint contract.  Leave deterministic capacity headroom
        # for that call instead of waiting until the provider input window is
        # nearly exhausted.  This budget is independent of, and must not
        # weaken, the main stream's liveness timeouts.
        trigger_ratio = _ratio_env("EVO_CONTEXT_COMPACTION_TRIGGER_RATIO", 0.65)
        post_ratio = _ratio_env("EVO_CONTEXT_POST_COMPACTION_RATIO", 0.35)
        return {
            "total_budget": total_budget,
            "reserved_output_tokens": reserved_output_tokens,
            "input_capacity": input_capacity,
            "message_capacity": message_capacity,
            "system_message": system_message,
            "system_tokens": system_tokens,
            "tool_tokens": tool_tokens,
            "visible": visible,
            "prompt_tokens": prompt_tokens,
            "trigger_tokens": int(input_capacity * trigger_ratio),
            "retain_recent_tokens": max(
                1,
                int(input_capacity * post_ratio) - system_tokens - tool_tokens,
            ),
            "trigger_ratio": trigger_ratio,
            "post_ratio": post_ratio,
        }

    def _finish_common(
        self,
        request: ModelRequest[Any],
        common: dict[str, Any],
        messages: list[Any],
        extra_stats: dict[str, Any],
    ) -> ModelRequest[Any]:
        return self._finalize(
            request,
            messages=messages,
            system_message=common["system_message"],
            total_budget=common["total_budget"],
            reserved_output_tokens=common["reserved_output_tokens"],
            input_capacity=common["input_capacity"],
            message_capacity=common["message_capacity"],
            system_tokens=common["system_tokens"],
            tool_tokens=common["tool_tokens"],
            extra_stats=extra_stats,
        )

    def _compose(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        common = self._common(request)
        if common["prompt_tokens"] <= common["trigger_tokens"]:
            return self._finish_common(
                request,
                common,
                common["visible"],
                {
                    "compaction_triggered": False,
                    "compaction_trigger_tokens": common["trigger_tokens"],
                    "compaction_trigger_ratio": common["trigger_ratio"],
                },
            )
        signature, lossless = self._compaction_signature_or_lossless(request, common)
        if lossless is not None:
            return lossless
        assert signature is not None
        retry, observation = self._advance_compaction_observation()
        retry_deferred = self._compaction_retry_deferred(
            retry,
            signature,
            observation=observation,
        )
        if retry_deferred and common["prompt_tokens"] <= common["input_capacity"]:
            extra_stats = {
                "compaction_triggered": False,
                "compaction_retry_deferred": True,
                "compaction_failure_count": int(retry.get("failure_count") or 0),
                "compaction_retry_after_observation": int(
                    retry.get("retry_after_observation") or 0
                ),
                "compaction_trigger_tokens": common["trigger_tokens"],
            }
            return self._finish_common(
                request,
                common,
                common["visible"],
                extra_stats,
            )
        compactor = SessionCompactor(
            self._summary_model or request.model,
            request_id=self._request_id,
            protocol=self._session_compaction_protocol,
        )
        try:
            result = compactor.compact(
                request.messages,
                previous_checkpoint=self._checkpoint(),
                retain_recent_tokens=common["retain_recent_tokens"],
            )
        except SessionCompactionError as exc:
            self._record_compaction_failure(
                retry,
                signature,
                exc,
                observation=observation,
            )
            if common["prompt_tokens"] <= common["input_capacity"]:
                return self._finish_common(
                    request,
                    common,
                    common["visible"],
                    {
                        "compaction_triggered": False,
                        "compaction_attempt_failed": True,
                        "compaction_error": str(exc),
                        "compaction_trigger_tokens": common["trigger_tokens"],
                    },
                )
            raise
        result_prompt_tokens = (
            common["system_tokens"]
            + common["tool_tokens"]
            + compose_model_messages(result.messages).stats["composed_tokens_estimated"]
        )
        if result_prompt_tokens > common["input_capacity"]:
            raise SessionCompactionError(
                "validated session checkpoint plus retained history still exceeds the real input capacity"
            )
        self._session_checkpoint_state.clear()
        self._session_checkpoint_state.update(result.checkpoint or {})
        self._clear_compaction_retry()
        return self._finish_common(
            request,
            common,
            list(result.messages),
            {
                **result.stats,
                "compaction_trigger_tokens": common["trigger_tokens"],
                "compaction_trigger_ratio": common["trigger_ratio"],
                "post_compaction_ratio": common["post_ratio"],
            },
        )

    async def _acompose(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        common = self._common(request)
        if common["prompt_tokens"] <= common["trigger_tokens"]:
            return self._finish_common(
                request,
                common,
                common["visible"],
                {
                    "compaction_triggered": False,
                    "compaction_trigger_tokens": common["trigger_tokens"],
                    "compaction_trigger_ratio": common["trigger_ratio"],
                },
            )
        signature, lossless = self._compaction_signature_or_lossless(request, common)
        if lossless is not None:
            return lossless
        assert signature is not None
        retry, observation = self._advance_compaction_observation()
        retry_deferred = self._compaction_retry_deferred(
            retry,
            signature,
            observation=observation,
        )
        if retry_deferred and common["prompt_tokens"] <= common["input_capacity"]:
            extra_stats = {
                "compaction_triggered": False,
                "compaction_retry_deferred": True,
                "compaction_failure_count": int(retry.get("failure_count") or 0),
                "compaction_retry_after_observation": int(
                    retry.get("retry_after_observation") or 0
                ),
                "compaction_trigger_tokens": common["trigger_tokens"],
            }
            return self._finish_common(
                request,
                common,
                common["visible"],
                extra_stats,
            )
        compactor = SessionCompactor(
            self._summary_model or request.model,
            request_id=self._request_id,
            protocol=self._session_compaction_protocol,
        )
        try:
            result = await compactor.acompact(
                request.messages,
                previous_checkpoint=self._checkpoint(),
                retain_recent_tokens=common["retain_recent_tokens"],
            )
        except SessionCompactionError as exc:
            self._record_compaction_failure(
                retry,
                signature,
                exc,
                observation=observation,
            )
            if common["prompt_tokens"] <= common["input_capacity"]:
                return self._finish_common(
                    request,
                    common,
                    common["visible"],
                    {
                        "compaction_triggered": False,
                        "compaction_attempt_failed": True,
                        "compaction_error": str(exc),
                        "compaction_trigger_tokens": common["trigger_tokens"],
                    },
                )
            raise
        result_prompt_tokens = (
            common["system_tokens"]
            + common["tool_tokens"]
            + compose_model_messages(result.messages).stats["composed_tokens_estimated"]
        )
        if result_prompt_tokens > common["input_capacity"]:
            raise SessionCompactionError(
                "validated session checkpoint plus retained history still exceeds the real input capacity"
            )
        self._session_checkpoint_state.clear()
        self._session_checkpoint_state.update(result.checkpoint or {})
        self._clear_compaction_retry()
        return self._finish_common(
            request,
            common,
            list(result.messages),
            {
                **result.stats,
                "compaction_trigger_tokens": common["trigger_tokens"],
                "compaction_trigger_ratio": common["trigger_ratio"],
                "post_compaction_ratio": common["post_ratio"],
            },
        )

    def wrap_model_call(self, request, handler):
        return handler(self._compose(request))

    async def awrap_model_call(self, request, handler):
        return await handler(await self._acompose(request))
