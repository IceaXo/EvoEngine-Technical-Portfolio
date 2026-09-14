from __future__ import annotations

import asyncio
import json

import pytest
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from src.agents import context_composer_middleware, native_agent_graph
from src.agents.context_composer_middleware import ContextComposerMiddleware
from src.config.settings import Settings
from src.schemas.api import AgentInvokeRequest
from src.services.context_composer import (
    MINIMUM_CONTEXT_WINDOW_TOKENS,
    compose_model_messages,
)
from src.services.orchestrator import compose_user_prompt, history_protocol_messages
from src.services.session_compaction import (
    REQUIRED_SECTIONS,
    SESSION_COMPACT_SCHEMA,
    JsonTextSessionCompactionTestProtocol,
    SessionCompactionError,
    SessionCompactionOutputLimitError,
    SessionCompactionProtocolResult,
    SessionCompactor,
    active_messages_with_checkpoint,
    checkpoint_model_message,
    plan_session_compaction,
)


def _closed_protocol_ledger(messages):
    ledger = native_agent_graph.CanonicalProtocolLedger()
    ledger.seed_if_empty(list(messages))
    assert ledger.is_closed()
    return ledger


def _tool_exchange(
    *,
    call_id: str,
    tool_name: str,
    args: dict,
    result: dict,
) -> list[object]:
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": call_id,
                    "name": tool_name,
                    "args": args,
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=json.dumps(result, ensure_ascii=False),
            tool_call_id=call_id,
            name=tool_name,
        ),
    ]


def _sections(marker: str = "") -> dict:
    return {
        "session_intent": ["EARLY_RESEARCH_GOAL", marker],
        "user_constraints": ["DO_NOT_REPEAT_SIDE_EFFECT", "KEEP_CITATIONS"],
        "current_task_state": ["phase=executing"],
        "completed_work_and_evidence": [],
        "key_decisions_and_rationale": ["use stable resources: lossless recovery"],
        "rejected_options": ["blind character truncation"],
        "resource_map": [],
        "citation_bindings": [],
        "side_effect_receipts": [],
        "open_questions_hitl": [],
        "remaining_work": ["continue"],
        "next_safe_action": "continue from retained messages",
    }


class _SummaryModel:
    def __init__(self) -> None:
        self.calls = 0

    def _response(self, prompt) -> AIMessage:
        self.calls += 1
        body = json.loads(str(prompt[-1].content))
        previous = body.get("previous_checkpoint") or {}
        marker = f"generation-{int(previous.get('generation') or 0) + 1}"
        sections = _sections(marker)
        if previous:
            sections["session_intent"] = list(previous["session_intent"]) + [marker]
            sections["user_constraints"] = list(previous["user_constraints"])
            sections["key_decisions_and_rationale"] = list(
                previous["key_decisions_and_rationale"]
            )
        return AIMessage(content=json.dumps(sections, ensure_ascii=False))

    def invoke(self, prompt, config=None):
        return self._response(prompt)

    async def ainvoke(self, prompt, config=None):
        return self._response(prompt)


class _InvalidSummaryModel:
    def invoke(self, prompt, config=None):
        return AIMessage(content="not-json")

    async def ainvoke(self, prompt, config=None):
        return AIMessage(content="not-json")


class _OutputLimitThenSuccessProtocol:
    def __init__(self, *, failures: int = 1) -> None:
        self.calls = 0
        self.failures = failures

    def invoke(self, model, prompt):
        self.calls += 1
        if self.calls <= self.failures:
            raise SessionCompactionOutputLimitError(
                "session compactor reached the provider output limit"
            )
        return SessionCompactionProtocolResult(
            sections=_sections("recovered"),
            stats={"compaction_protocol": "test_output_limit_then_success"},
        )

    async def ainvoke(self, model, prompt):
        return self.invoke(model, prompt)


def _model_request(messages, *, model=None) -> ModelRequest:
    return ModelRequest(
        model=model or object(),
        messages=list(messages),
        system_message=SystemMessage(content="System contract."),
        tools=[],
        state={"messages": list(messages)},
        runtime=None,
    )


def test_default_context_window_keeps_large_history_without_projection() -> None:
    payload = {
        "ok": True,
        "tool": "conversation_file_read",
        "complete": True,
        "has_more": False,
        "cursor": None,
        "content_text": "A" * 600_000 + "TAIL_FACT_MUST_REMAIN_INLINE",
    }
    messages = [
        HumanMessage(content="Read the complete bounded result."),
        *_tool_exchange(
            call_id="call-large",
            tool_name="conversation_file_read",
            args={"file_id": 42, "offset": 0},
            result=payload,
        ),
    ]

    composed = compose_model_messages(messages)

    assert MINIMUM_CONTEXT_WINDOW_TOKENS == 200_000
    assert composed.stats["message_token_budget"] >= 200_000
    assert 150_000 <= composed.stats["composed_tokens_estimated"] < 180_000
    assert tuple(messages) == composed.messages
    assert composed.stats["composed_message_count"] == len(messages)
    assert composed.stats["composed_chars"] == composed.stats["original_chars"]
    assert "TAIL_FACT_MUST_REMAIN_INLINE" in str(composed.messages[-1].content)


def test_tiny_observational_budget_cannot_reactivate_projection_or_deletion() -> None:
    messages = [
        HumanMessage(content="Keep everything."),
        *_tool_exchange(
            call_id="call-old-policy",
            tool_name="reader",
            args={},
            result={"head": "HEAD", "body": "X" * 200_000, "tail": "TAIL"},
        ),
    ]

    composed = compose_model_messages(
        messages,
        message_token_budget=2_000,
    )

    assert composed.stats["message_token_budget"] == 200_000
    assert composed.messages == tuple(messages)
    assert "HEAD" in str(composed.messages[-1].content)
    assert "TAIL" in str(composed.messages[-1].content)


def test_compaction_keeps_ai_tool_protocol_group_atomic() -> None:
    messages = [
        HumanMessage(content="old goal"),
        *_tool_exchange(
            call_id="call-closed",
            tool_name="reader",
            args={},
            result={"ok": True},
        ),
        HumanMessage(content="current goal"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "call-open", "name": "writer", "args": {}, "type": "tool_call"}
            ],
        ),
    ]

    plan = plan_session_compaction(
        messages,
        checkpoint=None,
        request_id="req-atomic",
        retain_recent_tokens=1,
    )

    assert messages[-1] in plan.retained_messages
    assert all(message not in plan.compacted_prefix for message in messages[-2:])


def test_three_cumulative_compactions_preserve_goal_receipts_and_resources() -> None:
    model = _SummaryModel()
    compactor = SessionCompactor(
        model,
        request_id="req-three-compacts",
        protocol=JsonTextSessionCompactionTestProtocol(),
    )
    messages: list = [HumanMessage(content="EARLY_RESEARCH_GOAL")]
    checkpoint = None
    resources = []
    for generation in range(1, 4):
        resource_id = f"resource:conversation-file:{700 + generation}"
        resources.append(resource_id)
        messages.extend(
            _tool_exchange(
                call_id=f"call-{generation}",
                tool_name="save_research_result",
                args={"generation": generation},
                result={
                    "ok": True,
                    "complete": True,
                    "has_more": False,
                    "resource_id": resource_id,
                    "resources": [
                        {
                            "resource_id": resource_id,
                            "uri": f"conversation-file://{700 + generation}",
                        }
                    ],
                    "citation_binding": {
                        "citation_key": f"source:{generation}",
                        "claim_id": f"claim:{generation}",
                    },
                    "side_effect_receipt": {"idempotency_key": f"write-{generation}"},
                },
            )
        )
        messages.append(HumanMessage(content=f"continue generation {generation}"))
        result = compactor.compact(
            messages,
            previous_checkpoint=checkpoint,
            retain_recent_tokens=1,
        )
        checkpoint = result.checkpoint

    assert model.calls == 3
    assert checkpoint["schema_version"] == SESSION_COMPACT_SCHEMA
    assert checkpoint["generation"] == 3
    assert "EARLY_RESEARCH_GOAL" in checkpoint["session_intent"]
    assert checkpoint["user_constraints"] == [
        "DO_NOT_REPEAT_SIDE_EFFECT",
        "KEEP_CITATIONS",
    ]
    mechanical = checkpoint["mechanical_state"]
    assert {row["resource_id"] for row in mechanical["resource_refs"]} == set(resources)
    assert len(mechanical["side_effect_receipts"]) == 3
    assert len(mechanical["citation_bindings"]) >= 3
    visible = active_messages_with_checkpoint(
        messages,
        checkpoint=checkpoint,
        request_id="req-three-compacts",
    )
    assert str(visible[0].additional_kwargs.get("checkpoint_id")) == checkpoint[
        "checkpoint_id"
    ]
    assert len(visible) < len(messages)


def test_checkpoint_validation_failure_preserves_original_below_real_window(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EVO_CONTEXT_COMPACTION_TRIGGER_RATIO", "0.001")
    monkeypatch.setenv("EVO_CONTEXT_POST_COMPACTION_RATIO", "0.001")
    messages = [
        HumanMessage(content="Do not lose this goal."),
        *_tool_exchange(
            call_id="call-failure",
            tool_name="reader",
            args={},
            result={"ok": True, "content": "X" * 20_000},
        ),
        HumanMessage(content="Continue."),
    ]
    checkpoint_state: dict = {}
    middleware = ContextComposerMiddleware(
        request_id="req-compact-failure",
        summary_model=_InvalidSummaryModel(),
        session_checkpoint_state=checkpoint_state,
        session_compaction_protocol=JsonTextSessionCompactionTestProtocol(),
    )

    result = middleware.wrap_model_call(_model_request(messages), lambda value: value)

    assert result.messages == messages
    assert checkpoint_state.get("schema_version") is None
    assert checkpoint_state["_runtime_compaction_retry_v1"]["failure_count"] == 1
    assert middleware._checkpoint() is None
    assert middleware.last_composition_stats["compaction_attempt_failed"] is True
    assert middleware.last_composition_stats["compaction_triggered"] is False


def test_compaction_failure_same_candidate_is_deferred_losslessly(monkeypatch) -> None:
    monkeypatch.setenv("EVO_CONTEXT_COMPACTION_TRIGGER_RATIO", "0.001")
    monkeypatch.setenv("EVO_CONTEXT_POST_COMPACTION_RATIO", "0.001")
    messages = [HumanMessage(content="A" * 4_000), HumanMessage(content="B" * 4_000)]
    checkpoint_state: dict = {}
    protocol = _OutputLimitThenSuccessProtocol(failures=99)
    middleware = ContextComposerMiddleware(
        request_id="req-compaction-backoff-same",
        summary_model=object(),
        session_checkpoint_state=checkpoint_state,
        session_compaction_protocol=protocol,
    )

    first = middleware.wrap_model_call(_model_request(messages), lambda value: value)
    second = middleware.wrap_model_call(_model_request(messages), lambda value: value)

    assert first.messages == messages
    assert second.messages == messages
    assert protocol.calls == 1
    assert middleware.last_composition_stats["compaction_retry_deferred"] is True


def test_output_limit_backoff_survives_changed_candidate(monkeypatch) -> None:
    monkeypatch.setenv("EVO_CONTEXT_COMPACTION_TRIGGER_RATIO", "0.001")
    monkeypatch.setenv("EVO_CONTEXT_POST_COMPACTION_RATIO", "0.001")
    messages = [HumanMessage(content="A" * 4_000), HumanMessage(content="B" * 4_000)]
    changed_messages = [*messages, HumanMessage(content="C" * 4_000)]
    protocol = _OutputLimitThenSuccessProtocol(failures=99)
    middleware = ContextComposerMiddleware(
        request_id="req-compaction-backoff-growth",
        summary_model=object(),
        session_checkpoint_state={},
        session_compaction_protocol=protocol,
    )

    middleware.wrap_model_call(_model_request(messages), lambda value: value)
    result = middleware.wrap_model_call(
        _model_request(changed_messages), lambda value: value
    )

    assert result.messages == changed_messages
    assert protocol.calls == 1
    assert middleware.last_composition_stats["compaction_retry_deferred"] is True


def test_compaction_backoff_is_overridden_above_input_capacity(monkeypatch) -> None:
    monkeypatch.setenv("EVO_CONTEXT_COMPACTION_TRIGGER_RATIO", "0.001")
    monkeypatch.setenv("EVO_CONTEXT_POST_COMPACTION_RATIO", "0.001")
    initial_messages = [
        HumanMessage(content="A" * 4_000),
        HumanMessage(content="B" * 4_000),
    ]
    oversized_messages = [
        *initial_messages,
        *(HumanMessage(content=f"{index}:" + "X" * 4_000) for index in range(400)),
    ]

    sync_protocol = _OutputLimitThenSuccessProtocol(failures=1)
    sync_middleware = ContextComposerMiddleware(
        request_id="req-capacity-forces-sync-retry",
        summary_model=object(),
        session_checkpoint_state={},
        session_compaction_protocol=sync_protocol,
    )
    sync_middleware.wrap_model_call(
        _model_request(initial_messages), lambda value: value
    )
    sync_result = sync_middleware.wrap_model_call(
        _model_request(oversized_messages), lambda value: value
    )

    assert sync_protocol.calls == 2
    assert sync_result.messages != oversized_messages
    assert sync_middleware.last_composition_stats["compaction_triggered"] is True

    async_protocol = _OutputLimitThenSuccessProtocol(failures=1)
    async_middleware = ContextComposerMiddleware(
        request_id="req-capacity-forces-async-retry",
        summary_model=object(),
        session_checkpoint_state={},
        session_compaction_protocol=async_protocol,
    )

    async def run_async():
        async def handler(value):
            return value

        await async_middleware.awrap_model_call(
            _model_request(initial_messages), handler
        )
        return await async_middleware.awrap_model_call(
            _model_request(oversized_messages), handler
        )

    async_result = asyncio.run(run_async())

    assert async_protocol.calls == 2
    assert async_result.messages != oversized_messages
    assert async_middleware.last_composition_stats["compaction_triggered"] is True


def test_semantic_checkpoint_hides_runtime_retry_state() -> None:
    retry_key = "_runtime_compaction_retry_v1"
    checkpoint_state = {
        "schema_version": SESSION_COMPACT_SCHEMA,
        "checkpoint_id": "session_checkpoint_runtime-private",
        retry_key: {
            "schema_version": "evoengine.session-compaction-retry/v1",
            "failure_count": 2,
        },
        **_sections(),
    }
    middleware = ContextComposerMiddleware(
        request_id="req-runtime-private",
        session_checkpoint_state=checkpoint_state,
    )

    semantic_checkpoint = middleware._checkpoint()

    assert semantic_checkpoint is not None
    assert retry_key in checkpoint_state
    assert retry_key not in semantic_checkpoint
    assert retry_key not in str(checkpoint_model_message(semantic_checkpoint).content)


def test_successful_compaction_clears_failure_backoff(monkeypatch) -> None:
    monkeypatch.setenv("EVO_CONTEXT_COMPACTION_TRIGGER_RATIO", "0.001")
    monkeypatch.setenv("EVO_CONTEXT_POST_COMPACTION_RATIO", "0.001")
    messages = [HumanMessage(content="A" * 4_000), HumanMessage(content="B" * 4_000)]
    checkpoint_state: dict = {}
    protocol = _OutputLimitThenSuccessProtocol(failures=1)
    middleware = ContextComposerMiddleware(
        request_id="req-compaction-backoff-clear",
        summary_model=object(),
        session_checkpoint_state=checkpoint_state,
        session_compaction_protocol=protocol,
    )

    middleware.wrap_model_call(_model_request(messages), lambda value: value)
    checkpoint_state["_runtime_compaction_retry_v1"][
        "retry_after_observation"
    ] = 2
    result = middleware.wrap_model_call(_model_request(messages), lambda value: value)

    assert protocol.calls == 2
    assert result.messages != messages
    assert checkpoint_state["schema_version"] == SESSION_COMPACT_SCHEMA
    assert "_runtime_compaction_retry_v1" not in checkpoint_state


def test_no_safe_compaction_prefix_passes_through_below_capacity(monkeypatch) -> None:
    monkeypatch.setenv("EVO_CONTEXT_COMPACTION_TRIGGER_RATIO", "0.001")
    messages = [HumanMessage(content="one indivisible active request " + "X" * 20_000)]
    checkpoint_state: dict = {}
    middleware = ContextComposerMiddleware(
        request_id="req-no-safe-prefix",
        summary_model=object(),
        session_checkpoint_state=checkpoint_state,
    )

    result = middleware.wrap_model_call(_model_request(messages), lambda value: value)

    assert result.messages == messages
    assert checkpoint_state == {}
    assert middleware.last_composition_stats["compaction_not_applicable"] is True
    assert middleware.last_composition_stats["compaction_triggered"] is False


def test_no_safe_compaction_prefix_fails_only_above_capacity(monkeypatch) -> None:
    monkeypatch.setenv("EVO_CONTEXT_COMPACTION_TRIGGER_RATIO", "0.001")
    messages = [HumanMessage(content="X" * 800_000)]
    middleware = ContextComposerMiddleware(
        request_id="req-no-safe-prefix-over-capacity",
        summary_model=object(),
        session_checkpoint_state={},
    )

    with pytest.raises(SessionCompactionError, match="no safely compactable prefix"):
        middleware.wrap_model_call(_model_request(messages), lambda value: value)


def test_default_compaction_trigger_keeps_headroom_for_checkpoint_model() -> None:
    middleware = ContextComposerMiddleware(request_id="req-default-trigger")
    common = middleware._common(_model_request([HumanMessage(content="continue")]))

    assert common["trigger_ratio"] == 0.65
    assert common["trigger_tokens"] < common["input_capacity"] - 60_000


def test_compactor_rejects_invalid_checkpoint_without_returning_partial_view() -> None:
    compactor = SessionCompactor(
        _InvalidSummaryModel(),
        request_id="req-invalid",
        protocol=JsonTextSessionCompactionTestProtocol(),
    )
    messages = [
        HumanMessage(content="goal"),
        HumanMessage(content="middle"),
        HumanMessage(content="latest"),
    ]

    with pytest.raises(SessionCompactionError):
        compactor.compact(messages, previous_checkpoint=None, retain_recent_tokens=1)

    assert [message.content for message in messages] == ["goal", "middle", "latest"]


def test_context_composer_middleware_applies_losslessly_sync_and_async() -> None:
    messages = [
        HumanMessage(content="Continue."),
        *_tool_exchange(
            call_id="call-lossless",
            tool_name="large_reader",
            args={"resource_id": "resource:conversation-file:999"},
            result={"ok": True, "result": {"body": "X" * 600_000}},
        ),
    ]
    request = _model_request(messages)
    middleware = ContextComposerMiddleware(
        request_id="req-context-composer",
        prompt_token_budget=12_000,
    )

    sync_result = middleware.wrap_model_call(request, lambda value: value)
    assert sync_result.messages == messages
    assert middleware.last_composition_stats["prompt_token_budget"] == 200_000
    assert middleware.last_composition_stats["compaction_triggered"] is False
    assert 150_000 <= middleware.last_composition_stats["prompt_tokens_estimated"] < 165_000

    async def async_handler(value):
        return value

    async_result = asyncio.run(middleware.awrap_model_call(request, async_handler))
    assert async_result.messages == messages


def test_context_composer_middleware_exposes_one_resource_access_ledger() -> None:
    messages = [
        HumanMessage(content="Continue from the unread position."),
        *_tool_exchange(
            call_id="call-ledger-read-1",
            tool_name="resource_read",
            args={"resource_id": "resource:conversation-file:1902"},
            result={
                "ok": True,
                "resource_id": "resource:conversation-file:1902",
                "content_text": "A" * 6_000,
                "offset": 0,
                "returned_chars": 6_000,
                "total_chars": 10_000,
                "complete": False,
                "has_more": True,
                "cursor": "conversation-file://1902?offset=6000",
            },
        ),
        *_tool_exchange(
            call_id="call-ledger-read-2",
            tool_name="resource_read",
            args={
                "resource_id": "resource:conversation-file:1902",
                "cursor": "conversation-file://1902?offset=6000",
            },
            result={
                "ok": True,
                "resource_id": "resource:conversation-file:1902",
                "content_text": "B" * 4_000,
                "offset": 6_000,
                "returned_chars": 4_000,
                "total_chars": 10_000,
                "complete": True,
                "has_more": False,
                "cursor": None,
            },
        ),
    ]
    request = _model_request(messages)
    shared_ledger: dict = {}
    middleware = ContextComposerMiddleware(
        request_id="req-resource-ledger",
        resource_access_ledger=shared_ledger,
    )

    composed_request = middleware.wrap_model_call(request, lambda value: value)
    system_text = str(composed_request.system_message.content)
    ledger_text = system_text.split("[RESOURCE_ACCESS_LEDGER]", 1)[1].split(
        "[/RESOURCE_ACCESS_LEDGER]", 1
    )[0]
    payload = json.loads(ledger_text)

    assert system_text.count("[RESOURCE_ACCESS_LEDGER]") == 1
    assert payload["resources"][0]["read_ranges"] == [[0, 10_000]]
    assert payload["resources"][0]["fully_read"] is True
    assert middleware.last_composition_stats["resource_access_resource_count"] == 1


def test_context_composer_default_window_is_at_least_200k(monkeypatch) -> None:
    assert context_composer_middleware._prompt_token_budget() >= 200_000


def test_history_rows_become_real_messages_with_turn_identity() -> None:
    messages = history_protocol_messages(
        [
            {"turn_id": "turn_41", "role": "user", "content": "question"},
            {"turn_id": "turn_41", "role": "assistant", "content": "answer"},
        ]
    )

    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].additional_kwargs["turn_id"] == "turn_41"
    assert messages[1].content == "answer"


def test_hitl_turn_checkpoint_carries_and_restores_session_checkpoint() -> None:
    session_checkpoint = {
        "schema_version": SESSION_COMPACT_SCHEMA,
        "checkpoint_id": "session_checkpoint_hitl",
        "generation": 2,
        "covered_through_turn_id": 12,
        **_sections(),
    }
    state = native_agent_graph._request_to_native_state(
        AgentInvokeRequest(
            text="goal",
            request_id="req-hitl-session-checkpoint",
            session_compact_checkpoint=session_checkpoint,
        )
    )
    turn_checkpoint = native_agent_graph._build_turn_checkpoint(
        state=state,
        protocol_ledger=_closed_protocol_ledger(
            [HumanMessage(content="goal")]
        ),
        pending_bundle={"bundle_id": "bundle-1", "questions": []},
        pending_sandbox_jobs=[],
    )
    restored = native_agent_graph._restore_runtime_from_turn_checkpoint(turn_checkpoint)

    assert turn_checkpoint["session_compact_checkpoint"]["checkpoint_id"] == (
        "session_checkpoint_hitl"
    )
    assert restored["session_compact_checkpoint"] == session_checkpoint


def test_hitl_turn_checkpoint_carries_and_restores_reference_context() -> None:
    reference_context = {
        "schema_version": "evoengine.reference-context/v1",
        "total": 1,
        "references": [
            {
                "schema_version": "evoengine.resolved-reference/v1",
                "reference_id": "project_asset:knowledge:41",
                "kind": "project_asset",
                "source_id": "41",
                "asset_scope": "knowledge",
                "display_label": "paper.pdf",
                "resource": {
                    "resource_id": "resource:project-asset:41",
                    "uri": "project-asset://41",
                    "kind": "project_asset",
                    "storage": "project_asset",
                    "media_type": "application/pdf",
                    "size_bytes": 2048,
                    "sha256": "a" * 64,
                },
                "document_ref": {"schema_version": "evoengine.document-ref/v1"},
            }
        ],
        "manifest": {
            "resource_id": "resource:reference-manifest:1",
            "uri": "reference-manifest://1",
            "kind": "reference_manifest",
            "storage": "reference_manifest",
            "media_type": "application/json",
            "size_bytes": 1024,
            "sha256": "b" * 64,
        },
    }
    state = native_agent_graph._request_to_native_state(
        AgentInvokeRequest(text="read it", reference_context=reference_context)
    )
    original_prompt = compose_user_prompt(
        "read it",
        reference_context=reference_context,
    )
    checkpoint = native_agent_graph._build_turn_checkpoint(
        state=state,
        protocol_ledger=_closed_protocol_ledger(
            [HumanMessage(content=original_prompt)]
        ),
        pending_bundle={"bundle_id": "bundle-reference", "questions": []},
        pending_sandbox_jobs=[],
    )
    restored = native_agent_graph._request_to_native_state(
        AgentInvokeRequest(
            text="selected answer",
            request_id="req-reference-child",
            hitl_parent_request_id="req-parent",
            task_authority={
                "schema_version": "evoengine.task-authority-control/v1",
                "transport_request_id": "req-reference-child",
                "task_authority_request_id": "req-reference-child",
                "task_run_id": None,
                "continuation_parent_request_id": "req-parent",
            },
            turn_checkpoint=checkpoint,
        )
    )

    assert checkpoint["reference_context"] == state["reference_context"]
    assert restored["reference_context"] == state["reference_context"]
    answer_reference_context = native_agent_graph._reference_context_for_current_prompt(
        restored,
        hitl_resume_messages=restored["hitl_resume_messages"],
    )
    answer_prompt = compose_user_prompt(
        "selected answer",
        reference_context=answer_reference_context,
    )
    combined_prompt = "\n".join(
        [
            *(str(message.content) for message in restored["hitl_resume_messages"]),
            answer_prompt,
        ]
    )
    assert answer_reference_context is None
    assert combined_prompt.count("[REFERENCE_CONTEXT]") == 1


def test_empty_reference_context_stays_absent_across_checkpoint_restore() -> None:
    state = native_agent_graph._request_to_native_state(
        AgentInvokeRequest(text="load a database capability")
    )
    checkpoint = native_agent_graph._build_turn_checkpoint(
        state=state,
        protocol_ledger=_closed_protocol_ledger(
            [HumanMessage(content="load a database capability")]
        ),
        pending_bundle={"bundle_id": "bundle-empty-reference", "questions": []},
        pending_sandbox_jobs=[],
    )
    restored = native_agent_graph._restore_runtime_from_turn_checkpoint(checkpoint)

    assert checkpoint["reference_context"] is None
    assert restored is not None
    assert restored["reference_context"] is None


def test_tool_message_diagnostics_separate_model_content_and_runtime_artifact() -> None:
    message = ToolMessage(
        content=json.dumps(
            {
                "ok": True,
                "raw_ref": "resource:conversation-file:3088",
                "resources": [
                    {
                        "resource_id": "resource:conversation-file:3088",
                        "uri": "conversation-file://3088",
                    }
                ],
                "result": {"blocks": [{"text": "bounded evidence"}]},
            }
        ),
        artifact={
            "schema_version": "evoengine.tool-runtime-artifact/v1",
            "machine_projection": {
                "resources": [
                    {
                        "resource_id": "resource:conversation-file:3088",
                        "uri": "conversation-file://3088",
                    }
                ],
                "citation_projection": [{"payload": "C" * 20_000}],
            },
        },
        tool_call_id="call-page",
        name="read_asset_page_parsed",
    )

    model_chars, artifact_chars = native_agent_graph._tool_message_projection_sizes(
        message
    )
    assert 0 < model_chars < artifact_chars
    assert native_agent_graph._tool_output_has_durable_resource(message) is True


def test_turn_checkpoint_roundtrips_canonical_tool_control_fields() -> None:
    state = native_agent_graph._request_to_native_state(
        AgentInvokeRequest(text="checkpoint exact protocol")
    )
    ai_message = AIMessage(
        id="ai-exact",
        content="",
        tool_calls=[
            {
                "id": "call-exact",
                "name": "typed_tool",
                "args": {"value": 1},
                "type": "tool_call",
            }
        ],
    )
    tool_message = ToolMessage(
        id="tool-exact",
        content=json.dumps({"ok": False, "status": "failed"}),
        tool_call_id="call-exact",
        name="typed_tool",
        status="error",
        artifact={
            "schema_version": "evoengine.tool-runtime-artifact/v1",
            "control_projection": {"completion_signal": {"accepted": False}},
            "runtime_control": {"kind": "capability_needs_input"},
            "machine_projection": {"large": "x" * 20_000},
        },
    )
    checkpoint = native_agent_graph._build_turn_checkpoint(
        state=state,
        protocol_ledger=_closed_protocol_ledger([ai_message, tool_message]),
        pending_bundle={"bundle_id": "bundle-exact", "questions": []},
        pending_sandbox_jobs=[],
    )

    restored = native_agent_graph._restore_runtime_from_turn_checkpoint(checkpoint)
    assert restored is not None
    restored_ai, restored_tool = restored["checkpoint_protocol_messages"]
    assert restored_ai.id == "ai-exact"
    assert restored_tool.id == "tool-exact"
    assert restored_tool.tool_call_id == "call-exact"
    assert restored_tool.status == "error"
    assert restored_tool.artifact == {
        "schema_version": "evoengine.tool-runtime-artifact/v1",
        "control_projection": {"completion_signal": {"accepted": False}},
        "runtime_control": {"kind": "capability_needs_input"},
    }


def test_native_turn_does_not_precompose_or_mutate_protocol(monkeypatch) -> None:
    class CaptureAgent:
        def __init__(self) -> None:
            self.payloads: list[list[object]] = []

        def astream_events(self, payload, version: str = "v2", **_kwargs):
            self.payloads.append(list(payload.get("messages") or []))

            async def events():
                yield {"event": "on_chat_model_start"}
                yield {
                    "event": "on_chat_model_stream",
                    "data": {"chunk": AIMessageChunk(content="Recovered.")},
                }
                yield {
                    "event": "on_chat_model_end",
                    "data": {"output": AIMessage(content="Recovered.")},
                }

            return events()

    protocol_messages = [
        HumanMessage(content="Complete the task."),
        *_tool_exchange(
            call_id="call-old",
            tool_name="large_reader",
            args={"resource_id": "resource:conversation-file:301"},
            result={
                "ok": True,
                "raw_ref": "resource:conversation-file:301",
                "result": {"body": "RECOVERY_BODY" * 10_000},
            },
        ),
    ]
    agent = CaptureAgent()
    monkeypatch.setattr(native_agent_graph, "build_dynamic_agent", lambda *_a, **_k: agent)
    state = native_agent_graph._request_to_native_state(
        AgentInvokeRequest(text="Continue.", request_id="req-composed-recovery")
    )
    state["continuation_messages"] = list(protocol_messages)
    emitted: list[dict] = []

    result = asyncio.run(
        native_agent_graph.run_native_agent_turn(
            state,
            native_agent_graph.build_runtime_context(
                Settings(
                    deepseek_api_key=None,
                    zhipu_api_key=None,
                    llm_model="deepseek-chat",
                ),
                object(),
                llm_factory=object,
            ),
            event_writer=emitted.append,
        )
    )

    assert result["status"] == "completed"
    assert agent.payloads[0][2].content == protocol_messages[2].content
    usage = next(event for event in emitted if event.get("type") == "context_usage")
    assert usage["context_composition"]["authority"] == "agent_middleware"
    assert usage["context_composition"]["precomposition_applied"] is False
