from __future__ import annotations

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.services.context_composer import compose_model_messages
from src.services.session_compaction import (
    REQUIRED_SECTIONS,
    SESSION_COMPACT_SCHEMA,
    SESSION_COMPACTION_MODEL_TAG,
    JsonTextSessionCompactionTestProtocol,
    SessionCompactionError,
    SessionCompactionFormatError,
    SessionCompactionOutputLimitError,
    SessionCompactionTransportError,
    SessionCompactor,
    active_messages_with_checkpoint,
    attach_completion_authority_to_checkpoint,
    completion_authority_for_request,
    plan_session_compaction,
)


def _exchange(index: int) -> list:
    resource_id = f"resource:conversation-file:{index}"
    call_id = f"call-{index}"
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": call_id,
                    "name": "save_result",
                    "args": {"index": index},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=json.dumps(
                {
                    "ok": True,
                    "complete": True,
                    "has_more": False,
                    "resource_id": resource_id,
                    "citation_binding": {
                        "citation_key": f"source:{index}",
                        "claim_id": f"claim:{index}",
                    },
                    "side_effect_receipt": {"idempotency_key": f"write-{index}"},
                }
            ),
            tool_call_id=call_id,
            name="save_result",
        ),
    ]


def _sections(previous: dict | None = None) -> dict:
    prior_intent = list((previous or {}).get("session_intent") or [])
    return {
        "session_intent": prior_intent or ["EARLY_GOAL"],
        "user_constraints": ["NO_REPLAY"],
        "current_task_state": ["phase=executing"],
        "completed_work_and_evidence": [],
        "key_decisions_and_rationale": ["ResourceRef: lossless"],
        "rejected_options": ["character truncation"],
        "resource_map": [],
        "citation_bindings": [],
        "side_effect_receipts": [],
        "open_questions_hitl": [],
        "remaining_work": ["continue"],
        "next_safe_action": "continue",
    }


class SummaryModel:
    def __init__(self):
        self.calls = 0
        self.prompts: list[list] = []

    def invoke(self, prompt, config=None):
        self.calls += 1
        self.prompts.append(list(prompt))
        body = json.loads(prompt[-1].content)
        return AIMessage(content=json.dumps(_sections(body.get("previous_checkpoint"))))

    async def ainvoke(self, prompt, config=None):
        return self.invoke(prompt, config=config)


class InvalidModel:
    def invoke(self, prompt, config=None):
        return AIMessage(content="invalid")


class _SyncTransport:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _AsyncTransport:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _StrictController:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0
        self.bindings: list[dict] = []
        self.call_configs: list[dict] = []
        self.transports: list[tuple[_SyncTransport, _AsyncTransport]] = []
        self.clone_updates: list[dict] = []


class _StatusError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status={status_code}")
        self.status_code = status_code


class RateLimitError(RuntimeError):
    pass


class _AsyncHang:
    pass


class _StrictBoundModel:
    def __init__(self, model: "_StrictFakeModel") -> None:
        self.model = model

    def invoke(self, prompt, config=None):
        controller = self.model.controller
        controller.call_configs.append(dict(config or {}))
        controller.calls += 1
        outcome = controller.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def ainvoke(self, prompt, config=None):
        controller = self.model.controller
        controller.call_configs.append(dict(config or {}))
        controller.calls += 1
        outcome = controller.outcomes.pop(0)
        if isinstance(outcome, _AsyncHang):
            await asyncio.sleep(60)
            raise AssertionError("async hang should be cancelled by the compaction timeout")
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _StrictFakeModel:
    def __init__(self, controller: _StrictController, **config) -> None:
        self.controller = controller
        self.api_base = config.get("api_base", "https://api.deepseek.com/v1")
        self.extra_body = config.get("extra_body")
        self.root_client = _SyncTransport()
        self.root_async_client = _AsyncTransport()
        controller.transports.append((self.root_client, self.root_async_client))

    def fresh_transport_copy(self, **updates):
        self.controller.clone_updates.append(dict(updates))
        return _StrictFakeModel(self.controller, **updates)

    def bind_tools(
        self,
        tools,
        *,
        tool_choice=None,
        strict=None,
        parallel_tool_calls=None,
        **kwargs,
    ):
        self.controller.bindings.append(
            {
                "tool_choice": tool_choice,
                "strict": strict,
                "parallel_tool_calls": parallel_tool_calls,
                "schema": tools[0].model_json_schema(),
                **kwargs,
            }
        )
        return _StrictBoundModel(self)


def _strict_response(sections: dict) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call-checkpoint",
                "name": "SubmitSessionCheckpoint",
                "args": sections,
                "type": "tool_call",
            }
        ],
    )


def _strict_compaction(outcomes: list[object]):
    controller = _StrictController(outcomes)
    model = _StrictFakeModel(controller)
    result = SessionCompactor(model, request_id="req-strict").compact(
        [
            HumanMessage(content="old goal"),
            HumanMessage(content="closed progress"),
            HumanMessage(content="current request"),
        ],
        previous_checkpoint=None,
        retain_recent_tokens=1,
    )
    return controller, result


def test_lossless_composer_clamps_observational_budget_without_projection() -> None:
    messages = [HumanMessage(content="goal"), *_exchange(1)]
    result = compose_model_messages(
        messages,
        message_token_budget=100,
    )
    assert result.messages == tuple(messages)
    assert result.stats["message_token_budget"] == 200_000
    assert result.stats["composed_message_count"] == len(messages)


def test_open_tool_protocol_is_never_compacted() -> None:
    open_call = AIMessage(
        content="",
        tool_calls=[
            {"id": "open", "name": "writer", "args": {}, "type": "tool_call"}
        ],
    )
    messages = [
        HumanMessage(content="historical question"),
        AIMessage(content="historical answer"),
        HumanMessage(content="current request"),
        *_exchange(1),
        open_call,
    ]
    plan = plan_session_compaction(
        messages,
        checkpoint=None,
        request_id="req-open",
        retain_recent_tokens=1,
    )
    assert open_call in plan.retained_messages
    assert open_call not in plan.compacted_prefix
    assert messages[2] in plan.retained_messages


def test_three_compactions_are_cumulative_and_prefix_is_verified() -> None:
    model = SummaryModel()
    compactor = SessionCompactor(
        model,
        request_id="req-three",
        protocol=JsonTextSessionCompactionTestProtocol(),
    )
    messages: list = [HumanMessage(content="EARLY_GOAL")]
    checkpoint = None
    for index in range(1, 4):
        messages.extend(_exchange(index))
        messages.append(HumanMessage(content=f"continue-{index}"))
        result = compactor.compact(
            messages,
            previous_checkpoint=checkpoint,
            retain_recent_tokens=1,
        )
        checkpoint = result.checkpoint

    assert checkpoint["schema_version"] == SESSION_COMPACT_SCHEMA
    assert checkpoint["generation"] == 3
    assert checkpoint["session_intent"] == ["EARLY_GOAL"]
    assert len(checkpoint["mechanical_state"]["tool_receipts"]) == 3
    assert len(checkpoint["mechanical_state"]["side_effect_receipts"]) == 3
    assert {
        row["resource_id"]
        for row in checkpoint["mechanical_state"]["resource_refs"]
    } == {
        "resource:conversation-file:1",
        "resource:conversation-file:2",
        "resource:conversation-file:3",
    }
    visible = active_messages_with_checkpoint(
        messages,
        checkpoint=checkpoint,
        request_id="req-three",
    )
    assert visible[0].additional_kwargs["checkpoint_id"] == checkpoint[
        "checkpoint_id"
    ]
    assert len(visible) < len(messages)


def test_large_task_complete_summary_and_server_receipt_restore_exactly_once() -> None:
    summary = "BEGIN_FINAL\n" + ("完整终态内容。" * 700) + "\nEND_FINAL"
    receipt = {
        "schema_version": "evoengine.task-completion-receipt/v1",
        "authority": "task_tree_internal_api",
        "receipt_id": "completion-compaction-unique",
        "run_id": "run-compaction-unique",
        "request_id": "req-completion-compaction",
        "summary_sha256": "a" * 64,
        "actions": [
            {
                "node_id": "node-runtime-only",
                "tool_name": "skill_runtime_only",
                "capability_id": "skill.runtime.only",
                "capability_version": "1.0.0",
                "call_id": "call-runtime-only",
                "status": "succeeded",
            }
        ],
        "deliverables": [
            {
                "conversation_file_id": 99,
                "task_node_id": "node-runtime-only",
                "tool_run_id": "call-runtime-only",
                "file_name": "runtime-only.txt",
                "artifact_key": "runtime_only_file",
                "drawer_section": "result_file",
                "mime_type": "text/plain",
                "size_bytes": 64,
                "sha256": "b" * 64,
            }
        ],
        "supersessions": [],
    }
    messages = [
        HumanMessage(content="完成这项长任务"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-task-complete",
                    "name": "task_complete",
                    "args": {"summary": summary},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=json.dumps(
                {
                    "ok": True,
                    "accepted": True,
                    "summary": summary,
                    "completion_receipt": receipt,
                },
                ensure_ascii=False,
            ),
            tool_call_id="call-task-complete",
            name="task_complete",
        ),
        HumanMessage(content="当前保留消息"),
    ]

    model = SummaryModel()
    result = SessionCompactor(
        model,
        request_id="req-completion-compaction",
        protocol=JsonTextSessionCompactionTestProtocol(),
    ).compact(
        messages,
        previous_checkpoint=None,
        retain_recent_tokens=1,
    )
    checkpoint = result.checkpoint
    assert checkpoint is not None
    summary_prompt_text = "\n".join(
        str(message.content) for prompt in model.prompts for message in prompt
    )
    assert receipt["receipt_id"] not in summary_prompt_text
    assert summary not in summary_prompt_text
    assert "node-runtime-only" not in summary_prompt_text
    assert "runtime-only.txt" not in summary_prompt_text
    assert checkpoint["runtime_control"]["completion_authority"] == {
        "schema_version": "evoengine.agent-terminal-seal/v1",
        "tool_call_id": "call-task-complete",
        "tool_name": "task_complete",
        "summary": summary,
        "completion_receipt": receipt,
    }

    checkpoint_json = json.dumps(checkpoint, ensure_ascii=False, sort_keys=True)
    assert checkpoint_json.count(receipt["receipt_id"]) == 1
    encoded_summary = json.dumps(summary, ensure_ascii=False)
    assert checkpoint_json.count(encoded_summary) == 1
    visible = active_messages_with_checkpoint(
        messages,
        checkpoint=checkpoint,
        request_id="req-completion-compaction",
    )
    checkpoint_messages = [
        message
        for message in visible
        if message.additional_kwargs.get("evo_context_kind")
        == "session_compact_checkpoint"
    ]
    assert len(checkpoint_messages) == 1
    assert receipt["receipt_id"] not in checkpoint_messages[0].content
    assert summary not in checkpoint_messages[0].content
    assert "runtime_control" not in checkpoint_messages[0].content
    assert "deliverables" not in checkpoint_messages[0].content
    assert "node-runtime-only" not in checkpoint_messages[0].content
    assert "runtime-only.txt" not in checkpoint_messages[0].content
    assert not any(
        isinstance(message, ToolMessage) and message.name == "task_complete"
        for message in visible
    )
    assert active_messages_with_checkpoint(
        messages,
        checkpoint=checkpoint,
        request_id="req-completion-compaction",
    ) == visible
    persisted_checkpoint = json.loads(
        json.dumps(checkpoint, ensure_ascii=False, sort_keys=True)
    )
    assert persisted_checkpoint["runtime_control"]["completion_authority"] == (
        checkpoint["runtime_control"]["completion_authority"]
    )
    restored_visible = active_messages_with_checkpoint(
        messages,
        checkpoint=persisted_checkpoint,
        request_id="req-completion-compaction",
    )
    assert len(
        [
            message
            for message in restored_visible
            if message.additional_kwargs.get("evo_context_kind")
            == "session_compact_checkpoint"
        ]
    ) == 1
    assert receipt["receipt_id"] not in restored_visible[0].content
    assert summary not in restored_visible[0].content


def test_completion_authority_attaches_only_to_an_existing_valid_checkpoint() -> None:
    request_id = "req-attach-completion-authority"
    source_messages = [
        HumanMessage(content="先完成历史工作"),
        *_exchange(41),
        HumanMessage(content="当前保留消息"),
    ]
    base = SessionCompactor(
        SummaryModel(),
        request_id=request_id,
        protocol=JsonTextSessionCompactionTestProtocol(),
    ).compact(
        source_messages,
        previous_checkpoint=None,
        retain_recent_tokens=1,
    ).checkpoint
    assert base is not None
    assert "runtime_control" not in base
    summary = "已有语义 checkpoint 后产生的最终摘要。"
    receipt = {
        "schema_version": "evoengine.task-completion-receipt/v1",
        "authority": "task_tree_internal_api",
        "receipt_id": "completion-attached-authority",
        "run_id": "run-attached-authority",
        "request_id": request_id,
        "summary_sha256": "a" * 64,
        "actions": [],
        "deliverables": [],
        "supersessions": [],
    }
    seal = {
        "schema_version": "evoengine.agent-terminal-seal/v1",
        "tool_call_id": "call-attached-task-complete",
        "tool_name": "task_complete",
        "summary": summary,
        "completion_receipt": receipt,
    }
    task_authority = {
        "schema_version": "evoengine.task-authority-control/v1",
        "transport_request_id": request_id,
        "task_authority_request_id": request_id,
        "task_run_id": receipt["run_id"],
        "continuation_parent_request_id": None,
    }
    attached = attach_completion_authority_to_checkpoint(
        base,
        completion_authority=seal,
        request_id=request_id,
        task_authority=task_authority,
    )
    assert attached is not None
    assert attached["checkpoint_id"] != base["checkpoint_id"]
    assert attached["runtime_control"]["completion_authority"] == seal
    assert attached["runtime_control"]["task_authority"] == task_authority
    serialized = json.dumps(attached, ensure_ascii=False, sort_keys=True)
    assert serialized.count(summary) == 1
    assert serialized.count(receipt["receipt_id"]) == 1
    visible = active_messages_with_checkpoint(
        source_messages,
        checkpoint=attached,
        request_id=request_id,
    )
    assert len(
        [
            message
            for message in visible
            if message.additional_kwargs.get("evo_context_kind")
            == "session_compact_checkpoint"
        ]
    ) == 1
    assert summary not in visible[0].content
    assert receipt["receipt_id"] not in visible[0].content
    assert "runtime_control" not in visible[0].content

    repeated = attach_completion_authority_to_checkpoint(
        attached,
        completion_authority=seal,
        request_id=request_id,
        task_authority=task_authority,
    )
    assert repeated == attached
    for invalid_coverage in (None, {"request_id": request_id, "message_count": 0}):
        invalid = json.loads(json.dumps(base))
        if invalid_coverage is None:
            invalid.pop("active_request_coverage")
        else:
            invalid["active_request_coverage"] = invalid_coverage
        assert (
            attach_completion_authority_to_checkpoint(
                invalid,
                completion_authority=seal,
                request_id=request_id,
                task_authority=task_authority,
            )
            is None
        )


def test_hitl_checkpoint_keeps_transport_owner_and_task_receipt_authority() -> None:
    transport_request_id = "req_hitl_continue_checkpoint"
    authority_request_id = "req_parent_task_authority"
    parent_request_id = "req_parent_waiting_human"
    run_id = "run_parent_task_authority"
    source_messages = [
        HumanMessage(content="继续父任务"),
        *_exchange(42),
        HumanMessage(content="保留当前步骤"),
    ]
    base = SessionCompactor(
        SummaryModel(),
        request_id=transport_request_id,
        protocol=JsonTextSessionCompactionTestProtocol(),
    ).compact(
        source_messages,
        previous_checkpoint=None,
        retain_recent_tokens=1,
    ).checkpoint
    assert base is not None
    receipt = {
        "schema_version": "evoengine.task-completion-receipt/v1",
        "authority": "task_tree_internal_api",
        "receipt_id": "completion-hitl-authority",
        "run_id": run_id,
        "request_id": authority_request_id,
        "summary_sha256": "b" * 64,
        "actions": [],
        "deliverables": [],
        "supersessions": [],
    }
    seal = {
        "schema_version": "evoengine.agent-terminal-seal/v1",
        "tool_call_id": "call-hitl-task-complete",
        "tool_name": "task_complete",
        "summary": "父任务在 HITL continuation 中权威完成。",
        "completion_receipt": receipt,
    }
    task_authority = {
        "schema_version": "evoengine.task-authority-control/v1",
        "transport_request_id": transport_request_id,
        "task_authority_request_id": authority_request_id,
        "task_run_id": run_id,
        "continuation_parent_request_id": parent_request_id,
    }

    attached = attach_completion_authority_to_checkpoint(
        base,
        completion_authority=seal,
        request_id=transport_request_id,
        task_authority=task_authority,
    )

    assert attached is not None
    assert attached["created_by_request_id"] == transport_request_id
    assert attached["active_request_coverage"]["request_id"] == transport_request_id
    assert attached["runtime_control"]["task_authority"] == task_authority
    assert completion_authority_for_request(
        attached,
        request_id=transport_request_id,
        task_authority=task_authority,
    ) == seal

    for field, wrong_value in (
        ("transport_request_id", transport_request_id + "_prefix_other"),
        ("task_authority_request_id", authority_request_id + "_prefix_other"),
        ("task_run_id", run_id + "_other"),
        ("continuation_parent_request_id", transport_request_id),
    ):
        tampered = json.loads(json.dumps(attached))
        tampered["runtime_control"]["task_authority"][field] = wrong_value
        with pytest.raises(SessionCompactionError):
            completion_authority_for_request(
                tampered,
                request_id=transport_request_id,
                task_authority=task_authority,
            )


def test_zero_coverage_checkpoint_is_not_injected_or_inherited() -> None:
    messages = [
        HumanMessage(content="new goal"),
        HumanMessage(content="closed work"),
        HumanMessage(content="current request"),
    ]
    stale_checkpoint = {
        "schema_version": SESSION_COMPACT_SCHEMA,
        "checkpoint_id": "session_checkpoint_stale",
        "generation": 9,
        "session_intent": ["STALE_GOAL"],
        "mechanical_state": {},
        "active_request_coverage": {
            "request_id": "req-zero-coverage",
            "message_count": 1,
            "prefix_digest": "not-the-current-prefix",
        },
    }

    visible = active_messages_with_checkpoint(
        messages,
        checkpoint=stale_checkpoint,
        request_id="req-zero-coverage",
    )
    result = SessionCompactor(
        SummaryModel(),
        request_id="req-zero-coverage",
        protocol=JsonTextSessionCompactionTestProtocol(),
    ).compact(
        messages,
        previous_checkpoint=stale_checkpoint,
        retain_recent_tokens=1,
    )

    assert visible == tuple(messages)
    assert result.checkpoint is not None
    assert result.checkpoint["generation"] == 1
    assert result.checkpoint["previous_checkpoint_id"] is None
    assert result.checkpoint["session_intent"] == ["EARLY_GOAL"]


def test_invalid_semantic_output_does_not_mutate_source_messages() -> None:
    messages = [HumanMessage(content="goal"), HumanMessage(content="middle"), HumanMessage(content="latest")]
    original = tuple(messages)
    with pytest.raises(SessionCompactionError):
        SessionCompactor(
            InvalidModel(),
            request_id="req-invalid",
            protocol=JsonTextSessionCompactionTestProtocol(),
        ).compact(
            messages,
            previous_checkpoint=None,
            retain_recent_tokens=1,
        )
    assert tuple(messages) == original


def test_required_checkpoint_sections_are_explicit() -> None:
    assert len(REQUIRED_SECTIONS) == 12
    assert "user_constraints" in REQUIRED_SECTIONS
    assert "side_effect_receipts" in REQUIRED_SECTIONS
    assert "open_questions_hitl" in REQUIRED_SECTIONS


def test_strict_protocol_forces_schema_disables_thinking_and_retries_format_once() -> None:
    controller, result = _strict_compaction(
        [AIMessage(content=""), _strict_response(_sections())]
    )

    assert controller.calls == 2
    assert result.stats["compaction_format_retry_count"] == 1
    assert result.stats["compaction_transport_reconnect_count"] == 0
    assert all(binding["strict"] is True for binding in controller.bindings)
    assert all(
        binding["parallel_tool_calls"] is False for binding in controller.bindings
    )
    assert all(
        binding["tool_choice"] == "SubmitSessionCheckpoint"
        for binding in controller.bindings
    )
    assert controller.clone_updates[0]["extra_body"]["thinking"] == {
        "type": "disabled"
    }
    assert all(
        SESSION_COMPACTION_MODEL_TAG in config.get("tags", [])
        for config in controller.call_configs
    )
    schema = controller.bindings[0]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(REQUIRED_SECTIONS)


def test_strict_protocol_never_uses_more_than_one_format_correction() -> None:
    controller = _StrictController(
        [AIMessage(content=""), AIMessage(content=""), _strict_response(_sections())]
    )
    model = _StrictFakeModel(controller)

    with pytest.raises(SessionCompactionFormatError):
        SessionCompactor(model, request_id="req-format-limit").compact(
            [HumanMessage(content="old"), HumanMessage(content="middle"), HumanMessage(content="latest")],
            previous_checkpoint=None,
            retain_recent_tokens=1,
        )

    assert controller.calls == 2


def test_transport_reconnect_budget_is_separate_and_uses_fresh_clients() -> None:
    controller, result = _strict_compaction(
        [TimeoutError(), TimeoutError(), TimeoutError(), _strict_response(_sections())]
    )

    assert controller.calls == 4
    assert result.stats["compaction_format_retry_count"] == 0
    assert result.stats["compaction_transport_reconnect_count"] == 3
    # One template plus one independently owned client for each generation.
    owned_transports = controller.transports[1:]
    assert len(owned_transports) == 4
    assert len({id(sync) for sync, _async in owned_transports}) == 4
    assert all(
        sync.closed and async_transport.closed
        for sync, async_transport in owned_transports
    )


@pytest.mark.parametrize("status_code", [409, 429])
def test_non_transport_status_does_not_rebuild_connection(status_code: int) -> None:
    controller = _StrictController(
        [_StatusError(status_code), _strict_response(_sections())]
    )
    model = _StrictFakeModel(controller)

    with pytest.raises(SessionCompactionError, match="provider request failed"):
        SessionCompactor(model, request_id="req-no-fake-reconnect").compact(
            [HumanMessage(content="old"), HumanMessage(content="middle"), HumanMessage(content="latest")],
            previous_checkpoint=None,
            retain_recent_tokens=1,
        )

    assert controller.calls == 1
    assert len(controller.transports[1:]) == 1


def test_5xx_status_rebuilds_one_fresh_connection() -> None:
    controller, result = _strict_compaction(
        [_StatusError(502), _strict_response(_sections())]
    )

    assert controller.calls == 2
    assert result.stats["compaction_transport_reconnect_count"] == 1
    assert len({id(sync) for sync, _async in controller.transports[1:]}) == 2


def test_rate_limit_error_does_not_rebuild_connection() -> None:
    controller = _StrictController(
        [RateLimitError("limited"), _strict_response(_sections())]
    )
    model = _StrictFakeModel(controller)

    with pytest.raises(SessionCompactionError, match="provider request failed"):
        SessionCompactor(model, request_id="req-rate-limit").compact(
            [HumanMessage(content="old"), HumanMessage(content="middle"), HumanMessage(content="latest")],
            previous_checkpoint=None,
            retain_recent_tokens=1,
        )

    assert controller.calls == 1
    assert len(controller.transports[1:]) == 1


def test_transport_stops_after_three_reconnects() -> None:
    controller = _StrictController(
        [TimeoutError(), TimeoutError(), TimeoutError(), TimeoutError()]
    )
    model = _StrictFakeModel(controller)

    with pytest.raises(SessionCompactionTransportError):
        SessionCompactor(model, request_id="req-transport-limit").compact(
            [HumanMessage(content="old"), HumanMessage(content="middle"), HumanMessage(content="latest")],
            previous_checkpoint=None,
            retain_recent_tokens=1,
        )

    assert controller.calls == 4


def test_output_limit_is_not_misclassified_as_format_or_transport() -> None:
    response = AIMessage(content="", response_metadata={"finish_reason": "length"})
    controller = _StrictController([response, _strict_response(_sections())])
    model = _StrictFakeModel(controller)

    with pytest.raises(SessionCompactionOutputLimitError):
        SessionCompactor(model, request_id="req-output-limit").compact(
            [HumanMessage(content="old"), HumanMessage(content="middle"), HumanMessage(content="latest")],
            previous_checkpoint=None,
            retain_recent_tokens=1,
        )

    assert controller.calls == 1


def test_async_strict_protocol_uses_the_same_independent_budgets() -> None:
    controller = _StrictController(
        [AIMessage(content=""), TimeoutError(), _strict_response(_sections())]
    )
    model = _StrictFakeModel(controller)

    result = asyncio.run(
        SessionCompactor(model, request_id="req-async-strict").acompact(
            [HumanMessage(content="old"), HumanMessage(content="middle"), HumanMessage(content="latest")],
            previous_checkpoint=None,
            retain_recent_tokens=1,
        )
    )

    assert controller.calls == 3
    assert result.stats["compaction_format_retry_count"] == 1
    assert result.stats["compaction_transport_reconnect_count"] == 1
    assert all(
        sync.closed and async_transport.closed
        for sync, async_transport in controller.transports[1:]
    )


def test_async_compaction_timeout_is_bounded_and_reuses_transport_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVO_SESSION_COMPACTION_REQUEST_TIMEOUT_SEC", "0.05")
    controller = _StrictController([_AsyncHang(), _AsyncHang(), _AsyncHang(), _AsyncHang()])
    model = _StrictFakeModel(controller)

    with pytest.raises(SessionCompactionTransportError):
        asyncio.run(
            SessionCompactor(model, request_id="req-async-timeout").acompact(
                [
                    HumanMessage(content="old"),
                    HumanMessage(content="middle"),
                    HumanMessage(content="latest"),
                ],
                previous_checkpoint=None,
                retain_recent_tokens=1,
            )
        )

    assert controller.calls == 4
    assert all(
        sync.closed and async_transport.closed
        for sync, async_transport in controller.transports[1:]
    )
