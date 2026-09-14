from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.agents.canonical_protocol_ledger import (
    CanonicalProtocolLedger,
    CanonicalProtocolMiddleware,
)


def _model_call(*calls: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=list(calls))


def _call(call_id: str, name: str, args: dict | None = None) -> dict:
    return {
        "id": call_id,
        "name": name,
        "args": dict(args or {}),
        "type": "tool_call",
    }


def _result(call_id: str, name: str, payload: dict) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(payload),
        tool_call_id=call_id,
        name=name,
    )


def test_parallel_same_name_results_keep_declared_call_order() -> None:
    ledger = CanonicalProtocolLedger()
    ledger.observe_model_message(
        _model_call(
            _call("call-1", "same_tool", {"value": 1}),
            _call("call-2", "same_tool", {"value": 2}),
        )
    )

    ledger.commit_tool_result(
        call_id="call-2",
        tool_name="same_tool",
        tool_input={"value": 2},
        result=_result("call-2", "same_tool", {"ok": True, "value": 2}),
    )
    ledger.commit_tool_result(
        call_id="call-1",
        tool_name="same_tool",
        tool_input={"value": 1},
        result=_result("call-1", "same_tool", {"ok": True, "value": 1}),
    )

    messages = ledger.snapshot_messages()
    assert [message.tool_call_id for message in messages[1:]] == [
        "call-1",
        "call-2",
    ]
    assert ledger.pending_calls() == {}


def test_transition_waits_for_entire_model_declared_batch() -> None:
    ledger = CanonicalProtocolLedger()
    ledger.observe_model_message(
        _model_call(
            _call("load", "load_tools", {"tool_ids": ["alpha"]}),
            _call("read", "read_resource", {}),
        )
    )
    ledger.commit_tool_result(
        call_id="load",
        tool_name="load_tools",
        tool_input={"tool_ids": ["alpha"]},
        result=_result(
            "load",
            "load_tools",
            {"ok": True, "loaded_tool_ids": ["alpha"]},
        ),
    )

    assert ledger.pop_ready_transition() is None

    ledger.commit_tool_result(
        call_id="read",
        tool_name="read_resource",
        tool_input={},
        result=_result("read", "read_resource", {"ok": True}),
    )
    assert ledger.pop_ready_transition() == (
        "reload_tools",
        {"tool_ids": ["alpha"]},
    )


def test_outer_middleware_commits_short_circuit_and_handler_error() -> None:
    ledger = CanonicalProtocolLedger()
    middleware = CanonicalProtocolMiddleware(ledger)
    ledger.observe_model_message(
        _model_call(
            _call("short", "tree_gate", {}),
            _call("error", "writer", {}),
        )
    )

    short_result = _result(
        "short",
        "tree_gate",
        {"ok": False, "status": "failed"},
    )
    returned = middleware.wrap_tool_call(
        SimpleNamespace(tool_call=_call("short", "tree_gate")),
        lambda _request: short_result,
    )

    def _raise(_request):
        raise ValueError("invalid")

    error_result = middleware.wrap_tool_call(
        SimpleNamespace(tool_call=_call("error", "writer")),
        _raise,
    )

    assert returned is short_result
    assert error_result.tool_call_id == "error"
    assert error_result.status == "error"
    assert ledger.pending_calls() == {}


def test_task_complete_seal_preserves_server_receipt() -> None:
    server_summary = "\nServer 已持久化的权威摘要。\n"
    receipt = {
        "schema_version": "evoengine.task-completion-receipt/v1",
        "authority": "task_tree_internal_api",
        "receipt_id": "completion_receipt_1",
        "run_id": "run-1",
        "deliverables": [],
    }
    ledger = CanonicalProtocolLedger()
    ledger.observe_model_message(
        _model_call(
            _call("complete", "task_complete", {"summary": "model draft"})
        )
    )
    ledger.commit_tool_result(
        call_id="complete",
        tool_name="task_complete",
        tool_input={"summary": "model draft"},
        result=ToolMessage(
            content=json.dumps(
                {
                    "ok": True,
                    "accepted": True,
                    "completion_receipt": receipt,
                }
            ),
            artifact={
                "schema_version": "evoengine.tool-runtime-artifact/v1",
                "control_projection": {
                    "completion_receipt": receipt,
                    "completion_summary": server_summary,
                },
            },
            tool_call_id="complete",
            name="task_complete",
        ),
    )

    kind, seal = ledger.pop_ready_transition()
    assert kind == "terminal_complete"
    assert seal["summary"] == server_summary
    assert seal["completion_receipt"] == receipt


def test_exact_call_declaration_replay_is_idempotent() -> None:
    ledger = CanonicalProtocolLedger()
    first = _model_call(_call("same", "reader", {"page": 1}))
    replay = _model_call(_call("same", "reader", {"page": 1}))

    ledger.observe_model_message(first)
    ledger.observe_model_message(replay)

    assert ledger.pending_calls() == {"same": "reader"}
    assert len(ledger.snapshot_messages()) == 1


def test_call_batch_replay_with_different_model_payload_fails_closed() -> None:
    ledger = CanonicalProtocolLedger()
    ledger.observe_model_message(
        AIMessage(
            content="first declaration",
            tool_calls=[_call("same", "reader", {"page": 1})],
        )
    )

    with pytest.raises(
        RuntimeError,
        match="CANONICAL_PROTOCOL_CONFLICTING_CALL_DECLARATION",
    ):
        ledger.observe_model_message(
            AIMessage(
                content="different declaration",
                tool_calls=[_call("same", "reader", {"page": 1})],
            )
        )


@pytest.mark.parametrize(
    "replay",
    [
        _model_call(_call("same", "other_reader", {"page": 1})),
        _model_call(_call("same", "reader", {"page": 2})),
    ],
)
def test_call_id_reuse_with_different_identity_fails_closed(
    replay: AIMessage,
) -> None:
    ledger = CanonicalProtocolLedger()
    ledger.observe_model_message(
        _model_call(_call("same", "reader", {"page": 1}))
    )

    with pytest.raises(
        RuntimeError,
        match="CANONICAL_PROTOCOL_CONFLICTING_CALL_DECLARATION",
    ):
        ledger.observe_model_message(replay)


def test_partial_call_id_reuse_fails_before_appending_new_unit() -> None:
    ledger = CanonicalProtocolLedger()
    ledger.observe_model_message(_model_call(_call("known", "reader", {})))

    with pytest.raises(
        RuntimeError,
        match="CANONICAL_PROTOCOL_PARTIAL_CALL_ID_REUSE",
    ):
        ledger.observe_model_message(
            _model_call(
                _call("known", "reader", {}),
                _call("new", "writer", {}),
            )
        )

    assert ledger.pending_calls() == {"known": "reader"}


def test_tool_request_identity_is_checked_before_handler() -> None:
    ledger = CanonicalProtocolLedger()
    middleware = CanonicalProtocolMiddleware(ledger)
    ledger.observe_model_message(
        _model_call(_call("bound", "writer", {"value": 1}))
    )
    handler_called = False

    def _handler(_request):
        nonlocal handler_called
        handler_called = True
        return _result("bound", "writer", {"ok": True})

    with pytest.raises(
        RuntimeError,
        match="CANONICAL_PROTOCOL_TOOL_REQUEST_MISMATCH",
    ):
        middleware.wrap_tool_call(
            SimpleNamespace(
                tool_call=_call("bound", "writer", {"value": 2})
            ),
            _handler,
        )

    assert handler_called is False


def test_duplicate_result_compares_status_and_runtime_artifact() -> None:
    ledger = CanonicalProtocolLedger()
    ledger.observe_model_message(_model_call(_call("result", "writer", {})))
    first = ToolMessage(
        content='{"ok":true}',
        tool_call_id="result",
        name="writer",
        status="success",
        artifact={"receipt": {"id": "first"}},
    )
    ledger.commit_tool_result(
        call_id="result",
        tool_name="writer",
        tool_input={},
        result=first,
    )
    conflicting = first.model_copy(
        update={"artifact": {"receipt": {"id": "second"}}},
        deep=True,
    )

    with pytest.raises(
        RuntimeError,
        match="CANONICAL_PROTOCOL_CONFLICTING_RESULT",
    ):
        ledger.commit_tool_result(
            call_id="result",
            tool_name="writer",
            tool_input={},
            result=conflicting,
        )
