from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents import native_agent_graph
from src.agents.canonical_protocol_ledger import (
    CanonicalProtocolLedger,
    CanonicalRuntimeTransition,
)


def test_runtime_rebuild_transitions_are_iterative_and_close_owned_context(
    monkeypatch,
) -> None:
    active_depth = 0
    max_active_depth = 0
    calls: list[tuple[int, int]] = []
    closed_contexts: list[dict] = []
    recovered_context = {
        "llm_transport_generation": 1,
        "llm_transport_owner": {"state": "open"},
    }

    async def fake_once(state, context, *, event_writer=None):
        del event_writer
        nonlocal active_depth, max_active_depth
        active_depth += 1
        max_active_depth = max(max_active_depth, active_depth)
        try:
            attempt = int(state.get("attempt") or 0)
            calls.append((attempt, int(context.get("llm_transport_generation") or 0)))
            if attempt < 4:
                transition = CanonicalRuntimeTransition(
                    "retry_turn",
                    {"reason": f"retry-{attempt}"},
                )
                transition.next_state = {
                    **state,
                    "attempt": attempt + 1,
                }
                if attempt == 1:
                    transition.next_context = recovered_context
                raise transition
            return {"status": "completed", "attempt": attempt}
        finally:
            active_depth -= 1

    async def fake_close(context):
        closed_contexts.append(context)
        return {"owned": True, "attempted": 1, "closed": 1, "errors": []}

    monkeypatch.setattr(native_agent_graph, "_run_native_agent_turn_once", fake_once)
    monkeypatch.setattr(native_agent_graph, "_close_owned_llm_transport", fake_close)

    result = asyncio.run(
        native_agent_graph.run_native_agent_turn(
            {"request_id": "request-iterative", "attempt": 0},
            {"llm_transport_generation": 0},
        )
    )

    assert result == {"status": "completed", "attempt": 4}
    assert max_active_depth == 1
    assert calls == [(0, 0), (1, 0), (2, 1), (3, 1), (4, 1)]
    assert closed_contexts == [recovered_context]


def test_turn_checkpoint_requires_closed_canonical_ledger() -> None:
    ledger = CanonicalProtocolLedger()
    ledger.seed_if_empty(
        [
            HumanMessage(content="goal"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "open-call",
                        "name": "writer",
                        "args": {},
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )

    with pytest.raises(
        RuntimeError,
        match="CANONICAL_PROTOCOL_CHECKPOINT_REQUIRES_CLOSED_LEDGER",
    ):
        native_agent_graph._build_turn_checkpoint(
            state={"request_id": "request-open"},
            protocol_ledger=ledger,
            pending_bundle={},
            pending_sandbox_jobs=[],
        )


def test_only_server_task_receipt_is_public() -> None:
    generic = {
        "schema_version": "evoengine.completion-receipt-generic/v1",
        "accepted": True,
    }
    task_receipt = {
        "schema_version": "evoengine.task-completion-receipt/v1",
        "authority": "task_tree_internal_api",
        "receipt_id": "completion_receipt_1",
    }

    assert native_agent_graph._public_task_completion_receipt(generic) is None
    assert native_agent_graph._public_task_completion_receipt(task_receipt) == (
        task_receipt
    )
