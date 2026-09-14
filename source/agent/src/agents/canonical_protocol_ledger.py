"""Single in-process authority for model/tool protocol state.

LangGraph owns execution, but callback events are observational and may omit a
middleware short-circuit.  This ledger is therefore written by the outermost
middleware around the actual model/tool handlers.  Timeline, recovery and
checkpoint code may project this state; they must not reconstruct it from
``on_tool_*`` events.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import threading
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.types import Command


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _deep_copy(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:  # noqa: BLE001
        return value


def _call_identity(call: dict[str, Any]) -> str:
    return _json_text(
        {
            "id": str(call.get("id") or "").strip(),
            "name": str(call.get("name") or "").strip(),
            "args": call.get("args"),
            "invalid": bool(call.get("_evo_invalid")),
        }
    )


def _tool_result_identity(message: ToolMessage) -> str:
    return _json_text(message.model_dump(mode="json", exclude_none=False))


def _model_message_identity(message: AIMessage) -> str:
    return _json_text(message.model_dump(mode="json", exclude_none=False))


def _copy_message(message: BaseMessage) -> BaseMessage:
    copier = getattr(message, "model_copy", None)
    if callable(copier):
        try:
            return copier(deep=True)
        except Exception:  # noqa: BLE001
            pass
    return _deep_copy(message)


def _content_payload(value: Any) -> dict[str, Any]:
    content = getattr(value, "content", value)
    if isinstance(content, dict):
        return dict(content)
    if not isinstance(content, str):
        return {}
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _result_ok(value: Any) -> bool:
    if isinstance(value, ToolMessage) and str(
        getattr(value, "status", "success") or "success"
    ).lower() == "error":
        return False
    payload = _content_payload(value)
    return not payload or (
        payload.get("ok") is not False
        and payload.get("success") is not False
        and str(payload.get("status") or "").lower()
        not in {"failed", "error", "cancelled", "canceled"}
    )


def _tool_message_from_command(
    value: Command[Any],
    *,
    call_id: str,
) -> ToolMessage | None:
    update = value.update
    if isinstance(update, dict):
        candidates = update.get("messages")
    else:
        candidates = None
    if not isinstance(candidates, (list, tuple)):
        return None
    for candidate in candidates:
        if not isinstance(candidate, ToolMessage):
            continue
        if str(getattr(candidate, "tool_call_id", "") or "").strip() == call_id:
            return candidate
    return None


def _exact_tool_message(
    value: Any,
    *,
    call_id: str,
    tool_name: str,
) -> ToolMessage:
    message = (
        _tool_message_from_command(value, call_id=call_id)
        if isinstance(value, Command)
        else value
    )
    if isinstance(message, ToolMessage):
        updates: dict[str, Any] = {}
        if str(getattr(message, "tool_call_id", "") or "").strip() != call_id:
            updates["tool_call_id"] = call_id
        if str(getattr(message, "name", "") or "").strip() != tool_name:
            updates["name"] = tool_name or None
        return message.model_copy(update=updates) if updates else message
    content = (
        _json_text(message)
        if isinstance(message, (dict, list, tuple))
        else str(message or "")
    )
    return ToolMessage(
        content=content,
        tool_call_id=call_id,
        name=tool_name or None,
    )


def _tool_error_message(
    exc: BaseException,
    *,
    call_id: str,
    tool_name: str,
) -> ToolMessage:
    missing_fields: list[str] = []
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            for item in errors() or []:
                if not isinstance(item, dict) or str(item.get("type") or "") != "missing":
                    continue
                loc = item.get("loc")
                field_name = (
                    ".".join(str(part) for part in loc if str(part).strip())
                    if isinstance(loc, (list, tuple))
                    else str(loc or "").strip()
                )
                if field_name and field_name not in missing_fields:
                    missing_fields.append(field_name)
        except Exception:
            missing_fields = []
    validation_error = bool(
        missing_fields or "validation" in exc.__class__.__name__.lower()
    )
    payload: dict[str, Any] = {
        "ok": False,
        "success": False,
        "status": "failed",
        "error_kind": "validation_error" if validation_error else "tool_execution_error",
        "error_type": exc.__class__.__name__,
        "summary": (
            "工具参数未通过 schema 校验；请修正参数后重试。"
            if validation_error
            else "工具执行已明确失败；不要假定副作用已经发生。"
        ),
        "complete": True,
        "has_more": False,
    }
    if missing_fields:
        payload["missing_fields"] = missing_fields
    return ToolMessage(
        content=_json_text(payload),
        tool_call_id=call_id,
        name=tool_name or None,
        status="error",
    )


def _invalid_call_message(call: dict[str, Any]) -> ToolMessage:
    return ToolMessage(
        content=_json_text(
            {
                "ok": False,
                "success": False,
                "status": "failed",
                "error_kind": "runtime_tool_call_invalid",
                "summary": (
                    "工具参数无法按 schema 解析，工具未执行。"
                    "请修正参数后重新调用；批量参数过长时拆成更小批次。"
                ),
                "complete": True,
                "has_more": False,
            }
        ),
        tool_call_id=str(call.get("id") or "").strip(),
        name=str(call.get("name") or "").strip() or None,
        status="error",
    )


def _completion_seal(
    *,
    tool_name: str,
    call_id: str,
    tool_input: Any,
    result: ToolMessage,
) -> dict[str, Any] | None:
    if tool_name != "task_complete" or not _result_ok(result):
        return None
    artifact = getattr(result, "artifact", None)
    control = (
        artifact.get("control_projection")
        if isinstance(artifact, dict)
        and str(artifact.get("schema_version") or "")
        == "evoengine.tool-runtime-artifact/v1"
        else None
    )
    if not isinstance(control, dict):
        control = {}
    payload = _content_payload(result)
    signal = control.get("completion_signal")
    if not isinstance(signal, dict):
        signal = payload.get("completion_signal") or payload.get("completion")
    if not isinstance(signal, dict):
        # The resident task_complete tool historically returned its validated
        # acceptance envelope directly.  Treat only that exact tool result as
        # a signal; other tools still require the typed completion envelope.
        signal = payload
    acceptance = signal.get("acceptance")
    if not isinstance(acceptance, dict) and isinstance(signal.get("accepted"), bool):
        acceptance = {"accepted": bool(signal.get("accepted"))}
    if not (
        bool(
            signal.get("should_stop_tool_loop")
            or signal.get("accepted")
        )
        and isinstance(acceptance, dict)
        and bool(acceptance.get("accepted"))
    ):
        return None

    receipt = control.get("completion_receipt")
    if not isinstance(receipt, dict):
        receipt = signal.get("completion_receipt")
    if not isinstance(receipt, dict):
        receipt = acceptance.get("completion_receipt")
    if not isinstance(receipt, dict):
        receipt = payload.get("completion_receipt")
    if not isinstance(receipt, dict):
        # Server rollout may temporarily return the accepted control envelope
        # before its immutable receipt object. Preserve that envelope exactly;
        # do not infer actions, files or lineage in Agent.
        receipt = {
            "schema_version": "evoengine.completion-receipt-generic/v1",
            "accepted": True,
            **(
                {"run": str(payload.get("task_tree_run_id") or "")}
                if str(payload.get("task_tree_run_id") or "").strip()
                else {}
            ),
            **(
                {"deliverables": list(payload.get("deliverable_files") or [])}
                if isinstance(payload.get("deliverable_files"), list)
                else {}
            ),
        }
    if not str(receipt.get("schema_version") or "").strip():
        return None
    if receipt.get("accepted") is False:
        return None
    model_summary = (
        str(tool_input.get("summary") or "")
        if isinstance(tool_input, dict)
        else ""
    )
    summary = model_summary
    if (
        str(receipt.get("schema_version") or "")
        == "evoengine.task-completion-receipt/v1"
    ):
        runtime_summary = control.get("completion_summary")
        if not isinstance(runtime_summary, str) or not runtime_summary.strip():
            runtime_summary = payload.get("summary")
        if isinstance(runtime_summary, str) and runtime_summary.strip():
            summary = runtime_summary
    return {
        "schema_version": "evoengine.agent-terminal-seal/v1",
        "tool_call_id": call_id,
        "tool_name": tool_name,
        "summary": summary if summary.strip() else "任务已完成",
        "completion_receipt": dict(receipt),
    }


def server_task_completion_seal_from_tool_result(
    *,
    tool_name: str,
    call_id: str,
    tool_input: Any,
    result: ToolMessage,
) -> dict[str, Any] | None:
    """Return only an exact Server-issued TaskTree completion seal.

    Generic no-tree acceptance may stop an in-process tool loop, but it is not
    durable Server authority and must never be restored from a session compact
    checkpoint as if it were a TaskTree CompletionReceipt.
    """

    seal = _completion_seal(
        tool_name=tool_name,
        call_id=call_id,
        tool_input=tool_input,
        result=result,
    )
    if not isinstance(seal, dict):
        return None
    receipt = seal.get("completion_receipt")
    if not isinstance(receipt, dict):
        return None
    if (
        str(receipt.get("schema_version") or "")
        != "evoengine.task-completion-receipt/v1"
        or str(receipt.get("authority") or "") != "task_tree_internal_api"
        or not str(receipt.get("receipt_id") or "").strip()
        or not str(receipt.get("run_id") or "").strip()
    ):
        return None
    return _deep_copy(seal)


def _validated_restored_terminal_seal(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if (
        str(value.get("schema_version") or "")
        != "evoengine.agent-terminal-seal/v1"
        or str(value.get("tool_name") or "") != "task_complete"
        or not str(value.get("tool_call_id") or "").strip()
        or not str(value.get("summary") or "").strip()
    ):
        return None
    receipt = value.get("completion_receipt")
    if not isinstance(receipt, dict):
        return None
    if (
        str(receipt.get("schema_version") or "")
        != "evoengine.task-completion-receipt/v1"
        or str(receipt.get("authority") or "") != "task_tree_internal_api"
        or not str(receipt.get("receipt_id") or "").strip()
        or not str(receipt.get("run_id") or "").strip()
    ):
        return None
    return _deep_copy(value)


@dataclass
class _ProtocolUnit:
    message: BaseMessage
    call_order: tuple[str, ...] = ()
    calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    results: dict[str, ToolMessage] = field(default_factory=dict)

    @property
    def closed(self) -> bool:
        return not self.call_order or all(call_id in self.results for call_id in self.call_order)


class CanonicalRuntimeTransition(RuntimeError):
    """Typed request to rebuild runtime state without recursively calling it."""

    def __init__(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        super().__init__(kind)
        self.kind = str(kind or "").strip()
        self.payload = dict(payload or {})
        self.next_state: dict[str, Any] | None = None
        self.next_context: dict[str, Any] | None = None


class CanonicalProtocolLedger:
    """Thread-safe ordered model/tool transcript with exact call identities."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._units: list[_ProtocolUnit] = []
        self._call_units: dict[str, _ProtocolUnit] = {}
        # Retain the original objects, not only their numeric ids.  The ledger
        # stores a deep copy in each protocol unit, so an otherwise-unreferenced
        # model message may be collected and its id reused by a different
        # message before the next middleware observation.
        self._observed_model_objects: dict[int, AIMessage] = {}
        self._transitions: list[tuple[str, dict[str, Any]]] = []
        self._terminal_seal: dict[str, Any] | None = None

    def seed_if_empty(self, messages: list[BaseMessage]) -> None:
        with self._lock:
            if self._units:
                return
            index = 0
            while index < len(messages):
                message = messages[index]
                calls = self._declared_calls(message)
                if not isinstance(message, AIMessage) or not calls:
                    if isinstance(message, ToolMessage):
                        call_id = str(
                            getattr(message, "tool_call_id", "") or ""
                        ).strip()
                        raise RuntimeError(
                            "CANONICAL_PROTOCOL_UNKNOWN_TOOL_RESULT:"
                            + (call_id or "missing")
                        )
                    self._units.append(
                        _ProtocolUnit(message=_copy_message(message))
                    )
                    index += 1
                    continue
                unit = self._new_model_unit(
                    message,
                    calls,
                    repair_invalid=False,
                )
                index += 1
                while index < len(messages) and isinstance(messages[index], ToolMessage):
                    result = messages[index]
                    call_id = str(getattr(result, "tool_call_id", "") or "").strip()
                    if call_id in unit.calls:
                        existing = unit.results.get(call_id)
                        if existing is None:
                            declared_name = str(
                                unit.calls[call_id].get("name") or ""
                            ).strip()
                            exact_result = _exact_tool_message(
                                result,
                                call_id=call_id,
                                tool_name=declared_name,
                            )
                            unit.results[call_id] = _copy_message(exact_result)
                        elif _tool_result_identity(existing) != _tool_result_identity(
                            _exact_tool_message(
                                result,
                                call_id=call_id,
                                tool_name=str(
                                    unit.calls[call_id].get("name") or ""
                                ).strip(),
                            )
                        ):
                            raise RuntimeError(
                                f"CANONICAL_PROTOCOL_CONFLICTING_RESULT:{call_id}"
                            )
                    else:
                        raise RuntimeError(
                            f"CANONICAL_PROTOCOL_UNKNOWN_TOOL_RESULT:{call_id or 'missing'}"
                        )
                    index += 1

    @staticmethod
    def _declared_calls(message: Any) -> list[dict[str, Any]]:
        if not isinstance(message, AIMessage):
            return []
        calls: list[dict[str, Any]] = []
        for invalid, rows in (
            (False, list(getattr(message, "tool_calls", None) or [])),
            (True, list(getattr(message, "invalid_tool_calls", None) or [])),
        ):
            for raw_call in rows:
                if not isinstance(raw_call, dict):
                    continue
                call_id = str(raw_call.get("id") or "").strip()
                if not call_id:
                    continue
                call = _deep_copy(dict(raw_call))
                call["id"] = call_id
                call["name"] = str(call.get("name") or "").strip()
                call["_evo_invalid"] = invalid
                calls.append(call)
        return calls

    def _new_model_unit(
        self,
        message: AIMessage,
        calls: list[dict[str, Any]],
        *,
        repair_invalid: bool = True,
    ) -> _ProtocolUnit:
        call_order = tuple(str(call.get("id") or "").strip() for call in calls)
        if len(call_order) != len(set(call_order)):
            duplicate = next(
                call_id
                for index, call_id in enumerate(call_order)
                if call_id in call_order[:index]
            )
            raise RuntimeError(
                f"CANONICAL_PROTOCOL_DUPLICATE_CALL_ID:{duplicate}"
            )
        for call in calls:
            call_id = str(call.get("id") or "").strip()
            if call_id in self._call_units:
                raise RuntimeError(
                    f"CANONICAL_PROTOCOL_DUPLICATE_CALL_ID:{call_id}"
                )
        unit = _ProtocolUnit(
            message=_copy_message(message),
            call_order=call_order,
            calls={
                str(call.get("id") or "").strip(): _deep_copy(dict(call))
                for call in calls
            },
        )
        self._units.append(unit)
        for call_id in call_order:
            self._call_units[call_id] = unit
        invalid_ids = {
            str(call.get("id") or "").strip() for call in calls
            if bool(call.get("_evo_invalid"))
        }
        if repair_invalid:
            for call_id in invalid_ids:
                if call_id and call_id in unit.calls:
                    unit.results[call_id] = _invalid_call_message(unit.calls[call_id])
        return unit

    def observe_model_message(self, message: Any) -> None:
        if not isinstance(message, AIMessage):
            return
        with self._lock:
            object_identity = id(message)
            if self._observed_model_objects.get(object_identity) is message:
                return
            calls = self._declared_calls(message)
            call_ids = tuple(str(call.get("id") or "").strip() for call in calls)
            if call_ids and any(call_id in self._call_units for call_id in call_ids):
                if not all(call_id in self._call_units for call_id in call_ids):
                    raise RuntimeError(
                        "CANONICAL_PROTOCOL_PARTIAL_CALL_ID_REUSE:"
                        + ",".join(call_ids)
                    )
                units = {id(self._call_units[call_id]) for call_id in call_ids}
                unit = self._call_units[call_ids[0]]
                exact_replay = (
                    len(units) == 1
                    and unit.call_order == call_ids
                    and isinstance(unit.message, AIMessage)
                    and _model_message_identity(unit.message)
                    == _model_message_identity(message)
                    and all(
                        _call_identity(unit.calls[call_id])
                        == _call_identity(call)
                        for call_id, call in zip(call_ids, calls, strict=True)
                    )
                )
                if not exact_replay:
                    raise RuntimeError(
                        "CANONICAL_PROTOCOL_CONFLICTING_CALL_DECLARATION:"
                        + ",".join(call_ids)
                    )
                self._observed_model_objects[object_identity] = message
                return
            if calls:
                self._new_model_unit(message, calls)
            else:
                self._units.append(
                    _ProtocolUnit(message=_copy_message(message))
                )
            self._observed_model_objects[object_identity] = message

    def validate_tool_request(
        self,
        *,
        call_id: str,
        tool_name: str,
        tool_input: Any,
    ) -> None:
        exact_call_id = str(call_id or "").strip()
        with self._lock:
            unit = self._call_units.get(exact_call_id)
            if unit is None:
                raise RuntimeError(
                    f"CANONICAL_PROTOCOL_UNKNOWN_TOOL_CALL:{exact_call_id or 'missing'}"
                )
            declared = unit.calls[exact_call_id]
            actual = {
                "id": exact_call_id,
                "name": str(tool_name or "").strip(),
                "args": tool_input,
                "_evo_invalid": bool(declared.get("_evo_invalid")),
            }
            if _call_identity(declared) != _call_identity(actual):
                raise RuntimeError(
                    f"CANONICAL_PROTOCOL_TOOL_REQUEST_MISMATCH:{exact_call_id}"
                )

    def commit_tool_result(
        self,
        *,
        call_id: str,
        tool_name: str,
        tool_input: Any,
        result: Any,
    ) -> ToolMessage:
        exact_call_id = str(call_id or "").strip()
        exact_tool_name = str(tool_name or "").strip()
        if not exact_call_id:
            raise RuntimeError("CANONICAL_PROTOCOL_TOOL_CALL_ID_MISSING")
        message = _exact_tool_message(
            result,
            call_id=exact_call_id,
            tool_name=exact_tool_name,
        )
        with self._lock:
            self.validate_tool_request(
                call_id=exact_call_id,
                tool_name=exact_tool_name,
                tool_input=tool_input,
            )
            unit = self._call_units.get(exact_call_id)
            if unit is None:
                raise RuntimeError(
                    f"CANONICAL_PROTOCOL_UNKNOWN_TOOL_CALL:{exact_call_id}"
                )
            existing = unit.results.get(exact_call_id)
            if existing is not None:
                if _tool_result_identity(existing) != _tool_result_identity(message):
                    raise RuntimeError(
                        f"CANONICAL_PROTOCOL_CONFLICTING_RESULT:{exact_call_id}"
                    )
                return _copy_message(existing)
            stored_message = _copy_message(message)
            unit.results[exact_call_id] = stored_message
            self._record_transition(
                tool_name=exact_tool_name,
                call_id=exact_call_id,
                tool_input=tool_input,
                result=stored_message,
            )
            return _copy_message(stored_message)

    def commit_tool_error(
        self,
        *,
        call_id: str,
        tool_name: str,
        tool_input: Any,
        error: BaseException,
    ) -> ToolMessage:
        message = _tool_error_message(
            error,
            call_id=str(call_id or "").strip(),
            tool_name=str(tool_name or "").strip(),
        )
        return self.commit_tool_result(
            call_id=call_id,
            tool_name=tool_name,
            tool_input=tool_input,
            result=message,
        )

    def _record_transition(
        self,
        *,
        tool_name: str,
        call_id: str,
        tool_input: Any,
        result: ToolMessage,
    ) -> None:
        seal = _completion_seal(
            tool_name=tool_name,
            call_id=call_id,
            tool_input=tool_input,
            result=result,
        )
        if seal is not None:
            if self._terminal_seal is None:
                self._terminal_seal = _deep_copy(seal)
            return
        if not _result_ok(result):
            return
        if tool_name == "request_human_input":
            self._queue_transition(
                "suspend_human",
                {
                    "tool_input": dict(tool_input)
                    if isinstance(tool_input, dict)
                    else {},
                },
            )
            return
        if tool_name == "load_tools":
            payload = _content_payload(result)
            loaded = payload.get("loaded_tool_ids")
            if loaded is None:
                loaded = payload.get("effective_tool_ids")
            tool_ids = (
                [str(item) for item in loaded if str(item).strip()]
                if isinstance(loaded, list)
                else []
            )
            if tool_ids:
                self._queue_transition(
                    "reload_tools",
                    {"tool_ids": tool_ids},
                )
            return
        if tool_name in {
            "task_tree_add_node",
            "task_tree_add_nodes",
        }:
            self._queue_transition("refresh_tree", {})
            return
        if tool_name == "task_tree_update_node":
            status = (
                str(tool_input.get("status") or "").strip().lower()
                if isinstance(tool_input, dict)
                else ""
            )
            if status in {"completed", "partial"}:
                self._queue_transition("refresh_tree", {})

    def _queue_transition(self, kind: str, payload: dict[str, Any]) -> None:
        identity = hashlib.sha256(
            (kind + "\n" + _json_text(payload)).encode("utf-8")
        ).hexdigest()
        existing = {
            hashlib.sha256((item_kind + "\n" + _json_text(item_payload)).encode("utf-8")).hexdigest()
            for item_kind, item_payload in self._transitions
        }
        if identity not in existing:
            self._transitions.append((kind, dict(payload)))

    def pop_ready_transition(self) -> tuple[str, dict[str, Any]] | None:
        with self._lock:
            if not self.is_closed():
                return None
            if self._terminal_seal is not None:
                return "terminal_complete", dict(self._terminal_seal)
            if not self._transitions:
                return None
            return self._transitions.pop(0)

    def is_closed(self) -> bool:
        with self._lock:
            return all(unit.closed for unit in self._units)

    def pending_calls(self) -> dict[str, str]:
        with self._lock:
            return {
                call_id: str(unit.calls[call_id].get("name") or "")
                for unit in self._units
                for call_id in unit.call_order
                if call_id not in unit.results
            }

    def completed_calls(self) -> list[dict[str, Any]]:
        """Return immutable projections for runtime/UI bookkeeping.

        The ToolMessage objects themselves remain the canonical LangChain
        protocol values.  Callers may inspect them, but must not append or
        rematch results using this projection.
        """
        with self._lock:
            return [
                {
                    "tool_call_id": call_id,
                    "tool_name": str(unit.calls[call_id].get("name") or ""),
                    "tool_input": _deep_copy(unit.calls[call_id].get("args")),
                    "result": _copy_message(unit.results[call_id]),
                }
                for unit in self._units
                for call_id in unit.call_order
                if call_id in unit.results
            ]

    def snapshot_messages(self) -> list[BaseMessage]:
        with self._lock:
            messages: list[BaseMessage] = []
            for unit in self._units:
                messages.append(_copy_message(unit.message))
                for call_id in unit.call_order:
                    result = unit.results.get(call_id)
                    if result is not None:
                        messages.append(_copy_message(result))
            return messages

    def terminal_seal(self) -> dict[str, Any] | None:
        with self._lock:
            return (
                _deep_copy(self._terminal_seal)
                if self._terminal_seal is not None
                else None
            )

    def restore_terminal_seal(self, value: Any) -> None:
        """Restore one immutable Server completion authority.

        This is deliberately independent of transcript reconstruction: a
        compact checkpoint may have removed the original closed TaskComplete
        protocol unit while retaining its exact Server receipt privately.
        """

        seal = _validated_restored_terminal_seal(value)
        if seal is None:
            raise RuntimeError("CANONICAL_PROTOCOL_RESTORED_TERMINAL_SEAL_INVALID")
        with self._lock:
            if self._terminal_seal is None:
                self._terminal_seal = seal
                return
            if _json_text(self._terminal_seal) != _json_text(seal):
                raise RuntimeError(
                    "CANONICAL_PROTOCOL_CONFLICTING_RESTORED_TERMINAL_SEAL"
                )


class CanonicalProtocolMiddleware(AgentMiddleware):
    """Outermost middleware that commits every handler outcome exactly once."""

    def __init__(self, ledger: CanonicalProtocolLedger) -> None:
        self._ledger = ledger

    @staticmethod
    def _state_messages(state: Any) -> list[Any]:
        messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
        return list(messages) if isinstance(messages, (list, tuple)) else []

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        del state, runtime
        transition = self._ledger.pop_ready_transition()
        if transition is not None:
            raise CanonicalRuntimeTransition(*transition)
        return None

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        del runtime
        messages = self._state_messages(state)
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                self._ledger.observe_model_message(message)
                invalid_ids = {
                    str(call.get("id") or "").strip()
                    for call in list(message.invalid_tool_calls or [])
                    if isinstance(call, dict)
                    and str(call.get("id") or "").strip()
                }
                if invalid_ids:
                    repairs = [
                        row["result"]
                        for row in self._ledger.completed_calls()
                        if str(row.get("tool_call_id") or "") in invalid_ids
                    ]
                    if repairs:
                        return {"messages": [message, *repairs]}
                break
        return None

    @staticmethod
    def _call(request: ToolCallRequest) -> tuple[str, str, Any]:
        call = request.tool_call if isinstance(request.tool_call, dict) else {}
        return (
            str(call.get("id") or "").strip(),
            str(call.get("name") or "").strip(),
            call.get("args"),
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        call_id, tool_name, tool_input = self._call(request)
        self._ledger.validate_tool_request(
            call_id=call_id,
            tool_name=tool_name,
            tool_input=tool_input,
        )
        try:
            result = handler(request)
        except Exception as exc:
            return self._ledger.commit_tool_error(
                call_id=call_id,
                tool_name=tool_name,
                tool_input=tool_input,
                error=exc,
            )
        self._ledger.commit_tool_result(
            call_id=call_id,
            tool_name=tool_name,
            tool_input=tool_input,
            result=result,
        )
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        call_id, tool_name, tool_input = self._call(request)
        self._ledger.validate_tool_request(
            call_id=call_id,
            tool_name=tool_name,
            tool_input=tool_input,
        )
        try:
            result = await handler(request)
        except Exception as exc:
            return self._ledger.commit_tool_error(
                call_id=call_id,
                tool_name=tool_name,
                tool_input=tool_input,
                error=exc,
            )
        self._ledger.commit_tool_result(
            call_id=call_id,
            tool_name=tool_name,
            tool_input=tool_input,
            result=result,
        )
        return result
