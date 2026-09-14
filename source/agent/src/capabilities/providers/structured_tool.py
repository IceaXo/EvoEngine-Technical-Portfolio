"""Invoke StructuredTool implementations without emitting nested tool events."""

from __future__ import annotations

import asyncio
from typing import Any


def normalize_structured_tool_arguments(tool: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is None or not hasattr(args_schema, "model_validate"):
        return dict(arguments)
    validated = args_schema.model_validate(dict(arguments))
    return validated.model_dump(mode="python")


async def invoke_structured_tool_function(
    tool: Any,
    arguments: dict[str, Any],
    *,
    timeout_seconds: int,
) -> Any:
    normalized = normalize_structured_tool_arguments(tool, arguments)
    raw_coroutine = getattr(tool, "coroutine", None)
    raw_func = getattr(tool, "func", None)
    if raw_coroutine is not None:
        return await asyncio.wait_for(
            raw_coroutine(**normalized),
            timeout=timeout_seconds,
        )
    if raw_func is not None:
        return await asyncio.wait_for(
            asyncio.to_thread(raw_func, **normalized),
            timeout=timeout_seconds,
        )
    raise TypeError("structured tool has no invoker")
