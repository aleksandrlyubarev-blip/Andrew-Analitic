"""
Concurrent tool orchestration primitives inspired by the Claude Code batching pattern.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator

from pydantic import BaseModel

from core.tools.base import AbstractTool, ToolResult, ToolUseContext

MAX_CONCURRENCY = 10


@dataclass
class ToolCall:
    name: str
    input: BaseModel | dict[str, Any]
    call_id: str | None = None


@dataclass
class ToolBatch:
    is_concurrent: bool
    tool_calls: list[ToolCall]


@dataclass
class ToolExecution:
    call: ToolCall
    result: ToolResult[Any]


def partition_tool_calls(
    calls: list[ToolCall],
    tools: dict[str, AbstractTool[Any, Any]],
) -> list[ToolBatch]:
    """
    Partition tool calls into concurrent read-only batches and serial writes.

    Unknown tools, schema-invalid inputs, and non-concurrency-safe tools are
    treated conservatively as serial calls.
    """

    batches: list[ToolBatch] = []
    current_concurrent: list[ToolCall] = []

    for call in calls:
        tool = tools.get(call.name)
        can_run_concurrently = False

        if tool is not None:
            try:
                parsed = tool.parse_input(call.input)
            except Exception:
                parsed = None
            if parsed is not None:
                can_run_concurrently = tool.is_concurrency_safe(parsed)

        if can_run_concurrently:
            current_concurrent.append(call)
            continue

        if current_concurrent:
            batches.append(ToolBatch(is_concurrent=True, tool_calls=current_concurrent))
            current_concurrent = []
        batches.append(ToolBatch(is_concurrent=False, tool_calls=[call]))

    if current_concurrent:
        batches.append(ToolBatch(is_concurrent=True, tool_calls=current_concurrent))

    return batches


async def _execute_single(
    call: ToolCall,
    tools: dict[str, AbstractTool[Any, Any]],
    context: ToolUseContext,
) -> ToolExecution:
    tool = tools.get(call.name)
    if tool is None:
        return ToolExecution(
            call=call,
            result=ToolResult(
                error=f"Unknown tool: {call.name}",
                metadata={"tool_name": call.name, "call_id": call.call_id},
            ),
        )

    result = await tool.run(call.input, context)
    result.metadata.setdefault("call_id", call.call_id)
    result.metadata.setdefault("tool_name", call.name)
    return ToolExecution(call=call, result=result)


async def run_tools(
    calls: list[ToolCall],
    tools: dict[str, AbstractTool[Any, Any]],
    context: ToolUseContext,
) -> AsyncIterator[ToolExecution]:
    """
    Execute tool calls with Claude-style batching semantics.

    - consecutive concurrency-safe reads run in parallel
    - writes or unsafe calls run serially
    - yielded order matches the order the LLM requested, not completion order
    """

    for batch in partition_tool_calls(calls, tools):
        if batch.is_concurrent:
            semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

            async def run_one(call: ToolCall) -> ToolExecution:
                async with semaphore:
                    return await _execute_single(call, tools, context)

            results = await asyncio.gather(*(run_one(call) for call in batch.tool_calls))
            for execution in results:
                yield execution
        else:
            for call in batch.tool_calls:
                yield await _execute_single(call, tools, context)
