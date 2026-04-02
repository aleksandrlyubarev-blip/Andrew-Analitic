"""
Tool orchestration tests for concurrent and serial batching behavior.
"""
import asyncio
import os
import sys
import time

import pytest
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.orchestration import ToolCall, partition_tool_calls, run_tools
from core.tools.base import AbstractTool, ToolResult, ToolUseContext


class DummyInput(BaseModel):
    label: str
    delay: float = Field(default=0.0, ge=0.0)


class DummyTool(AbstractTool[DummyInput, str]):
    def __init__(self, name: str, *, concurrent: bool):
        self.name = name
        self._concurrent = concurrent

    def input_schema(self) -> type[DummyInput]:
        return DummyInput

    async def prompt(self) -> str:
        return f"Dummy prompt for {self.name}"

    async def call(self, args: DummyInput, context: ToolUseContext) -> ToolResult[str]:
        await asyncio.sleep(args.delay)
        return ToolResult(output=args.label, output_to_model=args.label)

    def is_read_only(self, args: DummyInput) -> bool:
        return self._concurrent

    def is_concurrency_safe(self, args: DummyInput) -> bool:
        return self._concurrent


def _registry() -> dict[str, DummyTool]:
    return {
        "read": DummyTool("read", concurrent=True),
        "write": DummyTool("write", concurrent=False),
    }


def test_partition_tool_calls_groups_read_batches_around_writes():
    tools = _registry()
    calls = [
        ToolCall(name="read", input={"label": "r1"}),
        ToolCall(name="read", input={"label": "r2"}),
        ToolCall(name="write", input={"label": "w1"}),
        ToolCall(name="read", input={"label": "r3"}),
        ToolCall(name="read", input={"label": "r4"}),
    ]

    batches = partition_tool_calls(calls, tools)

    assert [batch.is_concurrent for batch in batches] == [True, False, True]
    assert [len(batch.tool_calls) for batch in batches] == [2, 1, 2]


@pytest.mark.asyncio
async def test_run_tools_preserves_llm_request_order():
    tools = _registry()
    calls = [
        ToolCall(name="read", input={"label": "first", "delay": 0.12}, call_id="1"),
        ToolCall(name="read", input={"label": "second", "delay": 0.01}, call_id="2"),
        ToolCall(name="write", input={"label": "third", "delay": 0.01}, call_id="3"),
        ToolCall(name="read", input={"label": "fourth", "delay": 0.02}, call_id="4"),
    ]

    observed = []
    async for execution in run_tools(calls, tools, ToolUseContext()):
        observed.append((execution.call.call_id, execution.result.output))

    assert observed == [
        ("1", "first"),
        ("2", "second"),
        ("3", "third"),
        ("4", "fourth"),
    ]


@pytest.mark.asyncio
async def test_concurrent_reads_are_faster_than_serial_execution():
    tools = _registry()
    calls = [
        ToolCall(name="read", input={"label": "a", "delay": 0.15}),
        ToolCall(name="read", input={"label": "b", "delay": 0.15}),
        ToolCall(name="read", input={"label": "c", "delay": 0.15}),
    ]

    started = time.perf_counter()
    results = [execution async for execution in run_tools(calls, tools, ToolUseContext())]
    elapsed = time.perf_counter() - started

    assert len(results) == 3
    assert elapsed < 0.30, f"expected concurrent batch, got elapsed={elapsed:.3f}s"


@pytest.mark.asyncio
async def test_mixed_batches_match_target_pattern():
    tools = _registry()
    calls = [
        *[
            ToolCall(name="read", input={"label": f"r{i}", "delay": 0.01}, call_id=f"r{i}")
            for i in range(5)
        ],
        ToolCall(name="write", input={"label": "w", "delay": 0.01}, call_id="w"),
        *[
            ToolCall(name="read", input={"label": f"x{i}", "delay": 0.01}, call_id=f"x{i}")
            for i in range(3)
        ],
    ]

    batches = partition_tool_calls(calls, tools)
    assert [len(batch.tool_calls) for batch in batches] == [5, 1, 3]
    assert [batch.is_concurrent for batch in batches] == [True, False, True]

    ordered_ids = []
    async for execution in run_tools(calls, tools, ToolUseContext()):
        ordered_ids.append(execution.call.call_id)

    assert ordered_ids == ["r0", "r1", "r2", "r3", "r4", "w", "x0", "x1", "x2"]


@pytest.mark.asyncio
async def test_unknown_tool_returns_structured_error():
    tools = _registry()
    calls = [ToolCall(name="missing", input={"label": "x"}, call_id="missing-1")]

    results = [execution async for execution in run_tools(calls, tools, ToolUseContext())]

    assert len(results) == 1
    assert results[0].result.success is False
    assert "Unknown tool" in (results[0].result.error or "")
