"""
Runtime integration tests for executor/supervisor access to the new tool layer.
"""
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.andrew_swarm import AndrewExecutor
from core.orchestration import ToolCall
from core.supervisor import SwarmSupervisor


def _sqlite_fixture() -> tuple[str, dict[str, dict[str, str]]]:
    fd, raw_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    path = Path(raw_path)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE sales (id INTEGER, region TEXT, revenue REAL, quantity INTEGER)"
        )
        conn.executemany(
            "INSERT INTO sales VALUES (?, ?, ?, ?)",
            [
                (1, "US", 100.0, 1),
                (2, "EU", 80.0, 1),
                (3, "US", 20.0, 2),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    schema = {
        "sales": {
            "id": "int",
            "region": "text",
            "revenue": "float",
            "quantity": "int",
        }
    }
    return f"sqlite:///{path.as_posix()}", schema


def test_andrew_executor_exposes_foundation_tools():
    executor = AndrewExecutor()

    assert executor.available_tools() == ["file_read", "python_exec", "sql_query"]


def test_andrew_executor_run_tool_calls_sync_executes_runtime_plan():
    db_url, schema = _sqlite_fixture()
    executor = AndrewExecutor(db_url=db_url)
    executor._schema = schema

    with tempfile.TemporaryDirectory() as tmpdir:
        notes = Path(tmpdir) / "notes.txt"
        notes.write_text("pilot\nfactory", encoding="utf-8")

        executions = executor.run_tool_calls_sync(
            [
                ToolCall(name="file_read", input={"path": "notes.txt"}, call_id="file-1"),
                ToolCall(
                    name="sql_query",
                    input={"query": "SELECT sales.id, sales.region FROM sales ORDER BY sales.id"},
                    call_id="sql-1",
                ),
                ToolCall(
                    name="python_exec",
                    input={
                        "code": "result = {'rows': int(len(df))}\nprint(json.dumps(result))",
                        "data": [{"value": 1}, {"value": 2}],
                    },
                    call_id="py-1",
                ),
            ],
            working_directory=tmpdir,
            metadata={"source": "test"},
        )

    assert [execution.call.call_id for execution in executions] == ["file-1", "sql-1", "py-1"]
    assert executions[0].result.output == "pilot\nfactory"
    assert executions[1].result.metadata["row_count"] == 3
    assert executions[2].result.output is not None
    assert '"rows": 2' in executions[2].result.output


@pytest.mark.asyncio
async def test_andrew_executor_sync_runtime_method_rejects_active_event_loop():
    executor = AndrewExecutor()

    with pytest.raises(RuntimeError):
        executor.run_tool_calls_sync([])


def test_supervisor_delegates_tool_runtime_to_andrew_executor():
    db_url, schema = _sqlite_fixture()
    supervisor = SwarmSupervisor(db_url=db_url)
    supervisor._schema = schema

    with tempfile.TemporaryDirectory() as tmpdir:
        notes = Path(tmpdir) / "notes.txt"
        notes.write_text("swarm", encoding="utf-8")

        executions = supervisor.run_tool_calls_sync(
            [ToolCall(name="file_read", input={"path": "notes.txt"}, call_id="file-1")],
            working_directory=tmpdir,
        )

    assert supervisor.available_tools() == ["file_read", "python_exec", "sql_query"]
    assert executions[0].result.output == "swarm"
