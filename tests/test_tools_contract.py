"""
Tool contract tests for the new Andrew tool abstraction layer.
"""
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.tools import FileReadTool, PythonExecTool, SQLQueryTool, ToolUseContext


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


@pytest.mark.asyncio
async def test_sql_tool_schema_validation_rejects_empty_query():
    tool = SQLQueryTool()
    ctx = ToolUseContext(schema_context={"sales": {"id": "int"}})

    result = await tool.run({}, ctx)

    assert result.success is False
    assert "Input validation failed" in (result.error or "")


@pytest.mark.asyncio
async def test_sql_tool_permission_blocks_destructive_query():
    tool = SQLQueryTool()
    ctx = ToolUseContext(schema_context={"sales": {"id": "int"}})

    result = await tool.run({"query": "DROP TABLE sales"}, ctx)

    assert result.success is False
    assert "Security block" in (result.error or "")


@pytest.mark.asyncio
async def test_sql_tool_call_happy_path_returns_rows():
    db_url, schema = _sqlite_fixture()
    tool = SQLQueryTool()
    ctx = ToolUseContext(db_url=db_url, schema_context=schema)

    result = await tool.run(
        {
            "query": (
                "SELECT sales.id, sales.region, sales.revenue "
                "FROM sales ORDER BY sales.id"
            )
        },
        ctx,
    )

    assert result.success is True
    assert result.output is not None
    assert len(result.output) == 3
    assert result.metadata["row_count"] == 3
    assert "EU" in (result.output_to_model or "")


@pytest.mark.asyncio
async def test_python_exec_tool_blocks_network_import():
    tool = PythonExecTool()
    ctx = ToolUseContext()

    result = await tool.run({"code": "import requests\nprint('x')"}, ctx)

    assert result.success is False
    assert "Network access" in (result.error or "")


@pytest.mark.asyncio
async def test_python_exec_tool_happy_path_executes_in_sandbox():
    tool = PythonExecTool()
    ctx = ToolUseContext()
    code = (
        "result = {'rows': int(len(df)), 'total': int(df['value'].sum())}\n"
        "print(json.dumps(result))"
    )

    result = await tool.run(
        {
            "code": code,
            "data": [{"value": 2}, {"value": 3}, {"value": 5}],
        },
        ctx,
    )

    assert result.success is True
    assert result.output is not None
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["rows"] == 3
    assert payload["total"] == 10


@pytest.mark.asyncio
async def test_file_read_tool_happy_path_reads_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "notes.txt"
        file_path.write_text("alpha\nbeta\ngamma", encoding="utf-8")
        tool = FileReadTool()
        ctx = ToolUseContext(working_directory=tmpdir)

        result = await tool.run({"path": "notes.txt"}, ctx)

        assert result.success is True
        assert result.output == "alpha\nbeta\ngamma"
        assert result.metadata["truncated"] is False


@pytest.mark.asyncio
async def test_file_read_tool_blocks_escape_outside_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        outside = Path(tmpdir).parent / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        tool = FileReadTool()
        ctx = ToolUseContext(working_directory=tmpdir)

        result = await tool.run({"path": str(outside)}, ctx)

        assert result.success is False
        assert "outside working directory" in (result.error or "")


@pytest.mark.asyncio
async def test_tool_prompts_are_non_empty_and_specific():
    sql_prompt = await SQLQueryTool().prompt()
    py_prompt = await PythonExecTool().prompt()
    file_prompt = await FileReadTool().prompt()

    assert "read-only SQL" in sql_prompt
    assert "sandbox" in py_prompt
    assert "read-only" in file_prompt
