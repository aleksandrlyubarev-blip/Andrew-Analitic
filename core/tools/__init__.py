"""
Reusable tool primitives for Andrew Swarm.
"""
from core.tools.base import (
    AbstractTool,
    PermissionResult,
    ToolResult,
    ToolUseContext,
    ValidationResult,
)
from core.tools.file_read import FileReadInput, FileReadTool
from core.tools.ltx_generate import LtxGenerateTool, make_ltx_generate_tool
from core.tools.python_exec import PythonExecInput, PythonExecTool
from core.tools.sql_query import SQLQueryInput, SQLQueryTool


def build_default_tool_registry():
    return {
        "sql_query": SQLQueryTool(),
        "python_exec": PythonExecTool(),
        "file_read": FileReadTool(),
        "ltx_generate": make_ltx_generate_tool(),
    }


__all__ = [
    "AbstractTool",
    "PermissionResult",
    "ToolResult",
    "ToolUseContext",
    "ValidationResult",
    "FileReadInput",
    "FileReadTool",
    "LtxGenerateTool",
    "make_ltx_generate_tool",
    "PythonExecInput",
    "PythonExecTool",
    "SQLQueryInput",
    "SQLQueryTool",
    "build_default_tool_registry",
]
