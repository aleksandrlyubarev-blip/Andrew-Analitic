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
from core.tools.python_exec import PythonExecInput, PythonExecTool
from core.tools.sql_query import SQLQueryInput, SQLQueryTool


def build_default_tool_registry():
    return {
        "sql_query": SQLQueryTool(),
        "python_exec": PythonExecTool(),
        "file_read": FileReadTool(),
    }


__all__ = [
    "AbstractTool",
    "PermissionResult",
    "ToolResult",
    "ToolUseContext",
    "ValidationResult",
    "FileReadInput",
    "FileReadTool",
    "PythonExecInput",
    "PythonExecTool",
    "SQLQueryInput",
    "SQLQueryTool",
    "build_default_tool_registry",
]
