"""
Python execution tool wrapping the existing AST safety and sandbox flow.
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from core.andrew_swarm import sandbox_execute, validate_python_static
from core.tools.base import (
    AbstractTool,
    PermissionResult,
    ToolResult,
    ToolUseContext,
    ValidationResult,
)

_NETWORK_PATTERNS = (
    re.compile(r"\bimport\s+requests\b"),
    re.compile(r"\bfrom\s+requests\b"),
    re.compile(r"\bimport\s+urllib\b"),
    re.compile(r"\bfrom\s+urllib\b"),
)


class PythonExecInput(BaseModel):
    code: str = Field(min_length=1)
    data: list[dict[str, Any]] = Field(default_factory=list)


class PythonExecTool(AbstractTool[PythonExecInput, str]):
    name = "python_exec"
    max_result_size_chars = 60_000

    def input_schema(self) -> type[PythonExecInput]:
        return PythonExecInput

    async def prompt(self) -> str:
        return (
            "Execute analyst-generated Python code inside the sandbox. "
            "The code is validated with AST safety checks before execution and "
            "receives tabular input as pandas DataFrame `df`."
        )

    async def check_permissions(
        self, args: PythonExecInput, ctx: ToolUseContext
    ) -> PermissionResult:
        lowered = args.code.lower()
        for pattern in _NETWORK_PATTERNS:
            if pattern.search(lowered):
                return PermissionResult(
                    behavior="deny",
                    reason="Network access is not allowed in python_exec",
                )
        return PermissionResult(behavior="allow")

    async def validate_input(
        self, args: PythonExecInput, ctx: ToolUseContext
    ) -> ValidationResult:
        state = {
            "python_code": args.code,
            "confidence": 0.5,
            "intent_contract": {"must_not_train_model_unless_requested": False},
            "warnings": [],
            "audit_log": [],
            "error_message": "",
        }
        result = validate_python_static(state)
        if result.get("error_message"):
            return ValidationResult(ok=False, reason=result["error_message"])
        return ValidationResult(ok=True)

    def is_read_only(self, args: PythonExecInput) -> bool:
        return False

    def is_concurrency_safe(self, args: PythonExecInput) -> bool:
        return False

    def to_classifier_input(self, args: PythonExecInput) -> str:
        return args.code

    async def call(
        self, args: PythonExecInput, context: ToolUseContext
    ) -> ToolResult[str]:
        state = {
            "python_code": args.code,
            "query_results": args.data,
            "warnings": [],
            "audit_log": [],
            "error_message": "",
        }

        validated = validate_python_static(state)
        if validated.get("error_message"):
            return ToolResult(
                error=validated["error_message"],
                metadata={"stage": "validate_python_static"},
            )

        executed = sandbox_execute(state)
        if executed.get("error_message"):
            return ToolResult(
                error=executed["error_message"],
                metadata={"stage": "sandbox_execute"},
            )

        output = executed.get("sandbox_output", "")
        return ToolResult(
            output=output,
            output_to_model=output,
            metadata={"warnings": state.get("warnings", [])},
        )
