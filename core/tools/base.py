"""
Foundational tool contract for Andrew Swarm.

This module introduces a reusable abstraction layer for tools without forcing
an immediate rewrite of the existing LangGraph pipeline. The goal is to make
new orchestration, permissions, and agent work land on a stable contract while
the current engine remains operational.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, Optional, TypeVar

from pydantic import BaseModel, Field, ValidationError

TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput")


class ToolResult(BaseModel, Generic[TOutput]):
    """Structured tool output passed back to orchestration and, later, the LLM."""

    output: Optional[TOutput] = None
    error: str | None = None
    output_to_model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None


class PermissionResult(BaseModel):
    behavior: str = "allow"  # allow | deny | ask
    reason: str = ""
    updated_input: dict[str, Any] | None = None


class ValidationResult(BaseModel):
    ok: bool = True
    reason: str = ""


@dataclass
class ToolUseContext:
    """
    Minimal execution context shared across tools.

    This stays intentionally small so it can be extended later for agents,
    permissions, memory, and Moltis integration without destabilizing the
    existing core.
    """

    db_url: str = ""
    schema_context: dict[str, dict[str, str]] = field(default_factory=dict)
    working_directory: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    available_tools: dict[str, Any] = field(default_factory=dict, repr=False)


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    omitted = len(text) - max_chars
    return (
        text[:half]
        + f"\n\n... [{omitted} chars truncated] ...\n\n"
        + text[-half:]
    )


class AbstractTool(ABC, Generic[TInput, TOutput]):
    """
    Python analogue of a buildTool()-style contract.

    Concrete tools implement:
    - input schema parsing
    - prompt description
    - permission hook
    - business validation
    - execution
    """

    name: str = "abstract_tool"
    max_result_size_chars: int = 100_000

    @abstractmethod
    def input_schema(self) -> type[TInput]:
        raise NotImplementedError

    @abstractmethod
    async def call(self, args: TInput, context: ToolUseContext) -> ToolResult[TOutput]:
        raise NotImplementedError

    @abstractmethod
    async def prompt(self) -> str:
        raise NotImplementedError

    def parse_input(self, raw_args: TInput | Mapping[str, Any]) -> TInput:
        schema = self.input_schema()
        if isinstance(raw_args, schema):
            return raw_args
        if isinstance(raw_args, BaseModel):
            raw_args = raw_args.model_dump()
        return schema.model_validate(raw_args)

    async def check_permissions(
        self, args: TInput, ctx: ToolUseContext
    ) -> PermissionResult:
        return PermissionResult(behavior="allow")

    async def validate_input(
        self, args: TInput, ctx: ToolUseContext
    ) -> ValidationResult:
        return ValidationResult(ok=True)

    def is_read_only(self, args: TInput) -> bool:
        return False

    def is_concurrency_safe(self, args: TInput) -> bool:
        return False

    def is_destructive(self, args: TInput) -> bool:
        return False

    def to_classifier_input(self, args: TInput) -> str:
        return json.dumps(args.model_dump(), default=str, ensure_ascii=False)

    async def run(
        self,
        raw_args: TInput | Mapping[str, Any],
        context: ToolUseContext,
    ) -> ToolResult[TOutput]:
        """
        Full lifecycle runner used by orchestration.

        This centralizes schema parsing, permission checks, business validation,
        and result truncation so each concrete tool only owns its core behavior.
        """

        try:
            args = self.parse_input(raw_args)
        except ValidationError as exc:
            return ToolResult(
                error=f"Input validation failed: {exc.errors()}",
                metadata={"stage": "input_schema", "tool_name": self.name},
            )

        permission = await self.check_permissions(args, context)
        if permission.updated_input:
            try:
                args = self.parse_input(permission.updated_input)
            except ValidationError as exc:
                return ToolResult(
                    error=f"Permission hook returned invalid input: {exc.errors()}",
                    metadata={"stage": "check_permissions", "tool_name": self.name},
                )

        if permission.behavior == "deny":
            return ToolResult(
                error=permission.reason or "Permission denied",
                metadata={"stage": "check_permissions", "tool_name": self.name},
            )

        if permission.behavior == "ask":
            return ToolResult(
                error=permission.reason or "Permission escalation required",
                metadata={
                    "stage": "check_permissions",
                    "tool_name": self.name,
                    "permission_behavior": "ask",
                },
            )

        validation = await self.validate_input(args, context)
        if not validation.ok:
            return ToolResult(
                error=validation.reason or "Business validation failed",
                metadata={"stage": "validate_input", "tool_name": self.name},
            )

        result = await self.call(args, context)
        result.metadata.setdefault("tool_name", self.name)
        result.output_to_model = self._prepare_output_to_model(result)
        return result

    def _prepare_output_to_model(self, result: ToolResult[TOutput]) -> str | None:
        if result.output_to_model is None:
            if result.output is None:
                return None
            try:
                result.output_to_model = json.dumps(
                    result.output, default=str, ensure_ascii=False
                )
            except TypeError:
                result.output_to_model = str(result.output)

        if result.output_to_model is None:
            return None
        return truncate_text(result.output_to_model, self.max_result_size_chars)
