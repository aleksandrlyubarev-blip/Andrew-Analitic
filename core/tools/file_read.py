"""
Read-only file tool for agent workflows and prompt/tool context assembly.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from core.tools.base import (
    AbstractTool,
    PermissionResult,
    ToolResult,
    ToolUseContext,
    ValidationResult,
)


class FileReadInput(BaseModel):
    path: str = Field(min_length=1)
    max_chars: int = 20_000


class FileReadTool(AbstractTool[FileReadInput, str]):
    name = "file_read"
    max_result_size_chars = 20_000

    def input_schema(self) -> type[FileReadInput]:
        return FileReadInput

    async def prompt(self) -> str:
        return (
            "Read a text file from the project working directory. "
            "The tool is strictly read-only and rejects path traversal outside "
            "the configured workspace root."
        )

    def _resolve_path(self, args: FileReadInput, ctx: ToolUseContext) -> Path:
        requested = Path(args.path)
        if requested.is_absolute():
            return requested.resolve()
        base = Path(ctx.working_directory or ".").resolve()
        return (base / requested).resolve()

    async def check_permissions(
        self, args: FileReadInput, ctx: ToolUseContext
    ) -> PermissionResult:
        resolved = self._resolve_path(args, ctx)
        if ctx.working_directory:
            root = Path(ctx.working_directory).resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                return PermissionResult(
                    behavior="deny",
                    reason=f"File read outside working directory is not allowed: {resolved}",
                )
        return PermissionResult(behavior="allow")

    async def validate_input(
        self, args: FileReadInput, ctx: ToolUseContext
    ) -> ValidationResult:
        resolved = self._resolve_path(args, ctx)
        if not resolved.exists():
            return ValidationResult(ok=False, reason=f"File does not exist: {resolved}")
        if not resolved.is_file():
            return ValidationResult(ok=False, reason=f"Path is not a file: {resolved}")
        return ValidationResult(ok=True)

    def is_read_only(self, args: FileReadInput) -> bool:
        return True

    def is_concurrency_safe(self, args: FileReadInput) -> bool:
        return True

    def to_classifier_input(self, args: FileReadInput) -> str:
        return args.path

    async def call(self, args: FileReadInput, context: ToolUseContext) -> ToolResult[str]:
        resolved = self._resolve_path(args, context)
        content = resolved.read_text(encoding="utf-8", errors="replace")
        limited = content[: args.max_chars]
        return ToolResult(
            output=limited,
            output_to_model=limited,
            metadata={"path": str(resolved), "truncated": len(content) > len(limited)},
        )
