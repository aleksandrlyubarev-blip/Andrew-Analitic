"""
SQL query tool built on top of the existing Andrew validation + execution flow.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from core.andrew_swarm import (
    BLOCKED_SQL_KEYWORDS,
    execute_sql_load_df,
    validate_sql,
)
from core.tools.base import (
    AbstractTool,
    PermissionResult,
    ToolResult,
    ToolUseContext,
    ValidationResult,
)


class SQLQueryInput(BaseModel):
    query: str = Field(min_length=1)


class SQLQueryTool(AbstractTool[SQLQueryInput, list[dict[str, Any]]]):
    name = "sql_query"
    max_result_size_chars = 40_000

    def input_schema(self) -> type[SQLQueryInput]:
        return SQLQueryInput

    async def prompt(self) -> str:
        return (
            "Execute a read-only SQL query against the configured analytics database. "
            "Only SELECT/WITH style reads are permitted. Validate against the live "
            "schema before execution and return a compact preview to the model."
        )

    async def check_permissions(
        self, args: SQLQueryInput, ctx: ToolUseContext
    ) -> PermissionResult:
        upper = args.query.upper()
        for keyword in BLOCKED_SQL_KEYWORDS:
            if keyword in upper:
                return PermissionResult(
                    behavior="deny",
                    reason=f"Security block: '{keyword}'",
                )
        return PermissionResult(behavior="allow")

    async def validate_input(
        self, args: SQLQueryInput, ctx: ToolUseContext
    ) -> ValidationResult:
        if not ctx.schema_context:
            return ValidationResult(ok=False, reason="Schema context missing")

        state = {
            "sql_query": args.query,
            "schema_context": ctx.schema_context,
            "intent_contract": {"allowed_tables": list(ctx.schema_context.keys())},
            "warnings": [],
            "audit_log": [],
        }
        result = validate_sql(state)
        if result.get("error_message"):
            return ValidationResult(ok=False, reason=result["error_message"])
        return ValidationResult(ok=True)

    def is_read_only(self, args: SQLQueryInput) -> bool:
        stripped = args.query.lstrip().upper()
        return stripped.startswith("SELECT") or stripped.startswith("WITH")

    def is_concurrency_safe(self, args: SQLQueryInput) -> bool:
        return self.is_read_only(args)

    def to_classifier_input(self, args: SQLQueryInput) -> str:
        return args.query

    async def call(
        self, args: SQLQueryInput, context: ToolUseContext
    ) -> ToolResult[list[dict[str, Any]]]:
        state = {
            "sql_query": args.query,
            "schema_context": context.schema_context,
            "intent_contract": {"allowed_tables": list(context.schema_context.keys())},
            "warnings": [],
            "audit_log": [],
            "db_url": context.db_url,
            "error_message": "",
        }

        validated = validate_sql(state)
        if validated.get("error_message"):
            return ToolResult(
                error=validated["error_message"],
                metadata={"stage": "validate_sql"},
            )

        executed = execute_sql_load_df(state)
        if executed.get("error_message"):
            return ToolResult(
                error=executed["error_message"],
                metadata={"stage": "execute_sql"},
            )

        rows = executed.get("query_results", [])
        preview = json.dumps(rows[:20], default=str, ensure_ascii=False)
        return ToolResult(
            output=rows,
            output_to_model=preview,
            metadata={
                "row_count": len(rows),
                "sql_result_path": executed.get("sql_result_path"),
                "warnings": state.get("warnings", []),
            },
        )
