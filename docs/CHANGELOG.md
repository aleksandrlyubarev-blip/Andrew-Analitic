# Changelog

## v1.0.0-rc1 (2026-03-15) — Release Candidate

### Core Engine (andrew_swarm.py)
- Weighted 48-keyword query router with 3 lanes (reasoning_math, analytics_fastlane, standard)
- Model registry via environment variables — zero hardcoded vendor strings
- Provider parameter adapter (Grok, Claude, OpenAI each get correct params)
- SQL generation with sqlglot `qualify(schema=...)` validation
- Column-level schema verification (catches hallucinated tables and columns)
- Blocked keywords: DROP, DELETE, TRUNCATE, ALTER, GRANT, REVOKE, INSERT, UPDATE, CREATE, ATTACH, DETACH, COPY, PRAGMA, CALL, EXEC, EXECUTE
- Statement whitelist: only SELECT and WITH allowed
- Python AST safety analysis (blocks os, subprocess, socket, eval, exec, open)
- Data leakage detection (fit_transform before train_test_split)
- Pandera numerical validation on SQL results
- Semantic guardrails (revenue/monthly/top-N intent vs. actual SQL)
- Intent contract system (constrains what SQL/Python stages may do)
- LiteLLM-based cost tracking with $1.00 hard budget guard
- LangGraph native RetryPolicy on LLM and sandbox nodes
- SHA-256 state hash for deterministic replay
- Structured audit trail (every node logs stage + status + details)

### Moltis Bridge (moltis_bridge.py)
- FastAPI webhook server (port 8100) with /analyze, /webhook/moltis, /schedule, /health
- MoltisClient: async HTTP/GraphQL client for Moltis runtime
- Memory integration: recall past analyses before execution, store results after
- Channel-aware formatting (Telegram/Discord message length limits)
- Cron scheduling for recurring analytical reports
- Moltis Docker sandbox as E2B replacement (zero cloud cost)
- Hook configuration generator for Moltis MessageReceived events

### Deployment
- Docker Compose: Moltis + Andrew bridge + optional PostgreSQL
- Dockerfile for bridge container
- Local fallback (subprocess sandbox when E2B/Moltis unavailable)

### Tests
- 12 routing smoke tests
- 15 validation tests (SQL safety, Python AST, leakage detection)

### Known Limitations
- No formal adversarial fuzz testing yet
- No HITL escalation for low-confidence results
- Moltis GraphQL schema may differ from assumed queries (verify on your version)
- Romeo PhD integration not yet implemented

## Pre-release History

### Sprint 4 (v0.4)
- Weighted routing, model registry, provider adapters

### Sprint 3 (v0.3)
- sqlglot qualify, AST safety visitor, Pandera, semantic guardrails, audit trail

### Sprint 2 (v0.2)
- LiteLLM direct, E2B sandbox, retry loops, schema discovery

### Sprint 1 (v0.1)
- Initial LangGraph pipeline, ROMA executor bridge, multi-file architecture
