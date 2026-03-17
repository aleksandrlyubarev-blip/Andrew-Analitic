# Changelog

## v1.4.0 (2026-03-17) — Sprint 9: Double Diamond Analytics Workflow

### Core Engine (`core/andrew_swarm.py`)
- New dataclasses: `ColumnProfile`, `TableProfile`, `DataProfile` — structured statistical profile of the database
- `_build_data_profile(db_url, schema)` — offline SQLAlchemy profiler:
  - Row count per table; quality flag `empty_table` when count = 0
  - Per-column: dtype, null rate, MIN/MAX/AVG (numeric), top-5 values (categorical)
  - Quality flags: `all_null`, `zero_variance`, `high_null_rate`
  - 50 000-row cap + 5-second SQLAlchemy pool timeout; returns `DataProfile(error=…)` on failure
- `_dataprofile_to_dict()` — serialises `DataProfile` to plain dict for state propagation
- `profile_schema(state)` node — **Phase 1 (Explore Data)**:
  - No-op when `db_url` or `schema_context` is absent
  - Merges profile warnings into `state["warnings"]`
  - Writes `data_profile` dict to state
- `hypothesis_gate(state)` node — **Phase 2 (Define Hypothesis)**:
  - No-op when `error_message` present or `data_profile` is None
  - Empty-table penalty: lowers `confidence` by 0.2 per empty table; appends warning
  - High-null-rate penalty: warns on any column with null_rate > 0.5 that appears in the user request
- `validate_results(state)` enhanced — **Phase 4 (Validate Results)**:
  - Empty DataFrame: lowers `confidence` by 0.2; appends `"0 rows returned"` warning
  - Zero-variance numeric column: appends `"zero variance in <col>"` warning
  - All-null column: appends `"<col> is entirely null"` warning
- LangGraph pipeline extended:
  ```
  START → profile_schema → route_query_intent → build_intent_contract
        → hypothesis_gate → generate_sql → … (existing) → validate_results → END
  ```

### Tests
- 25 new tests in `tests/test_double_diamond.py` (offline, SQLite in-process):
  - Dataclass defaults and `_dataprofile_to_dict` serialisation
  - `_build_data_profile`: row count, numeric stats, categorical top-values, empty table, all-null column, zero-variance, bad URL, empty schema
  - `profile_schema`: no-op paths, profile returned, warnings appended
  - `hypothesis_gate`: empty-table penalty, high-null warning, clean-data noop, skip on error/no profile
  - `validate_results`: empty DF, zero variance, all-null, clean data
- Total test count: **166 passing**

---

## v1.3.0 (2026-03-16) — Sprint 6: Security Hardening

### Bridge rate limiting (new)
- `SlidingWindowRateLimiter` — per-IP, per-endpoint sliding-window counter
- Three independent limiter instances configurable via env vars:
  - `RATE_LIMIT_ANALYZE` (default `10/60` — 10 req/min)
  - `RATE_LIMIT_EDUCATE` (default `20/60`)
  - `RATE_LIMIT_WEBHOOK` (default `30/60`)
- All three endpoints (`/analyze`, `/educate`, `/webhook/moltis`) enforce limit; return HTTP 429 + `Retry-After` header on excess
- Set limit to `0/N` to disable per-endpoint
- Thread-safe (RLock on per-key deque); no external dependency (Redis-free)
- `_parse_rate_spec`, `_build_limiter`, `_client_ip`, `_check_rate_limit` helpers
- 22 new tests in `tests/test_rate_limit.py`: allow/deny, window expiry, thread safety, env override, disabled limiter, per-key isolation

### CHANGELOG catchup: Sprints 6-8 (delivered 2026-03-15)

The items below were delivered in a prior session but not documented here.

#### Sprint 7 — HITL Escalation
- `hitl_escalate` LangGraph node fires when `confidence < HITL_CONFIDENCE_THRESHOLD`
- `POST /analyze` returns HTTP 202 (instead of 200) when `hitl_required=true`
- `HITL_CONFIDENCE_THRESHOLD` env var (default `0.35`)
- `AndrewResult.hitl_required`, `AndrewResult.hitl_reason` fields
- 13 tests in `tests/test_hitl.py`

#### Sprint 6 — Adversarial Test Suite (20 tests, 8 threat vectors)
- `tests/test_adversarial.py` — offline, no LLM calls
- Threat 1: hallucinated table → `validate_sql` rejects via sqlglot qualify
- Threat 2: hallucinated column → same
- Threat 3: destructive SQL → DROP/DELETE/TRUNCATE/INSERT/UPDATE/ALTER all blocked
- Threat 4: intent mismatch → `semantic_guardrails` fires, confidence drops
- Threat 5: data leakage → `fit_transform` before `train_test_split` flagged
- Threat 6: dangerous Python → os/subprocess/eval/exec/socket/open/__import__ blocked
- Threat 7: budget exhaustion → `_budget_ok` returns False at/above `MAX_COST_USD`
- Threat 8: prompt injection → adversarial payloads blocked or sanitised by validation layers

#### Sprint 8 — Romeo PhD + Vue 3 UI
- `core/romeo_phd.py` — `RomeoExecutor`, `RomeoResult`, educational agent for ML/stats
- `POST /educate` endpoint — returns Markdown-formatted explanations
- Vue 3 + Vite frontend in `frontend/` — split-view (chat + data panel)
- Multi-stage Dockerfile: Node builds UI → Python bridge serves `bridge/static/`
- `MODEL_ROMEO` env var (default `gpt-4o-mini`)

---

## v1.2.0 (2026-03-16) — Sprint 5: Memory Architecture

### Semantic Router (`core/semantic_router.py`)
- `CapabilityRecord` dataclass: agent_id, description, examples, cost_weight, version
- `CapabilityRegistry`: in-process store of per-agent capability records
- `SemanticRouter`: cosine + keyword + session context scoring with α/β/γ/δ weights
- `RoutingLog` dataclass: full decision audit (scores, fallback_used, model, token counts)
- `init_registry()`: embeds all capability descriptions at startup; keyword fallback when API unavailable
- `ProceduralRecord` / `ProceduralStore`: per-agent routing outcome history
  - `record()`: merges at cosine ≥ 0.95; inserts otherwise
  - `bias()`: weighted success-rate over top-k similar past queries (threshold 0.75)
- `SemanticRouter.record_outcome()`: fire-and-forget procedural feedback
- Routing score gains ε=0.10 × procedural_bias term (§5.4)

### Memory System (`core/memory.py`) — new
- `SemanticRecord`: content, embedding, ttl_days, tombstoned, stale_flagged
- `InProcessSemanticStore`: thread-safe in-process mirror of Moltis semantic store
  - dedup at cosine ≥ 0.92 (§5.4); merges content on collision
  - `search()` updates `last_accessed_at` (LRU touch)
  - `tombstone()` for hard deletion
- `SweepResult` dataclass
- `ConsolidationEngine`:
  - `consolidate_session()`: LLM summarises episodic window → embed → dedup-upsert
  - `staleness_sweep()`: two-pass (flag → tombstone after grace period)

### Bridge (`bridge/moltis_bridge.py`)
- `AndrewMoltisBridge.handle_query()`:
  - Episodic memory write on every turn
  - Semantic memory recall before execution
  - Procedural feedback fire-and-forget after execution
  - In-process semantic store mirror write after Moltis store
- `end_session()` + `POST /session/{id}/end` endpoint
- Daily staleness sweep asyncio background task (configurable via `STALENESS_SWEEP_INTERVAL_S`)

### Core (`core/andrew_swarm.py`)
- `AndrewExecutor.__init__()` calls `init_registry()` on startup
- `AndrewResult.routing_log` attribute exposes routing decision details

### Tests
- 28 new tests in `tests/test_memory.py`: consolidation, staleness sweep, dedup, procedural store
- 12 new tests in `tests/test_semantic_router.py`: scoring, registry, keyword fallback, routing log

---

## v1.0.0-rc1 (2026-03-15) — MVP

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

---

## Pre-release History

### Sprint 4 (v0.4)
- Weighted routing, model registry, provider adapters

### Sprint 3 (v0.3)
- sqlglot qualify, AST safety visitor, Pandera, semantic guardrails, audit trail

### Sprint 2 (v0.2)
- LiteLLM direct, E2B sandbox, retry loops, schema discovery

### Sprint 1 (v0.1)
- Initial LangGraph pipeline, ROMA executor bridge, multi-file architecture
