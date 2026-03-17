# Architecture

## System Overview

Andrew Swarm is a two-layer system: a Python analytical brain (LangGraph) and a Rust delivery runtime (Moltis).

```
User (Telegram / Discord / Web UI / API)
         |
    [Moltis Runtime — Rust, port 13131]
    Channels | Memory | Sandbox | Cron
         |
    [Andrew Bridge — Python, port 8100]
    FastAPI | Rate limiting | Memory pipeline | Query enrichment | Formatting
         |
         +──────────────────+
         |                  |
    [Andrew Core]      [Romeo PhD]
    LangGraph          LiteLLM
    SQL → Python       Educational
    analytics          explanations
```

## Core Engine Pipeline

```
START
  -> route_query_intent      [SemanticRouter: cosine + keyword + procedural bias]
  -> build_intent_contract   [constrain allowed tables/ops/metrics]
  -> generate_sql            [LLM, model from router] [retry_policy: 3x]
  -> validate_sql            [sqlglot qualify + schema + blocklist]
  -> [error? -> END]
  -> execute_sql_load_df     [SQLAlchemy, results to CSV]
  -> generate_python         [LLM, model from router] [retry_policy: 3x]
  -> validate_python_static  [AST safety + leakage detection]
  -> [error? -> END]
  -> sandbox_execute          [Moltis Docker / E2B / subprocess] [retry_policy: 2x]
  -> [error? -> END]
  -> validate_results         [Pandera + completeness checks]
  -> semantic_guardrails      [intent-to-query alignment]
  -> hitl_escalate            [fires when confidence < HITL_CONFIDENCE_THRESHOLD]
  -> finalize_state           [SHA-256 hash + audit log]
  -> END
```

## Routing Lanes

| Lane | Score Threshold | Trigger Terms | Default Model |
|---|---|---|---|
| reasoning_math | >= 4 or heavy ML term | ARIMA, neural network, Monte Carlo, regression | grok-4 |
| analytics_fastlane | any hit + light analytics | average + month, CAGR + region | gpt-4o-mini |
| standard | no math keywords | bar chart, group by, show | claude-sonnet |

### Semantic Router (`core/semantic_router.py`)

Replaces the original keyword-only router with a scored combination:

```
score(agent) = α × cosine(query, capability_description)
             + β × max_cosine(query, capability_examples)
             + γ × cost_weight(agent)            [inverse: cheaper = higher]
             + δ × cosine(query, session_context) [0 when no session]
             + ε × procedural_bias(query, agent)  [learned routing prior]
```

Weights: α=0.30, β=0.60, γ=0.10, δ=0.15, ε=0.10 (all tunable).

Falls back to keyword scoring when the embedding API is unavailable.
Falls back to LLM classification when cosine score is below threshold (default 0.72).

**Procedural memory (§5.1):** `ProceduralStore` records past `query→agent_id` routing
outcomes. After each successful/failed execution the bridge records the outcome via
`record_outcome()`. The router applies the learned success-rate as a bias term (ε).

## Memory Architecture

Three memory tiers (§5.1):

| Tier | Store | Lifetime | Consumer |
|---|---|---|---|
| Episodic | `_episodic_store` (in-process dict) | Session | Bridge query enrichment |
| Semantic | `InProcessSemanticStore` + Moltis backend | 90 days (TTL) | Bridge + consolidation |
| Procedural | `ProceduralStore` (in-process) | Process lifetime | SemanticRouter only |

### Consolidation (`core/memory.py`)

- **`consolidate_session()`** — called at `POST /session/{id}/end`:
  1. LLM summarises episodic window (concat fallback when API unavailable)
  2. Embeds the summary
  3. Cosine dedup against existing semantic records (threshold 0.92): update or insert
- **`staleness_sweep()`** — daily asyncio background task:
  - Pass 1: flag records idle beyond TTL (`stale_flagged = True`)
  - Pass 2 (next run): tombstone records that were flagged and past grace period

## Validation Layers (no LLM)

1. **SQL blocklist:** 16 dangerous keywords (DROP, DELETE, TRUNCATE …)
2. **Statement whitelist:** only SELECT/WITH
3. **sqlglot qualify:** resolves table.column against real schema
4. **Post-qualify audit:** every table and column verified
5. **Intent contract check:** unauthorized tables blocked
6. **Python AST analysis:** dangerous imports/calls blocked (os, subprocess, eval …)
7. **Leakage detection:** fit_transform before train_test_split flagged
8. **Pandera:** numerical constraints on output DataFrames
9. **Semantic guardrails:** intent vs. actual SQL alignment

All 9 layers are exercised by the adversarial test suite (20 tests, 8 threat vectors).

## HITL Escalation

When `confidence < HITL_CONFIDENCE_THRESHOLD` (default 0.35) after `semantic_guardrails`:
- `hitl_escalate` node sets `hitl_required=True` and records `hitl_reason`
- Bridge returns HTTP 202 instead of 200
- Caller must check `hitl_required` before acting on the result

## Bridge Architecture (`bridge/moltis_bridge.py`)

FastAPI server (port 8100) handling the full request lifecycle:

```
1. Rate limit check (SlidingWindowRateLimiter, per-IP)
2. Recall semantic memories relevant to query (cosine threshold 0.65)
3. Retrieve session episodic window (last N turns)
4. Enrich query with recalled context
5. Run Andrew Core pipeline (asyncio.to_thread)
6. Procedural feedback: record_outcome() fire-and-forget
7. Store result in Moltis memory + InProcessSemanticStore mirror
8. Format response for channel (Telegram/Discord length limits)
9. Return HTTP 200 (success) / 202 (hitl_required) / 429 (rate limited)
```

### Rate Limiting

Per-IP sliding-window counters with no external dependency:

| Endpoint | Default | Env var |
|---|---|---|
| `POST /analyze` | 10 req/60 s | `RATE_LIMIT_ANALYZE` |
| `POST /educate` | 20 req/60 s | `RATE_LIMIT_EDUCATE` |
| `POST /webhook/moltis` | 30 req/60 s | `RATE_LIMIT_WEBHOOK` |

Returns HTTP 429 + `Retry-After` header. Set `N=0` to disable per-endpoint.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness + internal state (store sizes, uptime, last sweep) |
| `POST` | `/analyze` | Submit analytical query → Andrew |
| `POST` | `/educate` | Ask educational question → Romeo PhD |
| `POST` | `/webhook/moltis` | Moltis hook receiver |
| `POST` | `/schedule` | Create recurring cron analysis |
| `POST` | `/session/{id}/end` | Close session + consolidate episodic → semantic |

## Romeo PhD (`core/romeo_phd.py`)

Stateless educational agent. Answers questions about ML, statistics, data science,
and programming. Always uses `MODEL_ROMEO` (default `gpt-4o-mini`); never touches
the database or runs code.

## Cost Control

- Hard budget: $1.00 per query (configurable via `ANDREW_MAX_COST`)
- `_budget_ok()` called before every LLM node; halts pipeline when exhausted
- LiteLLM `completion_cost()` tracks per-call spend
- Routing minimises cost: simple queries never hit expensive reasoning models

## Test Coverage

| File | Tests | Scope |
|---|---|---|
| `test_routing.py` | 12 | Keyword routing smoke tests |
| `test_validation.py` | 15 | SQL safety, Python AST, leakage detection |
| `test_adversarial.py` | 20 | 8 threat vectors end-to-end |
| `test_hitl.py` | 13 | HITL escalation, HTTP 202 |
| `test_semantic_router.py` | ~30 | Scoring, registry, fallback, routing log |
| `test_memory.py` | 28 | Consolidation, staleness sweep, dedup, procedural store |
| `test_rate_limit.py` | 22 | Allow/deny, window expiry, thread safety, env override |
