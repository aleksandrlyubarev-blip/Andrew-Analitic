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
    FastAPI | Memory recall | Query enrichment | Formatting
         |
    [Andrew Core — Python, LangGraph]
    Router | SQL | Validate | Python | Sandbox | Guardrails
```

## Core Engine Pipeline

```
START
  -> route_query_intent     [weighted 48-keyword scoring, 3 lanes]
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
  -> finalize_state           [SHA-256 hash + audit log]
  -> END
```

## Routing Lanes

| Lane | Score Threshold | Trigger Terms | Default Model |
|---|---|---|---|
| reasoning_math | >= 4 or heavy ML term | ARIMA, neural network, Monte Carlo, regression | grok-4 |
| analytics_fastlane | any hit + light analytics | average + month, CAGR + region | gpt-4o-mini |
| standard | no math keywords | bar chart, group by, show | claude-sonnet |

## Validation Layers (no LLM)

1. **SQL blocklist:** 16 dangerous keywords
2. **Statement whitelist:** only SELECT/WITH
3. **sqlglot qualify:** resolves table.column against schema
4. **Post-qualify audit:** every table and column verified
5. **Intent contract check:** unauthorized tables blocked
6. **Python AST analysis:** dangerous imports/calls blocked
7. **Leakage detection:** fit before split flagged
8. **Pandera:** numerical constraints on output DataFrames
9. **Semantic guardrails:** intent vs. actual SQL alignment

## Bridge Integration

The bridge (FastAPI, port 8100) sits between Moltis and Andrew Core:

1. **Inbound:** Moltis hook fires on MessageReceived -> POST /webhook/moltis
2. **Filter:** Bridge checks for analytical intent signals (15 keywords)
3. **Enrich:** Recalls relevant past analyses from Moltis memory
4. **Execute:** Runs Andrew Core pipeline via asyncio.to_thread
5. **Format:** Truncates to channel limits, adds confidence/cost metadata
6. **Store:** Saves successful results in Moltis memory
7. **Return:** Formatted message back to Moltis for channel delivery

## Cost Control

- Hard budget: $1.00 per query (configurable via ANDREW_MAX_COST)
- Checked before every LLM call
- LiteLLM completion_cost() tracks actual spend
- Routing minimizes cost: simple queries never touch expensive models
