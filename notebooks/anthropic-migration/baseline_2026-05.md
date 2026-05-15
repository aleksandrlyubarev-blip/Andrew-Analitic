# Anthropic SDK-credit migration — baseline (May 2026)

> **ТЗ Phase 0.1.** Fill before 31.05.2026. Without these numbers Phases 2.4
> (Haiku vs 4o-mini A/B) and 5.1 (post-mortem M1–M6 delta) have no reference
> point.

## M1 — Monthly Anthropic API spend (May 2026)

Source: Anthropic Console → Usage → May 2026.

- [ ] Total: `$_____`
- [ ] Breakdown by model (paste rows):

| Model                         | Spend, $ | Tokens in | Tokens out |
| ----------------------------- | -------- | --------- | ---------- |
| claude-sonnet-4-20250514      |          |           |            |
| claude-sonnet-4-6             |          |           |            |
| claude-haiku-4-5              |          |           |            |
| (other)                       |          |           |            |

## M2 — Swarm call distribution by lane (last 30 days)

Source: Andrew-Analitic structured logs / `lcb/state.py` traces.

- [ ] Total calls: `N = _____`

| Lane                | Provider     | Calls, N | Share, % |
| ------------------- | ------------ | -------- | -------- |
| reasoning_math      | Grok-4       |          |          |
| analytics_fastlane  | GPT-4o-mini  |          |          |
| standard            | Claude       |          |          |
| sql_generation      | GPT-4o-mini  |          |          |
| orchestrator        | Claude       |          |          |

Notes on data source / time window:

## M3 — Test suite pass rate

Source: `pytest tests/ -v` on `main` as of cutoff.

- [ ] Pass: `_____ / _____`
- [ ] Note: ТЗ said "27/27" (legacy figure). Current suite is much larger —
      record the actual numbers below and update the ТЗ acceptance criterion.
- [ ] Last green commit SHA: `__________`

```
# pytest -q --tb=no, captured 2026-05-15 on claude/anthropic-sdk-migration-mPKYD:
# 379 passed, 2 skipped in 25.13s
```

## M4 — Average latency p50 / p95 per lane

Source: existing logs / OTEL spans. Window: last 7 days, business hours only.

| Lane                | p50, ms | p95, ms |
| ------------------- | ------- | ------- |
| reasoning_math      |         |         |
| analytics_fastlane  |         |         |
| standard            |         |         |
| sql_generation      |         |         |
| orchestrator        |         |         |

## M5 — Interactive Claude subscription utilization (Max 5x)

Source: claude.ai → Settings → Usage. Take the snapshot **on 14.06** before
the auth flip — that is the last clean measurement of pre-migration baseline.

- [ ] Average daily usage, %: `_____`
- [ ] Days with throttling in last 30: `_____`

## M6 — Average $/task on Anthropic-routed traffic

Derived: `M1 / N(Claude calls in M2)`.

- [ ] `$/task = _____`
- [ ] Sanity check: should be in the same ballpark as the Anthropic invoice
      divided by the call count in lane-level logs. If the discrepancy is
      >20%, investigate before proceeding to Phase 1.

## Sign-off

- [ ] Baseline frozen by: `__________` on `__________`
- [ ] Linked commit on `claude/anthropic-sdk-migration-mPKYD`: `__________`
