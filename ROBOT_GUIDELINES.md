# ROBOT_GUIDELINES.md — Architectural Constitution for AI Agents

> Minimal, non-negotiable invariants. Every codex/claude agent MUST follow these
> rules before, during, and after code generation. ~100 lines by design:
> rules that grow long start to conflict and get ignored.

---

## 1. Data Model First

Do NOT write implementation code until the data model is verified.

- **Python**: update dataclasses / TypedDict / Pydantic models in the relevant
  module (`core/`, `bridge/schemas.py`, `lcb/state.py`) before touching logic.
- **TypeScript/Vue**: update Pinia store types or prop interfaces before
  touching components.
- **SQL**: write or update the migration before writing queries against new columns.
- If the task has no data-model impact, state that explicitly in the commit message.

## 2. Functional Core / Imperative Shell

All code follows a two-zone architecture:

| Zone | Allowed | Forbidden |
|------|---------|-----------|
| **Core** (pure functions) | Deterministic transforms, validation, scoring, formatting | I/O, network, DB, filesystem, timers, `random()` |
| **Shell** (adapters) | HTTP calls, DB queries, file reads, Telegram API, GitHub API | Business logic, scoring, decision-making |

Reference implementations in this repo:

- **Core zone**: `core/semantic_router.py` scoring functions, `core/memory.py`
  consolidation logic, `bridge/scene_ops.py` snapshot builder.
- **Shell zone**: `bridge/moltis_bridge.py` endpoint handlers,
  `bridge/client.py` HTTP client, `bridge/hitl.py` webhook dispatcher.

When adding new logic, ask: *"Can I test this without mocking I/O?"*
If no — split it.

## 3. Ports & Adapters

External systems (DB, Telegram, Supabase, GitHub API, Moltis, camera/V4L2)
are accessed ONLY through adapter modules with a clear interface.

- Never scatter `if provider == "supabase"` across components.
- One port = one Python protocol / TS interface.
- Adapters are swappable (e.g., `InProcessSemanticStore` vs future Qdrant).

## 4. Naming Discipline

| Element | Convention | Anti-pattern |
|---------|-----------|--------------|
| Functions | Verb phrase: `build_data_profile`, `validate_sql`, `route_query` | `manager`, `processor`, `handler`, `do_stuff` |
| Variables | Noun phrase: `cost_usd`, `cache_hit_rate`, `column_profile` | `data`, `result`, `tmp`, `x` |
| Booleans | Question form: `is_empty`, `hitl_required`, `has_session` | `flag`, `check`, `status` |
| Constants | `UPPER_SNAKE`: `ANDREW_MAX_COST`, `HITL_CONFIDENCE_THRESHOLD` | magic numbers inline |

Domain names are always preferred over generic CS names.
`scene_ops_snapshot` > `data_object`. `cost_usd_limit` > `threshold`.

## 5. Testing Contract

Every new logic module MUST have a testable pure-function slice.

- Test the **core zone** (pure functions), not the shell.
- No `unittest.mock.patch` on internal functions — if you need to mock it,
  the boundary is wrong; refactor to inject the dependency.
- Mock only at system boundaries: HTTP, DB, filesystem.
- All test data is **deterministic**: no `random()`, no `time.time()`,
  no `uuid4()` in assertions. Use seeds or fixtures.
- Reference: `tests/test_double_diamond.py` (pure dataclass tests),
  `tests/test_semantic_router.py` (scoring with fixed inputs).

## 6. No Symptom Fixes

> "Solve the cause, not the consequence." — Mokevnin

- If a test fails, find the root cause. Do not add `try/except` to silence it.
- If types don't match, fix the model. Do not cast with `# type: ignore`.
- If a function is too complex, decompose it. Do not add more parameters.
- Every fix MUST simplify the causal chain, not extend it.

## 7. Deterministic Demos and Mocks

UI demo modes, seed data, and mock traces MUST be reproducible:

- Use `demo/seed_demo_db.py` as the reference pattern: static, versioned data.
- No `Math.random()` / `random.random()` in demo/test paths.
- Mock traces use `buildTrace(seed)` or static JSON, never live randomness.
- If a component needs "simulated delay", use a fixed duration, not jitter.

## 8. CI as Gate, Not Ornament

- `pytest tests/` MUST pass before any merge. No `skip` without issue link.
- New modules get tests in the same PR. "Tests in follow-up" is not accepted.
- FinOps thresholds (`ANDREW_MAX_COST`, rate limits) are config, not code.
  Changes to limits go through `.env.example`, not hardcoded constants.

## 9. Scope Discipline

- Do not add features, refactors, or "improvements" beyond the task.
- Do not add docstrings/comments to code you didn't change.
- Do not introduce abstractions for single-use operations.
- Three similar lines > one premature abstraction.
- If the task says "fix X", fix X. Do not also clean up Y.

## 10. Protocol Boundaries (MCP / A2A)

- **MCP** = connection to tools and context (GitHub, filesystem, DB).
  Agents use MCP for tool access. Keep tool descriptions minimal and typed.
- **A2A** = agent-to-agent communication (Andrew ↔ Romeo, future agents).
  Use structured messages with typed payloads, not free-text relay.
- Never expose internal state through protocol boundaries.
  Pass only the minimal data the consumer needs.

---

*This document is the compile-time constitution. If a rule here conflicts with
a PR comment or ad-hoc instruction, this document wins unless explicitly
overridden with rationale.*
