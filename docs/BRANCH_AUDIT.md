# Branch Audit — Romeo Flex Vision Ecosystem

> Cross-repository architectural audit of codex/claude branches.
> Evaluated against ROBOT_GUIDELINES.md invariants.

---

## Scope

Two repositories in the `aleksandrlyubarev-blip` GitHub account:

| Repository | Role | Primary Stack |
|------------|------|---------------|
| **Andrew-Analitic** | Analytical engine + bridge + LCB runner | Python (LangGraph, FastAPI), Vue 3 |
| **RomeoFlexVision** | Product shell (RoboQC landing + integrations) | TypeScript (Vite/React), Telegram bot |

This audit covers feature branches with `codex/*` and `claude/*` prefixes
that carry architectural ideas intended for the multi-agent ecosystem.

---

## 1. Andrew-Analitic: `codex/agent-quality-foundations`

**What it does**: Strengthens the test and process contour. Grows test count,
formalizes release line, adds quality gates.

**Alignment with ROBOT_GUIDELINES**:

| Guideline | Status | Notes |
|-----------|--------|-------|
| §5 Testing Contract | **Strong** | Tests are pure-function focused, no LLM calls |
| §8 CI as Gate | **Strong** | `ci.yml` runs pytest matrix (3.11 + 3.12) |
| §2 Core/Shell | **Good** | Core tests don't mock I/O boundaries |
| §1 Data Model First | **Neutral** | No schema changes in this branch |

**Verdict**: This branch is the quality foundation. Merge-ready as reference.

---

## 2. RomeoFlexVision: `codex/finops-issue-alerts`

**What it does**: Adds FinOps cost/cache monitoring with auto-issue creation.

**Components**:
- `.env.example` additions: `FINOPS_COST_ALERT_USD`, `FINOPS_CACHE_HIT_FLOOR`
- `finops-alert.yml` workflow (manual + repository_dispatch)
- `scripts/finops-issue.mjs` — reads metrics JSON, checks thresholds, creates GitHub issue

**Alignment with ROBOT_GUIDELINES**:

| Guideline | Status | Issue |
|-----------|--------|-------|
| §2 Core/Shell | **Violation** | `finops-issue.mjs` mixes JSON parsing, threshold comparison, and HTTP issue creation in one function |
| §5 Testing Contract | **Missing** | No test for the threshold logic |
| §8 CI as Gate | **Partial** | Thresholds are config (good), but no test validates the evaluation logic |

**Required refactoring**:
1. Extract `evaluateFinops(metrics, policy) → {alert, reasons}` as pure function
2. Test it with fixed inputs (see AGENTS_ACADEMY Module 1)
3. Keep `openIssue(decision, token)` as the thin shell adapter

---

## 3. RomeoFlexVision: `claude/analyze-test-coverage-*`

**What it does**: Adds `npm test` step to deploy workflow + `TEST_COVERAGE_ANALYSIS.md`.

**Alignment with ROBOT_GUIDELINES**:

| Guideline | Status | Issue |
|-----------|--------|-------|
| §8 CI as Gate | **Risky** | Adding `npm test` before the test runner exists blocks deploys |
| §5 Testing Contract | **Directionally correct** | Analysis identifies pure slices for testing |
| §9 Scope Discipline | **OK** | Stays focused on coverage analysis |

**Required sequencing** (per §8):
1. First: add Vitest + at least one passing test
2. Then: wire `npm test` into deploy workflow
3. Then: expand coverage per the analysis document

**Never**: add CI test step without a passing test suite.

---

## 4. RomeoFlexVision: `claude/implement-todo-item-N3wMC`

**What it does**: Adds agent testing UI, dashboard enhancements, heatmap grid,
dynamic trace generation.

**Alignment with ROBOT_GUIDELINES**:

| Guideline | Status | Issue |
|-----------|--------|-------|
| §7 Deterministic Demos | **Violation** | `Math.random()` used for test pass/fail outcome |
| §2 Core/Shell | **Violation** | Cost calculations, heatmap generation, and trace building mixed into React components |
| §4 Naming | **Weak** | Generic state names in UI components |
| §1 Data Model First | **Missing** | No type definitions for new agent test states |

**Required fixes**:
1. Replace `Math.random()` with seeded PRNG or static scenarios
2. Extract `buildTrace(seed)` as pure function outside component
3. Extract cost calculations into `computeAgentCosts(agents, usage)` pure function
4. Define TypeScript types for agent test phases (`idle | running | passed | failed`)

---

## 5. RomeoFlexVision: `claude/implement-todo-item-nZbZf`

**What it does**: Adds custom SVG icons, updates agent identity system.

**Alignment with ROBOT_GUIDELINES**:

| Guideline | Status | Notes |
|-----------|--------|-------|
| §9 Scope Discipline | **Good** | Focused on visual identity only |
| §1 Data Model First | **OK** | Updates AGENTS registry with new icon identifiers |
| §4 Naming | **Good** | Domain-specific names (RomeoPhdIcon, BassitoDogIcon) |

**Verdict**: Clean, scoped change. Low architectural risk.

---

## 6. RomeoFlexVision: `claude/check-frontend-obnII`

**What it does**: Adds Supabase DB (migrations, RLS), `useTasks.ts` hook with
CRUD + realtime, localStorage demo mode, toast notifications, Profile page.

**Alignment with ROBOT_GUIDELINES**:

| Guideline | Status | Issue |
|-----------|--------|-------|
| §3 Ports & Adapters | **Violation** | `useTasks.ts` has inline Supabase/localStorage branching |
| §2 Core/Shell | **Violation** | CRUD logic, realtime subscription, and demo-mode switching in one hook |
| §1 Data Model First | **Partial** | Migration exists (good), but TypeScript types not derived from schema |
| §5 Testing Contract | **Missing** | No tests for task CRUD logic |

**Required refactoring** (see AGENTS_ACADEMY Module 2):
1. Define `TaskStore` interface (port)
2. Implement `SupabaseTaskStore` adapter (with RLS, realtime)
3. Implement `LocalTaskStore` adapter (localStorage, no realtime)
4. `useTasks(store: TaskStore)` — zero conditionals
5. Derive TypeScript types from DB migration schema

---

## 7. RomeoFlexVision: `telegram-bot/` (main)

**What it does**: Public Telegram entry point with webhook/long-polling modes.

**Alignment with ROBOT_GUIDELINES**:

| Guideline | Status | Notes |
|-----------|--------|-------|
| §3 Ports & Adapters | **Correct** | Bot is an adapter (drive port) |
| §2 Core/Shell | **At risk** | If command logic grows, it must stay in a separate core module |

**Architectural rule**: The bot handler file should contain ONLY message
parsing and response formatting. Any business logic (command interpretation,
data lookup, formatting) belongs in a shared core that both bot and web UI consume.

---

## Cross-Cutting Findings

### 1. Core/Shell Separation Is the #1 Issue

Four of seven branches violate §2 (Functional Core / Imperative Shell).
This is the single most impactful improvement: extract pure functions
from mixed modules, then test them.

### 2. Determinism in UI Demos

`Math.random()` in test/demo paths (branch N3wMC) makes quality assertions
impossible. The fix is simple: seeded PRNG or fixture data.

### 3. CI Sequencing Matters

Adding `npm test` before a test runner exists is a deployment blocker.
The correct order: runner → tests → CI gate.

### 4. Naming Is Architecture

Branches with domain-specific names (FinOps thresholds, agent phases)
produce cleaner code than branches with generic names. This is not
aesthetic — it directly affects how LLM agents interpret and extend code.

### 5. MCP + A2A Readiness

No branch currently implements MCP or A2A protocols explicitly.
The roadmap (Sprint 10+) includes "OpenAPI / MCP server for Claude Code".
When this lands, it must follow §10: typed tool descriptions,
no internal state leakage across protocol boundaries.

---

## Priority Matrix

| Priority | Branch/Area | Action | Effort |
|----------|-------------|--------|--------|
| **P0** | `codex/agent-quality-foundations` | Merge as quality baseline | Low |
| **P1** | `claude/check-frontend-obnII` | Refactor to ports & adapters | Medium |
| **P1** | `codex/finops-issue-alerts` | Extract pure `evaluateFinops()` + test | Low |
| **P2** | `claude/implement-todo-item-N3wMC` | Remove randomness, extract pure functions | Medium |
| **P2** | `claude/analyze-test-coverage-*` | Fix sequencing (runner before gate) | Low |
| **P3** | `claude/implement-todo-item-nZbZf` | Merge as-is (clean scope) | Low |
| **P3** | `telegram-bot/` | Monitor — enforce thin adapter rule | Ongoing |

---

*This audit is versioned alongside the code. Re-run when branches are updated.*
