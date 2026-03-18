# Andrew Analitic — Investor Brief

> **"60 % of enterprise dashboards are abandoned. Andrew turns natural language
> into validated, actionable analytics in seconds — without a single line of SQL
> from the business user."**

---

## 1. The Problem

### Data latency kills decisions

Modern enterprises are simultaneously data-rich and insight-poor:

- **Analyst bottleneck.** An analyst receives a question, writes SQL, builds a
  report, and delivers it in 2–5 business days. By delivery, the business
  context has shifted.
- **60 % abandonment.** Industry research shows six in ten deployed BI
  dashboards are never used. Analysts build for complexity; executives need
  clarity.
- **Static reports can't answer new questions.** Every new business question
  requires a new analyst ticket. This creates a permanent queue that never
  empties.

**The gap:** business decisions require sub-hour insight; traditional BI
delivers sub-week reports.

---

## 2. The Solution — Andrew Analitic

Andrew Analitic is a **conversational AI analytics agent** that transforms a
natural language question into a validated SQL query, a Python analysis, a
formatted chart, and an AIDA narrative — in under 10 seconds.

```
User: "Why did EMEA revenue drop in Q3?"
                    ↓
Andrew: routes → generates SQL → validates schema → executes safely →
        produces Python chart → delivers AIDA narrative with root cause +
        recommended next action
```

Two agents, one UI:

| Agent | Role | Cost tier |
|---|---|---|
| **Andrew Swarm** | Analytical: SQL, Python, statistics, forecasting | Cost-routed |
| **Romeo PhD** | Educational: explains methodology, concepts, formulas | Cheap |

---

## 3. Product Capabilities

### Core pipeline (v1.4.0 — 9 completed sprints)

| Capability | Detail |
|---|---|
| **Natural language → SQL** | sqlglot schema-qualified, validated against real table structure |
| **Natural language → Python** | pandas/scikit-learn, AST-sanitised before execution |
| **Double Diamond workflow** | Explore → Hypothesis → Analyse → Validate (4-phase quality gate) |
| **Semantic routing** | Embedding cosine scoring routes to cheap (GPT-4o-mini) or premium (Grok 4) models |
| **HITL escalation** | Low-confidence results return HTTP 202 + flag for human review |
| **Memory architecture** | Episodic + semantic + procedural; consolidation + staleness sweep |
| **Rate limiting** | Per-IP sliding-window; no Redis required |
| **Multi-channel delivery** | Telegram, Discord, Web UI via Moltis Rust runtime |
| **Scheduled analytics** | Cron-based recurring analysis |
| **Vue 3 web UI** | Split-view: chat + SQL table + Chart.js charts |

### Security posture

- SQL: blocklist (DROP/DELETE/TRUNCATE + 13 others) + sqlglot schema validation
- Python: AST analysis blocks `os`, `subprocess`, `exec`, `eval`, `open`
- Sandbox: Moltis runs generated code in per-session Docker containers
- Budget guard: hard cap per query (default $1.00)
- 20 adversarial tests across 8 threat vectors (all passing)

---

## 4. Technical Architecture

```
Browser
  ↓  Vue 3 UI (port 8100)
FastAPI Bridge  ←→  Andrew Swarm (LangGraph, Python)
                        ↓ semantic routing
                    cheap model ── GPT-4o-mini   ($)
                    standard    ── Claude Sonnet ($$)
                    math/ML     ── Grok 4       ($$$)
                ←→  Romeo PhD (LiteLLM)
                ←→  Moltis (Rust runtime)
                        ↓
                    Channels · Sandbox · Memory · Cron
```

**Stack:** Python 3.12, LangGraph, LiteLLM, FastAPI, Vue 3, Pinia, Chart.js,
Moltis (Rust), SQLite / PostgreSQL, Docker.

**Test coverage:** 166 tests, 8 modules, 100 % passing on Python 3.11 + 3.12.

**CI/CD:** GitHub Actions matrix builds + Docker smoke test + coverage upload.

---

## 5. Cost Model

Andrew is cost-conscious by design. The semantic router dynamically selects
the cheapest model capable of answering the query:

| Query type | Model | Typical cost |
|---|---|---|
| Simple BI (group by, totals) | GPT-4o-mini | ~$0.002 |
| Standard analytics (trends, funnels) | Claude Sonnet | ~$0.015 |
| Advanced ML / forecasting / math | Grok 4 | ~$0.08 |

A hard per-query budget cap (default $1.00) prevents runaway costs. The
`/metrics` endpoint exposes live cost-per-query and routing distribution for
operational visibility.

---

## 6. Go-to-Market

### Target buyers (B2B SaaS)

1. **Mid-market operations teams** (50–500 employees) without a dedicated BI
   team — pays for analytics that was previously outsourced.
2. **Analytics platform vendors** — white-label Andrew as an embedded
   conversational layer on top of existing dashboards.
3. **Enterprise data teams** — self-hosted deployment (Docker / Kubernetes) to
   keep data on-premises; pay per seat or per API call.

### Deployment modes

| Mode | Target | Revenue model |
|---|---|---|
| Cloud SaaS | SMB | Monthly subscription per seat |
| Self-hosted | Enterprise | Annual licence + support |
| API / OEM | Platforms | Per-query or per-seat royalty |

---

## 7. Competitive Differentiation

| Dimension | Andrew Analitic | Generic LLM chatbot | Traditional BI |
|---|---|---|---|
| Schema validation | ✓ sqlglot qualify | ✗ hallucinates | ✓ |
| Cost routing | ✓ 3-tier semantic | ✗ fixed model | N/A |
| Python sandbox | ✓ Docker isolation | ✗ no execution | ✗ |
| HITL escalation | ✓ HTTP 202 | ✗ | manual QA |
| Multi-channel | ✓ Telegram/Discord/Web | partial | web only |
| Time-to-insight | < 10 s | varies | 2–5 days |
| Setup | `docker compose up` | varies | weeks |

---

## 8. Traction & Validation

- **v1.4.0** shipped across 9 structured sprints in < 6 weeks
- **166 tests** covering routing, validation, adversarial, memory, HITL,
  double diamond — 100 % passing
- **Production-hardened**: adversarial test suite simulates prompt injection,
  hallucinated schemas, budget exhaustion, data leakage
- **Architecture validated** against Microsoft Fabric / Power BI patterns
  (Data2Speak methodology, AIDA storytelling framework)

---

## 9. Roadmap

| Sprint | Milestone |
|---|---|
| Sprint 10 | OpenAPI / MCP server exposure — let Claude Code, Cursor, other AI tools call Andrew directly |
| Sprint 11 | Streaming responses (Server-Sent Events) — real-time token-by-token output in UI |
| Sprint 12 | Multi-tenant auth — JWT / API-key per workspace, RLS passthrough to DB |
| Sprint 13 | Distributed rate limiting — Redis-backed limiter for multi-replica deployments |
| Sprint 14 | Vector store integration — Qdrant / Weaviate for semantic search over past analyses |
| Sprint 15 | Dashboard export — PDF / PPTX one-click export of chart + narrative |
| Sprint 16 | Formal security audit + penetration test |

---

## 10. The Ask

We are raising **[amount]** to:

1. **Go-to-market** — first 10 paying customers, content + SEO, product-led
   growth loop
2. **Team** — one senior backend engineer (Rust/Python), one product-focused
   ML engineer
3. **Infrastructure** — managed cloud deployment (Kubernetes), SOC 2 Type II
   certification

**Use of funds:** 50 % engineering, 30 % GTM, 20 % infrastructure & compliance.

---

## 11. Demo

```bash
# 1. Clone and configure
git clone https://github.com/YOUR_ORG/andrew-analitic.git
cd andrew-analitic
cp config/.env.example .env
# Add at least one LLM API key to .env

# 2. Seed the demo database
python demo/seed_demo_db.py          # creates demo/demo.db

# 3. Start (builds frontend automatically)
DATABASE_URL=sqlite:///demo/demo.db docker compose up -d

# 4. Run the live showcase (8 queries, metrics printout)
bash demo/demo_queries.sh

# 5. Open the web UI
open http://localhost:8100
```

Or just open `http://localhost:8100` and type:
> _"Why did EMEA revenue drop in Q3 2025 and what should we do about it?"_

---

*Andrew Analitic — v1.4.0 · MIT License*
