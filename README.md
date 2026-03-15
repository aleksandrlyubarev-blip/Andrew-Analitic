# Andrew & Romeo — AI Analytics Suite

> Andrew Swarm analyses data. Romeo PhD explains it. One dark-themed UI to rule them both.

**Release: v1.1.0** — Romeo PhD agent + Vue 3 web UI added.

## What It Does

**Andrew Swarm** is an analytical AI agent that turns natural language questions into validated SQL queries, Python analyses, and formatted reports. It routes simple BI tasks to cheap models and complex math/ML to reasoning models, enforcing a hard dollar budget per query.

**Romeo PhD** is the educational companion agent. Ask it to explain concepts — gradient descent, p-values, random forests, SQL joins — and it returns clear, Markdown-formatted answers with examples and analogies.

The **Vue 3 web UI** (port 8100, served by the bridge) gives both agents a split-view interface: chat on the left, SQL tables + charts on the right.

Moltis (a Rust runtime) provides the delivery layer: messaging channels (Telegram, Discord, Web UI), persistent memory, sandboxed code execution, and cron scheduling.

## What You Need

| Requirement | Minimum | Notes |
|---|---|---|
| **Python** | 3.11+ | 3.12 recommended |
| **LLM API key** | At least one | OpenAI, Anthropic, or OpenRouter |
| **Docker** | 24.0+ | For Moltis + sandbox. Podman also works. |
| **Database** | SQLite (default) | PostgreSQL optional for production |
| **Moltis** | Latest | Auto-installed via Docker. Or: `brew install moltis-org/tap/moltis` |
| **Node.js** | 20+ | Only needed for local frontend dev (`npm run dev`) |

**You do NOT need:** xAI/Grok key (optional for math routing), E2B account (Moltis sandbox replaces it), GPU.

## Compatibility Matrix

| Component | Supported |
|---|---|
| Python | 3.11, 3.12 |
| OS | Linux (x86_64, arm64), macOS (Apple Silicon, Intel) |
| Container | Docker 24+, Podman 4+, OrbStack |
| Database | SQLite 3.35+, PostgreSQL 14+ |
| Moltis | 0.x (latest from ghcr.io/moltis-org/moltis) |
| LLM Providers | OpenAI, Anthropic, OpenRouter, Ollama (local), any OpenAI-compatible |

## Quick Start

### Docker (recommended)

```bash
# 1. Clone
git clone https://github.com/YOUR_ORG/andrew-swarm.git
cd andrew-swarm

# 2. Configure
cp config/.env.example .env
# Edit .env — add at least one LLM API key

# 3. Launch  (builds frontend automatically via multi-stage Dockerfile)
docker compose up -d

# 4. Open the web UI
open http://localhost:8100

# 5. Or call the API directly
curl -X POST http://localhost:8100/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Total revenue by region"}'

curl -X POST http://localhost:8100/educate \
  -H "Content-Type: application/json" \
  -d '{"question": "What is gradient descent?"}'
```

### Local development

```bash
# Terminal 1: Moltis
brew install moltis-org/tap/moltis
moltis gateway

# Terminal 2: Python bridge
pip install -r requirements.txt
uvicorn bridge.moltis_bridge:app --reload --port 8100

# Terminal 3: Frontend dev server (with hot-reload)
cd frontend
npm install
npm run dev        # → http://localhost:3000  (proxies /api → :8100)
```

To build the frontend and serve it directly from the bridge (no separate dev server):

```bash
cd frontend && npm run build   # outputs to bridge/static/
# Now http://localhost:8100 serves the UI
```

## Architecture

```
User (Browser → http://localhost:8100)
         |
         v
+---------------------------+
|  Vue 3 UI  (bridge/static)|  Chat panel + SQL table + Chart.js charts
+-----------+---------------+
            | fetch /analyze or /educate
            v
+---------------------------+
|  Bridge (Python, :8100)   |  FastAPI. Serves UI + API. Memory recall,
|                           |  query enrichment, result formatting.
+------+------------+-------+
       |            |
       v            v
  Andrew Core   Romeo PhD
  (LangGraph)   (LiteLLM)
  SQL → Python  Educational
  analysis      explanations
       |
       v
+---------------------------+
|   Moltis (Rust, :13131)   |  Channels, memory, sandbox, cron
+---------------------------+
```

## How Routing Works

Andrew uses weighted keyword scoring (48 terms) to route queries to three model lanes:

| Lane | Trigger | Default Model | Cost |
|---|---|---|---|
| **reasoning_math** | ARIMA, Monte Carlo, neural network, regression, score >= 4 | Grok 4 (or env override) | $$$ |
| **analytics_fastlane** | average + monthly, CAGR + region (light math + BI context) | GPT-4o-mini | $ |
| **standard** | Bar chart, group by, show totals (pure BI) | Claude Sonnet | $$ |

Romeo PhD always uses `MODEL_ROMEO` (default: `gpt-4o-mini`).

## Security Posture

This is an MVP. The following guardrails are implemented but not formally audited:

- **SQL safety:** sqlglot `qualify(schema=...)` validates every table and column against the real schema. Blocked keywords: DROP, DELETE, TRUNCATE, ALTER, and 13 others. Only SELECT/WITH statements allowed.
- **Python safety:** AST analysis blocks `import os`, `subprocess`, `exec()`, `eval()`, `open()` before code reaches the sandbox. Data leakage detection flags `fit_transform` before `train_test_split`.
- **Sandbox isolation:** Moltis runs generated code in per-session Docker containers. No host filesystem access.
- **Budget guard:** Hard cap at $1.00 per query (configurable). Stops LLM calls when exhausted.
- **Semantic guardrails:** Checks that SQL output matches user intent (revenue question must reference revenue column, monthly question must have GROUP BY).

**Not yet implemented:** formal threat model testing, adversarial fuzz suite, HITL escalation for low-confidence results, rate limiting on the bridge API.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI (served from `bridge/static/`) |
| `GET` | `/health` | Health check (Andrew + Moltis) |
| `POST` | `/analyze` | Submit analytical query (Andrew) |
| `POST` | `/educate` | Ask an educational question (Romeo PhD) |
| `POST` | `/webhook/moltis` | Moltis hook receiver (filters for analytical intent) |
| `POST` | `/schedule` | Create recurring analysis (cron) |

## Project Structure

```
andrew-swarm/
  core/
    andrew_swarm.py       # LangGraph analytical engine
    romeo_phd.py          # Romeo PhD educational agent
  bridge/
    moltis_bridge.py      # FastAPI bridge: API + static UI server
    static/               # Compiled Vue UI (git-ignored, built by Dockerfile)
  frontend/               # Vue 3 + Vite source
    src/
      App.vue             # Root layout (split view)
      components/         # Header, AgentTabs, ChatPanel, DataPanel, charts
      stores/agent.js     # Pinia store (messages, API calls, data panel state)
      api/index.js        # fetch wrappers for /analyze + /educate
    vite.config.js        # Dev proxy + build output config
  config/
    .env.example
  tests/
    test_routing.py
    test_validation.py
  docs/
    ARCHITECTURE.md
    CHANGELOG.md
  docker-compose.yml
  Dockerfile              # Multi-stage: Node builds UI, Python serves it
  requirements.txt
  README.md
  LICENSE
```

## Environment Variables

```env
# LLM providers (at least one required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Model overrides (all optional)
MODEL_REASONING_MATH=grok-4
MODEL_ORCHESTRATOR=anthropic/claude-sonnet-4-20250514
MODEL_PYTHON=anthropic/claude-sonnet-4-20250514
MODEL_ANALYTICS=gpt-4o-mini
MODEL_SQL=gpt-4o-mini
MODEL_ROMEO=gpt-4o-mini        # Romeo PhD model

# Moltis connection
MOLTIS_HOST=127.0.0.1
MOLTIS_PORT=13131
MOLTIS_PASSWORD=your-password

# Database
DATABASE_URL=sqlite:///andrew.db

# Limits
ANDREW_MAX_COST=1.00
ROMEO_MAX_TOKENS=2000
HITL_CONFIDENCE_THRESHOLD=0.35   # results below this trigger HTTP 202 + hitl_required=true
```

## Roadmap

- [x] Sprint 1-2: LangGraph pipeline + LiteLLM routing
- [x] Sprint 3: Hardened validation (sqlglot qualify, AST safety, Pandera)
- [x] Sprint 4: Weighted 48-keyword routing, model registry, provider adapters
- [x] Sprint 5: Moltis integration (channels, memory, sandbox, scheduling)
- [x] Sprint 8: Romeo PhD educational agent + Vue 3 split-view web UI
- [x] Sprint 6: Adversarial test suite — 20 tests covering all 8 threat model cases
- [x] Sprint 7: HITL escalation — `hitl_escalate` LangGraph node, HTTP 202, 13 tests

## License

MIT
