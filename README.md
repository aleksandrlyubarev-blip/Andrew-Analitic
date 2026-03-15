# Andrew Swarm

> A data science AI agent with hardened analytics pipelines, budget-aware multi-model routing, and Moltis-powered delivery.

**Release: v1.0.0-rc1** — MVP release candidate. Functional, tested internally, not yet battle-tested in production.

## What It Does

Andrew Swarm is an analytical AI agent that turns natural language questions into validated SQL queries, Python analyses, and formatted reports. It routes simple BI tasks to cheap models and complex math/ML to reasoning models, enforcing a hard dollar budget per query.

Moltis (a Rust runtime) provides the delivery layer: messaging channels (Telegram, Discord, Web UI), persistent memory, sandboxed code execution, and cron scheduling.

## What You Need

| Requirement | Minimum | Notes |
|---|---|---|
| **Python** | 3.11+ | 3.12 recommended |
| **LLM API key** | At least one | OpenAI, Anthropic, or OpenRouter |
| **Docker** | 24.0+ | For Moltis + sandbox. Podman also works. |
| **Database** | SQLite (default) | PostgreSQL optional for production |
| **Moltis** | Latest | Auto-installed via Docker. Or: `brew install moltis-org/tap/moltis` |
| **Hardware** | 2GB RAM, 2 cores | Moltis itself uses <100MB |

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

# 3. Launch
docker compose up -d

# 4. Test
curl -X POST http://localhost:8100/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Total revenue by region"}'

# 5. Open Moltis Web UI
open http://localhost:13131
```

### Local (without Docker)

```bash
# Terminal 1: Moltis
brew install moltis-org/tap/moltis
moltis gateway

# Terminal 2: Andrew bridge
pip install -r requirements.txt
python bridge/moltis_bridge.py

# Terminal 3: Test
curl -X POST http://localhost:8100/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Average revenue by month"}'
```

## Architecture

```
User (Telegram / Discord / Web UI / API)
         |
         v
+-------------------------+
|   Moltis (Rust, :13131) |  Channels, memory, sandbox, cron
|   Hook: MessageReceived  |
+-----------+-------------+
            | POST /webhook/moltis
            v
+-------------------------+
| Bridge (Python, :8100)  |  FastAPI. Memory recall, query enrichment,
|                         |  result formatting, memory store.
+-----------+-------------+
            | executor.execute()
            v
+-------------------------+
| Andrew Core (LangGraph) |  Weighted router -> SQL gen -> sqlglot qualify
|                         |  -> execute -> Python gen -> AST safety
|                         |  -> sandbox -> Pandera -> semantic guardrails
+-------------------------+
```

## How Routing Works

Andrew uses weighted keyword scoring (48 terms) to route queries to three model lanes:

| Lane | Trigger | Default Model | Cost |
|---|---|---|---|
| **reasoning_math** | ARIMA, Monte Carlo, neural network, regression, score >= 4 | Grok 4 (or env override) | $$$ |
| **analytics_fastlane** | average + monthly, CAGR + region (light math + BI context) | GPT-4o-mini | $ |
| **standard** | Bar chart, group by, show totals (pure BI) | Claude Sonnet | $$ |

All models are configurable via environment variables. No hardcoded vendor lock-in.

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
| `GET` | `/health` | Health check (Andrew + Moltis) |
| `POST` | `/analyze` | Submit analytical query |
| `POST` | `/webhook/moltis` | Moltis hook receiver (filters for analytical intent) |
| `POST` | `/schedule` | Create recurring analysis (cron) |

## Project Structure

```
andrew-swarm/
  core/
    andrew_swarm.py       # LangGraph analytical engine (883 lines)
  bridge/
    moltis_bridge.py      # Moltis integration layer (430 lines)
  config/
    .env.example          # Environment template
    models.yaml           # Model registry (optional)
  tests/
    test_routing.py       # Routing smoke tests
    test_validation.py    # SQL + Python safety tests
  docs/
    ARCHITECTURE.md       # Detailed architecture notes
    CHANGELOG.md          # Release history
  docker-compose.yml
  Dockerfile
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

# Moltis connection
MOLTIS_HOST=127.0.0.1
MOLTIS_PORT=13131
MOLTIS_PASSWORD=your-password

# Database
DATABASE_URL=sqlite:///andrew.db

# Limits
ANDREW_MAX_COST=1.00
BRIDGE_PORT=8100
```

## Roadmap

- [x] Sprint 1-2: LangGraph pipeline + LiteLLM routing
- [x] Sprint 3: Hardened validation (sqlglot qualify, AST safety, Pandera)
- [x] Sprint 4: Weighted 48-keyword routing, model registry, provider adapters
- [x] Sprint 5: Moltis integration (channels, memory, sandbox, scheduling)
- [ ] Sprint 6: Adversarial test suite (8 test cases from threat model)
- [ ] Sprint 7: HITL escalation for low-confidence results
- [ ] Sprint 8: Romeo PhD shared supervisor (educational + analytical agents)

## Companion Agent

Andrew Swarm is the analytical half of a two-agent system. **Romeo PhD** handles educational queries. A shared LangGraph supervisor (planned for Sprint 8) routes between them based on intent.

## License

MIT
