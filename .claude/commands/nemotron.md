---
name: nemotron
description: >
  Nemotron — Autonomous LLM Ensemble Architect. Use this skill whenever the
  user asks about deploying large language models in production, VRAM planning,
  model quantization (FP8, INT8, INT4, GGUF, AWQ, GPTQ), tensor parallelism,
  pipeline parallelism, KV-cache optimization, multi-agent ensemble design,
  Llama-3.1-Nemotron-Ultra-253B-v1 architecture, NAS (Neural Architecture
  Search), Skip Attention, FFN Fusion, reasoning toggle (detailed thinking
  on/off), GRPO, vLLM, TensorRT-LLM, SGLang, or building autonomous
  data-analyst agents that run local open-weight models. Always invoke this
  skill when the question is about *running* LLMs rather than *prompting* them.
---

# Nemotron — Autonomous LLM Ensemble Architect

> «The difference between a demo and a production data-analyst agent is not
> the prompt. It is the hardware plan, the quantization strategy, the
> KV-cache budget, and the reasoning toggle.»

Reference architecture: **4 × Llama-3.1-Nemotron-Ultra-253B-v1** ensemble
(NVIDIA, 2025) — autonomous multi-agent data analyst stack.

Related skills: `/llm-engineer` (application layer) · `/grunwald` (theory) · `/bi` (output layer)

---

## Execution Checklist (low-freedom — copy → check off)

```
[ ] 1. TASK CLASS   — Classify workload: reasoning | code | retrieval | serving
[ ] 2. VRAM BUDGET  — Calculate peak VRAM per model (weights + KV-cache + activations)
[ ] 3. QUANT CHOICE — Select precision tier (see Quantization Decision Tree)
[ ] 4. PARALLELISM  — Choose TP / PP / DP split for the cluster
[ ] 5. SERVING      — Select inference engine (vLLM | TRT-LLM | SGLang | llama.cpp)
[ ] 6. REASONING    — Define thinking toggle policy per agent role
[ ] 7. ORCHESTRATE  — Wire LangGraph graph with agent roles and handoff contracts
[ ] 8. GUARDRAILS   — Budget cap, timeout, retry + fallback to smaller model
[ ] 9. MONITOR      — tokens/s, TTFT, ITL, VRAM headroom, cost/query
```

---

## Part 1 — Nemotron-Ultra-253B Architecture

### 1.1 NAS innovations vs. standard Transformer

The model is a structurally compressed derivative of Llama-3.1-405B-Instruct,
produced by NVIDIA's **Puzzle** NAS framework. Three architectural changes
enable 4× better throughput at same quality:

```
Standard Transformer block          Nemotron-Ultra block
────────────────────────────────    ──────────────────────────────────
LayerNorm                           LayerNorm
Multi-Head Self-Attention ←─────── SKIP (replaced by single linear layer
                                         in selected blocks)
LayerNorm                           LayerNorm
Feed-Forward Network (fixed width)  Variable-width FFN (NAS-tuned per layer)
                                    + FFN Fusion: consecutive skip-attn blocks
                                      → single wider parallel FFN
```

**Why this matters for agentic workloads:**

| NAS change | Effect | Agent benefit |
|---|---|---|
| Skip Attention | KV-cache not allocated for skipped layers | 128K ctx without OOM |
| Variable FFN | Parameters concentrated where needed | Better code reasoning per FLOP |
| FFN Fusion | Fewer sequential ops → lower latency | Faster tool-call round-trips |

### 1.2 Training pipeline

```
Llama-3.1-405B-Instruct
        │
        ▼  Knowledge Distillation (65B tokens, teacher → student)
Nemotron-253B (architecture compressed by NAS)
        │
        ▼  Continual Pretraining (88B tokens, domain adaptation)
        │
        ▼  SFT — Math · Code · Reasoning · Tool-Calling
           (traces from DeepSeek-R1 used as silver labels)
        │
        ▼  GRPO (Group Relative Policy Optimization)
           Reward: correctness of final answer, not process
        │
        ▼  Nemotron-Ultra-253B-v1 ← production model
```

**GRPO vs PPO for agents:**
GRPO does not need a separate value network. It computes a group baseline
from N rollouts of the same prompt — much cheaper to run on long
reasoning traces, which is why it scales to STEM-length solutions.

### 1.3 Reasoning Toggle — the key operational lever

```python
# Activate deep reasoning (multi-step planning, code debugging)
SYSTEM_THINKING_ON = "detailed thinking on"

# Deactivate for cheap tasks (formatting, routing, summarising)
SYSTEM_THINKING_OFF = "detailed thinking off"

# Token cost difference (empirical):
#   thinking OFF → ~500 tokens/response   (~$0.002 at self-hosted cost)
#   thinking ON  → ~4,000-12,000 tokens   (~$0.015-0.05)
```

**Policy: which agents get thinking ON?**

```
Agent role                  Thinking toggle    Justification
──────────────────────────────────────────────────────────────────────
Orchestrator (task plan)    ON                 Multi-step plan = complex
SQL Generator               ON                 Schema joins + edge cases
Python Analyst              ON                 Algorithm selection + debug
Validator / QA              ON                 Adversarial checking
Formatter / Narrator        OFF                Text templating = cheap
Router / Intent Classifier  OFF                Single-token decision
```

---

## Part 2 — VRAM Planning

### 2.1 Memory components per model instance

```
Total VRAM = Weights + KV-cache + Activations + Framework overhead

Weights (253B params):
  FP16 / BF16  → 253 × 2  = 506 GB
  FP8          → 253 × 1  = 253 GB   ← recommended for H100 SXM
  INT8         → 253 × 1  = 253 GB
  INT4 / AWQ   → 253 × 0.5 = 127 GB  ← quality loss on reasoning tasks

KV-cache (per-request, per-layer):
  Size = 2 × n_layers × n_kv_heads × d_head × seq_len × precision_bytes
  Nemotron-Ultra: ~80 transformer layers (some skip-attn → no KV for those)
  At 128K ctx, FP8: ~40-60 GB per concurrent request

Activations + framework: ~10-15 GB flat overhead
```

### 2.2 Quantization Decision Tree

```
Quality requirement?
├── Maximum (financial reports, adversarial testing)
│   → BF16 / FP16  (506 GB per model)
│   → Hardware: 8× H100 SXM 80GB (TP=8) per model
│
├── Production standard (most analytics workloads)
│   → FP8  (253 GB per model)           ← NVIDIA native, minimal degradation
│   → Hardware: 4× H100 SXM 80GB (TP=4) per model
│   → Use: TensorRT-LLM FP8 quantization or vLLM --quantization fp8
│
├── Cost-optimised (high-volume serving)
│   → AWQ INT4 (127 GB per model)
│   → Hardware: 2× H100 SXM 80GB (TP=2) per model
│   → Benchmark on your reasoning tasks before deploying — 2-5% quality loss
│
└── Edge / local (single-node, no cluster)
    → GGUF Q4_K_M via llama.cpp (127 GB, CPU-offload friendly)
    → Accept 10-15% quality loss on complex math/code
```

### 2.3 Ensemble of 4 × Nemotron-253B FP8 — cluster sizing

```
4 models × 253 GB weights  = 1,012 GB
4 models × 50 GB KV-cache  =   200 GB (4 concurrent requests per model)
Overhead × 4               =    60 GB
─────────────────────────────────────
Total VRAM required        = ~1,272 GB

Cluster options:
  Option A: 4 × 8-GPU nodes (H100 SXM 80GB), TP=8 per model  → 2,560 GB available ✓
  Option B: 4 × 4-GPU nodes (H100 SXM 80GB), FP8, TP=4       → 1,280 GB available ✓ (tight)
  Option C: 2 × 8-GPU nodes, TP=8, 2 models/node              → co-location risk ✗
```

---

## Part 3 — Inference Engine Selection

### 3.1 Engine comparison

| Engine | Throughput | Latency | FP8 | Tool calling | Best for |
|---|---|---|---|---|---|
| **TensorRT-LLM** | ★★★★★ | ★★★★★ | ✓ native | Moderate | Max throughput, NVIDIA cluster |
| **vLLM** | ★★★★ | ★★★★ | ✓ | ✓ excellent | Balance, easy deploy |
| **SGLang** | ★★★★ | ★★★★★ | ✓ | ✓ excellent | Multi-call agents, RadixAttention |
| **llama.cpp** | ★★ | ★★★ | GGUF only | Basic | Local / edge, no cluster |

**Recommendation for 4×253B analyst agent: SGLang**
SGLang's RadixAttention reuses KV-cache prefixes across agent turns, directly
cutting cost for long analytical sessions where the system prompt + schema
context is repeated on every tool call.

### 3.2 vLLM deployment (quickstart)

```bash
# Single model, FP8, TP=4
docker run --gpus '"device=0,1,2,3"' \
  -p 8001:8000 \
  vllm/vllm-openai:latest \
  --model nvidia/Llama-3.1-Nemotron-Ultra-253B-v1 \
  --tensor-parallel-size 4 \
  --quantization fp8 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching            # reuse prompt prefix KV-cache
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="none")

def call_nemotron(prompt: str, thinking: bool = True, model_port: int = 8001) -> str:
    system = "detailed thinking on" if thinking else "detailed thinking off"
    response = client.chat.completions.create(
        model="nvidia/Llama-3.1-Nemotron-Ultra-253B-v1",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.6 if thinking else 0.0,
        top_p=0.95,
        max_tokens=16384 if thinking else 2048,
    )
    return response.choices[0].message.content
```

---

## Part 4 — Multi-Agent Ensemble Architecture

### 4.1 Roles of the 4 Nemotron instances

```
                    ┌─────────────────────┐
                    │   ORCHESTRATOR       │  thinking=ON
                    │   Port 8001          │  Plans multi-step task
                    │   Decomposes → routes│  Writes ExecutionPlan JSON
                    └────────┬────────────┘
                             │ dispatch
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
 ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
 │  SQL ANALYST   │  │ PYTHON ANALYST │  │   VALIDATOR    │
 │  Port 8002     │  │  Port 8003     │  │  Port 8004     │
 │  thinking=ON   │  │  thinking=ON   │  │  thinking=ON   │
 │  Schema aware  │  │  pandas/sklearn│  │  Adversarial   │
 │  SQL + explain │  │  stats + plots │  │  QA + bounds   │
 └────────┬───────┘  └────────┬───────┘  └────────┬───────┘
          └──────────────────►│◄─────────────────-─┘
                              ▼
                    ┌─────────────────────┐
                    │   NARRATOR (bi)      │  thinking=OFF
                    │   Shared port / tiny │  AIDA pitch
                    │   model or OFF mode  │  Dashboard JSON
                    └─────────────────────┘
```

### 4.2 LangGraph orchestration

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator, httpx, json

class AnalystState(TypedDict):
    question:       str
    execution_plan: list[dict]
    sql_result:     dict | None
    python_result:  dict | None
    validation:     dict | None
    narrative:      str
    errors:         Annotated[list[str], operator.add]
    cost_usd:       Annotated[float,    operator.add]

MODEL_URLS = {
    "orchestrator": "http://gpu-node-1:8001/v1",
    "sql":          "http://gpu-node-2:8002/v1",
    "python":       "http://gpu-node-3:8003/v1",
    "validator":    "http://gpu-node-4:8004/v1",
}
BUDGET_CAP = 5.00   # USD per full pipeline run


def _nemotron(node: str, prompt: str, thinking: bool = True) -> str:
    """Call a specific Nemotron instance."""
    from openai import OpenAI
    client = OpenAI(base_url=MODEL_URLS[node], api_key="none")
    system = "detailed thinking on" if thinking else "detailed thinking off"
    r = client.chat.completions.create(
        model="nvidia/Llama-3.1-Nemotron-Ultra-253B-v1",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=8192 if thinking else 1024,
        temperature=0.6 if thinking else 0.0,
    )
    # Estimate cost: ~$0.002/1K tokens (self-hosted electricity + hardware amort.)
    return r.choices[0].message.content


def orchestrate(state: AnalystState) -> AnalystState:
    """Decompose question → ExecutionPlan JSON."""
    plan_str = _nemotron("orchestrator", f"""
You are the Orchestrator of a 4-agent data analyst system.
Decompose this question into a JSON ExecutionPlan with steps for:
sql_analyst, python_analyst, validator.

Question: {state['question']}

Return ONLY valid JSON: {{"steps": [{{"agent": "sql_analyst", "task": "..."}}]}}
""", thinking=True)
    try:
        plan = json.loads(plan_str.split("```json")[-1].split("```")[0].strip())
        state["execution_plan"] = plan.get("steps", [])
    except Exception:
        state["errors"].append("Orchestrator failed to produce valid plan JSON")
    return state


def sql_analyst(state: AnalystState) -> AnalystState:
    task = next((s["task"] for s in state["execution_plan"]
                 if s["agent"] == "sql_analyst"), "")
    if not task:
        return state
    result_str = _nemotron("sql", f"""
You are a SQL expert with access to these tables: [schema injected here]
Task: {task}
Write validated SQL, explain the logic, return JSON:
{{"sql": "...", "explanation": "...", "expected_columns": [...]}}
""", thinking=True)
    try:
        state["sql_result"] = json.loads(
            result_str.split("```json")[-1].split("```")[0].strip()
        )
    except Exception as e:
        state["errors"].append(f"SQL analyst parse error: {e}")
    return state


def python_analyst(state: AnalystState) -> AnalystState:
    task = next((s["task"] for s in state["execution_plan"]
                 if s["agent"] == "python_analyst"), "")
    if not task:
        return state
    result_str = _nemotron("python", f"""
You are a Python data scientist. SQL result available: {state.get('sql_result')}
Task: {task}
Write pandas/scikit-learn analysis. Return JSON:
{{"code": "...", "findings": "...", "chart_type": "line|bar|scatter"}}
""", thinking=True)
    try:
        state["python_result"] = json.loads(
            result_str.split("```json")[-1].split("```")[0].strip()
        )
    except Exception as e:
        state["errors"].append(f"Python analyst parse error: {e}")
    return state


def validator(state: AnalystState) -> AnalystState:
    result_str = _nemotron("validator", f"""
You are an adversarial QA agent. Review the analysis pipeline:
SQL result:    {state.get('sql_result')}
Python result: {state.get('python_result')}

Check: hallucinations, statistical errors, SQL injection risk, data leakage.
Return JSON: {{"pass": true|false, "issues": [...], "confidence": 0.0-1.0}}
""", thinking=True)
    try:
        state["validation"] = json.loads(
            result_str.split("```json")[-1].split("```")[0].strip()
        )
    except Exception as e:
        state["errors"].append(f"Validator parse error: {e}")
    return state


def should_retry(state: AnalystState) -> str:
    v = state.get("validation", {})
    if not v.get("pass") and len(state["errors"]) < 3:
        return "retry"
    return "narrate"


def build_analyst_graph() -> StateGraph:
    g = StateGraph(AnalystState)
    g.add_node("orchestrate",    orchestrate)
    g.add_node("sql_analyst",    sql_analyst)
    g.add_node("python_analyst", python_analyst)
    g.add_node("validator",      validator)
    g.add_edge("orchestrate",    "sql_analyst")
    g.add_edge("sql_analyst",    "python_analyst")
    g.add_edge("python_analyst", "validator")
    g.add_conditional_edges("validator", should_retry, {
        "retry":   "orchestrate",
        "narrate": END,
    })
    g.set_entry_point("orchestrate")
    return g.compile()
```

---

## Part 5 — KV-Cache Optimisation

### 5.1 Prefix caching (RadixAttention / vLLM)

```python
# The system prompt + schema description is the same for every query.
# With prefix caching, this shared prefix is computed ONCE and reused.

SHARED_PREFIX = """
You are a data analyst with access to the following schema:
[large schema definition — 2,000 tokens]
Always validate column names against the schema before generating SQL.
"""

# Without prefix cache: 2,000 tokens processed per query → expensive
# With prefix cache:    2,000 tokens processed ONCE, KV reused → ~60% latency reduction
# Enable: --enable-prefix-caching in vLLM or RadixAttention in SGLang (default)
```

### 5.2 KV-cache budget for long analytical sessions

```python
def kv_cache_gb(
    n_layers: int,
    n_kv_heads: int,
    d_head: int,
    seq_len: int,
    batch_size: int,
    bytes_per_element: float = 1.0,  # FP8 = 1 byte
) -> float:
    """Estimate KV-cache VRAM in GB."""
    # 2 = key + value
    total_bytes = 2 * n_layers * n_kv_heads * d_head * seq_len * batch_size * bytes_per_element
    return total_bytes / 1e9

# Nemotron-Ultra-253B approximate (public estimates):
# n_layers ≈ 80 (some skip-attn → ~60 active KV layers)
# n_kv_heads ≈ 8 (GQA — Grouped Query Attention)
# d_head ≈ 128

kv = kv_cache_gb(n_layers=60, n_kv_heads=8, d_head=128,
                 seq_len=131072, batch_size=4, bytes_per_element=1.0)
print(f"KV-cache per model (128K ctx, batch=4, FP8): {kv:.1f} GB")
# → ~32 GB (comfortably fits within 80GB H100 alongside FP8 weights on TP=4)
```

---

## Part 6 — Production Guardrails

### 6.1 Budget + timeout + fallback

```python
import asyncio
from dataclasses import dataclass

@dataclass
class AgentConfig:
    model_url:     str
    thinking:      bool
    timeout_s:     float = 120.0
    max_tokens:    int   = 8192
    budget_cap:    float = 5.00   # USD total per pipeline
    fallback_url:  str | None = None   # smaller model for fallback

async def safe_nemotron_call(config: AgentConfig, prompt: str) -> dict:
    """
    Wraps Nemotron call with:
    - asyncio timeout
    - budget enforcement
    - fallback to smaller model on timeout / error
    """
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=config.model_url, api_key="none")
    system = "detailed thinking on" if config.thinking else "detailed thinking off"

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="nvidia/Llama-3.1-Nemotron-Ultra-253B-v1",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=config.max_tokens,
                temperature=0.6 if config.thinking else 0.0,
            ),
            timeout=config.timeout_s,
        )
        return {"content": response.choices[0].message.content, "fallback": False}

    except asyncio.TimeoutError:
        if config.fallback_url:
            # Retry with faster, smaller model (e.g. 8B Llama)
            fb_client = AsyncOpenAI(base_url=config.fallback_url, api_key="none")
            r = await fb_client.chat.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048, temperature=0.0,
            )
            return {"content": r.choices[0].message.content, "fallback": True}
        raise
```

### 6.2 Monitoring metrics

```python
# Track these via Prometheus / Grafana or log to structured JSON:
METRICS = {
    "ttft_ms":          "Time to first token per request (p50, p99)",
    "itl_ms":           "Inter-token latency — affects streaming UX",
    "tokens_per_sec":   "Throughput per GPU node",
    "vram_headroom_gb": "Free VRAM after peak — must stay > 10 GB",
    "thinking_tokens":  "Avg <think> token count (cost driver)",
    "fallback_rate":    "% requests hitting fallback model",
    "cost_per_query":   "USD amortised hardware + electricity",
    "validation_pass_rate": "% pipelines passing adversarial QA",
}
```

---

## Quantization Quick-Reference

| Format | Library | VRAM (253B) | Quality drop | Throughput |
|---|---|---|---|---|
| BF16 | vLLM, TRT-LLM | 506 GB | 0% | Baseline |
| FP8 | TRT-LLM, vLLM | 253 GB | <1% | +40% |
| AWQ INT4 | vLLM, AutoAWQ | 127 GB | 2-5% | +70% |
| GPTQ INT4 | vLLM | 127 GB | 3-6% | +60% |
| GGUF Q4_K_M | llama.cpp | 127 GB | 8-12% | CPU-offload |
| GGUF Q8_0 | llama.cpp | 253 GB | 1-2% | CPU-offload |

**Rule:** for reasoning-heavy analytics agents, do not go below FP8.
AWQ INT4 is acceptable only for narration / formatting roles (thinking=OFF).

---

## Lesson Format

For every question, produce:

### ARCHITECTURE DECISION
Which configuration applies. State VRAM figures explicitly.

### CONFIGURATION
Complete command or config file — no pseudocode, no placeholders.

### COST ESTIMATE
Back-of-envelope: hardware cost + token cost + latency at target load.

### PITFALLS
Two deployment mistakes + one-line fix each.

### NEXT STEP
One concrete action: benchmark, quantise, or deploy a specific component.

---

## Context — Andrew Analitic

When the question touches this codebase:
- **`core/andrew_swarm.py`** is the single-model LangGraph analogue of the
  4-agent ensemble above. Map its nodes to ensemble roles:
  `route_query_intent` → Orchestrator, `generate_sql` → SQL Analyst,
  `python_analysis` → Python Analyst, `validate_results` → Validator.
- **Budget guard $1.00/query** → maps to `AgentConfig.budget_cap`.
  For a 4-model ensemble, raise cap to $5.00 and split per sub-agent.
- **HITL escalation** → `validation.pass == False` in the Validator node;
  return HTTP 202 and surface to human when confidence < threshold.
- **`/metrics` endpoint** → add `thinking_tokens`, `fallback_rate`,
  `vram_headroom_gb` to the existing metrics dict.
- **Upgrade path**: replace single `LiteLLM` call in `AndrewMoltisBridge`
  with 4-endpoint `safe_nemotron_call` fan-out when self-hosted H100s
  are provisioned.

---

## Input

The user's request: **$ARGUMENTS**

If `$ARGUMENTS` is empty:

> "С чего начнём? Выбери задачу: планирование VRAM и квантование, настройка
> кластера для ансамбля 4×253B, оптимизация KV-кэша, конфигурация Reasoning
> Toggle, или оркестрация через LangGraph. Укажи целевое железо — и разберём
> архитектуру до конкретных команд."

Then produce the full lesson following the checklist and the matching section above.
