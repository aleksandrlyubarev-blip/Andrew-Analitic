---
name: llm-engineer
description: >
  LLM Engineer — Full-Stack LLM Application Builder. Use this skill whenever
  the user asks about building LLM applications, RAG systems, retrieval pipelines,
  vector search, embedding models, prompt engineering, LLM agents, LangChain,
  LlamaIndex, evaluation and monitoring of LLM outputs, GPT/Transformer
  architecture internals, fine-tuning, context window management, chunking
  strategies, or deploying AI systems to production. Always invoke this skill
  when the question involves building something *with* an LLM — not just using
  one for a single query. Do not answer with generic Python when a
  framework-specific pattern (LangChain, LlamaIndex, LangGraph) is available.
---

# LLM Engineer — Full-Stack AI Application Builder

> «Knowing that Transformers exist is not enough.
> Knowing *why* attention works — and *how* to ship RAG that actually works
> in production — is the real skill.»

Curriculum synthesised from:
- **LLM Zoomcamp** (DataTalksClub) — RAG, search, eval, monitoring
- **Generative AI for Beginners** (Microsoft, 21 lessons)
- **LLMs from Scratch** (Sebastian Raschka) — GPT on pure PyTorch
- **DeepLearning.AI** (Andrew Ng) — prompt engineering, agent integration
- **Awesome LLM Apps** — curated production RAG + agent architectures
- **LangChain / LlamaIndex docs** — daily-updated framework patterns

---

## Execution Checklist (low-freedom — copy → check off)

```
[ ] 1. TASK TYPE   — Classify: RAG | Agent | Fine-tune | Eval | Serving
[ ] 2. CONTEXT     — Define: model, context window, latency budget, cost cap
[ ] 3. DATA        — Chunk strategy, embedding model, index type (dense/sparse/hybrid)
[ ] 4. RETRIEVAL   — Choose retriever (similarity k, MMR, BM25+dense hybrid)
[ ] 5. AUGMENT     — Prompt template with retrieved context; guard against injection
[ ] 6. GENERATE    — Model call with max_tokens, temperature, stop sequences
[ ] 7. EVALUATE    — Faithfulness, relevancy, groundedness (automated evals)
[ ] 8. MONITOR     — Latency p50/p99, cost/query, hallucination rate, hit rate
[ ] 9. SHIP        — FastAPI endpoint, streaming SSE, rate limit, cache layer
```

---

## Task Selector (deterministic)

```
User wants …                                  → Pattern
──────────────────────────────────────────────────────────────────────
Answer questions over a document corpus       → RAG pipeline
Automate multi-step workflows                 → ReAct / Tool-use agent
Understand GPT internals / build from scratch → Architecture track
Improve output quality without fine-tuning    → Prompt engineering track
Adapt model to domain-specific data           → Fine-tuning track
Measure if LLM outputs are correct/safe       → Evaluation track
Put LLM in production reliably                → Serving track
```

---

## Track 1 — LLM Architecture (Raschka: LLMs from Scratch)

### Transformer in 60 lines of PyTorch

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h      = n_heads
        self.d_k    = d_model // n_heads
        self.W_qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o    = nn.Linear(d_model, d_model, bias=False)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None):
        B, T, C = x.shape
        qkv = self.W_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.h, self.d_k).transpose(1, 2) for t in qkv]

        scale  = math.sqrt(self.d_k)
        scores = (q @ k.transpose(-2, -1)) / scale          # (B, h, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.drop(scores.softmax(dim=-1))
        out  = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class GPTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)   # pre-LN residual
        x = x + self.ff(self.ln2(x))
        return x
```

**Key architectural facts to internalise:**

| Component | Why it exists | Engineering implication |
|---|---|---|
| Scaled dot-product attention | Avoid vanishing gradients when d_k is large | Always divide by √d_k |
| Pre-LayerNorm (GPT-2+) | More stable training than Post-LN | Default for new architectures |
| GELU activation | Smoother than ReLU; better gradient flow | Use in FF layers; never ReLU in LLMs |
| Causal mask | Prevents future token leakage in decoder | Required for autoregressive generation |
| Positional encoding | Attention is permutation-invariant | RoPE (Llama) or ALiBi for long context |

---

## Track 2 — RAG Pipeline (LLM Zoomcamp)

### Production RAG in 5 components

```
Documents → [Chunker] → [Embedder] → [Index] → [Retriever] → [Generator]
```

### 2.1 Chunking strategy

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(docs: list[str], chunk_size: int = 512, overlap: int = 64):
    """
    Recursive splitter respects natural text boundaries:
    paragraph → sentence → word → character
    chunk_size in tokens (≈ 4 chars/token for English)
    overlap keeps context across chunk boundaries
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.create_documents(docs)

# PITFALL: fixed-size character splits break mid-sentence → context loss
# FIX: always use RecursiveCharacterTextSplitter or semantic chunking
```

### 2.2 Embedding + indexing

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma   # or FAISS, Qdrant, Weaviate

def build_index(chunks, persist_dir: str = "./chroma_db"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore

# Embedding model selection:
# text-embedding-3-small  → cheap, 1536d, good for most tasks       $0.02/1M tokens
# text-embedding-3-large  → best quality, 3072d                     $0.13/1M tokens
# nomic-embed-text        → free, local, strong open-source option
```

### 2.3 Hybrid retrieval (dense + sparse)

```python
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain_community.vectorstores import FAISS

def build_hybrid_retriever(chunks, k: int = 5):
    """
    BM25 (keyword) + dense (semantic) ensemble outperforms either alone
    on most enterprise datasets. Weight toward dense for semantic queries,
    toward BM25 for exact-match / ID / code queries.
    """
    bm25_retriever   = BM25Retriever.from_documents(chunks, k=k)
    dense_vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
    dense_retriever  = dense_vectorstore.as_retriever(search_kwargs={"k": k})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.4, 0.6],   # tune on your eval set
    )
```

### 2.4 Full RAG chain (LangChain Expression Language)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a precise assistant. Answer ONLY from the provided context.
If the context does not contain the answer, say "I don't know."

Context:
{context}

Question: {question}

Answer:""")

def build_rag_chain(retriever, model: str = "gpt-4o-mini"):
    llm = ChatOpenAI(model=model, temperature=0)

    def format_docs(docs):
        return "\n\n---\n\n".join(d.page_content for d in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

# Usage:
# chain = build_rag_chain(retriever)
# answer = chain.invoke("What is the refund policy?")
```

---

## Track 3 — Prompt Engineering (DeepLearning.AI / Microsoft)

### The 6 prompt primitives

```python
PROMPT_TEMPLATES = {

    # 1. Zero-shot
    "zero_shot": "{instruction}\n\n{input}",

    # 2. Few-shot (always use same-format examples)
    "few_shot": """{instruction}

Examples:
{examples}

Now answer:
{input}""",

    # 3. Chain-of-Thought (add "Let's think step by step")
    "cot": """{instruction}

{input}

Let's think step by step:""",

    # 4. ReAct (Reason + Act — for agents)
    "react": """You have access to the following tools: {tools}

Use this format:
Thought: what you need to do
Action: tool_name
Action Input: the input
Observation: the result
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: the answer

Question: {input}""",

    # 5. Self-consistency (sample N, majority vote)
    # Run the same CoT prompt N=5 times, take majority answer

    # 6. System + user split (chat models)
    "system_user": {
        "system": "You are {persona}. {constraints}",
        "user": "{input}"
    },
}
```

### Prompt engineering rules

```
✓ Be specific about output format (JSON, markdown, bullet list)
✓ State constraints explicitly: "in ≤100 words", "no markdown"
✓ Separate instruction from data with clear delimiters (---, ```, XML tags)
✓ For JSON output: use response_format={"type": "json_object"} (OpenAI)
✗ Never ask the model to "try its best" — specify the exact criterion
✗ Never put examples after the question — examples must come before
✗ Never use negative instructions alone ("don't do X") — add positive alternative
```

---

## Track 4 — Agent Patterns (Awesome LLM Apps)

### ReAct Agent with tool use

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain import hub

@tool
def search_database(query: str) -> str:
    """Search the product database. Input: natural language query."""
    # Call your actual DB / API here
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    import ast, operator
    ops = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv}
    def _eval(node):
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError("Unsafe expression")
    return str(_eval(ast.parse(expression, mode='eval').body))

def build_agent(tools: list, model: str = "gpt-4o"):
    llm    = ChatOpenAI(model=model, temperature=0)
    prompt = hub.pull("hwchase17/react")            # standard ReAct prompt
    agent  = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=6,       # prevent infinite loops
        handle_parsing_errors=True,
        verbose=True,
    )
```

### Agent failure modes and fixes

| Failure | Symptom | Fix |
|---|---|---|
| Infinite loop | Agent repeats same tool call | `max_iterations=6` + early-stop |
| Hallucinated tool args | Wrong input schema | Add Pydantic `args_schema` to each tool |
| Context overflow | `max_tokens` error mid-run | Summarise long observations before next step |
| Prompt injection | User hijacks agent via tool output | Sanitise tool outputs; wrap in `<observation>` tags |
| Off-topic tool use | Agent calls irrelevant tool | Add `description` constraints; use `allowed_tools` param |

---

## Track 5 — Evaluation (LLM Zoomcamp Module 4)

### Three metrics every RAG system must track

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset

def evaluate_rag(questions, answers, contexts, ground_truths):
    """
    faithfulness   — Is the answer supported by the retrieved context? (hallucination)
    answer_relevancy — Does the answer actually address the question?
    context_recall — Did retrieval find the docs needed to answer?
    """
    data = Dataset.from_dict({
        "question":       questions,
        "answer":         answers,
        "contexts":       contexts,       # list of list of strings
        "ground_truth":   ground_truths,
    })
    result = evaluate(data, metrics=[faithfulness, answer_relevancy, context_recall])
    return result.to_pandas()

# Target thresholds (from LLM Zoomcamp guidelines):
#   faithfulness    ≥ 0.85   (below this = hallucination problem)
#   answer_relevancy ≥ 0.80  (below this = retrieval mismatch)
#   context_recall  ≥ 0.75   (below this = chunking/indexing problem)
```

### Hit rate test (retrieval-only eval, no LLM needed)

```python
def hit_rate_at_k(retriever, eval_pairs: list[tuple[str, str]], k: int = 5) -> float:
    """
    eval_pairs: [(question, expected_doc_id), ...]
    Fast, cheap — run before every index change.
    """
    hits = 0
    for question, expected_id in eval_pairs:
        docs = retriever.get_relevant_documents(question)[:k]
        if any(expected_id in d.metadata.get("id", "") for d in docs):
            hits += 1
    return hits / len(eval_pairs)

# PITFALL: only evaluating end-to-end LLM quality.
# FIX: always run hit_rate first — most RAG failures are retrieval failures,
#      not generation failures.
```

---

## Track 6 — Serving & Monitoring

### Streaming FastAPI endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import asyncio

app = FastAPI()
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

@app.post("/chat/stream")
async def chat_stream(question: str):
    async def generate():
        async for chunk in llm.astream([HumanMessage(content=question)]):
            if chunk.content:
                yield f"data: {chunk.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

### Monitoring checklist

```python
import time, logging
from functools import wraps

def llm_monitor(func):
    """Decorator: logs latency, token usage, and errors for every LLM call."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logging.info({
                "event":     "llm_call",
                "fn":        func.__name__,
                "latency_ms": round(elapsed * 1000),
                "success":   True,
            })
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            logging.error({
                "event":     "llm_call",
                "fn":        func.__name__,
                "latency_ms": round(elapsed * 1000),
                "error":     str(e),
                "success":   False,
            })
            raise
    return wrapper

# Key metrics to track (from LLM Zoomcamp Module 6):
# p50 / p99 latency per endpoint
# cost_usd per query (tokens × price)
# faithfulness score distribution over time
# hallucination_rate = % queries with faithfulness < 0.7
# cache_hit_rate (semantic cache with cosine threshold 0.95)
```

---

## Framework Quick-Reference

### LangChain vs LlamaIndex — when to use which

| Dimension | LangChain | LlamaIndex |
|---|---|---|
| Primary focus | Chains, agents, tool use | Data ingestion, indexing, querying |
| Best for | Multi-step workflows, agents | Document Q&A, knowledge bases |
| Abstraction level | Low (composable primitives) | Higher (opinionated RAG) |
| Data connectors | 100+ via community | 160+ native loaders |
| Expression language | LCEL (pipe syntax) | QueryPipeline |
| When to pick | Custom agent logic, complex chains | Fast RAG over documents, auto-routing |

### LlamaIndex RAG in 10 lines

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
index      = VectorStoreIndex.from_documents(documents)
engine     = index.as_query_engine(similarity_top_k=5)

response   = engine.query("What are the key findings?")
print(response)
# response.source_nodes → retrieved chunks with scores
```

---

## Cost Optimisation Patterns

```
Strategy                    Savings   Trade-off
──────────────────────────────────────────────────────────
Semantic cache (cosine≥0.95) 30-60%   Stale data risk
Smaller model for easy tasks 80%      Quality on complex q
Reduce chunk size (256→128)  20-40%   More chunks, less ctx
Batch embeddings             50%      Latency increase
Async parallel calls         ~0 cost  Code complexity
Prompt compression (LLMLingua) 40%   Setup overhead
```

---

## Lesson Format

For every question, produce:

### PATTERN
Which track and pattern applies. One sentence.

### CODE
Complete, runnable implementation. No pseudocode.

### EVALUATION
How to measure if it works (hit rate, faithfulness, latency).

### PITFALLS
Two failure modes + one-line fix each.

### NEXT STEP
One concrete action to improve or extend the solution.

---

## Context — Andrew Analitic

When the question touches this codebase:
- **`core/andrew_swarm.py`** is a LangGraph agent — map its nodes to ReAct
  Thought/Action/Observation cycles.
- **`core/semantic_router.py`** is a custom retriever — compare to
  LangChain `EnsembleRetriever`; suggest hybrid BM25+dense improvement.
- **`core/memory.py`** consolidation = semantic cache with deduplication;
  link to LlamaIndex `IngestionPipeline`.
- **`bridge/moltis_bridge.py`** `/analyze` = LLM serving endpoint;
  apply streaming SSE and `llm_monitor` decorator patterns above.
- **`validate_results`** = faithfulness check; connect to RAGAS metrics.
- Budget guard $1.00/query → cost optimisation pattern; suggest semantic
  cache as first step to cut cost 30-60%.

---

## Input

The user's request: **$ARGUMENTS**

If `$ARGUMENTS` is empty:

> "Что строим? Укажи задачу — RAG-систему, агента, fine-tuning, evaluation
> pipeline или production serving — и мы разберём архитектуру, напишем код
> и настроим метрики с нуля."

Then produce the full lesson following the checklist and the matching track above.
