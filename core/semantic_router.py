"""
Semantic Router — Sprint 5 §3 (Capability Registry) + §4 (Routing Engine)

Replaces the keyword-based route_query_intent with embedding-based cosine
similarity scoring against a Capability Registry.

Scoring formula (§4.1):
  score(query, agent) =
      α × cosine(embed(query), agent.description_embedding)
    + β × max(cosine(embed(query), ex) for ex in agent.example_embeddings)
    + γ × agent.metadata.cost_weight
    + δ × cosine(embed(session_summary), agent.description_embedding)

Default weights (§4.2): α=0.30  β=0.60  γ=0.10  δ=0.15
Fallback threshold (§4.3): 0.72

If the top candidate scores below the threshold, an LLM classifier is called
via LiteLLM (structured JSON output).  When both embedding and LLM are
unavailable (e.g. no API key in test environments) a deterministic keyword
heuristic is used so the system never throws.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Cosine similarity ─────────────────────────────────────────────────────────

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity in [0, 1] between two embedding vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-9:
        return 0.0
    return float(np.clip(np.dot(va, vb) / denom, -1.0, 1.0))


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class CapabilityMeta:
    domain: str           # analytics | education | execution
    cost_weight: float    # 0.0 (cheapest) – 1.0 (most expensive)
    latency_hint: str     # fast | medium | slow
    requires_sandbox: bool = False


@dataclass
class CapabilityRecord:
    """§3.1 — one registered agent/tool with versioned embeddings."""
    agent_id: str
    version: int
    description: str                        # raw text; kept for hot-reload diff
    description_embedding: List[float]
    example_texts: List[str]                # kept for hot-reload diff
    example_embeddings: List[List[float]]
    metadata: CapabilityMeta


@dataclass
class RoutingLog:
    """§7.4 — full routing decision log entry."""
    timestamp: str
    query_hash: str
    selected_agent: str
    top_score: float
    runner_up_score: float
    registry_versions: Dict[str, int]
    fallback_used: bool
    session_length: int
    memory_records_retrieved: int
    memory_precision: Optional[float] = None   # filled post-hoc by bridge

    def as_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "query_hash": self.query_hash,
            "selected_agent": self.selected_agent,
            "top_score": self.top_score,
            "runner_up_score": self.runner_up_score,
            "registry_versions": self.registry_versions,
            "fallback_used": self.fallback_used,
            "session_length": self.session_length,
            "memory_records_retrieved": self.memory_records_retrieved,
            "memory_precision": self.memory_precision,
        }


# ── Procedural Memory (§5.1 table) ────────────────────────────────────────────

@dataclass
class ProceduralRecord:
    """One learned routing pattern: a past query embedding + outcome counts."""
    agent_id: str
    query_embedding: List[float]
    success_count: int = 0
    fail_count: int = 0
    last_updated: float = field(default_factory=lambda: __import__("time").time())

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total > 0 else 0.5


class ProceduralStore:
    """
    Stores past query→agent routing outcomes for use as a routing prior.

    Consumed only by the routing layer (§5.1).  Not passed to agents.
    Updated asynchronously after successful/failed completions (§5.4).
    """

    # Two thresholds:
    #   MERGE_SIM  — past record is "same query" → merge counts rather than insert
    #   BIAS_SIM   — minimum similarity for a past record to influence routing
    MERGE_SIM = 0.95
    BIAS_SIM = 0.75

    def __init__(self) -> None:
        self._records: List[ProceduralRecord] = []
        self._lock = threading.RLock()

    def record(self, query_embedding: List[float], agent_id: str, success: bool) -> None:
        """Record a routing outcome.  Merges with an existing similar record if found."""
        import time as _time
        with self._lock:
            for rec in self._records:
                if (rec.agent_id == agent_id
                        and cosine_similarity(query_embedding, rec.query_embedding) >= self.MERGE_SIM):
                    if success:
                        rec.success_count += 1
                    else:
                        rec.fail_count += 1
                    rec.last_updated = _time.time()
                    return
            self._records.append(ProceduralRecord(
                agent_id=agent_id,
                query_embedding=query_embedding,
                success_count=1 if success else 0,
                fail_count=0 if success else 1,
            ))

    def bias(
        self,
        query_embedding: List[float],
        agent_id: str,
        top_k: int = 5,
    ) -> float:
        """
        Return a score in [0, 1] representing how well this agent handled
        past queries similar to the current one.

        Returns 0.0 when no prior data exists (no bias either way).
        """
        with self._lock:
            candidates = [
                (cosine_similarity(query_embedding, r.query_embedding), r)
                for r in self._records
                if r.agent_id == agent_id
                and cosine_similarity(query_embedding, r.query_embedding) >= self.BIAS_SIM
            ]
        if not candidates:
            return 0.0
        # Weight success_rate by similarity, take top-k
        top = sorted(candidates, reverse=True)[:top_k]
        total_weight = sum(sim for sim, _ in top)
        if total_weight < 1e-9:
            return 0.0
        return sum(sim * rec.success_rate for sim, rec in top) / total_weight

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


# ── Capability Registry ───────────────────────────────────────────────────────

class CapabilityRegistry:
    """
    Thread-safe in-memory store of CapabilityRecords (§3.2).
    Versions are monotonic per-agent counters (§3.3).
    """

    def __init__(self) -> None:
        self._records: Dict[str, CapabilityRecord] = {}
        self._lock = threading.RLock()

    def register(self, record: CapabilityRecord) -> None:
        with self._lock:
            existing = self._records.get(record.agent_id)
            if existing:
                record.version = existing.version + 1
            self._records[record.agent_id] = record
        logger.debug(f"Registry: {record.agent_id} v{record.version} registered")

    def get(self, agent_id: str) -> Optional[CapabilityRecord]:
        with self._lock:
            return self._records.get(agent_id)

    def all_records(self) -> List[CapabilityRecord]:
        with self._lock:
            return list(self._records.values())

    def versions(self) -> Dict[str, int]:
        with self._lock:
            return {aid: r.version for aid, r in self._records.items()}

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


# ── Agent profiles (text; embeddings are computed at build time) ───────────────

_AGENT_PROFILES: Dict[str, Dict] = {
    "reasoning_math": {
        "description": (
            "Complex statistical modeling, time-series forecasting with ARIMA and Prophet, "
            "Monte Carlo simulations, neural networks, deep learning, optimization problems, "
            "regression analysis, clustering, dimensionality reduction, causal inference, "
            "Bayesian analysis, A/B testing, confidence intervals, hypothesis testing."
        ),
        "examples": [
            "Forecast next quarter revenue using ARIMA and confidence intervals",
            "Run Monte Carlo simulation for sales growth over 12 months",
            "Calculate CAGR and break-even point for product lines",
            "Build a neural network to predict churn probability",
            "Apply k-means clustering to customer segments",
            "Optimize marketing budget allocation using linear programming",
            "Perform Bayesian A/B test on conversion rate",
            "Detect anomalies in time-series data using isolation forest",
        ],
        "domain": "analytics",
        "cost_weight": 0.8,
        "latency_hint": "slow",
        "requires_sandbox": True,
    },
    "standard_analytics": {
        "description": (
            "SQL queries, data aggregation, GROUP BY, JOINs, filtering, sorting, "
            "business KPIs, pivot tables, cohort analysis, funnel analysis, "
            "revenue summaries, regional breakdowns, trend reports, customer segmentation."
        ),
        "examples": [
            "Total revenue by region for last quarter",
            "Top 10 customers by order value",
            "Monthly active users trend over 6 months",
            "Cohort retention for users who signed up in January",
            "Sales funnel conversion rates by product category",
            "Average order value broken down by channel",
            "Join orders with products and show profit margin per SKU",
            "Revenue by sales rep with ranking",
        ],
        "domain": "analytics",
        "cost_weight": 0.5,
        "latency_hint": "medium",
        "requires_sandbox": False,
    },
    "analytics_fastlane": {
        "description": (
            "Simple aggregations, quick totals, counts, averages, basic charts, "
            "bar charts, pie charts, line charts, single-metric summaries, "
            "straightforward data lookups with no complex joins."
        ),
        "examples": [
            "Show total revenue as a bar chart",
            "How many orders were placed this week?",
            "Average order value this month",
            "Pie chart of revenue by product category",
            "Count customers by country",
            "Line chart of daily signups",
            "Sum of sales by region",
            "What is the total number of active products?",
        ],
        "domain": "analytics",
        "cost_weight": 0.2,
        "latency_hint": "fast",
        "requires_sandbox": False,
    },
}

EmbedFn = Callable[[str], List[float]]


def build_default_registry(embed_fn: EmbedFn) -> CapabilityRegistry:
    """
    Build the default CapabilityRegistry by embedding all agent profiles.
    Call once at startup; hot-reloads call registry.register() again which
    auto-increments the version counter (§3.3).
    """
    registry = CapabilityRegistry()
    for agent_id, profile in _AGENT_PROFILES.items():
        desc_emb = embed_fn(profile["description"])
        example_embs = [embed_fn(ex) for ex in profile["examples"]]
        record = CapabilityRecord(
            agent_id=agent_id,
            version=1,
            description=profile["description"],
            description_embedding=desc_emb,
            example_texts=profile["examples"],
            example_embeddings=example_embs,
            metadata=CapabilityMeta(
                domain=profile["domain"],
                cost_weight=profile["cost_weight"],
                latency_hint=profile["latency_hint"],
                requires_sandbox=profile.get("requires_sandbox", False),
            ),
        )
        registry.register(record)
    logger.info(f"CapabilityRegistry built: {list(_AGENT_PROFILES.keys())}")
    return registry


# ── Semantic Router ───────────────────────────────────────────────────────────

class SemanticRouter:
    """
    Full routing pipeline: embed → score → decide → log (§4.4).

    Graceful degradation order:
      1. Embedding cosine scoring  (primary path, <50ms p95)
      2. LLM classifier fallback   (when top score < threshold, ~300ms)
      3. Keyword heuristic         (when embedding + LLM both unavailable)
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        embed_model: str = "text-embedding-3-small",
        threshold: Optional[float] = None,
        alpha: float = 0.30,
        beta: float = 0.60,
        gamma: float = 0.10,
        delta: float = 0.15,
        epsilon: float = 0.10,          # procedural memory bias weight
        procedural_store: Optional["ProceduralStore"] = None,
    ) -> None:
        self.registry = registry
        self.embed_model = embed_model
        self.threshold = threshold if threshold is not None else float(
            os.getenv("SEMANTIC_ROUTER_THRESHOLD", "0.72")
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.procedural_store = procedural_store

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed(self, text: str) -> Optional[List[float]]:
        """Call LiteLLM embedding API; returns None on any failure."""
        try:
            from litellm import embedding as litellm_embed
            resp = litellm_embed(model=self.embed_model, input=[text])
            return resp.data[0]["embedding"]
        except Exception as exc:
            logger.warning(f"SemanticRouter.embed failed ({type(exc).__name__}): {exc}")
            return None

    # ── Scoring formula (§4.1) ────────────────────────────────────────────────

    def score(
        self,
        query_emb: List[float],
        record: CapabilityRecord,
        session_emb: Optional[List[float]] = None,
    ) -> float:
        """Apply §4.1 scoring formula for one candidate record."""
        desc_sim = cosine_similarity(query_emb, record.description_embedding)
        ex_sim = max(
            (cosine_similarity(query_emb, ex) for ex in record.example_embeddings),
            default=0.0,
        )
        cost_term = record.metadata.cost_weight
        sess_term = (
            cosine_similarity(session_emb, record.description_embedding)
            if session_emb else 0.0
        )
        return (
            self.alpha * desc_sim
            + self.beta * ex_sim
            + self.gamma * cost_term
            + self.delta * sess_term
        )

    # ── Route (§4.4) ─────────────────────────────────────────────────────────

    def route(
        self,
        query: str,
        session_summary: Optional[str] = None,
        session_length: int = 0,
        memory_records_retrieved: int = 0,
    ) -> RoutingLog:
        """
        Full routing pipeline.  Returns a RoutingLog; read .selected_agent for
        the decision.  Never raises — falls back to keyword heuristic on error.
        """
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        records = self.registry.all_records()
        fallback_used = False
        runner_up_score = 0.0

        if not records:
            # Registry empty (e.g. embedding model not configured) → keyword
            selected_agent = self._keyword_fallback(query)
            top_score = 0.0
            fallback_used = True
        else:
            query_emb = self.embed(query)
            session_emb = self.embed(session_summary) if session_summary else None

            if query_emb is None:
                selected_agent = self._keyword_fallback(query)
                top_score = 0.0
                fallback_used = True
            else:
                # Base cosine scores
                base_scored = [
                    (rec.agent_id, self.score(query_emb, rec, session_emb))
                    for rec in records
                ]
                # Add procedural bias (ε term) when store has data
                if self.procedural_store:
                    scored = sorted(
                        [
                            (aid, s + self.epsilon * self.procedural_store.bias(query_emb, aid))
                            for aid, s in base_scored
                        ],
                        key=lambda x: x[1],
                        reverse=True,
                    )
                else:
                    scored = sorted(base_scored, key=lambda x: x[1], reverse=True)

                top_agent_id, top_score = scored[0]
                runner_up_score = scored[1][1] if len(scored) > 1 else 0.0

                if top_score >= self.threshold:
                    selected_agent = top_agent_id
                else:
                    selected_agent, top_score = self._llm_fallback(
                        query, [r[0] for r in scored]
                    )
                    fallback_used = True

        log = RoutingLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            query_hash=query_hash,
            selected_agent=selected_agent,
            top_score=round(top_score, 4),
            runner_up_score=round(runner_up_score, 4),
            registry_versions=self.registry.versions(),
            fallback_used=fallback_used,
            session_length=session_length,
            memory_records_retrieved=memory_records_retrieved,
            memory_precision=None,
        )
        logger.info(
            f"SemanticRouter → {selected_agent} "
            f"(score={top_score:.3f}, fallback={fallback_used}, "
            f"runner_up={runner_up_score:.3f})"
        )
        return log

    # ── Procedural feedback (§5.4) ────────────────────────────────────────────

    def record_outcome(self, query: str, agent_id: str, success: bool) -> None:
        """
        Record a completed routing outcome in the procedural store (§5.4).
        No-ops when the procedural store is not configured or embedding fails.
        Called asynchronously after task completion — never on the hot path.
        """
        if self.procedural_store is None:
            return
        emb = self.embed(query)
        if emb:
            self.procedural_store.record(emb, agent_id, success)
            logger.debug(
                f"Procedural: recorded {agent_id} {'OK' if success else 'FAIL'} "
                f"(store size={len(self.procedural_store)})"
            )

    # ── LLM fallback classifier (§4.3) ────────────────────────────────────────

    def _llm_fallback(self, query: str, candidates: List[str]) -> Tuple[str, float]:
        """LLM-based classifier via LiteLLM when cosine score < threshold."""
        try:
            from litellm import completion
            prompt = (
                f"Classify this analytics query to exactly one agent. "
                f"Agents: {', '.join(candidates)}. "
                f"Return JSON only: {{\"agent_id\": \"<id>\", \"confidence\": <0.0-1.0>}}\n\n"
                f"Query: {query}"
            )
            resp = completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=80,
            )
            data = json.loads(resp.choices[0].message.content)
            agent_id = data.get("agent_id", candidates[0])
            confidence = float(data.get("confidence", 0.5))
            if agent_id not in candidates:
                agent_id = candidates[0]
            logger.info(f"LLM fallback → {agent_id} (confidence={confidence:.2f})")
            return agent_id, confidence
        except Exception as exc:
            logger.warning(f"LLM fallback failed: {exc} — using keyword heuristic")
            return self._keyword_fallback(query), 0.0

    # ── Keyword heuristic (last resort) ───────────────────────────────────────

    _REASONING_MATH_KEYWORDS = frozenset({
        "arima", "forecast", "monte carlo", "simulation", "neural", "cluster",
        "regression", "optimize", "bayesian", "anomaly", "machine learning",
        "deep learning", "gradient", "bootstrap", "confidence interval",
        "lstm", "prophet", "linear programming", "eigenvalue", "causal",
    })
    _FASTLANE_KEYWORDS = frozenset({
        "bar chart", "pie chart", "line chart", "total", "how many",
        "sum of", "count", "average", "simple", "quick",
    })

    def _keyword_fallback(self, query: str) -> str:
        ql = query.lower()
        if any(kw in ql for kw in self._REASONING_MATH_KEYWORDS):
            return "reasoning_math"
        if any(kw in ql for kw in self._FASTLANE_KEYWORDS):
            return "analytics_fastlane"
        return "standard_analytics"


# ── Module-level singleton ────────────────────────────────────────────────────
# Built lazily: the registry is empty until build_default_registry() is called.
# route_query_intent calls .route() which gracefully falls back to keyword
# heuristic when the registry is empty (no API key / test environment).

_registry = CapabilityRegistry()
_procedural_store = ProceduralStore()
_semantic_router = SemanticRouter(registry=_registry, procedural_store=_procedural_store)


def init_registry(embed_fn: Optional[EmbedFn] = None) -> None:
    """
    Populate the module-level registry with embedded capability profiles.

    Call once at application startup (e.g. in AndrewExecutor.__init__).
    If embed_fn is None, a LiteLLM-backed embed function is used.
    No-ops if embedding fails; keyword fallback remains active.
    """
    global _registry, _semantic_router

    if embed_fn is None:
        def embed_fn(text: str) -> List[float]:
            try:
                from litellm import embedding as _emb
                return _emb(model="text-embedding-3-small", input=[text]).data[0]["embedding"]
            except Exception as exc:
                logger.warning(f"init_registry embed failed: {exc}")
                # Return a zero vector — cosine will be 0, keyword fallback takes over
                return [0.0] * 1536

    try:
        _registry = build_default_registry(embed_fn)
        _semantic_router = SemanticRouter(
            registry=_registry, procedural_store=_procedural_store
        )
        logger.info("SemanticRouter: registry initialised with embedding model")
    except Exception as exc:
        logger.warning(f"SemanticRouter: registry init failed ({exc}); keyword fallback active")
