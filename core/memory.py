"""
Sprint 5 §5.4 — Memory Consolidation & Staleness Sweep

Three subsystems:

1. InProcessSemanticStore
   An in-process semantic knowledge store that mirrors what gets written to
   the Moltis backend.  Needed for deduplication and staleness tracking
   because the Moltis API does not expose record-level list/scan endpoints.

2. ConsolidationEngine.consolidate_session()
   Called at session end (§6 step 6b, §5.4 "End of session"):
   - LLM summarises the episodic window (falls back to plain concat)
   - Embeds the summary
   - Dedup check: if cosine > 0.92 with an existing record → update it
   - Otherwise insert as a new semantic record

3. ConsolidationEngine.staleness_sweep()
   Scheduled daily (§5.4 "Staleness sweep"):
   - Flag records not accessed within TTL days
   - Tombstone records that were flagged in a previous sweep (grace period)
   - Returns a SweepResult with counts for observability
"""
from __future__ import annotations

import logging
import time
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.semantic_router import cosine_similarity, EmbedFn

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DEDUP_THRESHOLD = 0.92          # §5.4 — cosine threshold for merging records
DEFAULT_TTL_DAYS = 90           # §5.4 — semantic record lifetime
STALENESS_GRACE_DAYS = 7        # grace period before tombstoning
SESSION_SUMMARY_MAX_CHARS = 800 # max chars passed to the LLM summariser


# ── SemanticRecord ─────────────────────────────────────────────────────────────

@dataclass
class SemanticRecord:
    """
    One entry in the in-process semantic memory store.

    Mirrors what gets written to the Moltis backend so we can run dedup
    and staleness checks locally without requiring Moltis list endpoints.
    """
    record_id: str
    content: str
    embedding: List[float]
    created_at: float
    last_accessed_at: float
    ttl_days: int = DEFAULT_TTL_DAYS
    tombstoned: bool = False
    stale_flagged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400

    @property
    def idle_days(self) -> float:
        return (time.time() - self.last_accessed_at) / 86400

    def touch(self) -> None:
        self.last_accessed_at = time.time()


# ── InProcessSemanticStore ────────────────────────────────────────────────────

class InProcessSemanticStore:
    """
    Thread-safe in-process semantic memory store (§5.1).

    Acts as a local mirror alongside the Moltis backend.  The bridge writes to
    both; this store enables dedup, staleness sweep, and precision tracking.
    """

    def __init__(self) -> None:
        self._records: Dict[str, SemanticRecord] = {}
        self._lock = threading.RLock()

    def upsert(
        self,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict] = None,
        ttl_days: int = DEFAULT_TTL_DAYS,
    ) -> Tuple[str, bool]:
        """
        Insert or merge a record.

        Returns (record_id, was_merged).
        was_merged=True means an existing similar record was updated instead of
        inserting a new one (dedup threshold §5.4: cosine > 0.92).
        """
        with self._lock:
            # Dedup check
            for rec in self._records.values():
                if rec.tombstoned:
                    continue
                sim = cosine_similarity(embedding, rec.embedding)
                if sim >= DEDUP_THRESHOLD:
                    # Merge: append new content if it adds information
                    if content not in rec.content:
                        rec.content = rec.content + " | " + content
                    rec.touch()
                    if metadata:
                        rec.metadata.update(metadata)
                    logger.debug(f"SemanticStore: merged into {rec.record_id} (cosine={sim:.3f})")
                    return rec.record_id, True
            # Insert new
            record_id = str(uuid.uuid4())[:12]
            now = time.time()
            self._records[record_id] = SemanticRecord(
                record_id=record_id,
                content=content,
                embedding=embedding,
                created_at=now,
                last_accessed_at=now,
                ttl_days=ttl_days,
                metadata=metadata or {},
            )
            logger.debug(f"SemanticStore: inserted {record_id}")
            return record_id, False

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.65,
    ) -> List[SemanticRecord]:
        """
        Retrieve top-k non-tombstoned records above the relevance threshold.
        Touches (updates last_accessed_at on) each returned record.
        """
        with self._lock:
            candidates = [
                (cosine_similarity(query_embedding, r.embedding), r)
                for r in self._records.values()
                if not r.tombstoned
            ]
        candidates.sort(reverse=True, key=lambda x: x[0])
        results = [r for sim, r in candidates[:top_k] if sim >= threshold]
        for r in results:
            r.touch()
        return results

    def all_active(self) -> List[SemanticRecord]:
        with self._lock:
            return [r for r in self._records.values() if not r.tombstoned]

    def tombstone(self, record_id: str) -> None:
        with self._lock:
            rec = self._records.get(record_id)
            if rec:
                rec.tombstoned = True
                logger.debug(f"SemanticStore: tombstoned {record_id}")

    def set_stale_flag(self, record_id: str, value: bool) -> None:
        with self._lock:
            rec = self._records.get(record_id)
            if rec:
                rec.stale_flagged = value

    def __len__(self) -> int:
        with self._lock:
            return sum(1 for r in self._records.values() if not r.tombstoned)


# ── SweepResult ───────────────────────────────────────────────────────────────

@dataclass
class SweepResult:
    """Observability output from a staleness sweep run."""
    scanned: int
    newly_flagged: int
    tombstoned: int
    already_tombstoned: int
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        return (
            f"SweepResult(scanned={self.scanned}, flagged={self.newly_flagged}, "
            f"tombstoned={self.tombstoned}, already_dead={self.already_tombstoned})"
        )


# ── ConsolidationEngine ───────────────────────────────────────────────────────

class ConsolidationEngine:
    """
    Handles the write path for semantic memory (§5.4).

    consolidate_session() — called at session end.
    staleness_sweep()     — called on a daily schedule.
    """

    def __init__(
        self,
        store: InProcessSemanticStore,
        embed_fn: Optional[EmbedFn] = None,
        llm_model: str = "gpt-4o-mini",
    ) -> None:
        self.store = store
        self._embed_fn = embed_fn
        self.llm_model = llm_model

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> Optional[List[float]]:
        if self._embed_fn:
            try:
                return self._embed_fn(text)
            except Exception as exc:
                logger.warning(f"ConsolidationEngine._embed failed: {exc}")
                return None
        try:
            from litellm import embedding as _emb
            return _emb(model="text-embedding-3-small", input=[text]).data[0]["embedding"]
        except Exception as exc:
            logger.warning(f"ConsolidationEngine._embed (litellm) failed: {exc}")
            return None

    # ── Summarisation ─────────────────────────────────────────────────────────

    def _summarise(self, episodic: List[Dict[str, Any]]) -> str:
        """
        Summarise episodic entries into a short paragraph using the LLM.
        Falls back to plain concatenation if LLM is unavailable.
        """
        if not episodic:
            return ""
        raw = "\n".join(
            f"[{e.get('role','?')}] {e.get('content','')[:200]}"
            for e in episodic
        )[:SESSION_SUMMARY_MAX_CHARS]

        try:
            from litellm import completion
            resp = completion(
                model=self.llm_model,
                messages=[{
                    "role": "user",
                    "content": (
                        "Summarise the following analytics session into 2–3 sentences. "
                        "Focus on the data questions asked and the key findings. "
                        "Be concise.\n\n" + raw
                    ),
                }],
                max_tokens=120,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning(f"LLM summarise failed ({exc}); using concat fallback")
            return raw[:400]

    # ── consolidate_session ───────────────────────────────────────────────────

    def consolidate_session(
        self,
        session_id: str,
        episodic: List[Dict[str, Any]],
        extra_metadata: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Summarise a session's episodic memory and merge into the semantic store.

        §5.4 "End of session":
          - summarise episodic window
          - embed the summary
          - if cosine > 0.92 with existing record → update (dedup)
          - otherwise → insert new record

        Returns the record_id of the inserted/updated record, or None if the
        episodic window was empty or embedding failed.
        """
        if not episodic:
            logger.debug(f"consolidate_session: episodic empty for {session_id}, skipping")
            return None

        summary = self._summarise(episodic)
        if not summary:
            return None

        emb = self._embed(summary)
        if emb is None:
            logger.warning(f"consolidate_session: embedding failed for session {session_id}")
            return None

        metadata = {"session_id": session_id, "consolidated_at": time.time()}
        if extra_metadata:
            metadata.update(extra_metadata)

        record_id, was_merged = self.store.upsert(summary, emb, metadata=metadata)
        action = "merged" if was_merged else "inserted"
        logger.info(f"consolidate_session: {action} → {record_id} (session={session_id})")
        return record_id

    # ── staleness_sweep ───────────────────────────────────────────────────────

    def staleness_sweep(
        self,
        ttl_days: int = DEFAULT_TTL_DAYS,
        grace_days: int = STALENESS_GRACE_DAYS,
    ) -> SweepResult:
        """
        §5.4 "Staleness sweep" — daily job.

        Pass 1: flag records idle beyond ttl_days (stale_flagged = True).
        Pass 2: tombstone records that were already flagged AND are now past
                the grace period (i.e. flagged more than grace_days ago).

        This two-pass approach avoids immediately destroying records that might
        just be temporarily unaccessed.
        """
        now = time.time()
        scanned = newly_flagged = tombstoned = already_dead = 0

        active = self.store.all_active()
        scanned = len(active)

        for rec in active:
            if rec.tombstoned:
                already_dead += 1
                continue

            idle = rec.idle_days

            if rec.stale_flagged:
                # Check if grace period has elapsed since it was flagged
                # We use last_accessed_at as a proxy for "flagged_at" —
                # once flagged, it won't be touched again so idle keeps growing
                if idle >= ttl_days + grace_days:
                    self.store.tombstone(rec.record_id)
                    tombstoned += 1
                    logger.info(
                        f"staleness_sweep: tombstoned {rec.record_id} "
                        f"(idle={idle:.1f}d)"
                    )
            elif idle >= ttl_days:
                self.store.set_stale_flag(rec.record_id, True)
                newly_flagged += 1
                logger.info(
                    f"staleness_sweep: flagged {rec.record_id} "
                    f"(idle={idle:.1f}d > ttl={ttl_days}d)"
                )

        result = SweepResult(
            scanned=scanned,
            newly_flagged=newly_flagged,
            tombstoned=tombstoned,
            already_tombstoned=already_dead,
        )
        logger.info(f"staleness_sweep: {result}")
        return result


# ── Store factory ─────────────────────────────────────────────────────────────

def get_semantic_store(mongo_uri: Optional[str] = None):
    """
    Return the appropriate semantic store backend.

    When MONGODB_URI is set (or mongo_uri is provided), returns a
    MongoDBSemanticStore for persistent, cross-restart memory.
    Falls back to InProcessSemanticStore when MongoDB is not configured.
    """
    import os
    uri = mongo_uri or os.getenv("MONGODB_URI", "")
    if uri:
        try:
            from core.mongodb_store import MongoDBSemanticStore
            db_name = os.getenv("MONGODB_DB", "romeoflexvision")
            store = MongoDBSemanticStore(mongo_uri=uri, db_name=db_name)
            logger.info("Semantic store: MongoDB backend active")
            return store
        except Exception as exc:
            logger.warning(
                f"MongoDBSemanticStore init failed ({exc}); "
                "falling back to in-process store"
            )
    logger.info("Semantic store: in-process backend (volatile)")
    return InProcessSemanticStore()


# ── Module-level singletons ───────────────────────────────────────────────────

_semantic_store = get_semantic_store()
_consolidation_engine = ConsolidationEngine(store=_semantic_store)
