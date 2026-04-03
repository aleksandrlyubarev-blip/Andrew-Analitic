"""
MongoDB-backed semantic memory store for Andrew-Analitic / RomeoFlexVision.

Drop-in replacement for InProcessSemanticStore when MONGODB_URI is set.
Adds persistence across restarts and Atlas Vector Search support.

Collections (all in the configured database, default: "romeoflexvision"):
  semantic_memory   — consolidated session memories with embeddings
  quality_control   — Robo QC defect records with embeddings

Atlas Vector Search index (create once in Atlas UI or via create_vector_index()):
  Collection: semantic_memory
  Field:      embedding  (knnVector, dimensions match your embed model)
  Similarity: cosine

Without Atlas, the store falls back to full-collection scan + in-Python cosine
similarity, which is fine for small stores (< ~10k records).
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from core.semantic_router import cosine_similarity

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

DEDUP_THRESHOLD = 0.92
DEFAULT_TTL_DAYS = 90
SEMANTIC_COLLECTION = "semantic_memory"
QC_COLLECTION = "quality_control"
VECTOR_INDEX_NAME = "semantic_vector_index"


# ── MongoDBSemanticStore ───────────────────────────────────────────────────────

class MongoDBSemanticStore:
    """
    Persistent semantic memory store backed by MongoDB.

    Implements the same interface as InProcessSemanticStore so it can be used
    as a drop-in replacement via the get_semantic_store() factory in memory.py.

    Deduplication uses cosine similarity computed in Python (full scan).
    When an Atlas Vector Search index named VECTOR_INDEX_NAME exists on the
    collection, search() uses $vectorSearch instead — call create_vector_index()
    once during setup to enable it.
    """

    def __init__(self, mongo_uri: str, db_name: str = "romeoflexvision") -> None:
        from pymongo import MongoClient, ASCENDING

        self._client = MongoClient(mongo_uri)
        self._db = self._client[db_name]
        self._col = self._db[SEMANTIC_COLLECTION]
        self._qc_col = self._db[QC_COLLECTION]

        # Basic indexes for fast lookups
        self._col.create_index([("tombstoned", ASCENDING)])
        self._col.create_index([("session_id", ASCENDING)])
        self._qc_col.create_index([("line_id", ASCENDING)])
        self._qc_col.create_index([("defect_code", ASCENDING)])
        self._qc_col.create_index([("created_at", ASCENDING)])

        self._atlas_search_available = False  # checked lazily on first search
        logger.info(
            f"MongoDBSemanticStore connected: "
            f"{self._client.address} / db={db_name}"
        )

    # ── Atlas Vector Search setup ──────────────────────────────────────────────

    def create_vector_index(self, dimensions: int = 1536) -> None:
        """
        Create Atlas Vector Search index on semantic_memory.embedding.

        Call once after creating the collection (requires Atlas M10+ or Atlas Local).
        dimensions=1536 matches text-embedding-3-small; adjust for other models.
        """
        try:
            self._col.create_search_index({
                "name": VECTOR_INDEX_NAME,
                "type": "vectorSearch",
                "definition": {
                    "fields": [{
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": dimensions,
                        "similarity": "cosine",
                    }]
                },
            })
            logger.info(
                f"Atlas Vector Search index '{VECTOR_INDEX_NAME}' created "
                f"(dimensions={dimensions})"
            )
        except Exception as exc:
            logger.warning(f"create_vector_index failed (non-Atlas?): {exc}")

    def _check_atlas_search(self) -> bool:
        """Probe whether $vectorSearch works on this cluster (cached)."""
        if self._atlas_search_available:
            return True
        try:
            list(self._col.aggregate([{
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": [0.0] * 3,
                    "numCandidates": 1,
                    "limit": 1,
                }
            }]))
            self._atlas_search_available = True
        except Exception:
            pass
        return self._atlas_search_available

    # ── SemanticRecord conversion ──────────────────────────────────────────────

    @staticmethod
    def _to_record(doc: Dict[str, Any]):
        """Convert a MongoDB document to a SemanticRecord-compatible dict."""
        from core.memory import SemanticRecord
        return SemanticRecord(
            record_id=doc["record_id"],
            content=doc["content"],
            embedding=doc["embedding"],
            created_at=doc["created_at"],
            last_accessed_at=doc["last_accessed_at"],
            ttl_days=doc.get("ttl_days", DEFAULT_TTL_DAYS),
            tombstoned=doc.get("tombstoned", False),
            stale_flagged=doc.get("stale_flagged", False),
            metadata=doc.get("metadata", {}),
        )

    # ── upsert ────────────────────────────────────────────────────────────────

    def upsert(
        self,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict] = None,
        ttl_days: int = DEFAULT_TTL_DAYS,
    ) -> Tuple[str, bool]:
        """
        Insert or merge a record (same semantics as InProcessSemanticStore).

        Dedup: cosine similarity against all active records; if any >= 0.92,
        merge content into the existing record and return (record_id, True).
        """
        active = list(self._col.find({"tombstoned": False}, {"_id": 0}))
        for doc in active:
            sim = cosine_similarity(embedding, doc["embedding"])
            if sim >= DEDUP_THRESHOLD:
                new_content = doc["content"]
                if content not in doc["content"]:
                    new_content = doc["content"] + " | " + content
                update: Dict[str, Any] = {
                    "$set": {
                        "content": new_content,
                        "last_accessed_at": time.time(),
                    }
                }
                if metadata:
                    update["$set"]["metadata"] = {**doc.get("metadata", {}), **metadata}
                self._col.update_one({"record_id": doc["record_id"]}, update)
                logger.debug(
                    f"MongoDBSemanticStore: merged into {doc['record_id']} "
                    f"(cosine={sim:.3f})"
                )
                return doc["record_id"], True

        # Insert new record
        record_id = str(uuid.uuid4())[:12]
        now = time.time()
        self._col.insert_one({
            "record_id": record_id,
            "content": content,
            "embedding": embedding,
            "created_at": now,
            "last_accessed_at": now,
            "ttl_days": ttl_days,
            "tombstoned": False,
            "stale_flagged": False,
            "metadata": metadata or {},
        })
        logger.debug(f"MongoDBSemanticStore: inserted {record_id}")
        return record_id, False

    # ── search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.65,
    ) -> list:
        """
        Retrieve top-k records above the relevance threshold.

        Uses Atlas $vectorSearch when the index is available, otherwise falls
        back to full-collection scan with in-Python cosine similarity.
        """
        if self._check_atlas_search():
            return self._search_atlas(query_embedding, top_k, threshold)
        return self._search_linear(query_embedding, top_k, threshold)

    def _search_atlas(
        self,
        query_embedding: List[float],
        top_k: int,
        threshold: float,
    ) -> list:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": top_k * 10,
                    "limit": top_k,
                    "filter": {"tombstoned": False},
                }
            },
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$match": {"score": {"$gte": threshold}}},
        ]
        docs = list(self._col.aggregate(pipeline))
        results = [self._to_record(d) for d in docs]
        self._touch_records([r.record_id for r in results])
        return results

    def _search_linear(
        self,
        query_embedding: List[float],
        top_k: int,
        threshold: float,
    ) -> list:
        active = list(self._col.find({"tombstoned": False}, {"_id": 0}))
        scored = [
            (cosine_similarity(query_embedding, doc["embedding"]), doc)
            for doc in active
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [
            self._to_record(doc)
            for sim, doc in scored[:top_k]
            if sim >= threshold
        ]
        self._touch_records([r.record_id for r in results])
        return results

    def _touch_records(self, record_ids: List[str]) -> None:
        if not record_ids:
            return
        self._col.update_many(
            {"record_id": {"$in": record_ids}},
            {"$set": {"last_accessed_at": time.time()}},
        )

    # ── all_active / tombstone ─────────────────────────────────────────────────

    def all_active(self) -> list:
        docs = list(self._col.find({"tombstoned": False}, {"_id": 0}))
        return [self._to_record(d) for d in docs]

    def tombstone(self, record_id: str) -> None:
        self._col.update_one(
            {"record_id": record_id},
            {"$set": {"tombstoned": True}},
        )
        logger.debug(f"MongoDBSemanticStore: tombstoned {record_id}")

    def __len__(self) -> int:
        return self._col.count_documents({"tombstoned": False})

    # ── Robo QC helpers ───────────────────────────────────────────────────────

    def insert_qc_result(
        self,
        line_id: str,
        defect_code: str,
        confidence: float,
        image_path: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a Robo QC defect record in the quality_control collection.

        Fields:
          line_id      — production line identifier (e.g. "line_07")
          defect_code  — defect classification code
          confidence   — model confidence score [0.0 – 1.0]
          image_path   — path or URI to the defect image
          embedding    — optional vision embedding for similarity search
          metadata     — arbitrary extra fields (batch_id, station, etc.)

        Returns the inserted qc_id.
        """
        qc_id = str(uuid.uuid4())
        doc = {
            "qc_id": qc_id,
            "line_id": line_id,
            "defect_code": defect_code,
            "confidence": confidence,
            "image_path": image_path,
            "embedding": embedding,
            "created_at": time.time(),
            "metadata": metadata or {},
        }
        self._qc_col.insert_one(doc)
        logger.debug(f"MongoDBSemanticStore: QC record {qc_id} inserted ({defect_code})")
        return qc_id

    def find_similar_defects(
        self,
        embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve QC records with similar vision embeddings.

        Uses Atlas $vectorSearch on quality_control.embedding when available,
        otherwise falls back to in-Python cosine similarity scan.

        To enable Atlas Vector Search on QC collection call:
            store.create_qc_vector_index(dimensions=<your_model_dim>)
        """
        if self._atlas_search_available:
            try:
                pipeline = [
                    {
                        "$vectorSearch": {
                            "index": "qc_vector_index",
                            "path": "embedding",
                            "queryVector": embedding,
                            "numCandidates": top_k * 10,
                            "limit": top_k,
                        }
                    },
                    {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
                    {"$match": {"score": {"$gte": threshold}}},
                    {"$project": {"_id": 0, "embedding": 0}},
                ]
                return list(self._qc_col.aggregate(pipeline))
            except Exception:
                pass

        # Linear fallback
        docs = list(self._qc_col.find(
            {"embedding": {"$ne": None}},
            {"_id": 0}
        ))
        scored = [
            (cosine_similarity(embedding, d["embedding"]), d)
            for d in docs
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {k: v for k, v in d.items() if k != "embedding"}
            for sim, d in scored[:top_k]
            if sim >= threshold
        ]

    def create_qc_vector_index(self, dimensions: int = 512) -> None:
        """
        Create Atlas Vector Search index on quality_control.embedding.

        Call once after first QC records are inserted (requires Atlas M10+).
        dimensions=512 is typical for vision embedding models.
        """
        try:
            self._qc_col.create_search_index({
                "name": "qc_vector_index",
                "type": "vectorSearch",
                "definition": {
                    "fields": [{
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": dimensions,
                        "similarity": "cosine",
                    }]
                },
            })
            logger.info(
                f"Atlas Vector Search index 'qc_vector_index' created "
                f"(dimensions={dimensions})"
            )
        except Exception as exc:
            logger.warning(f"create_qc_vector_index failed (non-Atlas?): {exc}")

    def close(self) -> None:
        self._client.close()
