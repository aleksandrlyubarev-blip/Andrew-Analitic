"""
scripts/setup_mongo.py
======================
MongoDB initialization script for Andrew-Analitic / RomeoFlexVision.

What it does:
  1. Connects to MongoDB using MONGODB_URI from environment (or --uri flag)
  2. Creates collections + basic indexes (idempotent)
  3. Optionally creates Atlas Vector Search indexes (--atlas flag)
  4. Prints a summary of all collections and index counts

Usage:
    # Basic setup (local or Atlas, no Vector Search):
    python scripts/setup_mongo.py

    # With Atlas Vector Search indexes (requires Atlas M10+ or Atlas Local):
    python scripts/setup_mongo.py --atlas

    # Custom URI / DB:
    python scripts/setup_mongo.py --uri "mongodb+srv://..." --db romeoflexvision

    # Just validate connection (no changes):
    python scripts/setup_mongo.py --check
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MongoDB setup for Andrew-Analitic")
    parser.add_argument(
        "--uri",
        default=os.getenv("MONGODB_URI", ""),
        help="MongoDB URI (default: $MONGODB_URI)",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("MONGODB_DB", "romeoflexvision"),
        help="Database name (default: romeoflexvision)",
    )
    parser.add_argument(
        "--atlas",
        action="store_true",
        help="Create Atlas Vector Search indexes (requires Atlas M10+ or Atlas Local)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only validate connection, make no changes",
    )
    parser.add_argument(
        "--semantic-dims",
        type=int,
        default=1536,
        dest="semantic_dims",
        help="Embedding dimensions for semantic_memory (default: 1536 = text-embedding-3-small)",
    )
    parser.add_argument(
        "--qc-dims",
        type=int,
        default=512,
        dest="qc_dims",
        help="Embedding dimensions for quality_control vision embeddings (default: 512)",
    )
    return parser.parse_args()


def check_connection(client, db_name: str) -> None:
    print(f"  Connecting to MongoDB... ", end="", flush=True)
    t0 = time.time()
    result = client.admin.command("ping")
    elapsed = time.time() - t0
    if result.get("ok") != 1:
        raise RuntimeError(f"ping returned: {result}")
    print(f"OK ({elapsed*1000:.0f}ms)")

    db = client[db_name]
    colls = db.list_collection_names()
    print(f"  Database: '{db_name}' — {len(colls)} existing collection(s)")
    for c in colls:
        count = db[c].count_documents({})
        print(f"    {c}: {count} document(s)")


def create_indexes(db) -> None:
    from pymongo import ASCENDING, DESCENDING

    print("\n  Creating indexes...")

    # ── semantic_memory ────────────────────────────────────────────────────────
    col = db["semantic_memory"]
    col.create_index([("tombstoned", ASCENDING)], name="idx_tombstoned")
    col.create_index([("stale_flagged", ASCENDING)], name="idx_stale_flagged")
    col.create_index([("session_id", ASCENDING)], name="idx_session_id")
    col.create_index([("record_id", ASCENDING)], unique=True, name="idx_record_id")
    col.create_index([("last_accessed_at", ASCENDING)], name="idx_last_accessed")
    print("    semantic_memory: 5 indexes")

    # ── quality_control ────────────────────────────────────────────────────────
    qc = db["quality_control"]
    qc.create_index([("qc_id", ASCENDING)], unique=True, name="idx_qc_id")
    qc.create_index([("line_id", ASCENDING)], name="idx_line_id")
    qc.create_index([("defect_code", ASCENDING)], name="idx_defect_code")
    qc.create_index([("created_at", DESCENDING)], name="idx_created_at")
    qc.create_index(
        [("line_id", ASCENDING), ("defect_code", ASCENDING)],
        name="idx_line_defect",
    )
    print("    quality_control: 5 indexes")

    # ── langgraph_checkpoints ──────────────────────────────────────────────────
    # LangGraph creates its own collection/indexes; we just ensure it exists.
    db.create_collection("langgraph_checkpoints", check_exists=True) if (
        "langgraph_checkpoints" not in db.list_collection_names()
    ) else None
    print("    langgraph_checkpoints: ready (LangGraph manages its own indexes)")


def create_atlas_indexes(db, semantic_dims: int, qc_dims: int) -> None:
    print("\n  Creating Atlas Vector Search indexes...")

    # ── semantic_memory ────────────────────────────────────────────────────────
    try:
        db["semantic_memory"].create_search_index({
            "name": "semantic_vector_index",
            "type": "vectorSearch",
            "definition": {
                "fields": [{
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": semantic_dims,
                    "similarity": "cosine",
                }]
            },
        })
        print(f"    semantic_memory.embedding: vectorSearch index (dims={semantic_dims}) — OK")
    except Exception as exc:
        if "already exists" in str(exc).lower():
            print(f"    semantic_memory.embedding: already exists — skipped")
        else:
            print(f"    semantic_memory.embedding: FAILED — {exc}")

    # ── quality_control ────────────────────────────────────────────────────────
    try:
        db["quality_control"].create_search_index({
            "name": "qc_vector_index",
            "type": "vectorSearch",
            "definition": {
                "fields": [{
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": qc_dims,
                    "similarity": "cosine",
                }]
            },
        })
        print(f"    quality_control.embedding: vectorSearch index (dims={qc_dims}) — OK")
    except Exception as exc:
        if "already exists" in str(exc).lower():
            print(f"    quality_control.embedding: already exists — skipped")
        else:
            print(f"    quality_control.embedding: FAILED — {exc}")


def print_summary(db) -> None:
    print("\n  Summary:")
    for name in ["semantic_memory", "quality_control", "langgraph_checkpoints"]:
        try:
            col = db[name]
            docs = col.count_documents({})
            indexes = col.index_information()
            print(f"    {name}: {docs} docs, {len(indexes)} index(es)")
        except Exception:
            print(f"    {name}: (empty)")


def main() -> None:
    args = parse_args()

    if not args.uri:
        print("ERROR: MONGODB_URI is not set.")
        print("  Set it in .env:  MONGODB_URI=mongodb://...")
        print("  Or pass:         --uri mongodb://...")
        sys.exit(1)

    try:
        from pymongo import MongoClient
    except ImportError:
        print("ERROR: pymongo not installed. Run: pip install pymongo")
        sys.exit(1)

    print(f"\nAndrew-Analitic — MongoDB Setup")
    print(f"{'='*45}")
    print(f"  URI: {args.uri[:40]}...")
    print(f"  DB:  {args.db}")

    client = MongoClient(args.uri)
    db = client[args.db]

    try:
        check_connection(client, args.db)

        if args.check:
            print("\n  --check mode: no changes made.")
            return

        create_indexes(db)

        if args.atlas:
            create_atlas_indexes(db, args.semantic_dims, args.qc_dims)
        else:
            print(
                "\n  Tip: run with --atlas to create Atlas Vector Search indexes "
                "(faster semantic search, requires Atlas M10+ or Atlas Local)"
            )

        print_summary(db)
        print(f"\n  Setup complete.\n")

    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
