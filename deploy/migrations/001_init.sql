-- deploy/migrations/001_init.sql
-- Postgres 16 schema for the Andrew Swarm beta on Cloud SQL.
--
-- Apply with:
--   psql "host=127.0.0.1 port=5433 user=andrew dbname=andrew" -f 001_init.sql
-- (after starting cloud-sql-proxy on port 5433)
--
-- Idempotent: every CREATE uses IF NOT EXISTS.

-- ── Extensions ─────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ── Schemas ────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS andrew;
CREATE SCHEMA IF NOT EXISTS moltis;

-- ── Sales fixture (mirrors scripts/setup_db.py) ────────────
-- Lets the existing demo flows work against Postgres without code changes.
CREATE TABLE IF NOT EXISTS andrew.sales (
    id       BIGINT PRIMARY KEY,
    product  TEXT      NOT NULL,
    revenue  DOUBLE PRECISION NOT NULL,
    quantity INTEGER   NOT NULL,
    date     DATE      NOT NULL,
    region   TEXT      NOT NULL
);

INSERT INTO andrew.sales VALUES
    (1,  'Widget A', 15000.0, 120, '2025-01-15', 'North'),
    (2,  'Widget B', 23000.0, 200, '2025-01-20', 'South'),
    (3,  'Widget A', 18000.0, 150, '2025-02-10', 'East'),
    (4,  'Widget C', 31000.0, 280, '2025-02-15', 'West'),
    (5,  'Widget B', 27000.0, 230, '2025-03-01', 'North'),
    (6,  'Widget A', 12000.0, 100, '2025-03-10', 'South'),
    (7,  'Widget C', 35000.0, 310, '2025-04-05', 'East'),
    (8,  'Widget B', 19000.0, 160, '2025-04-20', 'West'),
    (9,  'Widget A', 22000.0, 180, '2025-05-15', 'North'),
    (10, 'Widget C', 29000.0, 250, '2025-06-01', 'South')
ON CONFLICT (id) DO NOTHING;

-- ── LangGraph checkpoints (durable graph state) ────────────
-- Schema follows langgraph.checkpoint.postgres.PostgresSaver layout.
-- Tables are created on first checkpointer.setup() call at runtime, but we
-- declare them here so the migration is the single source of truth.
CREATE TABLE IF NOT EXISTS andrew.checkpoints (
    thread_id     TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type          TEXT,
    checkpoint    JSONB NOT NULL,
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS andrew.checkpoint_writes (
    thread_id     TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id       TEXT NOT NULL,
    idx           INTEGER NOT NULL,
    channel       TEXT NOT NULL,
    type          TEXT,
    blob          BYTEA,
    task_path     TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

-- ── Hybrid memory (vector + full-text) for both Andrew and Moltis ──
-- Default embedding dim 1536 = OpenAI text-embedding-3-small.
-- Switch to VECTOR(3072) for text-embedding-3-large; pgvector supports up to 16k.
CREATE TABLE IF NOT EXISTS andrew.memory_chunks (
    id         BIGSERIAL PRIMARY KEY,
    user_id    TEXT NOT NULL,
    namespace  TEXT NOT NULL DEFAULT 'andrew',
    content    TEXT NOT NULL,
    embedding  VECTOR(1536),
    metadata   JSONB NOT NULL DEFAULT '{}'::jsonb,
    ts         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ivfflat: fast approximate cosine search. lists=100 is a sane default for
-- <50k rows; rebuild with more lists once you exceed that.
CREATE INDEX IF NOT EXISTS memory_chunks_embedding_idx
    ON andrew.memory_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- gin (tsvector) for keyword recall in the hybrid retriever.
CREATE INDEX IF NOT EXISTS memory_chunks_content_fts_idx
    ON andrew.memory_chunks
    USING gin (to_tsvector('english', content));

-- gin (trgm) for fuzzy matching short identifiers (product names, etc.).
CREATE INDEX IF NOT EXISTS memory_chunks_content_trgm_idx
    ON andrew.memory_chunks
    USING gin (content gin_trgm_ops);

CREATE INDEX IF NOT EXISTS memory_chunks_user_ns_idx
    ON andrew.memory_chunks (user_id, namespace, ts DESC);

-- ── Beta-tester request log (for cost/usage attribution) ───
CREATE TABLE IF NOT EXISTS andrew.request_log (
    id           BIGSERIAL PRIMARY KEY,
    user_slug    TEXT NOT NULL,
    route        TEXT NOT NULL,
    model        TEXT,
    tokens_in    INTEGER,
    tokens_out   INTEGER,
    latency_ms   INTEGER,
    cost_usd     NUMERIC(10, 6),
    status_code  INTEGER,
    error        TEXT,
    ts           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS request_log_user_ts_idx
    ON andrew.request_log (user_slug, ts DESC);
