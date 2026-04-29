"""
scripts/setup_db.py — Create the Andrew Analytics sample database.

Routes by URL scheme so the same script works for local SQLite dev and the
GCP beta's Cloud SQL Postgres:

  sqlite:///path/to/andrew.db        → SQLite file at <path>
  postgresql://user:pw@host/db       → libpq Postgres
  postgresql+psycopg://user@/db      → SQLAlchemy-style Postgres (rewritten)

Usage:
    # Local (SQLite, default — same as before)
    python3 scripts/setup_db.py
    python3 scripts/setup_db.py /tmp/andrew.db

    # Postgres (Cloud SQL Auth Proxy on port 5433):
    DATABASE_URL=postgresql://andrew:$DB_PASS@127.0.0.1:5433/andrew \
        python3 scripts/setup_db.py

    # Postgres (Cloud Run unix socket):
    DATABASE_URL=postgresql+psycopg://andrew@/andrew?host=/cloudsql/proj:reg:inst \
        python3 scripts/setup_db.py

The Postgres path writes into schema ``andrew`` (created by
``deploy/migrations/001_init.sql``); falls back to creating the schema if
absent so this script alone is enough for ad-hoc dev environments.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

ROWS = [
    (1,  "Widget A", 15000.0, 120, "2025-01-15", "North"),
    (2,  "Widget B", 23000.0, 200, "2025-01-20", "South"),
    (3,  "Widget A", 18000.0, 150, "2025-02-10", "East"),
    (4,  "Widget C", 31000.0, 280, "2025-02-15", "West"),
    (5,  "Widget B", 27000.0, 230, "2025-03-01", "North"),
    (6,  "Widget A", 12000.0, 100, "2025-03-10", "South"),
    (7,  "Widget C", 35000.0, 310, "2025-04-05", "East"),
    (8,  "Widget B", 19000.0, 160, "2025-04-20", "West"),
    (9,  "Widget A", 22000.0, 180, "2025-05-15", "North"),
    (10, "Widget C", 29000.0, 250, "2025-06-01", "South"),
]


def _resolve_target(argv: list[str]) -> tuple[str, str]:
    """Return ``(backend, target)`` where backend is ``"sqlite"`` or ``"postgres"``.

    Precedence:
      1. CLI arg that looks like a URL  → use it
      2. CLI arg that's a path          → SQLite at that path
      3. ``DATABASE_URL`` env var       → use it
      4. Default                        → SQLite at <repo>/andrew.db
    """
    if len(argv) > 1:
        arg = argv[1]
        if "://" in arg:
            return _classify(arg)
        # Plain path → SQLite
        return "sqlite", arg

    env = os.getenv("DATABASE_URL", "").strip()
    if env:
        return _classify(env)

    default = str(Path(__file__).resolve().parent.parent / "andrew.db")
    return "sqlite", default


def _classify(url: str) -> tuple[str, str]:
    scheme = url.split("://", 1)[0].lower()
    base = scheme.split("+", 1)[0]  # strip SQLAlchemy driver suffix
    if base == "sqlite":
        parsed = urlparse(url)
        # urlparse drops the leading slash for SQLAlchemy-style triple-slash.
        path = parsed.path.lstrip("/") if parsed.netloc == "" else parsed.path
        # Empty path = in-memory; reject that as setup target.
        if not path:
            raise SystemExit("In-memory SQLite (':memory:') has no fixture target.")
        return "sqlite", path
    if base in {"postgresql", "postgres"}:
        return "postgres", url
    raise SystemExit(f"Unsupported DATABASE_URL scheme: {url!r}")


def _setup_sqlite(path: str) -> None:
    import sqlite3

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("DROP TABLE IF EXISTS sales")
        conn.execute(
            """
            CREATE TABLE sales (
                id       INTEGER PRIMARY KEY,
                product  TEXT    NOT NULL,
                revenue  FLOAT   NOT NULL,
                quantity INTEGER NOT NULL,
                date     DATE    NOT NULL,
                region   TEXT    NOT NULL
            )
            """
        )
        conn.executemany("INSERT INTO sales VALUES (?,?,?,?,?,?)", ROWS)
        conn.commit()
    finally:
        conn.close()
    print(f"SQLite database created: {path}")
    print(f"  {len(ROWS)} rows | 3 products (Widget A/B/C) | 4 regions")
    print()
    print("Next step:")
    print(f'  export DATABASE_URL="sqlite:///{path}"')


def _setup_postgres(url: str) -> None:
    try:
        import psycopg  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dep
        raise SystemExit(
            "psycopg not installed. Install with: pip install 'psycopg[binary]>=3.2.0'"
        ) from exc

    # PostgresSaver / psycopg expects libpq-style URLs. Strip SQLAlchemy
    # driver suffix (postgresql+psycopg → postgresql) so psycopg parses it.
    libpq_url = url
    if "+" in libpq_url.split("://", 1)[0]:
        scheme, rest = libpq_url.split("://", 1)
        libpq_url = scheme.split("+", 1)[0] + "://" + rest

    with psycopg.connect(libpq_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS andrew")
            # Match deploy/migrations/001_init.sql column types so the fixture
            # works whether or not the migration has run.
            cur.execute("DROP TABLE IF EXISTS andrew.sales")
            cur.execute(
                """
                CREATE TABLE andrew.sales (
                    id       BIGINT PRIMARY KEY,
                    product  TEXT      NOT NULL,
                    revenue  DOUBLE PRECISION NOT NULL,
                    quantity INTEGER   NOT NULL,
                    date     DATE      NOT NULL,
                    region   TEXT      NOT NULL
                )
                """
            )
            cur.executemany(
                "INSERT INTO andrew.sales (id, product, revenue, quantity, date, region) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                ROWS,
            )

    print(f"Postgres database seeded: {_redact(libpq_url)}")
    print(f"  schema=andrew, table=sales, {len(ROWS)} rows")
    print()
    print("Already pointed at this DB? You're done.")


def _redact(url: str) -> str:
    if "@" not in url:
        return url
    scheme, rest = url.split("://", 1) if "://" in url else ("", url)
    creds_host, _, tail = rest.partition("/")
    _, _, host = creds_host.rpartition("@")
    return f"{scheme}://***@{host}/{tail}" if scheme else f"***@{host}/{tail}"


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv
    backend, target = _resolve_target(argv)
    if backend == "sqlite":
        _setup_sqlite(target)
    else:
        _setup_postgres(target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
