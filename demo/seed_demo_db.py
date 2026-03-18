"""
demo/seed_demo_db.py — Seeds a realistic e-commerce SQLite database for the Andrew demo.

Creates demo.db with four tables:
  products   — product catalog with category and unit cost
  regions    — geographic regions with country and zone
  sales      — 24 months of daily orders (pre-joined flat view via VIEW)
  events     — web session events (pageview, add_to_cart, purchase, churn)

Usage:
  python demo/seed_demo_db.py            # creates demo/demo.db
  python demo/seed_demo_db.py --path /tmp/andrew_demo.db
"""

import argparse
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

SEED = 42
random.seed(SEED)

# ── Configuration ──────────────────────────────────────────────────────────────

PRODUCTS = [
    (1, "Pro Laptop",      "Electronics",  899.00),
    (2, "Wireless Mouse",  "Electronics",   29.99),
    (3, "USB-C Hub",       "Electronics",   49.99),
    (4, "Standing Desk",   "Furniture",    349.00),
    (5, "Ergonomic Chair", "Furniture",    299.00),
    (6, "Notebook",        "Stationery",     4.99),
    (7, "Ballpoint Pen",   "Stationery",     1.99),
    (8, "Monitor 27in",    "Electronics",  399.00),
    (9, "Webcam 4K",       "Electronics",   89.99),
    (10, "Desk Lamp",      "Furniture",     39.99),
]

REGIONS = [
    (1, "Germany",       "EMEA"),
    (2, "France",        "EMEA"),
    (3, "Poland",        "EMEA"),
    (4, "United States", "AMER"),
    (5, "Canada",        "AMER"),
    (6, "Brazil",        "LATAM"),
    (7, "Japan",         "APAC"),
    (8, "Australia",     "APAC"),
]

# Simulate a supply-chain disruption in Germany Q3 2025 (latency spike → revenue drop)
REGION_MONTHLY_FACTOR = {
    # (region_id, year, month) → multiplier on base volume
    (1, 2025, 7): 0.55,
    (1, 2025, 8): 0.50,
    (1, 2025, 9): 0.48,
}


def _base_volume(region_id: int, product_id: int) -> float:
    """Deterministic base daily order volume for a region-product pair."""
    random.seed(region_id * 100 + product_id)
    return random.uniform(0.5, 4.0)


def generate_sales(start: date, end: date):
    """Yield (date, product_id, region_id, units, revenue) rows."""
    day = start
    while day <= end:
        for product in PRODUCTS:
            for region in REGIONS:
                pid, _, _, price = product
                rid = region[0]
                base = _base_volume(rid, pid)
                # Apply disruption factor if applicable
                factor = REGION_MONTHLY_FACTOR.get((rid, day.year, day.month), 1.0)
                # Weekend dip
                weekend_factor = 0.6 if day.weekday() >= 5 else 1.0
                units = max(0, round(base * factor * weekend_factor + random.gauss(0, 0.3)))
                if units > 0:
                    yield (str(day), pid, rid, units, round(units * price, 2))
        day += timedelta(days=1)


EVENT_TYPES = ["pageview", "add_to_cart", "checkout_start", "purchase", "session_end"]


def generate_events(start: date, end: date, n_users: int = 300):
    """Yield (timestamp, user_id, session_id, event_type, product_id) rows."""
    import uuid
    day = start
    session_seq = 0
    while day <= end:
        daily_sessions = random.randint(40, 120)
        for _ in range(daily_sessions):
            user_id = random.randint(1, n_users)
            session_id = f"s{session_seq:06d}"
            session_seq += 1
            minute = random.randint(0, 1439)
            ts = f"{day}T{minute // 60:02d}:{minute % 60:02d}:00"
            # Funnel: pageview → maybe add_to_cart → maybe purchase
            yield (ts, user_id, session_id, "pageview", None)
            if random.random() < 0.40:
                pid = random.choice(PRODUCTS)[0]
                yield (ts, user_id, session_id, "add_to_cart", pid)
                if random.random() < 0.35:
                    yield (ts, user_id, session_id, "checkout_start", pid)
                    if random.random() < 0.70:
                        yield (ts, user_id, session_id, "purchase", pid)
            yield (ts, user_id, session_id, "session_end", None)
        day += timedelta(days=1)


def seed(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE products (
            product_id   INTEGER PRIMARY KEY,
            name         TEXT    NOT NULL,
            category     TEXT    NOT NULL,
            unit_price   REAL    NOT NULL
        );

        CREATE TABLE regions (
            region_id    INTEGER PRIMARY KEY,
            country      TEXT    NOT NULL,
            zone         TEXT    NOT NULL
        );

        CREATE TABLE sales (
            sale_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            sale_date    TEXT    NOT NULL,   -- YYYY-MM-DD
            product_id   INTEGER REFERENCES products(product_id),
            region_id    INTEGER REFERENCES regions(region_id),
            units        INTEGER NOT NULL,
            revenue      REAL    NOT NULL
        );

        CREATE TABLE events (
            event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            ts           TEXT    NOT NULL,   -- ISO 8601
            user_id      INTEGER NOT NULL,
            session_id   TEXT    NOT NULL,
            event_type   TEXT    NOT NULL,
            product_id   INTEGER
        );

        -- Flat analytical view — what Andrew reads directly (no JOINs needed)
        CREATE VIEW sales_flat AS
        SELECT
            s.sale_date,
            strftime('%Y', s.sale_date)        AS year,
            strftime('%m', s.sale_date)        AS month,
            strftime('%Y-%m', s.sale_date)     AS year_month,
            p.name                             AS product,
            p.category,
            p.unit_price,
            r.country,
            r.zone,
            s.units,
            s.revenue
        FROM sales s
        JOIN products  p ON p.product_id = s.product_id
        JOIN regions   r ON r.region_id  = s.region_id;
    """)

    cur.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?)", PRODUCTS
    )
    cur.executemany(
        "INSERT INTO regions VALUES (?, ?, ?)", REGIONS
    )

    start = date(2024, 1, 1)
    end   = date(2025, 12, 31)

    print("Generating sales rows …")
    sales_rows = list(generate_sales(start, end))
    cur.executemany(
        "INSERT INTO sales (sale_date, product_id, region_id, units, revenue) VALUES (?, ?, ?, ?, ?)",
        sales_rows,
    )
    print(f"  {len(sales_rows):,} sales rows inserted")

    print("Generating event rows …")
    event_rows = list(generate_events(start, end))
    cur.executemany(
        "INSERT INTO events (ts, user_id, session_id, event_type, product_id) VALUES (?, ?, ?, ?, ?)",
        event_rows,
    )
    print(f"  {len(event_rows):,} event rows inserted")

    conn.commit()
    conn.close()
    print(f"\nDemo database ready → {db_path.resolve()}")
    print("Set DATABASE_URL=sqlite:///" + str(db_path.resolve()) + " before starting the bridge.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Andrew demo database")
    parser.add_argument("--path", default="demo/demo.db", help="Output SQLite path")
    args = parser.parse_args()
    seed(Path(args.path))
