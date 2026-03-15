"""
scripts/setup_db.py — Create the Andrew Analytics sample database.

Usage:
    python3 scripts/setup_db.py [path/to/andrew.db]

Default path: andrew.db (project root)
After running, set:
    export DATABASE_URL="sqlite:///andrew.db"
"""

import sqlite3
import sys
import os

DB_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "andrew.db"
)

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

conn = sqlite3.connect(DB_PATH)
conn.execute("DROP TABLE IF EXISTS sales")
conn.execute("""
    CREATE TABLE sales (
        id       INTEGER PRIMARY KEY,
        product  TEXT    NOT NULL,
        revenue  FLOAT   NOT NULL,
        quantity INTEGER NOT NULL,
        date     DATE    NOT NULL,
        region   TEXT    NOT NULL
    )
""")
conn.executemany("INSERT INTO sales VALUES (?,?,?,?,?,?)", ROWS)
conn.commit()
conn.close()

print(f"Database created: {DB_PATH}")
print(f"  {len(ROWS)} rows | 3 products (Widget A/B/C) | 4 regions (North/South/East/West)")
print(f"\nNext step:")
print(f'  export DATABASE_URL="sqlite:///{DB_PATH}"')
