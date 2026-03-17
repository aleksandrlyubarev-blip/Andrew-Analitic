"""
Double Diamond workflow tests — Sprint 9.

Covers all four phases:

  Phase 1 — Explore Data     : profile_schema / _build_data_profile
  Phase 2 — Define Hypothesis: hypothesis_gate
  Phase 3 — Run Analysis     : (existing pipeline — not retested here)
  Phase 4 — Validate Results : enhanced validate_results stat checks

All tests are offline — no LLM calls, no real database connections.
profile_schema is tested via the pure-Python helpers; _build_data_profile
is unit-tested with an in-process SQLite DB.

Run: python -m pytest tests/test_double_diamond.py -v
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import sqlalchemy

from core.andrew_swarm import (
    ColumnProfile,
    TableProfile,
    DataProfile,
    _build_data_profile,
    _dataprofile_to_dict,
    profile_schema,
    hypothesis_gate,
    validate_results,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

SCHEMA = {
    "sales": {
        "id":       "int",
        "product":  "text",
        "revenue":  "float",
        "quantity": "int",
        "region":   "text",
    }
}


def _sqlite_url(tmp_path) -> str:
    return f"sqlite:///{tmp_path}/test.db"


def _make_sales_db(tmp_path, rows=10):
    """Create a minimal SQLite DB with a 'sales' table."""
    url = _sqlite_url(tmp_path)
    engine = sqlalchemy.create_engine(url)
    with engine.connect() as conn:
        conn.execute(sqlalchemy.text(
            "CREATE TABLE sales "
            "(id INTEGER, product TEXT, revenue REAL, quantity INTEGER, region TEXT)"
        ))
        for i in range(rows):
            conn.execute(sqlalchemy.text(
                "INSERT INTO sales VALUES (:id, :prod, :rev, :qty, :reg)"
            ), {"id": i, "prod": f"P{i%3}", "rev": float(i * 100),
                "qty": i % 5, "reg": "EU" if i % 2 == 0 else "US"})
        conn.commit()
    return url


def _make_empty_db(tmp_path):
    url = _sqlite_url(tmp_path)
    engine = sqlalchemy.create_engine(url)
    with engine.connect() as conn:
        conn.execute(sqlalchemy.text(
            "CREATE TABLE sales "
            "(id INTEGER, product TEXT, revenue REAL, quantity INTEGER, region TEXT)"
        ))
        conn.commit()
    return url


def _make_null_db(tmp_path):
    """DB where revenue is 100% NULL."""
    url = f"sqlite:///{tmp_path}/null.db"
    engine = sqlalchemy.create_engine(url)
    with engine.connect() as conn:
        conn.execute(sqlalchemy.text(
            "CREATE TABLE sales (id INTEGER, revenue REAL)"
        ))
        for i in range(5):
            conn.execute(sqlalchemy.text(
                "INSERT INTO sales VALUES (:id, NULL)"
            ), {"id": i})
        conn.commit()
    return url


def _make_zeroval_db(tmp_path):
    """DB where revenue has zero variance (all same value)."""
    url = f"sqlite:///{tmp_path}/zeroval.db"
    engine = sqlalchemy.create_engine(url)
    with engine.connect() as conn:
        conn.execute(sqlalchemy.text(
            "CREATE TABLE sales (id INTEGER, revenue REAL)"
        ))
        for i in range(5):
            conn.execute(sqlalchemy.text(
                "INSERT INTO sales VALUES (:id, 42.0)"
            ), {"id": i})
        conn.commit()
    return url


# ── DataProfile dataclasses ───────────────────────────────────────────────────

def test_column_profile_defaults():
    cp = ColumnProfile(dtype="float", null_rate=0.1, row_count=90)
    assert cp.min_val is None
    assert cp.top_values == []
    assert cp.quality_flags == []


def test_table_profile_defaults():
    tp = TableProfile(row_count=100)
    assert tp.columns == {}
    assert tp.quality_flags == []


def test_data_profile_defaults():
    dp = DataProfile()
    assert dp.tables == {}
    assert dp.error is None
    assert dp.warnings == []
    assert dp.profiled_at <= time.time()


def test_dataprofile_to_dict_structure():
    dp = DataProfile(
        tables={
            "sales": TableProfile(
                row_count=10,
                columns={"revenue": ColumnProfile(
                    dtype="float", null_rate=0.0, row_count=10,
                    min_val=0.0, max_val=100.0, mean_val=50.0,
                )},
            )
        }
    )
    d = _dataprofile_to_dict(dp)
    assert "tables" in d
    assert d["tables"]["sales"]["row_count"] == 10
    assert d["tables"]["sales"]["columns"]["revenue"]["mean_val"] == 50.0
    assert "profiled_at" in d
    assert d["error"] is None


# ── _build_data_profile ───────────────────────────────────────────────────────

def test_build_profile_row_count(tmp_path):
    url = _make_sales_db(tmp_path, rows=10)
    dp = _build_data_profile(url, SCHEMA)
    assert dp.error is None
    assert dp.tables["sales"].row_count == 10


def test_build_profile_numeric_stats(tmp_path):
    url = _make_sales_db(tmp_path, rows=10)
    dp = _build_data_profile(url, SCHEMA)
    rev = dp.tables["sales"].columns["revenue"]
    assert rev.min_val == 0.0
    assert rev.max_val == 900.0
    assert rev.mean_val is not None
    assert "float" in rev.dtype.lower() or "real" in rev.dtype.lower()


def test_build_profile_categorical_top_values(tmp_path):
    url = _make_sales_db(tmp_path, rows=10)
    dp = _build_data_profile(url, SCHEMA)
    region = dp.tables["sales"].columns["region"]
    assert len(region.top_values) <= 5
    assert "EU" in region.top_values or "US" in region.top_values


def test_build_profile_empty_table(tmp_path):
    url = _make_empty_db(tmp_path)
    dp = _build_data_profile(url, SCHEMA)
    assert dp.tables["sales"].row_count == 0
    assert "empty_table" in dp.tables["sales"].quality_flags
    assert any("empty" in w for w in dp.warnings)


def test_build_profile_all_null_column(tmp_path):
    url = _make_null_db(tmp_path)
    schema = {"sales": {"id": "int", "revenue": "float"}}
    dp = _build_data_profile(url, schema)
    rev = dp.tables["sales"].columns["revenue"]
    assert rev.null_rate == 1.0
    assert "all_null" in rev.quality_flags


def test_build_profile_zero_variance(tmp_path):
    url = _make_zeroval_db(tmp_path)
    schema = {"sales": {"id": "int", "revenue": "float"}}
    dp = _build_data_profile(url, schema)
    rev = dp.tables["sales"].columns["revenue"]
    assert "zero_variance" in rev.quality_flags


def test_build_profile_bad_url_returns_error():
    dp = _build_data_profile("sqlite:///nonexistent/path/db.db", SCHEMA)
    assert dp.error is not None


def test_build_profile_empty_schema(tmp_path):
    url = _make_sales_db(tmp_path)
    dp = _build_data_profile(url, {})
    assert dp.tables == {}
    assert dp.error is None


# ── profile_schema node ───────────────────────────────────────────────────────

def test_profile_schema_no_db_url_is_noop():
    state = {"schema_context": SCHEMA, "warnings": [], "audit_log": []}
    result = profile_schema(state)
    assert result == {}


def test_profile_schema_no_schema_is_noop(tmp_path):
    url = _make_sales_db(tmp_path)
    state = {"db_url": url, "schema_context": {}, "warnings": [], "audit_log": []}
    result = profile_schema(state)
    assert result == {}


def test_profile_schema_returns_data_profile(tmp_path):
    url = _make_sales_db(tmp_path, rows=5)
    state = {
        "db_url": url,
        "schema_context": SCHEMA,
        "warnings": [],
        "audit_log": [],
    }
    result = profile_schema(state)
    assert "data_profile" in result
    assert result["data_profile"]["tables"]["sales"]["row_count"] == 5


def test_profile_schema_appends_warnings(tmp_path):
    url = _make_empty_db(tmp_path)
    state = {
        "db_url": url,
        "schema_context": SCHEMA,
        "warnings": ["pre-existing"],
        "audit_log": [],
    }
    result = profile_schema(state)
    warnings = result.get("warnings", [])
    assert "pre-existing" in warnings
    assert any("empty" in w.lower() for w in warnings)


# ── hypothesis_gate node ──────────────────────────────────────────────────────

def _gate_state(data_profile, goal="show revenue by region", intent_tables=None):
    return {
        "data_profile": data_profile,
        "user_request": goal,
        "intent_contract": {"allowed_tables": intent_tables or ["sales"]},
        "warnings": [],
        "confidence": 0.5,
        "audit_log": [],
    }


def test_hypothesis_gate_empty_table_warns():
    profile = {
        "tables": {"sales": {"row_count": 0, "quality_flags": ["empty_table"], "columns": {}}},
        "warnings": [],
        "error": None,
    }
    state = _gate_state(profile)
    result = hypothesis_gate(state)
    assert any("empty" in w.lower() for w in result["warnings"])
    assert result.get("confidence", 0.5) < 0.5


def test_hypothesis_gate_high_null_warns():
    profile = {
        "tables": {
            "sales": {
                "row_count": 100,
                "quality_flags": [],
                "columns": {
                    "revenue": {"null_rate": 0.9, "quality_flags": ["high_null_rate"],
                                "dtype": "float", "row_count": 10},
                },
            }
        },
        "error": None, "warnings": [],
    }
    state = _gate_state(profile, goal="show total revenue by region")
    result = hypothesis_gate(state)
    assert any("revenue" in w.lower() for w in result["warnings"])


def test_hypothesis_gate_no_warnings_clean_data():
    profile = {
        "tables": {
            "sales": {
                "row_count": 100,
                "quality_flags": [],
                "columns": {
                    "revenue": {"null_rate": 0.0, "quality_flags": [],
                                "dtype": "float", "row_count": 100},
                },
            }
        },
        "error": None, "warnings": [],
    }
    state = _gate_state(profile)
    result = hypothesis_gate(state)
    assert result.get("warnings", []) == []
    assert "confidence" not in result  # no penalty


def test_hypothesis_gate_skips_on_error():
    state = {"error_message": "something broke", "data_profile": {}, "audit_log": []}
    result = hypothesis_gate(state)
    assert result == {}


def test_hypothesis_gate_skips_on_no_profile():
    state = {"data_profile": None, "warnings": [], "audit_log": [], "user_request": "x",
             "intent_contract": {}, "confidence": 0.5}
    result = hypothesis_gate(state)
    assert result == {}


# ── validate_results: enhanced stat checks ────────────────────────────────────

def _csv_state(df: pd.DataFrame, tmp_path) -> dict:
    path = str(tmp_path / "result.csv")
    df.to_csv(path, index=False)
    return {
        "sql_result_path": path,
        "sandbox_output": "some output",
        "query_results": [{"a": 1}],
        "confidence": 0.6,
        "warnings": [],
        "audit_log": [],
        "error_message": "",
    }


def test_validate_results_empty_df_lowers_confidence(tmp_path):
    state = _csv_state(pd.DataFrame(columns=["revenue", "region"]), tmp_path)
    result = validate_results(state)
    assert result["confidence"] < 0.6
    assert any("0 rows" in w for w in state["warnings"])


def test_validate_results_zero_variance_warns(tmp_path):
    df = pd.DataFrame({"revenue": [42.0, 42.0, 42.0], "region": ["EU", "US", "EU"]})
    state = _csv_state(df, tmp_path)
    result = validate_results(state)
    assert any("zero variance" in w for w in state["warnings"])


def test_validate_results_all_null_column_warns(tmp_path):
    df = pd.DataFrame({"revenue": [None, None, None], "region": ["EU", "US", "EU"]})
    state = _csv_state(df, tmp_path)
    validate_results(state)
    assert any("entirely null" in w for w in state["warnings"])


def test_validate_results_clean_df_no_stat_warnings(tmp_path):
    df = pd.DataFrame({
        "revenue": [100.0, 200.0, 300.0],
        "region": ["EU", "US", "EU"],
    })
    state = _csv_state(df, tmp_path)
    validate_results(state)
    stat_warnings = [w for w in state["warnings"]
                     if "zero variance" in w or "entirely null" in w or "0 rows" in w]
    assert stat_warnings == []
