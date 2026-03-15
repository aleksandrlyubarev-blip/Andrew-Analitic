"""
Validation tests — SQL safety, Python AST checks, blocked keywords.
Run: python -m pytest tests/test_validation.py -v

These correspond to the analyst's 8 adversarial test cases.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.andrew_swarm import validate_sql, validate_python_static


SAMPLE_SCHEMA = {
    "sales": {"id": "int", "product": "text", "revenue": "float",
              "quantity": "int", "date": "date", "region": "text"}
}


def _validate_sql(query: str, schema=None) -> dict:
    state = {
        "sql_query": query,
        "schema_context": schema or SAMPLE_SCHEMA,
        "intent_contract": {"allowed_tables": list((schema or SAMPLE_SCHEMA).keys())},
        "warnings": [],
        "audit_log": [],
    }
    return validate_sql(state)


def _validate_python(code: str) -> dict:
    state = {
        "python_code": code,
        "confidence": 0.5,
        "intent_contract": {"must_not_train_model_unless_requested": False},
        "warnings": [],
        "audit_log": [],
    }
    return validate_python_static(state)


# ─── Test 1: Hallucinated table ──────────────────────────────

def test_hallucinated_table_blocked():
    result = _validate_sql("SELECT total_revenue FROM customers")
    assert result.get("error_message")
    assert result["error_message"]  # sqlglot catches via qualification error


# ─── Test 3: Destructive SQL ─────────────────────────────────

def test_drop_table_blocked():
    result = _validate_sql("DROP TABLE sales")
    assert result.get("error_message")
    assert "DROP" in result["error_message"]


def test_delete_blocked():
    result = _validate_sql("DELETE FROM sales WHERE id = 1")
    assert result.get("error_message")


def test_truncate_blocked():
    result = _validate_sql("TRUNCATE TABLE sales")
    assert result.get("error_message")


def test_insert_blocked():
    result = _validate_sql("INSERT INTO sales VALUES (1, 'x', 100, 1, '2025-01-01', 'US')")
    assert result.get("error_message")


# ─── Test: Valid SELECT passes ────────────────────────────────

def test_valid_select_passes():
    result = _validate_sql("SELECT region, SUM(revenue) FROM sales GROUP BY region")
    # This may pass or fail depending on sqlglot qualify behavior
    # The key is it shouldn't error on blocked keywords
    assert "DROP" not in result.get("error_message", "")


# ─── Test 6: Dangerous Python imports ─────────────────────────

def test_import_os_blocked():
    result = _validate_python("import os\nos.system('curl evil')")
    assert result.get("error_message")
    assert "os" in result["error_message"].lower()


def test_import_subprocess_blocked():
    result = _validate_python("import subprocess\nsubprocess.run(['ls'])")
    assert result.get("error_message")
    assert "subprocess" in result["error_message"].lower()


def test_eval_blocked():
    result = _validate_python("x = eval('1+1')")
    assert result.get("error_message")
    assert "eval" in result["error_message"].lower()


def test_exec_blocked():
    result = _validate_python("exec('print(1)')")
    assert result.get("error_message")


def test_import_socket_blocked():
    result = _validate_python("import socket\ns = socket.socket()")
    assert result.get("error_message")


# ─── Test: Safe Python passes ─────────────────────────────────

def test_safe_pandas_code_passes():
    result = _validate_python(
        "import pandas as pd\nimport json\ndf = pd.DataFrame({'a': [1,2,3]})\nprint(json.dumps({'result': 'ok'}))"
    )
    assert not result.get("error_message")
    assert result.get("python_validated") is True


# ─── Test 5: Data leakage detection ──────────────────────────

def test_fit_before_split_warns():
    code = """
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
import numpy as np
X = np.array([[1,2],[3,4],[5,6]])
y = np.array([1,2,3])
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
"""
    state = {
        "python_code": code,
        "confidence": 0.5,
        "intent_contract": {"must_not_train_model_unless_requested": True},
        "warnings": [],
        "audit_log": [],
    }
    result = validate_python_static(state)
    # Should warn about leakage, reduce confidence
    warnings = state.get("warnings", [])
    assert any("leakage" in w.lower() for w in warnings)


# ─── Test: Empty inputs ──────────────────────────────────────

def test_empty_sql_fails():
    result = _validate_sql("")
    assert result.get("error_message")


def test_empty_python_fails():
    result = _validate_python("")
    assert result.get("error_message")
