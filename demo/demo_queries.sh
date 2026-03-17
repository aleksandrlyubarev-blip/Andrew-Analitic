#!/usr/bin/env bash
# demo/demo_queries.sh — Live showcase of Andrew Swarm + Romeo PhD
#
# Prerequisites:
#   1. python demo/seed_demo_db.py          (creates demo/demo.db)
#   2. DATABASE_URL=sqlite:///demo/demo.db  (in .env or exported)
#   3. docker compose up -d                 (or uvicorn bridge.moltis_bridge:app)
#
# Usage:  bash demo/demo_queries.sh [BASE_URL]
#         BASE_URL defaults to http://localhost:8100

BASE="${1:-http://localhost:8100}"
SEP="──────────────────────────────────────────────────────────────"

_ask() {
  local label="$1" endpoint="$2" payload="$3"
  echo ""
  echo "$SEP"
  echo "▶  $label"
  echo "$SEP"
  curl -s -X POST "$BASE/$endpoint" \
    -H "Content-Type: application/json" \
    -d "$payload" | python3 -m json.tool --no-ensure-ascii
  echo ""
}

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║          Andrew & Romeo  —  Live Demo                   ║"
echo "║          Target: $BASE"
echo "╚══════════════════════════════════════════════════════════╝"

# ── Health check ─────────────────────────────────────────────
echo ""
echo "$SEP"
echo "▶  Health check"
echo "$SEP"
curl -s "$BASE/health" | python3 -m json.tool

# ── 1. Revenue trend — standard BI lane ──────────────────────
_ask \
  "1 / Revenue trend by month (standard BI lane — cheap model)" \
  "analyze" \
  '{"query": "Show me total revenue by month for 2025", "session_id": "demo-001"}'

# ── 2. EMEA anomaly — triggers anomaly detection ─────────────
_ask \
  "2 / EMEA revenue by zone and quarter" \
  "analyze" \
  '{"query": "Compare total revenue by zone (EMEA vs AMER vs APAC) per quarter in 2025", "session_id": "demo-001"}'

# ── 3. Germany supply-chain drop — drill-down ────────────────
_ask \
  "3 / Germany Q3 anomaly — month-by-month" \
  "analyze" \
  '{"query": "Show monthly revenue for Germany vs France in 2025 to see if there is a divergence", "session_id": "demo-001"}'

# ── 4. Product category mix ───────────────────────────────────
_ask \
  "4 / Top products by revenue — Electronics vs Furniture" \
  "analyze" \
  '{"query": "Which product categories generated the most revenue in 2025? Show top 5 products.", "session_id": "demo-001"}'

# ── 5. Conversion funnel — analytics fastlane ────────────────
_ask \
  "5 / Web conversion funnel (analytics fastlane)" \
  "analyze" \
  '{"query": "Calculate the funnel conversion rate from pageview to purchase in 2025", "session_id": "demo-002"}'

# ── 6. Cohort retention — reasoning/math lane ────────────────
_ask \
  "6 / Monthly cohort retention (reasoning/math lane — premium model)" \
  "analyze" \
  '{"query": "Build a monthly cohort retention analysis for 2025: what % of users who first purchased in Jan returned each subsequent month?", "session_id": "demo-003"}'

# ── 7. Forecast with statistical confidence ───────────────────
_ask \
  "7 / Q4 revenue forecast with confidence interval" \
  "analyze" \
  '{"query": "Based on 2024-2025 trends, forecast total revenue for Q1 2026 with a 90% confidence interval using linear regression", "session_id": "demo-003"}'

# ── 8. Romeo PhD — educational explanation ───────────────────
_ask \
  "8 / Romeo PhD — explain cohort retention to a non-technical executive" \
  "educate" \
  '{"question": "Explain cohort retention analysis in plain English for an executive audience. What business decisions can it drive?"}'

# ── Metrics snapshot ─────────────────────────────────────────
echo ""
echo "$SEP"
echo "▶  Operational metrics after demo"
echo "$SEP"
curl -s "$BASE/metrics" | python3 -m json.tool

echo ""
echo "Demo complete. Open $BASE in a browser to explore the UI."
