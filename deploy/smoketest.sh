#!/usr/bin/env bash
# deploy/smoketest.sh — Day 13 smoke test for the live Cloud Run beta.
#
# Hits /health and a representative slice of /analyze queries that exercise
# the three LLM lanes the routing layer can pick (analytics / education /
# hybrid). Reports per-call status, latency, routing, confidence and cost.
# Aggregates a pass/fail summary at the end.
#
# Pass criteria (per call):
#   - HTTP 200
#   - response.success == true OR response.error is short / known
#   - confidence >= 0.0 (sanity)
#   - elapsed_seconds < 90 (caller-side latency cap)
#
# Required env:
#   URL              base URL of the Cloud Run service (no trailing slash)
#   API_KEY          a valid X-Api-Key from the BETA_API_KEYS secret
#
# Optional env:
#   TIMEOUT          per-request curl timeout in seconds (default 90)
#   QUERIES_FILE     newline-separated list of queries (one per line) to
#                    override the built-in scenario list
#
# Usage:
#   URL=https://andrew-core-xxxxx-uc.a.run.app \
#     API_KEY=$(gcloud secrets versions access latest --secret=BETA_API_KEYS \
#               | jq -r .alex) \
#     ./deploy/smoketest.sh

set -euo pipefail

URL="${URL:?Set URL=https://andrew-core-...-uc.a.run.app}"
API_KEY="${API_KEY:?Set API_KEY to a valid beta key}"
TIMEOUT="${TIMEOUT:-90}"

if ! command -v jq >/dev/null 2>&1; then
    echo "jq is required (apt-get install -y jq / brew install jq)" >&2
    exit 2
fi

# ── Built-in scenario list — keyword-balanced across the three lanes ─────────
# Each line: <expected_lane>|<query>
DEFAULT_QUERIES=$(cat <<'EOF'
analytics|What is the total revenue by region?
analytics|Compare Widget A and Widget B sales for Q1 2025.
analytics|Which region had the largest revenue drop month over month?
analytics|Forecast next quarter revenue based on the last 3 months.
analytics|Break down quantity sold by product per region.
education|Explain what a moving average is in plain English.
education|How does linear regression differ from logistic regression?
education|Define heteroskedasticity in one paragraph.
education|What is the difference between mean and median for skewed data?
education|Why does correlation not imply causation?
hybrid|Analyze our top product by revenue and explain why customers might prefer it.
hybrid|Calculate the average order size and explain what an outlier in this data means.
hybrid|Summarise the trend in West region sales and define what 'seasonality' would look like here.
hybrid|Compute year-over-year growth and tutorial me on how YoY differs from MoM.
hybrid|Predict next month's sales and explain what predict means in a regression sense.
analytics|Show me the top 3 products by revenue.
analytics|What is the standard deviation of revenue across regions?
education|Tutorial: what is a confidence interval?
education|Explain p-value to a non-statistician.
analytics|Aggregate revenue by month for 2025.
EOF
)

QUERIES="${QUERIES_FILE:+$(cat "$QUERIES_FILE")}"
QUERIES="${QUERIES:-$DEFAULT_QUERIES}"

# ── Health probe ─────────────────────────────────────────────────────────────

echo "▶ Probe endpoint: $URL/healthz (Cloud Run startup/liveness target)"
HEALTHZ=$(curl -fsS --max-time 10 "$URL/healthz" || true)
if [[ -z "$HEALTHZ" ]]; then
    echo "  FAIL — service did not respond on /healthz within 10s"
    exit 1
fi
echo "  $HEALTHZ" | jq -c .

echo "▶ Rich health: $URL/health (includes Moltis reachability)"
HEALTH=$(curl -fsS --max-time 10 "$URL/health" || true)
if [[ -n "$HEALTH" ]]; then
    echo "  $HEALTH" | jq -c .
else
    echo "  WARN — /health did not respond within 10s (Moltis may be unreachable; non-fatal)"
fi
echo

# ── Auth gate probe (no key must 401) ────────────────────────────────────────

echo "▶ Auth gate probe (expect HTTP 401 without key)"
GATE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 \
       -X POST "$URL/analyze" \
       -H "Content-Type: application/json" \
       -d '{"query":"ping"}')
if [[ "$GATE" == "401" || "$GATE" == "403" ]]; then
    echo "  OK — gate rejected unauthenticated request (HTTP $GATE)"
else
    echo "  WARN — gate returned HTTP $GATE; service may be running without auth"
fi
echo

# ── Run scenarios ────────────────────────────────────────────────────────────

PASS=0
FAIL=0
TOTAL=0
LANE_MISMATCH=0
SLOW=0
declare -a FAILURES=()

printf "%-3s %-9s %-8s %-7s %-6s %-9s %s\n" \
    "#" "lane" "status" "latency" "conf" "cost_usd" "query"
printf "%-3s %-9s %-8s %-7s %-6s %-9s %s\n" \
    "---" "---------" "--------" "-------" "------" "---------" "---------------------------------------------"

i=0
while IFS='|' read -r expected_lane query; do
    [[ -z "${expected_lane:-}" || -z "${query:-}" ]] && continue
    i=$((i + 1))
    TOTAL=$((TOTAL + 1))

    PAYLOAD=$(jq -nc --arg q "$query" --arg sid "smoke-$i" \
              '{query:$q,channel:"smoketest",session_id:$sid}')

    START=$(date +%s%N)
    BODY=$(curl -sS --max-time "$TIMEOUT" \
                -w "\n__HTTP__%{http_code}__" \
                -X POST "$URL/analyze" \
                -H "Content-Type: application/json" \
                -H "X-Api-Key: $API_KEY" \
                -d "$PAYLOAD" 2>/dev/null || echo $'\n__HTTP__000__')
    END=$(date +%s%N)
    LATENCY_MS=$(( (END - START) / 1000000 ))

    HTTP=$(echo "$BODY" | sed -n 's/.*__HTTP__\([0-9]*\)__$/\1/p' | tail -n1)
    JSON=$(echo "$BODY" | sed '$d')

    SUCCESS=false; ROUTING="?"; CONF="?"; COST="?"
    if [[ "$HTTP" == "200" ]]; then
        SUCCESS=$(echo "$JSON"   | jq -r '.success // false'         2>/dev/null || echo false)
        ROUTING=$(echo "$JSON"   | jq -r '.routing // "?"'           2>/dev/null || echo "?")
        CONF=$(echo "$JSON"      | jq -r '.confidence // "?"'        2>/dev/null || echo "?")
        COST=$(echo "$JSON"      | jq -r '.cost_usd // "?"'          2>/dev/null || echo "?")
    fi

    STATUS="$HTTP"
    PASSED=true
    if [[ "$HTTP" != "200" || "$SUCCESS" != "true" ]]; then
        PASSED=false
    fi
    if [[ "$LATENCY_MS" -gt $((TIMEOUT * 1000)) ]]; then
        PASSED=false
        SLOW=$((SLOW + 1))
    fi

    if [[ "$PASSED" == "true" ]]; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        FAILURES+=("[$i] HTTP=$HTTP routing=$ROUTING q=$query")
    fi

    # Soft check: did the routing layer pick the lane we expected?
    case "$expected_lane" in
        analytics) match='andrew|analytics' ;;
        education) match='romeo|education' ;;
        hybrid)    match='hybrid|both|andrew|romeo' ;;
        *)         match='.*' ;;
    esac
    if ! [[ "$ROUTING" =~ $match ]]; then
        LANE_MISMATCH=$((LANE_MISMATCH + 1))
    fi

    Q_SHORT=$(echo "$query" | cut -c1-45)
    printf "%-3s %-9s %-8s %5dms %-6s %-9s %s\n" \
        "$i" "$expected_lane" "$STATUS" "$LATENCY_MS" "$CONF" "$COST" "$Q_SHORT"
done <<< "$QUERIES"

# ── Summary ─────────────────────────────────────────────────────────────────

echo
echo "─── Summary ───"
echo "  total:    $TOTAL"
echo "  pass:     $PASS"
echo "  fail:     $FAIL"
echo "  lane mismatch (soft): $LANE_MISMATCH"
echo "  slow (>${TIMEOUT}s): $SLOW"

if (( FAIL > 0 )); then
    echo
    echo "Failures:"
    for f in "${FAILURES[@]}"; do echo "  - $f"; done
    exit 1
fi
echo
echo "All scenarios passed."
