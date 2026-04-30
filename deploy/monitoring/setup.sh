#!/usr/bin/env bash
# deploy/monitoring/setup.sh
#
# Idempotent setup of Cloud Monitoring artifacts:
#   1. Log-based metrics that surface bridge/llm_telemetry events
#   2. Email notification channel (if not already there)
#   3. Two alert policies — cost spike + error rate
#
# Run AFTER deploy.sh (the metrics need at least one log entry to render
# in the Console, but creation works against an empty filter just fine).
#
# Required env:
#   PROJECT_ID            GCP project
#   ALERT_EMAIL           where alerts go (e.g. alex@example.com)
#
# Optional env:
#   COST_THRESHOLD_USD    5-min spend that triggers the cost alert (default 0.50)
#   ERROR_THRESHOLD_COUNT 5-min llm_error count that triggers (default 5)

set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
ALERT_EMAIL="${ALERT_EMAIL:?Set ALERT_EMAIL=you@domain.com}"
COST_THRESHOLD_USD="${COST_THRESHOLD_USD:-0.50}"
ERROR_THRESHOLD_COUNT="${ERROR_THRESHOLD_COUNT:-5}"

HERE="$(cd "$(dirname "$0")" && pwd)"

gcloud config set project "${PROJECT_ID}" >/dev/null

# ── 1. Log-based metrics ────────────────────────────────────

upsert_metric () {
  local name="$1"
  local config_file="$2"
  if gcloud logging metrics describe "${name}" >/dev/null 2>&1; then
    echo "    metric ${name} exists; updating from ${config_file}"
    gcloud logging metrics update "${name}" \
      --config-from-file="${config_file}"
  else
    echo ">>> Creating metric ${name}"
    gcloud logging metrics create "${name}" \
      --config-from-file="${config_file}"
  fi
}

upsert_metric andrew_llm_cost_usd     "${HERE}/llm-cost-metric.yaml"
upsert_metric andrew_llm_error_count  "${HERE}/llm-error-metric.yaml"

# ── 2. Notification channel ────────────────────────────────

# Look up an existing email channel matching ALERT_EMAIL; create if missing.
CHANNEL_NAME=$(gcloud alpha monitoring channels list \
  --filter="type=email AND labels.email_address=${ALERT_EMAIL}" \
  --format='value(name)' 2>/dev/null | head -n1 || true)

if [[ -z "${CHANNEL_NAME}" ]]; then
  echo ">>> Creating email notification channel for ${ALERT_EMAIL}"
  CHANNEL_NAME=$(gcloud alpha monitoring channels create \
    --display-name="Andrew beta on-call (${ALERT_EMAIL})" \
    --type=email \
    --channel-labels="email_address=${ALERT_EMAIL}" \
    --format='value(name)')
else
  echo "    notification channel exists: ${CHANNEL_NAME}"
fi

# ── 3. Alert policies ──────────────────────────────────────

# Strip the _README field (extra keys are valid JSON but the policy schema
# may reject them) and substitute placeholders, then apply.
render_policy () {
  local src="$1"
  local threshold_field="$2"
  local threshold_value="$3"
  python3 - "$src" "$threshold_field" "$threshold_value" "${CHANNEL_NAME}" <<'PY'
import json
import sys

src, field, value, channel = sys.argv[1:]
with open(src) as f:
    policy = json.load(f)
policy.pop("_README", None)
# Find the matching condition (by display name suffix) and patch the threshold.
for cond in policy.get("conditions", []):
    if field in cond.get("displayName", ""):
        cond["conditionThreshold"]["thresholdValue"] = float(value)
policy["notificationChannels"] = [channel]
print(json.dumps(policy, indent=2))
PY
}

upsert_policy () {
  local display_name="$1"
  local rendered_json="$2"
  local tmp; tmp=$(mktemp); echo "${rendered_json}" >"${tmp}"
  local existing
  existing=$(gcloud alpha monitoring policies list \
    --filter="displayName:'${display_name}'" \
    --format='value(name)' 2>/dev/null | head -n1 || true)
  if [[ -n "${existing}" ]]; then
    echo "    policy '${display_name}' exists; updating"
    gcloud alpha monitoring policies update "${existing}" --policy-from-file="${tmp}"
  else
    echo ">>> Creating policy '${display_name}'"
    gcloud alpha monitoring policies create --policy-from-file="${tmp}"
  fi
  rm -f "${tmp}"
}

COST_POLICY=$(render_policy \
  "${HERE}/llm-cost-alert.policy.json" \
  "llm_call cost sum" \
  "${COST_THRESHOLD_USD}")
upsert_policy "Andrew — LLM cost spike (>\$0.50 / 5 min)" "${COST_POLICY}"

ERROR_POLICY=$(render_policy \
  "${HERE}/llm-error-alert.policy.json" \
  "llm_error count" \
  "${ERROR_THRESHOLD_COUNT}")
upsert_policy "Andrew — LLM error rate (>5 / 5 min)" "${ERROR_POLICY}"

echo
echo ">>> Monitoring setup complete."
echo "    Metrics:  andrew_llm_cost_usd, andrew_llm_error_count"
echo "    Channel:  ${CHANNEL_NAME}"
echo "    Policies: 'Andrew — LLM cost spike', 'Andrew — LLM error rate'"
echo
echo "Console dashboard for the metrics:"
echo "  https://console.cloud.google.com/monitoring/metrics-explorer?project=${PROJECT_ID}"
