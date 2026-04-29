#!/usr/bin/env bash
# deploy/pause.sh — Emergency stop. Halts Cloud Run scaling and stops Cloud SQL.
#
# Use when:
#   - Billing alert fires above expected threshold
#   - Beta is suspended for the weekend
#   - You discover a runaway loop hammering an LLM API
#
# Cost while paused:
#   - Cloud Run:    $0 (max-instances=0 blocks new instances)
#   - Cloud SQL:    storage only (~$0.06/day for 10 GB) — vCPU/RAM stop billing
#   - Secrets/AR:   pennies/month, leave as-is

set -euo pipefail
PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-andrew-core}"
SQL_INSTANCE="${SQL_INSTANCE:-andrew-pg}"

gcloud config set project "${PROJECT_ID}" >/dev/null

echo ">>> Pausing Cloud Run service: ${SERVICE}"
gcloud run services update "${SERVICE}" \
  --region="${REGION}" \
  --max-instances=0

echo ">>> Stopping Cloud SQL instance: ${SQL_INSTANCE}"
gcloud sql instances patch "${SQL_INSTANCE}" --activation-policy=NEVER --quiet

echo ">>> Paused. Resume with: ./deploy/resume.sh"
