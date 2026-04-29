#!/usr/bin/env bash
# deploy/resume.sh — Reverse pause.sh. Restores Cloud SQL and Cloud Run scaling.

set -euo pipefail
PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-andrew-core}"
SQL_INSTANCE="${SQL_INSTANCE:-andrew-pg}"
MAX_INSTANCES="${MAX_INSTANCES:-3}"

gcloud config set project "${PROJECT_ID}" >/dev/null

echo ">>> Starting Cloud SQL instance: ${SQL_INSTANCE} (~30s)"
gcloud sql instances patch "${SQL_INSTANCE}" --activation-policy=ALWAYS --quiet

echo ">>> Restoring Cloud Run max-instances=${MAX_INSTANCES}: ${SERVICE}"
gcloud run services update "${SERVICE}" \
  --region="${REGION}" \
  --max-instances="${MAX_INSTANCES}"

URL=$(gcloud run services describe "${SERVICE}" --region="${REGION}" --format='value(status.url)')
echo ">>> Resumed. URL: ${URL}"
