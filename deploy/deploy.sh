#!/usr/bin/env bash
# deploy/deploy.sh — Day 6: build image, push to Artifact Registry, deploy to Cloud Run.
#
# Run after bootstrap.sh and provision.sh. Idempotent.
#
# Required env:
#   PROJECT_ID            GCP project
#
# Optional env:
#   REGION                default us-central1
#   TAG                   default v1.0.0-rc1 (or pass any tag)
#   SQL_INSTANCE          default andrew-pg
#   SERVICE               default andrew-core
#   MIN_INSTANCES         default 0  (set to 1 to keep one warm; ~$27/2-week beta)
#   MAX_INSTANCES         default 3  (cost cap)
#   ALLOW_UNAUTHENTICATED default true  (set false for service-to-service only)

set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-us-central1}"
TAG="${TAG:-v1.0.0-rc1}"
SQL_INSTANCE="${SQL_INSTANCE:-andrew-pg}"
SERVICE="${SERVICE:-andrew-core}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-3}"
ALLOW_UNAUTHENTICATED="${ALLOW_UNAUTHENTICATED:-true}"

CORE_SA="andrew-core-sa@${PROJECT_ID}.iam.gserviceaccount.com"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/andrew/andrew-core:${TAG}"
SQL_CONN="${PROJECT_ID}:${REGION}:${SQL_INSTANCE}"

echo ">>> Building image via Cloud Build (no local Docker required)"
gcloud builds submit \
  --tag="${IMAGE}" \
  --project="${PROJECT_ID}" \
  -f deploy/Dockerfile.cloudrun .

echo ">>> Deploying to Cloud Run: ${SERVICE} (${REGION})"

AUTH_FLAG="--allow-unauthenticated"
[[ "${ALLOW_UNAUTHENTICATED}" == "false" ]] && AUTH_FLAG="--no-allow-unauthenticated"

gcloud run deploy "${SERVICE}" \
  --image="${IMAGE}" \
  --region="${REGION}" \
  --platform=managed \
  --execution-environment=gen2 \
  --service-account="${CORE_SA}" \
  --cpu=1 \
  --memory=1Gi \
  --cpu-boost \
  --min-instances="${MIN_INSTANCES}" \
  --max-instances="${MAX_INSTANCES}" \
  --concurrency=4 \
  --timeout=3600 \
  --add-cloudsql-instances="${SQL_CONN}" \
  --set-env-vars="DB_USER=andrew,DB_NAME=andrew,DB_HOST=/cloudsql/${SQL_CONN},LITELLM_LOG=INFO,DATABASE_URL=postgresql+psycopg://andrew@/andrew?host=/cloudsql/${SQL_CONN}" \
  --set-secrets="OPENAI_API_KEY=OPENAI_API_KEY:latest,ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest,XAI_API_KEY=XAI_API_KEY:latest,E2B_API_KEY=E2B_API_KEY:latest,DB_PASSWORD=DB_PASSWORD:latest,MOLTIS_PASSWORD=MOLTIS_PASSWORD:latest,BETA_API_KEYS=BETA_API_KEYS:latest" \
  ${AUTH_FLAG}

URL=$(gcloud run services describe "${SERVICE}" --region="${REGION}" --format='value(status.url)')
echo
echo ">>> Deployed: ${URL}"
echo ">>> Smoke test: curl -sS \"${URL}/health\""
echo ">>> Authenticated request: curl -sS -H \"X-Api-Key: <key>\" \"${URL}/analyze\" -d '{...}'"
