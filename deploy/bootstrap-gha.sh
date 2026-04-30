#!/usr/bin/env bash
# deploy/bootstrap-gha.sh
#
# One-time setup for GitHub Actions → Cloud Run deploys via Workload Identity
# Federation. Run this AFTER deploy/bootstrap.sh (which creates the
# cloudbuild-deployer service account that GHA will impersonate).
#
# This script DOES NOT mint any long-lived JSON keys — auth from GHA to GCP
# is via short-lived OIDC tokens exchanged at runtime.
#
# Idempotent: safe to re-run on threshold tweaks or to re-bind a new repo.
#
# Required env:
#   PROJECT_ID            GCP project (e.g. andrew-beta-2026)
#   GITHUB_REPO           owner/repo (e.g. aleksandrlyubarev-blip/Andrew-Analitic)
#
# Optional env:
#   POOL_ID               default: andrew-gha
#   PROVIDER_ID           default: github
#   DEPLOYER_SA           default: cloudbuild-deployer@${PROJECT_ID}.iam.gserviceaccount.com

set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
GITHUB_REPO="${GITHUB_REPO:?Set GITHUB_REPO=owner/repo}"
POOL_ID="${POOL_ID:-andrew-gha}"
PROVIDER_ID="${PROVIDER_ID:-github}"
DEPLOYER_SA="${DEPLOYER_SA:-cloudbuild-deployer@${PROJECT_ID}.iam.gserviceaccount.com}"

PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')

gcloud config set project "${PROJECT_ID}" >/dev/null

# ── 0. Verify deployer SA exists (created by bootstrap.sh) ──
if ! gcloud iam service-accounts describe "${DEPLOYER_SA}" >/dev/null 2>&1; then
    echo "ERROR: ${DEPLOYER_SA} not found. Run ./deploy/bootstrap.sh first." >&2
    exit 1
fi

# ── 1. Workload Identity Pool ──────────────────────────────
if gcloud iam workload-identity-pools describe "${POOL_ID}" \
     --location=global >/dev/null 2>&1; then
    echo "    workload-identity-pool ${POOL_ID} exists"
else
    echo ">>> Creating workload-identity-pool: ${POOL_ID}"
    gcloud iam workload-identity-pools create "${POOL_ID}" \
        --location=global \
        --display-name="Andrew GitHub Actions" \
        --description="Pool for GHA → Cloud Run deploys"
fi

# ── 2. OIDC provider (GitHub Actions) ──────────────────────
if gcloud iam workload-identity-pools providers describe "${PROVIDER_ID}" \
     --location=global --workload-identity-pool="${POOL_ID}" >/dev/null 2>&1; then
    echo "    OIDC provider ${PROVIDER_ID} exists"
else
    echo ">>> Creating OIDC provider: ${PROVIDER_ID}"
    # Attribute mapping: bind the GitHub `repository` claim onto a GCP
    # principal so we can scope IAM to one repo.
    # The attribute-condition is the actual security gate — without it any
    # GitHub repo on the planet could mint tokens against this pool.
    gcloud iam workload-identity-pools providers create-oidc "${PROVIDER_ID}" \
        --location=global \
        --workload-identity-pool="${POOL_ID}" \
        --display-name="GitHub Actions" \
        --issuer-uri="https://token.actions.githubusercontent.com" \
        --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
        --attribute-condition="assertion.repository == '${GITHUB_REPO}'"
fi

# ── 3. Bind GHA repo principal → impersonate deployer SA ───
WIF_PRINCIPAL="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/attribute.repository/${GITHUB_REPO}"

echo ">>> Granting ${WIF_PRINCIPAL} → impersonate ${DEPLOYER_SA}"
gcloud iam service-accounts add-iam-policy-binding "${DEPLOYER_SA}" \
    --role=roles/iam.workloadIdentityUser \
    --member="${WIF_PRINCIPAL}" \
    --quiet >/dev/null

# ── 4. Print the values to paste into GitHub repo Variables ──
PROVIDER_RESOURCE="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/providers/${PROVIDER_ID}"

cat <<EOF

>>> WIF setup complete.

Set the following as repository VARIABLES (not secrets — they're not sensitive):
  Settings → Secrets and variables → Actions → Variables

  GCP_PROJECT_ID                  ${PROJECT_ID}
  GCP_REGION                      \${REGION:-us-central1}     # set explicitly
  GCP_WORKLOAD_IDENTITY_PROVIDER  ${PROVIDER_RESOURCE}
  GCP_SERVICE_ACCOUNT             ${DEPLOYER_SA}
  GCP_SQL_INSTANCE                andrew-pg
  GCP_SERVICE                     andrew-core

Then push to main, or trigger Actions → Deploy to Cloud Run → Run workflow.

To revoke access later (e.g. rotating to a new repo):
  gcloud iam service-accounts remove-iam-policy-binding ${DEPLOYER_SA} \\
    --role=roles/iam.workloadIdentityUser \\
    --member="${WIF_PRINCIPAL}"
EOF
