#!/usr/bin/env bash
# deploy/bootstrap.sh — Day 1-2 of the 14-day plan.
#
# Idempotent. Enables APIs, creates runtime service accounts with least-privilege
# IAM bindings, and sets a default region. Safe to re-run.
#
# Prereqs:
#   - gcloud CLI installed and authenticated (`gcloud auth login`)
#   - PROJECT_ID set, billing linked
#
# Usage:
#   PROJECT_ID=andrew-beta-2026 REGION=us-central1 ./deploy/bootstrap.sh

set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID, e.g. andrew-beta-2026}"
REGION="${REGION:-us-central1}"

echo ">>> Setting active project: ${PROJECT_ID} (region ${REGION})"
gcloud config set project "${PROJECT_ID}"
gcloud config set run/region "${REGION}"
gcloud config set compute/region "${REGION}"

echo ">>> Enabling required APIs (this can take 60-90 seconds)"
gcloud services enable \
  run.googleapis.com \
  sqladmin.googleapis.com \
  secretmanager.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  iamcredentials.googleapis.com \
  storage.googleapis.com \
  --project="${PROJECT_ID}"

create_sa () {
  local name="$1"
  local desc="$2"
  if gcloud iam service-accounts describe "${name}@${PROJECT_ID}.iam.gserviceaccount.com" \
       --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "    SA ${name} already exists, skipping create"
  else
    echo ">>> Creating service account: ${name}"
    gcloud iam service-accounts create "${name}" \
      --display-name="${desc}" \
      --project="${PROJECT_ID}"
  fi
}

bind_role () {
  local sa="$1"; local role="$2"
  echo "    binding ${role} to ${sa}"
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${sa}" \
    --role="${role}" \
    --condition=None \
    --quiet >/dev/null
}

create_sa "andrew-core-sa"   "Andrew core runtime"
create_sa "moltis-bridge-sa" "Moltis bridge runtime (if used)"
create_sa "cloudbuild-deployer" "Cloud Build → Cloud Run deployer"

CORE_SA="andrew-core-sa@${PROJECT_ID}.iam.gserviceaccount.com"
MOLTIS_SA="moltis-bridge-sa@${PROJECT_ID}.iam.gserviceaccount.com"
DEPLOYER_SA="cloudbuild-deployer@${PROJECT_ID}.iam.gserviceaccount.com"

echo ">>> Binding least-privilege roles"
for ROLE in roles/cloudsql.client \
            roles/secretmanager.secretAccessor \
            roles/storage.objectAdmin \
            roles/logging.logWriter \
            roles/monitoring.metricWriter; do
  bind_role "${CORE_SA}" "${ROLE}"
done

for ROLE in roles/cloudsql.client \
            roles/secretmanager.secretAccessor \
            roles/logging.logWriter \
            roles/monitoring.metricWriter; do
  bind_role "${MOLTIS_SA}" "${ROLE}"
done

# Cloud Build's default SA also needs run.admin + iam.serviceAccountUser
# to deploy revisions to Cloud Run, but we use a dedicated deployer SA so
# the default SA stays scoped to building only.
for ROLE in roles/run.admin \
            roles/iam.serviceAccountUser \
            roles/artifactregistry.writer; do
  bind_role "${DEPLOYER_SA}" "${ROLE}"
done

# Cloud Build's actual service account ($PROJECT_NUMBER@cloudbuild.gserviceaccount.com)
# needs permission to impersonate the deployer SA when triggers fire.
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')
CB_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
echo ">>> Allowing Cloud Build SA (${CB_SA}) to act as the deployer SA"
gcloud iam service-accounts add-iam-policy-binding "${DEPLOYER_SA}" \
  --member="serviceAccount:${CB_SA}" \
  --role="roles/iam.serviceAccountUser" \
  --project="${PROJECT_ID}" \
  --quiet >/dev/null

echo ">>> Bootstrap complete."
echo "    CORE_SA    = ${CORE_SA}"
echo "    MOLTIS_SA  = ${MOLTIS_SA}"
echo "    DEPLOYER_SA= ${DEPLOYER_SA}"
echo
echo "Next: ./deploy/provision.sh"
