#!/usr/bin/env bash
# deploy/provision.sh — Day 4-5: Artifact Registry + Cloud SQL + Secret Manager.
#
# Idempotent. Creates everything stateful. Run once, then run deploy.sh to
# build/push/deploy the Cloud Run service.
#
# Required env (or pass via flags):
#   PROJECT_ID            GCP project (must already have billing linked)
#   OPENAI_API_KEY        secret value to seed Secret Manager
#   ANTHROPIC_API_KEY     ditto
#   XAI_API_KEY           ditto (set "" if you don't use Grok yet)
#   E2B_API_KEY           ditto
#   MOLTIS_PASSWORD       ditto (only required if running the Moltis sidecar)
#   BETA_USERS            space-separated list of beta tester slugs, e.g. "alex sergei"
#
# Optional:
#   REGION                default us-central1
#   SQL_INSTANCE          default andrew-pg
#   SQL_TIER              default db-f1-micro  (Enterprise edition only)

set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-us-central1}"
SQL_INSTANCE="${SQL_INSTANCE:-andrew-pg}"
SQL_TIER="${SQL_TIER:-db-f1-micro}"
SQL_DB="${SQL_DB:-andrew}"
SQL_USER="${SQL_USER:-andrew}"
BETA_USERS="${BETA_USERS:-alex}"

gcloud config set project "${PROJECT_ID}" >/dev/null

# ── Artifact Registry ────────────────────────────────────────
if gcloud artifacts repositories describe andrew \
     --location="${REGION}" >/dev/null 2>&1; then
  echo "    Artifact Registry repo 'andrew' already exists"
else
  echo ">>> Creating Artifact Registry repo: andrew (${REGION})"
  gcloud artifacts repositories create andrew \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Andrew Swarm container images"
fi

# ── Cloud SQL Postgres ───────────────────────────────────────
if gcloud sql instances describe "${SQL_INSTANCE}" >/dev/null 2>&1; then
  echo "    Cloud SQL instance ${SQL_INSTANCE} already exists"
else
  echo ">>> Creating Cloud SQL Postgres 16 (${SQL_TIER}, ENTERPRISE, ${REGION})"
  echo "    This takes ~5-7 minutes."
  gcloud sql instances create "${SQL_INSTANCE}" \
    --database-version=POSTGRES_16 \
    --tier="${SQL_TIER}" \
    --edition=ENTERPRISE \
    --region="${REGION}" \
    --storage-size=10 \
    --storage-type=SSD \
    --availability-type=zonal \
    --backup-start-time=04:00 \
    --no-deletion-protection
fi

if gcloud sql databases describe "${SQL_DB}" --instance="${SQL_INSTANCE}" >/dev/null 2>&1; then
  echo "    Database ${SQL_DB} already exists"
else
  echo ">>> Creating database: ${SQL_DB}"
  gcloud sql databases create "${SQL_DB}" --instance="${SQL_INSTANCE}"
fi

# ── DB user + password (rotate password on every provision run) ──
DB_PASS=$(openssl rand -base64 24 | tr -d '=+/' | cut -c1-32)
if gcloud sql users list --instance="${SQL_INSTANCE}" \
     --format='value(name)' | grep -qx "${SQL_USER}"; then
  echo ">>> Resetting password for existing DB user: ${SQL_USER}"
  gcloud sql users set-password "${SQL_USER}" --instance="${SQL_INSTANCE}" \
    --password="${DB_PASS}"
else
  echo ">>> Creating DB user: ${SQL_USER}"
  gcloud sql users create "${SQL_USER}" --instance="${SQL_INSTANCE}" \
    --password="${DB_PASS}"
fi

# ── Secret Manager: API keys + per-user beta keys ────────────
upsert_secret () {
  local name="$1"; local value="$2"
  if [[ -z "${value}" ]]; then
    echo "    Skipping ${name} (empty value)"
    return
  fi
  if gcloud secrets describe "${name}" >/dev/null 2>&1; then
    echo "    Adding new version to existing secret: ${name}"
  else
    echo ">>> Creating secret: ${name}"
    gcloud secrets create "${name}" --replication-policy=automatic >/dev/null
  fi
  printf '%s' "${value}" | gcloud secrets versions add "${name}" --data-file=-
}

upsert_secret OPENAI_API_KEY    "${OPENAI_API_KEY:-}"
upsert_secret ANTHROPIC_API_KEY "${ANTHROPIC_API_KEY:-}"
upsert_secret XAI_API_KEY       "${XAI_API_KEY:-}"
upsert_secret E2B_API_KEY       "${E2B_API_KEY:-}"
upsert_secret MOLTIS_PASSWORD   "${MOLTIS_PASSWORD:-}"
upsert_secret DB_PASSWORD       "${DB_PASS}"

# Per-tester API keys, packaged as a single JSON-encoded secret so the FastAPI
# middleware can load them with one Secret Manager fetch at startup.
KEYS_JSON='{'
sep=''
echo
echo ">>> Generating per-tester beta API keys (write these down — only shown once):"
for USER in ${BETA_USERS}; do
  KEY=$(openssl rand -hex 32)
  echo "    ${USER}: ${KEY}"
  KEYS_JSON+="${sep}\"${USER}\":\"${KEY}\""
  sep=','
done
KEYS_JSON+='}'
upsert_secret BETA_API_KEYS "${KEYS_JSON}"

# Grant the runtime SA access to each secret.
CORE_SA="andrew-core-sa@${PROJECT_ID}.iam.gserviceaccount.com"
for SECRET in OPENAI_API_KEY ANTHROPIC_API_KEY XAI_API_KEY E2B_API_KEY \
              MOLTIS_PASSWORD DB_PASSWORD BETA_API_KEYS; do
  if gcloud secrets describe "${SECRET}" >/dev/null 2>&1; then
    gcloud secrets add-iam-policy-binding "${SECRET}" \
      --member="serviceAccount:${CORE_SA}" \
      --role="roles/secretmanager.secretAccessor" \
      --quiet >/dev/null
  fi
done

# ── Apply migrations via Cloud SQL Auth Proxy ────────────────
if [[ -f deploy/migrations/001_init.sql ]]; then
  echo ">>> Applying schema migration via cloud-sql-proxy"
  if ! command -v cloud-sql-proxy >/dev/null 2>&1; then
    echo "    cloud-sql-proxy not on PATH — skipping. Install with:"
    echo "    curl -o cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.13.0/cloud-sql-proxy.linux.amd64 && chmod +x cloud-sql-proxy"
  else
    cloud-sql-proxy "${PROJECT_ID}:${REGION}:${SQL_INSTANCE}" --port 5433 &
    PROXY_PID=$!
    sleep 5
    PGPASSWORD="${DB_PASS}" psql "host=127.0.0.1 port=5433 user=${SQL_USER} dbname=${SQL_DB}" \
      -v ON_ERROR_STOP=1 -f deploy/migrations/001_init.sql
    kill "${PROXY_PID}" 2>/dev/null || true
  fi
fi

echo
echo ">>> Provision complete."
echo "    Cloud SQL: ${PROJECT_ID}:${REGION}:${SQL_INSTANCE}"
echo "    DB:        ${SQL_DB} (user ${SQL_USER})"
echo "    DB pass is in Secret Manager as DB_PASSWORD."
echo
echo "Next: ./deploy/deploy.sh"
