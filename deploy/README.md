# `deploy/` — Andrew Swarm GCP Beta

Deployment artifacts for the 2-week beta on Cloud Run, sized for **5-10 named
testers on a $300 cap**. See the planning doc for the full rationale; this
README maps that plan to the files in this directory.

## Files at a glance

| File | What it is | Plan section |
|---|---|---|
| `Dockerfile.cloudrun` | Cloud-Run-tuned image: listens on `$PORT`, single uvicorn worker, gen2 exec env. | §1 architecture, §3 Day 3 |
| `cloudbuild.yaml` | Cloud Build config: build → push to Artifact Registry → deploy a `--no-traffic` revision tagged `rev-<sha>`. Promote with `update-traffic --to-latest`. | §3 Day 10 |
| `bootstrap.sh` | Day 1-2: enable APIs, create `andrew-core-sa` / `moltis-bridge-sa` / `cloudbuild-deployer` with least-privilege roles. Idempotent. | §3 Day 1-2 |
| `provision.sh` | Day 4-5: Artifact Registry + Cloud SQL Postgres 16 (`db-f1-micro` Enterprise) + Secret Manager seeding + per-tester beta keys + apply schema migration. | §3 Day 4-5 |
| `deploy.sh` | Day 6: `gcloud builds submit` + `gcloud run deploy`. Reads tag/min-instances/max-instances from env. | §3 Day 6, §9 |
| `pause.sh` / `resume.sh` | Emergency stop & restart. Sets Cloud Run `max-instances=0` and Cloud SQL `activation-policy=NEVER`. | §4 cost-protection |
| `migrations/001_init.sql` | Postgres schema: `pgvector` + `pg_trgm`, LangGraph checkpoint tables, `memory_chunks` (vector + tsvector + trigram indexes), `request_log`, sales fixture. | §3 Day 5, §3 Day 8 |
| `moltis-gce/` | Optional GCE VM startup script + README for running Moltis with `docker.sock`. **Skip unless E2B fallback is insufficient.** | §5.5 option (c) |
| `BETA_RUNBOOK.md` | Operational playbook: rollback, key rotation, log tailing, DB surgery, teardown. | §3 Day 14 |

## End-to-end first-time setup

```bash
# 1) Make scripts executable
chmod +x deploy/*.sh deploy/moltis-gce/*.sh

# 2) Set required env (Day 1: free trial activation, billing link, $50 budget alert
#    are manual Console steps — see plan §3 Day 1 and §6).
export PROJECT_ID=andrew-beta-2026
export REGION=us-central1
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export XAI_API_KEY=xai-...
export E2B_API_KEY=e2b_...
export MOLTIS_PASSWORD=$(openssl rand -hex 32)   # only matters if you use Moltis
export BETA_USERS="alex sergei novik kuk"

# 3) Create runtime SAs and bind IAM
./deploy/bootstrap.sh

# 4) Cloud SQL + Artifact Registry + Secret Manager + schema
./deploy/provision.sh

# 5) Build + deploy the bridge
./deploy/deploy.sh
```

After step 5 you'll get a public Cloud Run URL. The FastAPI app at
`bridge/api.py` should validate the `X-Api-Key` header against the
`BETA_API_KEYS` secret (a JSON object `{user_slug: key}`). If that middleware
isn't wired up yet, the bridge will accept any caller until you add it — track
this as the one piece of application-side work the Cloud Run deploy depends on.

## What the deploy does NOT do (yet)

These are intentionally manual, either because they need a human in the loop
or because they're once-per-project Console steps:

- Activating the **$300 / 90-day free trial** — only one billing identity per
  user; do this at https://cloud.google.com/free with the account you intend
  to use long-term. (Plan §6.)
- Setting **billing budget alerts** at $50 / $100 / $200 — Console → Billing
  → Budgets & alerts. The single most valuable cost-protection step.
- Connecting the **Cloud Build GitHub trigger** to this repo — Console →
  Cloud Build → Triggers → "Connect repository". After that, every push to
  `claude/deploy-andrew-gcp-beta-f5UVU` (or whatever branch you pick) builds
  and deploys via `cloudbuild.yaml`.
- Setting the **Telegram webhook** — see plan §7.2. Only relevant if you
  ship Telegram access during the beta.
- Distributing per-tester API keys to the team (printed by `provision.sh`).

## What's deliberately not in this directory

- **Moltis bridge as a second Cloud Run service.** The repo's existing
  `docker-compose.yml` mounts `/var/run/docker.sock` into the Moltis container
  to support per-session sandbox containers. Cloud Run forbids privileged
  containers and DinD; the deploy scripts here run only the Andrew bridge
  (this repo) on Cloud Run and rely on the bridge's E2B fallback for code
  execution. If you need real Moltis sandboxes, see `moltis-gce/`.
- **Serverless VPC connector / private IP Cloud SQL.** Not needed when Cloud
  Run reaches Cloud SQL via the Auth Proxy unix socket
  (`/cloudsql/<connection-name>`), which is what `--add-cloudsql-instances`
  sets up. Saves $10-30/month for the beta.
- **HTTPS Load Balancer / IAP.** Cloud Run gives every service a managed
  HTTPS URL. Auth happens in FastAPI middleware against `BETA_API_KEYS`.
  HTTPS LB is ~$18/month and worth it for production, not a 2-week beta.

## Cost expectation (14-day beta)

- `min-instances=0` everywhere → **~$7** (mostly Cloud SQL `db-f1-micro` at
  ~$0.013/hr × 336 h ≈ $4.40, plus storage and small egress).
- `min-instances=1` on the bridge for warm starts → **~$33**.
- The $300 free trial covers either path with ~10× headroom.

If you blow past $50, `./deploy/pause.sh` and investigate before touching
production again. The runbook §6 covers how.
