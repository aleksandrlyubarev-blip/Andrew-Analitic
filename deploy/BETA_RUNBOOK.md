# Andrew Swarm — Beta Runbook

Operational playbook for the 2-week GCP beta. Bookmark this; it's what you
run during an incident or at end-of-day.

Assumed env vars in your shell:
```bash
export PROJECT_ID=andrew-beta-2026
export REGION=us-central1
export SERVICE=andrew-core
export SQL_INSTANCE=andrew-pg
gcloud config set project $PROJECT_ID
```

---

## 1. Health checks

Two endpoints serve different purposes:

| Endpoint | Use | Cost | Calls Moltis? |
|---|---|---|---|
| `/healthz` | Cloud Run startup + liveness probes | <1 ms | no |
| `/health`  | Human "is the whole stack reachable" | depends on Moltis | yes |

```bash
URL=$(gcloud run services describe $SERVICE --region=$REGION --format='value(status.url)')
curl -sS "$URL/healthz" | jq .   # cheap process-alive check
curl -sS "$URL/health"  | jq .   # rich check; calls Moltis (will fail in
                                 # Cloud Run unless the GCE Moltis sidecar
                                 # is wired up — that's expected)
```

Expected `/healthz`: `{"status": "ok"}` within 2 s warm, ~6-10 s cold.

If unhealthy:
```bash
gcloud run services logs read $SERVICE --region=$REGION --limit=200
```

### Probe failures (`STARTUP HTTP probe failed` / restart loop)

Cloud Run logs `STARTUP HTTP probe failed` when the container can't serve
`/healthz` within `failureThreshold × periodSeconds` (currently 50 s). Common
causes:

| Symptom in logs | Likely cause | Fix |
|---|---|---|
| `Cannot import bridge.api` | Missing dep or broken Dockerfile | Check `pip install` step in build logs |
| `bind: address already in use` | App didn't read `$PORT` env | Verify `gunicorn ... --bind 0.0.0.0:$PORT` in Dockerfile CMD |
| Probe times out at exactly 4 s | Sync work in `lifespan` startup | Move heavy init off the import path |
| Liveness probe failing on warm instance | `/healthz` handler regressed | Confirm it returns immediately with no deps |

To inspect the current probe config of a deployed revision:
```bash
gcloud run services describe $SERVICE --region=$REGION \
  --format='yaml(spec.template.spec.containers[0].startupProbe,
                 spec.template.spec.containers[0].livenessProbe)'
```

---

## 2. Tail live logs

```bash
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE" \
  --format="value(timestamp, severity, textPayload, jsonPayload.message)"
```

Filter to one tester:
```bash
gcloud logging tail "resource.type=cloud_run_revision AND \
  resource.labels.service_name=$SERVICE AND \
  jsonPayload.user_slug=alex"
```

Per-LLM-call cost (one row per `litellm.completion` invocation, emitted by
`bridge/llm_telemetry.py`):
```bash
gcloud logging read 'resource.type=cloud_run_revision AND
  resource.labels.service_name='$SERVICE' AND
  jsonPayload.event="llm_call"' \
  --limit=200 --format='value(timestamp,jsonPayload.model,jsonPayload.cost_usd,jsonPayload.tokens_in,jsonPayload.tokens_out,jsonPayload.latency_ms)'
```

Top expensive calls in the last hour:
```bash
gcloud logging read 'resource.type=cloud_run_revision AND
  resource.labels.service_name='$SERVICE' AND
  jsonPayload.event="llm_call" AND
  jsonPayload.cost_usd > 0.05 AND
  timestamp >= "'$(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%SZ)'"' \
  --limit=50
```

---

## 3. Roll back a bad deploy

Cloud Run keeps every revision. Roll back to the previous one:

```bash
# List revisions, newest first
gcloud run revisions list --service=$SERVICE --region=$REGION \
  --format='table(metadata.name, status.conditions[0].lastTransitionTime, spec.containers[0].image)'

# Send 100% traffic to the previous revision
PREV=$(gcloud run revisions list --service=$SERVICE --region=$REGION \
  --format='value(metadata.name)' --limit=2 | tail -n1)
gcloud run services update-traffic $SERVICE --region=$REGION --to-revisions=${PREV}=100
```

Canary instead of cutover:
```bash
gcloud run services update-traffic $SERVICE --region=$REGION \
  --to-revisions=${PREV}=90,LATEST=10
```

---

## 4. Rotate a beta tester's API key

```bash
# Pull current keys
gcloud secrets versions access latest --secret=BETA_API_KEYS > /tmp/keys.json

# Edit /tmp/keys.json — replace the user's key with a fresh hex token:
NEW=$(openssl rand -hex 32)
jq --arg user alex --arg new "$NEW" '.[$user] = $new' /tmp/keys.json > /tmp/keys.new.json

# Push as a new version
gcloud secrets versions add BETA_API_KEYS --data-file=/tmp/keys.new.json
shred -u /tmp/keys.json /tmp/keys.new.json

# Force the running service to pick up the new version on next cold start.
# Either redeploy, or set min-instances=0 briefly:
gcloud run services update $SERVICE --region=$REGION --min-instances=0
```

If your FastAPI middleware caches `BETA_API_KEYS` at startup (recommended),
the new key won't be live until the next instance starts. To force immediate
rotation, redeploy:
```bash
gcloud run deploy $SERVICE --region=$REGION \
  --image=$(gcloud run services describe $SERVICE --region=$REGION \
    --format='value(spec.template.spec.containers[0].image)')
```

---

## 5. Rotate an LLM provider key (compromised, expired, switching account)

```bash
echo -n "sk-..." | gcloud secrets versions add OPENAI_API_KEY --data-file=-
gcloud run services update $SERVICE --region=$REGION   # forces revision
```

`gcloud run services update` with no flags still creates a new revision that
re-reads `:latest` of every secret reference.

---

## 6. Pause everything (weekend / runaway cost)

```bash
./deploy/pause.sh
```

That's:
- Cloud Run `--max-instances=0` (no new instances start, in-flight finish)
- Cloud SQL `--activation-policy=NEVER` (vCPU/RAM stop billing; storage continues)

Resume:
```bash
./deploy/resume.sh
```

Cloud SQL takes ~30-60 s to start. Cloud Run is instant.

---

## 7. Database surgery

Point cloud-sql-proxy at your laptop:
```bash
cloud-sql-proxy ${PROJECT_ID}:${REGION}:${SQL_INSTANCE} --port 5433 &
PGPASSWORD=$(gcloud secrets versions access latest --secret=DB_PASSWORD) \
  psql "host=127.0.0.1 port=5433 user=andrew dbname=andrew"
```

The `andrew.request_log` table is populated by `bridge/request_log.py`'s
`RequestLogMiddleware`, which writes one row per non-probe request. It's
fail-open: a Cloud SQL outage drops log rows but never blocks `/analyze`.

Common queries:

```sql
-- Cost / usage by tester (last 24h)
SELECT user_slug, COUNT(*), SUM(cost_usd)::numeric(10,4) AS spend_usd,
       AVG(latency_ms)::int AS p50_ms
FROM andrew.request_log
WHERE ts > now() - interval '24 hours'
GROUP BY 1 ORDER BY spend_usd DESC;

-- Top 5xx routes (last hour)
SELECT route, status_code, COUNT(*)
FROM andrew.request_log
WHERE ts > now() - interval '1 hour' AND status_code >= 500
GROUP BY 1,2 ORDER BY 3 DESC;

-- Auth-failure pattern (potential abuse / leaked key)
SELECT date_trunc('hour', ts) AS hr, COUNT(*) AS attempts
FROM andrew.request_log
WHERE status_code IN (401, 403) AND ts > now() - interval '24 hours'
GROUP BY 1 ORDER BY 1 DESC;

-- Memory usage check (pgvector index health)
SELECT pg_size_pretty(pg_total_relation_size('andrew.memory_chunks'));
SELECT COUNT(*) FROM andrew.memory_chunks;
```

---

## 8. Snapshot the DB before a risky change

```bash
gcloud sql backups create --instance=$SQL_INSTANCE \
  --description="pre-migration $(date +%Y%m%d-%H%M)"
gcloud sql backups list --instance=$SQL_INSTANCE
```

Restore:
```bash
gcloud sql backups restore $BACKUP_ID --restore-instance=$SQL_INSTANCE
```

---

## 9. Cost check

```bash
# Current month-to-date for this project
gcloud billing accounts list
# Then open Console → Billing → Reports, filter to project=$PROJECT_ID

# Cloud Run instance time (cost proxy)
gcloud monitoring time-series list \
  --filter='metric.type="run.googleapis.com/container/billable_instance_time"' \
  --format=json --interval-end-time=$(date -Iseconds) \
  --interval-start-time=$(date -d '24 hours ago' -Iseconds)
```

The Day-1 budget alert (set at $50, $100, $200) is the actual safety net.

---

## 10. End-of-beta teardown

```bash
# Snapshot first
gcloud sql backups create --instance=$SQL_INSTANCE --description="end-of-beta-$(date +%Y%m%d)"
gsutil mb -l $REGION gs://andrew-beta-archive
gcloud sql export sql $SQL_INSTANCE gs://andrew-beta-archive/final.sql.gz --database=andrew

# Delete the expensive things
gcloud sql instances delete $SQL_INSTANCE --quiet
gcloud run services delete $SERVICE --region=$REGION --quiet

# Optional: delete the project entirely (stops every line item billing)
gcloud projects delete $PROJECT_ID
```

---

## Phone-a-friend matrix

| Symptom | Most likely cause | First check |
|---|---|---|
| 502s on `/analyze` | Cold start exceeded probe timeout | `gcloud run services logs read … --limit=50`, raise `--timeout` |
| Steady 401s | Beta key not in `BETA_API_KEYS` secret | `gcloud secrets versions access latest --secret=BETA_API_KEYS` |
| 500 with "could not connect to server" | Cloud SQL paused or restarted | `gcloud sql instances describe $SQL_INSTANCE` → state |
| LLM call returns rate-limit 429 | Provider quota | Provider dashboard; switch lane in `core/routing.py` |
| Bill spike >$10/day | Min-instances ratchet OR loop bug | `gcloud run services describe $SERVICE` → `minScale`; check `request_log` for repeats |
| Telegram bot silent | Webhook not reset after redeploy | `curl https://api.telegram.org/bot$TOK/getWebhookInfo`; re-run `setWebhook` |
