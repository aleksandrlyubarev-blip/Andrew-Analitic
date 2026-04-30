# `deploy/monitoring/` — early-warning telemetry for the GCP beta

The Cloud Billing budget alert (set in plan §3 Day 1) fires when daily spend
crosses $50/$100/$200. **That's a next-day signal**: at $0.50/hour of
runaway, you'd find out at sunup with $12 already spent.

This directory ships two sharper signals built on top of `bridge/llm_telemetry.py`:

| Signal | Lead time | Catches |
|---|---|---|
| Log-based metric `andrew_llm_cost_usd` + alert | ~5 min | Retry loop, runaway agent, accidentally-pricier model |
| Log-based metric `andrew_llm_error_count` + alert | ~5 min | Bad API key, provider outage, expired secret |

Both alerts auto-close after 30 minutes once the condition clears.

## Files

| File | What |
|---|---|
| `llm-cost-metric.yaml` | DISTRIBUTION metric — extracts `cost_usd` from `event=llm_call` logs, labelled by `model` |
| `llm-error-metric.yaml` | DELTA counter — counts `event=llm_error` logs, labelled by `model` |
| `llm-cost-alert.policy.json` | Alert: 5-min sum of `cost_usd` > $0.50 (≈100× steady state) |
| `llm-error-alert.policy.json` | Alert: 5-min `llm_error` count > 5 (filters single-failure noise) |
| `setup.sh` | Idempotent — creates/updates metrics + email channel + both policies |

## Usage

```bash
# Prereq: bridge already deployed via deploy/deploy.sh, and at least one
# /analyze call has happened so the log filter has data to match.
PROJECT_ID=andrew-beta-2026 \
  ALERT_EMAIL=alex@example.com \
  ./deploy/monitoring/setup.sh
```

Override thresholds:
```bash
PROJECT_ID=... ALERT_EMAIL=... \
  COST_THRESHOLD_USD=1.50 \
  ERROR_THRESHOLD_COUNT=10 \
  ./deploy/monitoring/setup.sh
```

## Why these thresholds?

Plan §2 cost model:
- 5-10 testers × 10 LLM calls/day × ~$0.005/call ≈ **$0.50/day total**
- Per-minute steady state ≈ $0.0003

So:
- **$0.50 in 5 minutes ≈ 100× normal** → unmistakably a loop or wedged agent
- **5 errors in 5 minutes** → sustained, not a single transient retry

Tune up if alerts get too noisy after the first weekend, but the defaults
should be quiet during steady-state usage.

## Manual alert configuration (alternative to setup.sh)

If `gcloud alpha monitoring policies create` ever rejects the JSON (the
`alpha` channel breaks occasionally), you can recreate the same alert via
the Console:

1. Apply the metrics: `gcloud logging metrics create andrew_llm_cost_usd \
   --config-from-file=deploy/monitoring/llm-cost-metric.yaml`
2. Console → Monitoring → Alerting → Create Policy
3. Add condition → Threshold → metric `logging/user/andrew_llm_cost_usd`
4. Aggregation: 5 min, sum
5. Threshold: 0.50
6. Notification channel: pick your email

## Triage flow when an alert fires

`llm-cost-alert.policy.json` and `llm-error-alert.policy.json` both
include `documentation.content` with concrete `gcloud logging read`
commands you can run from the alert email. Same commands also documented
in `BETA_RUNBOOK.md` under "Alert response".

## Cost of the monitoring itself

Log-based metrics: **free up to 10 metrics per project** (we use 2). After
that they're billed under Cloud Logging analysed-data fees, but the beta
won't get there.

Alert policies: free for the first 150 conditions per project per month.
Two policies × 1 condition each × 720 hours = ~1440 condition-hours, well
inside free.

Notification channels (email): free.

Net add to the $7-33 / 14-day estimate: **$0**.
