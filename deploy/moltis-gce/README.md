# Optional: Moltis sandbox VM on GCE

This directory exists for **plan §5.5 option (c)** — running Moltis on a small
GCE VM so its Docker-in-Docker code-execution path keeps working. **You almost
certainly don't need this for the 2-week beta.** Use it only if E2B fallback
isn't enough.

## Decision tree

```
Do beta scenarios actually require Moltis to spawn its own sandbox containers?
├── No  → Skip this directory. Cloud Run bridge calls E2B directly. Done.
└── Yes → Provision a single e2-small VM with the startup script here.
```

## Cost

- `e2-small` 24/7: **~$13/month** in us-central1
- Single 10 GB pd-standard disk: **~$0.40/month**
- Egress: negligible (Cloud Run ↔ VM stays inside GCP networking)

That's ~$6/2-week beta on top of the existing $7-33 estimate, still well
inside the $300 budget.

## Provisioning

Prereqs:
- `bootstrap.sh` already ran (creates `moltis-bridge-sa`)
- `MOLTIS_PASSWORD` already in Secret Manager (provision.sh handles this)

```bash
PROJECT_ID=andrew-beta-2026
ZONE=us-central1-a

gcloud compute instances create moltis-sandbox \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --machine-type=e2-small \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=10GB \
  --boot-disk-type=pd-standard \
  --metadata-from-file=startup-script=deploy/moltis-gce/startup.sh \
  --metadata=moltis-password-secret=projects/${PROJECT_ID}/secrets/MOLTIS_PASSWORD/versions/latest \
  --service-account=moltis-bridge-sa@${PROJECT_ID}.iam.gserviceaccount.com \
  --scopes=cloud-platform \
  --tags=moltis-sandbox \
  --no-address          # private IP only — accessed via VPC connector
```

## Connecting Cloud Run → VM

Cloud Run can't reach a private VM IP without a Serverless VPC Access connector.
Cheapest path that still works:

```bash
gcloud compute networks vpc-access connectors create andrew-connector \
  --region=us-central1 \
  --network=default \
  --range=10.8.0.0/28 \
  --min-instances=2 --max-instances=3 \
  --machine-type=e2-micro
```

Then redeploy the bridge with:

```bash
gcloud run services update andrew-core \
  --region=us-central1 \
  --vpc-connector=andrew-connector \
  --vpc-egress=private-ranges-only \
  --set-env-vars=MOLTIS_HOST=10.128.0.X,MOLTIS_PORT=13131
```

(Substitute `10.128.0.X` with the VM's actual private IP from
`gcloud compute instances describe moltis-sandbox`.)

**Connector cost:** ~$10/month minimum. If this pushes you past the budget,
reconsider Cloud Run + E2B-only mode.

## Tearing it down

```bash
gcloud compute instances delete moltis-sandbox --zone=us-central1-a --quiet
gcloud compute networks vpc-access connectors delete andrew-connector \
  --region=us-central1 --quiet
gcloud run services update andrew-core --region=us-central1 \
  --clear-vpc-connector --remove-env-vars=MOLTIS_HOST,MOLTIS_PORT
```
