#!/usr/bin/env bash
# deploy/moltis-gce/startup.sh
#
# GCE instance startup script for the optional Moltis sandbox VM (plan §5.5
# option (c)). Cloud Run cannot host Moltis because Moltis launches per-session
# Docker containers via /var/run/docker.sock — privileged container runtimes
# are forbidden on Cloud Run (gen1 gVisor + gen2 unprivileged sandbox).
#
# When to use this VM:
#   - You need the Moltis Rust runtime *and* its DinD code-execution path
#   - The bridge's E2B fallback isn't sufficient for your beta scenarios
#
# When NOT to use this VM:
#   - The bridge already falls back to E2B-only mode (commit 3221784).
#     For 5-10 testers in a 2-week beta this is the right answer.
#
# Cost: e2-small @ $0.0167/hr ≈ $13/month if left running 24/7.
#
# Provision the VM with:
#   gcloud compute instances create moltis-sandbox \
#     --machine-type=e2-small \
#     --zone=us-central1-a \
#     --image-family=debian-12 --image-project=debian-cloud \
#     --metadata-from-file=startup-script=deploy/moltis-gce/startup.sh \
#     --metadata=moltis-password-secret=projects/${PROJECT_ID}/secrets/MOLTIS_PASSWORD/versions/latest \
#     --service-account=moltis-bridge-sa@${PROJECT_ID}.iam.gserviceaccount.com \
#     --scopes=cloud-platform \
#     --tags=moltis-sandbox \
#     --no-address                        # private IP only; reach via IAP tunnel
#
# Then expose to Cloud Run via Serverless VPC Access connector OR just
# allowlist Cloud Run NAT egress IPs on tag moltis-sandbox.

set -euxo pipefail

# Wait for network
until ping -c1 -W2 deb.debian.org >/dev/null 2>&1; do sleep 2; done

apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg jq

# Install Docker (the whole point of this VM).
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | \
    gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/debian $(. /etc/os-release && echo ${VERSION_CODENAME}) stable" \
    > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io
systemctl enable --now docker

# Fetch Moltis password from Secret Manager (instance SA must have accessor role).
SECRET_REF=$(curl -fsS -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/moltis-password-secret || true)
if [[ -n "${SECRET_REF}" ]]; then
    TOKEN=$(curl -fsS -H 'Metadata-Flavor: Google' \
      http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token \
      | jq -r .access_token)
    MOLTIS_PASSWORD=$(curl -fsS -H "Authorization: Bearer ${TOKEN}" \
      "https://secretmanager.googleapis.com/v1/${SECRET_REF}:access" \
      | jq -r .payload.data | base64 -d)
fi
: "${MOLTIS_PASSWORD:?MOLTIS_PASSWORD missing — set the moltis-password-secret instance metadata}"

# Run Moltis with persistent volumes.
docker volume create moltis-config
docker volume create moltis-data
docker rm -f andrew-moltis 2>/dev/null || true
docker run -d \
    --name andrew-moltis \
    --restart unless-stopped \
    -p 13131:13131 -p 13132:13132 \
    -v moltis-config:/home/moltis/.config/moltis \
    -v moltis-data:/home/moltis/.moltis \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -e MOLTIS_PASSWORD="${MOLTIS_PASSWORD}" \
    ghcr.io/moltis-org/moltis:latest

# Simple health probe loop, logs to journald.
echo "Moltis container started; tailing logs"
docker logs -f andrew-moltis &
