# ── Stage 1: Build the Vue 3 frontend ───────────────────────
FROM node:20-slim AS ui-builder
WORKDIR /ui
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build   # outputs to ../bridge/static (vite.config.js: build.outDir)

# ── Stage 2: Python bridge + pre-built static files ─────────
FROM python:3.12-slim
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core/   core/
COPY bridge/ bridge/

# Copy the compiled UI into bridge/static/ so FastAPI can serve it
# Vite outDir is '../bridge/static' relative to /ui → resolves to /bridge/static
COPY --from=ui-builder /bridge/static/ bridge/static/

EXPOSE 8100
CMD ["uvicorn", "bridge.moltis_bridge:app", "--host", "0.0.0.0", "--port", "8100"]
