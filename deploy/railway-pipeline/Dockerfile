# Phase 4 (E) — monthly fraud-lead validation pipeline, as a Railway cron service.
# Build context = repo ROOT (so it can COPY anomaly/). Set in Railway:
#   build.builder = DOCKERFILE
#   build.dockerfilePath = deploy/railway-pipeline/Dockerfile
#   deploy.cronSchedule = 0 3 1 * *      (1st of month, 03:00 UTC)
#   deploy.restartPolicyType = NEVER     (run-to-completion, don't loop)
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY deploy/railway-pipeline/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Pipeline code.
COPY anomaly/ ./anomaly/

# Frozen static detection artifacts (derived once from the 2013-2023 CMS PUF):
# the flag panel + a slim per-provider volume table. The job re-downloads the
# LIVE OIG LEIE + CMS Revoked ground truth itself on every run, so only these
# two static files need to ship in the image (~43 MB).
COPY deploy/railway-pipeline/seed/flags.parquet        ./local_pipeline/anomaly/flags.parquet
COPY deploy/railway-pipeline/seed/npi_profiles.parquet ./local_pipeline/anomaly/npi_profiles.parquet

# Runs: refresh LEIE+CMS -> labels -> lead queue -> backtest -> scorecard ->
# point-in-time (honest) -> self-eval, then exits. Results go to stdout (Railway
# logs). Outputs are ephemeral by design; mount a volume at /app/local_pipeline
# if you want the self-eval history / dated snapshots to persist across runs.
CMD ["python", "-m", "anomaly.run_pipeline"]
