# Railway cron service — monthly fraud-lead validation pipeline

Runs `python -m anomaly.run_pipeline` on a monthly schedule, fully independent of
any local machine: refresh OIG LEIE + CMS Revoked ground truth → rebuild labels →
lead queue → retrospective backtest → scorecard → **leakage-corrected point-in-time
validation** → self-eval. Results are written to stdout (Railway logs).

## Architecture
- **Static seed (~43 MB, baked into the image):** `seed/flags.parquet` (the flag
  panel) + `seed/npi_profiles.parquet` (slim per-provider volume). Derived once
  from the frozen 2013–2023 CMS PUF; regenerate with `make_seed.py`.
- **Live ground truth (downloaded each run):** OIG LEIE (`--insecure`, ~10 MB) +
  CMS Revoked (~1.3 MB). No Claude API key needed (no brief generation).
- **No volume by default:** outputs are ephemeral and logged. To persist the
  self-eval history / dated snapshots across runs, mount a Railway volume at
  `/app/local_pipeline` (the entrypoint/CMD writes there).

## Cost
Cron services bill only during execution (not idle). One ~2–5 min run/month at
~1–2 GB RAM ≈ **a fraction of a cent of compute/month**; with no volume there is
no storage cost. Effectively absorbed by the existing Railway plan.

## Deploy / redeploy
From the repo root, with the local pipeline data present in `local_pipeline/`:

```bash
python deploy/railway-pipeline/make_seed.py          # refresh the 43 MB seed
railway add --service medicare-fraud-validation       # once
railway environment edit --service-config medicare-fraud-validation build.builder DOCKERFILE
railway environment edit --service-config medicare-fraud-validation build.dockerfilePath deploy/railway-pipeline/Dockerfile
railway environment edit --service-config medicare-fraud-validation deploy.restartPolicyType NEVER
railway up --service medicare-fraud-validation --detach -m "fraud-validation pipeline"
# verify the run in logs, then schedule it monthly:
railway environment edit --service-config medicare-fraud-validation deploy.cronSchedule "0 3 1 * *"
```

`railway up` uploads only `anomaly/` + `deploy/` (see the repo-root `.railwayignore`).
The `seed/` parquet is gitignored — `make_seed.py` regenerates it locally before deploy.

Remove the schedule: set `deploy.cronSchedule` to `""`. Tear down: delete the service in the Railway dashboard.
