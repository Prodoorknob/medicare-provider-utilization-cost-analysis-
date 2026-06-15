"""Regenerate the slim static seed the Railway pipeline image bakes in.

The monthly Railway job needs only two static files from the local pipeline:
the flag panel (subset of columns) and a slim per-provider volume table. They
are derived from the frozen 2013-2023 CMS PUF and never change month to month;
the live OIG LEIE / CMS Revoked ground truth is downloaded by the job at runtime.

Run from the repo root before `railway up`:
    python deploy/railway-pipeline/make_seed.py
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "local_pipeline", "anomaly")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed")
os.makedirs(OUT, exist_ok=True)

FLAG_COLS = ["Rndrng_NPI", "year", "specialty", "state", "flag_type", "flag_metric", "severity"]

pd.read_parquet(os.path.join(SRC, "flags.parquet"), columns=FLAG_COLS) \
    .to_parquet(os.path.join(OUT, "flags.parquet"), compression="zstd")
pd.read_parquet(os.path.join(SRC, "npi_profiles.parquet"), columns=["Rndrng_NPI", "year", "total_services"]) \
    .to_parquet(os.path.join(OUT, "npi_profiles.parquet"), compression="zstd")

for f in ("flags.parquet", "npi_profiles.parquet"):
    p = os.path.join(OUT, f)
    print(f"  wrote {f}: {os.path.getsize(p)/1e6:.1f} MB")
print(f"seed -> {OUT}")
