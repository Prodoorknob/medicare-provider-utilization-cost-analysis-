"""
Export specialty-level yearly averages from LSTM sequences data.

Reads local_pipeline/lstm/sequences.parquet, aggregates by
(specialty_idx, year) across all states and buckets, and produces:
  1. A SQL file to create + populate the specialty_yearly_avg table
  2. A CSV for inspection

Usage:
    python scripts/export_specialty_history.py
"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SEQUENCES_PATH = os.path.join("local_pipeline", "lstm", "sequences.parquet")
OUTPUT_DIR = os.path.join("local_pipeline", "_upload_sql")
SQL_FILE = os.path.join(OUTPUT_DIR, "specialty_yearly_avg.sql")
CSV_FILE = os.path.join(OUTPUT_DIR, "specialty_yearly_avg.csv")


def main():
    print(f"Reading {SEQUENCES_PATH}...")
    df = pq.read_table(SEQUENCES_PATH).to_pandas()
    print(f"  {len(df)} sequence rows, {df['Rndrng_Prvdr_Type_idx'].nunique()} specialties")

    # Explode years + target_seq into (specialty_idx, year, allowed_amt) rows
    rows = []
    for _, row in df.iterrows():
        spec = int(row["Rndrng_Prvdr_Type_idx"])
        years = row["years"]
        targets = row["target_seq"]
        for yr, val in zip(years, targets):
            rows.append({"specialty_idx": spec, "year": int(yr), "mean_allowed": float(val)})

    exploded = pd.DataFrame(rows)
    print(f"  {len(exploded)} exploded rows")

    # Aggregate: mean of mean_allowed per (specialty_idx, year)
    agg = (
        exploded.groupby(["specialty_idx", "year"])["mean_allowed"]
        .mean()
        .reset_index()
        .sort_values(["specialty_idx", "year"])
    )
    print(f"  {len(agg)} aggregated rows ({agg['specialty_idx'].nunique()} specialties x {agg['year'].nunique()} years)")

    # Save CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    agg.to_csv(CSV_FILE, index=False)
    print(f"  Saved CSV: {CSV_FILE}")

    # Generate SQL
    lines = []
    lines.append("-- specialty_yearly_avg: mean allowed amount per specialty per year")
    lines.append("-- Aggregated from LSTM sequences (across all states and HCPCS buckets)")
    lines.append("")
    lines.append("CREATE TABLE IF NOT EXISTS specialty_yearly_avg (")
    lines.append("    specialty_idx INT NOT NULL,")
    lines.append("    year INT NOT NULL,")
    lines.append("    mean_allowed FLOAT NOT NULL,")
    lines.append("    PRIMARY KEY (specialty_idx, year)")
    lines.append(");")
    lines.append("")
    lines.append("TRUNCATE TABLE specialty_yearly_avg;")
    lines.append("")

    # Batch inserts (500 rows per INSERT)
    batch_size = 500
    values = []
    for _, r in agg.iterrows():
        values.append(f"({int(r['specialty_idx'])},{int(r['year'])},{r['mean_allowed']:.2f})")

    for i in range(0, len(values), batch_size):
        batch = values[i : i + batch_size]
        lines.append(
            f"INSERT INTO specialty_yearly_avg (specialty_idx, year, mean_allowed) VALUES"
        )
        lines.append(",\n".join(batch) + ";")
        lines.append("")

    with open(SQL_FILE, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved SQL: {SQL_FILE} ({len(values)} rows)")
    print("Done. Upload with: python upload_sql_to_supabase.py")


if __name__ == "__main__":
    main()
