"""
04_eda_local.py — Local EDA
Reads per-state silver parquets from local_pipeline/silver/{STATE}.parquet,
samples across all states, and saves plots/CSVs to local_pipeline/eda/.

Usage:
    python notebooks/04_eda_local.py
    python notebooks/04_eda_local.py --silver-dir local_pipeline/silver --sample 0.1
"""

import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SILVER  = os.path.join(_PROJECT_ROOT, "local_pipeline", "silver")
DEFAULT_EDA_DIR = os.path.join(_PROJECT_ROOT, "local_pipeline", "eda")

NUM_COLS = [
    "Tot_Benes", "Tot_Srvcs", "Avg_Sbmtd_Chrg", "Avg_Mdcr_Alowd_Amt",
    # Avg_Mdcr_Alowd_Amt and Avg_Mdcr_Stdzd_Amt excluded — data leakage with target
]


def run_eda(silver_dir: str, eda_dir: str, sample_frac: float):
    os.makedirs(eda_dir, exist_ok=True)

    silver_files = sorted(glob.glob(os.path.join(os.path.abspath(silver_dir), "*.parquet")))
    if not silver_files:
        raise FileNotFoundError(f"No silver parquets found under '{silver_dir}'.")

    print(f"Sampling {sample_frac*100:.0f}% from each of {len(silver_files)} state files...")
    chunks = []
    for f in silver_files:
        state_df = pd.read_parquet(f)
        chunks.append(state_df.sample(frac=sample_frac, random_state=42)
                      if sample_frac < 1.0 else state_df)
    df = pd.concat(chunks, ignore_index=True)
    print(f"  EDA sample size: {len(df):,} rows")

    # ── 1. Target & charge distributions ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, col in zip(axes, ["Avg_Mdcr_Alowd_Amt", "Avg_Sbmtd_Chrg", "Tot_Srvcs"]):
        if col in df.columns:
            df[col].dropna().plot.hist(bins=60, ax=ax, title=col, color="steelblue", edgecolor="white")
            ax.set_xlabel(col)
    plt.suptitle("Distribution of Key Cost & Volume Metrics", y=1.02)
    plt.tight_layout()
    out = os.path.join(eda_dir, "01_distributions.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")

    # ── 2. Correlation heatmap ─────────────────────────────────────────────────
    present = [c for c in NUM_COLS if c in df.columns]
    corr = df[present].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
    plt.title("Correlation Matrix — Medicare Cost Features")
    plt.tight_layout()
    out = os.path.join(eda_dir, "02_correlation_heatmap.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")

    # ── 3. Top 20 provider types by median payment ────────────────────────────
    if "Rndrng_Prvdr_Type" in df.columns and "Avg_Mdcr_Alowd_Amt" in df.columns:
        top_types = (
            df.groupby("Rndrng_Prvdr_Type")["Avg_Mdcr_Alowd_Amt"]
            .median()
            .nlargest(20)
            .reset_index()
        )
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_types, x="Avg_Mdcr_Alowd_Amt", y="Rndrng_Prvdr_Type", palette="Blues_r")
        plt.xlabel("Median Avg Medicare Payment ($)")
        plt.title("Top 20 Provider Types by Median Avg Medicare Payment")
        plt.tight_layout()
        out = os.path.join(eda_dir, "03_top_provider_types.png")
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved -> {out}")

    # ── 4. State-level summary CSV ────────────────────────────────────────────
    if "Rndrng_Prvdr_State_Abrvtn" in df.columns and "Avg_Mdcr_Alowd_Amt" in df.columns:
        state_summary = (
            df.groupby("Rndrng_Prvdr_State_Abrvtn")["Avg_Mdcr_Alowd_Amt"]
            .agg(mean="mean", median="median", count="count")
            .sort_values("mean", ascending=False)
            .reset_index()
        )
        out = os.path.join(eda_dir, "04_state_summary.csv")
        state_summary.to_csv(out, index=False)
        print(f"  Saved -> {out}")
        print(state_summary.head(10).to_string(index=False))

    print(f"\nEDA complete. All outputs in {eda_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDA on Silver parquet and save plots locally.")
    parser.add_argument("--silver-dir", default=DEFAULT_SILVER,
                        help=f"Directory of per-state silver parquets (default: {DEFAULT_SILVER})")
    parser.add_argument("--eda-dir",    default=DEFAULT_EDA_DIR,
                        help=f"Output directory for plots/CSVs (default: {DEFAULT_EDA_DIR})")
    parser.add_argument("--sample", type=float, default=0.05,
                        help="Fraction to sample per state for plotting (default: 0.05)")
    args = parser.parse_args()
    run_eda(args.silver_dir, args.eda_dir, args.sample)
