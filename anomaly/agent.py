"""
agent.py -- Provider Anomaly Investigation Agent orchestrator (spec section 9)

Picks the top-N flagged NPI-years by composite severity, builds a
ProviderContext, runs rule checks, and generates an investigation brief
(dry-run by default, or live via --live).

Usage:
    # Dry-run on the top 10 composite-severity NPIs: prints the prompts
    python anomaly/agent.py --top-n 10

    # Live run against Claude (requires ANTHROPIC_API_KEY)
    python anomaly/agent.py --top-n 10 --live \
        --env-path "C:/Users/rajas/Documents/ADS/coverdrive_pred_11/.env"

    # Target a specific list of (NPI, year) pairs
    python anomaly/agent.py --targets 1710906219:2018,1033474374:2018 --live ...
"""

import os
import json
import argparse
import time
from pathlib import Path

import pandas as pd

from retrieve_context import ContextRetriever
from check_rules import evaluate_all
from generate_brief import generate_brief, DEFAULT_MODEL

_PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ANOMALY  = os.path.join(_PROJECT_ROOT, "local_pipeline", "anomaly")
DEFAULT_BRIEFS   = os.path.join(DEFAULT_ANOMALY, "briefs")


def rank_flags(flags_path: str, top_n: int, year: int | None = None) -> list[tuple[str, int]]:
    """Rank flagged NPI-years by composite severity (sum of severity across flags)."""
    flags = pd.read_parquet(flags_path)
    if year is not None:
        before = len(flags)
        flags = flags[flags["year"] == year]
        print(f"Filtered flags to year={year}: {before:,} -> {len(flags):,}")
    comp = (
        flags.groupby(["Rndrng_NPI", "year"])
             .agg(composite_severity=("severity", "sum"),
                  n_flags=("severity", "size"),
                  methods=("flag_type", lambda x: ",".join(sorted(set(x)))))
             .reset_index()
             .sort_values(["composite_severity", "n_flags"], ascending=[False, False])
             .head(top_n)
    )
    print(f"Top {len(comp)} flagged NPI-years by composite severity:")
    for _, row in comp.iterrows():
        print(f"  NPI {row['Rndrng_NPI']} year={row['year']}: "
              f"severity={row['composite_severity']:.2f}, n_flags={row['n_flags']}, "
              f"methods=[{row['methods']}]")
    return [(str(r["Rndrng_NPI"]), int(r["year"])) for _, r in comp.iterrows()]


def parse_targets(arg: str) -> list[tuple[str, int]]:
    out = []
    for pair in arg.split(","):
        pair = pair.strip()
        if not pair:
            continue
        npi, year = pair.split(":")
        out.append((npi.strip(), int(year.strip())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags",        default=os.path.join(DEFAULT_ANOMALY, "flags.parquet"))
    ap.add_argument("--output-dir",   default=DEFAULT_BRIEFS)
    ap.add_argument("--top-n",        type=int, default=10,
                    help="Number of top-severity NPI-years to process")
    ap.add_argument("--year",         type=int, default=None,
                    help="Restrict ranking to a single year (e.g. 2023)")
    ap.add_argument("--targets",      type=str, default=None,
                    help="Override ranking with explicit NPI:year pairs, comma-separated")
    ap.add_argument("--live",         action="store_true",
                    help="Call Claude API (costs money). Default is dry-run.")
    ap.add_argument("--model",        default=DEFAULT_MODEL)
    ap.add_argument("--env-path",     default=None,
                    help="Path to .env containing ANTHROPIC_API_KEY")
    ap.add_argument("--sleep-seconds", type=float, default=0.0,
                    help="Delay between API calls (for rate-limit safety)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Pick targets
    if args.targets:
        targets = parse_targets(args.targets)
        print(f"Targets from CLI: {len(targets)}")
    else:
        targets = rank_flags(args.flags, args.top_n, year=args.year)

    # 2. Load retriever once (profiles + benchmarks amortized across all briefs)
    retriever = ContextRetriever()

    # 3. Generate briefs
    usage_total = {"input": 0, "output": 0, "cache_read": 0, "cache_create": 0}
    briefs_summary = []

    for i, (npi, year) in enumerate(targets, 1):
        print(f"\n=== [{i}/{len(targets)}] NPI {npi} year {year} ===")
        ctx = retriever.get_context(npi, year)
        if ctx is None:
            print("  [SKIP] not found in profiles")
            continue
        print(f"  specialty={ctx.specialty}, state={ctx.state}, "
              f"risk_score={ctx.risk_score}, trend={ctx.trend_direction}")

        rules = evaluate_all(ctx)
        n_triggered = sum(1 for r in rules if r.triggered)
        n_unavail  = sum(1 for r in rules if not r.available)
        print(f"  rules: {n_triggered} triggered, "
              f"{len(rules) - n_triggered - n_unavail} not triggered, "
              f"{n_unavail} not evaluable")

        t0 = time.time()
        brief = generate_brief(
            ctx, rules,
            live=args.live,
            model=args.model,
            env_path=args.env_path,
        )
        dur = time.time() - t0

        if args.live:
            u = brief.evidence_summary.get("usage", {})
            usage_total["input"]        += u.get("input_tokens", 0)
            usage_total["output"]       += u.get("output_tokens", 0)
            usage_total["cache_read"]   += u.get("cache_read_input_tokens", 0)
            usage_total["cache_create"] += u.get("cache_creation_input_tokens", 0)
            print(f"  brief: risk={brief.risk_classification} "
                  f"score={brief.risk_score:.0f}/100  ({dur:.1f}s)  "
                  f"tokens in={u.get('input_tokens', 0)} "
                  f"cache_read={u.get('cache_read_input_tokens', 0)} "
                  f"out={u.get('output_tokens', 0)}")
        else:
            print(f"  [DRY-RUN] prompt formatted ({dur:.2f}s). "
                  f"Re-run with --live to generate the brief.")

        # Write artifacts
        base = f"{npi}_{year}"
        json_path = os.path.join(args.output_dir, f"{base}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(brief.to_dict(), f, indent=2, default=str)

        # If live, write the markdown brief separately for easy reading
        if args.live:
            md = brief.evidence_summary.get("full_markdown", "")
            if md:
                md_path = os.path.join(args.output_dir, f"{base}.md")
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(md)
        else:
            prompt = brief.evidence_summary.get("formatted_prompt", "")
            if prompt:
                p_path = os.path.join(args.output_dir, f"{base}_prompt.md")
                with open(p_path, "w", encoding="utf-8") as f:
                    f.write(prompt)

        briefs_summary.append({
            "npi":                 npi,
            "year":                year,
            "specialty":           ctx.specialty,
            "state":               ctx.state,
            "risk_classification": brief.risk_classification,
            "risk_score":          brief.risk_score,
        })

        if args.live and args.sleep_seconds:
            time.sleep(args.sleep_seconds)

    # 4. Summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "targets":        len(targets),
            "generated":      len(briefs_summary),
            "live":           args.live,
            "model":          args.model,
            "usage":          usage_total if args.live else None,
            "briefs":         briefs_summary,
        }, f, indent=2)

    print(f"\n-- Done. Wrote {len(briefs_summary)} brief(s) to {args.output_dir}")
    if args.live:
        print(f"   Total usage: in={usage_total['input']:,}  "
              f"cache_read={usage_total['cache_read']:,}  "
              f"cache_create={usage_total['cache_create']:,}  "
              f"out={usage_total['output']:,}")
        # Rough cost estimate for Sonnet 4.6 ($3/1M in, $15/1M out, ~$0.30/1M cache read, ~$3.75/1M cache create)
        est = (usage_total["input"] * 3 +
               usage_total["output"] * 15 +
               usage_total["cache_read"] * 0.30 +
               usage_total["cache_create"] * 3.75) / 1_000_000
        print(f"   Est. cost (Sonnet 4.6 list price): ${est:.4f}")


if __name__ == "__main__":
    main()
