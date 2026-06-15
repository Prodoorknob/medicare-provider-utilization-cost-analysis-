"""
run_pipeline.py -- Phase 4 (E): the autonomous fraud-lead validation loop.

Runs the full deterministic pipeline a scheduler can fire on a cadence:

    1. refresh OIG LEIE          (monthly source)        [anomaly.external.leie_loader]
    2. refresh CMS Revoked       (quarterly source)      [anomaly.groundtruth.cms_revoked]
    3. rebuild unified labels    (+ dated snapshot)      [anomaly.groundtruth.build_labels]
    4. rebuild analyst lead queue                        [anomaly.validation.build_lead_queue]
    5. re-run the temporal backtest                      [anomaly.validation.run_validation]
   5b. agent scorecard + leakage-corrected eval          [anomaly.validation.scorecard, .point_in_time]
    6. record self-eval + dashboard (lift / lead-time)   [anomaly.validation.self_eval]

Steps 1-2 pull fresh government ground truth; 3-6 re-score the existing detector
flags against it and track drift. The detector flags themselves come from the
(static, annual) silver layer and are NOT recomputed here.

Recommended cadence: monthly (LEIE refreshes monthly; CMS quarterly -- a monthly
run simply re-pulls the current CMS snapshot, which is cheap and keeps the
self-eval history dense). Output is for authorized analyst review, never an
accusation.

Usage:
  python -m anomaly.run_pipeline                 # full loop (refresh + re-eval)
  python -m anomaly.run_pipeline --skip-refresh  # re-eval against existing snapshots
  python -m anomaly.run_pipeline --no-leie        # skip just the LEIE refresh
  python -m anomaly.run_pipeline --secure         # don't pass --insecure to the LEIE loader
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_step(name: str, module: str, args: list[str], critical: bool = True) -> bool:
    """Run `python -m <module> <args>` from the project root. Returns success."""
    print(f"\n{'='*70}\n[pipeline] {name}\n{'='*70}", flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-m", module, *args], cwd=_PROJECT_ROOT)
    dur = time.time() - t0
    ok = proc.returncode == 0
    status = "OK" if ok else f"FAILED (exit {proc.returncode})"
    print(f"[pipeline] {name}: {status} in {dur:.1f}s", flush=True)
    if not ok and critical:
        print(f"[pipeline] ABORT: critical step '{name}' failed.", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-refresh", action="store_true",
                    help="Skip the LEIE + CMS ground-truth refresh; re-eval existing snapshots")
    ap.add_argument("--no-leie", action="store_true", help="Skip only the LEIE refresh")
    ap.add_argument("--no-cms", action="store_true", help="Skip only the CMS Revoked refresh")
    ap.add_argument("--secure", action="store_true",
                    help="Do NOT pass --insecure to the LEIE loader (OIG TLS chain may fail locally)")
    args = ap.parse_args()

    t_start = time.time()
    results: list[tuple[str, bool]] = []

    # 1-2. Refresh government ground truth.
    if not args.skip_refresh:
        if not args.no_leie:
            leie_args = [] if args.secure else ["--insecure"]
            # LEIE refresh is best-effort: if OIG is unreachable we still re-eval
            # against the last good snapshot rather than aborting the whole loop.
            results.append(("refresh LEIE", run_step("Refresh OIG LEIE", "anomaly.external.leie_loader", leie_args, critical=False)))
        if not args.no_cms:
            results.append(("refresh CMS Revoked", run_step("Refresh CMS Revoked", "anomaly.groundtruth.cms_revoked", [], critical=False)))
    else:
        print("[pipeline] --skip-refresh: re-evaluating against existing snapshots")

    # 3. Rebuild unified labels (+ snapshot). Critical.
    if not run_step("Build unified labels", "anomaly.groundtruth.build_labels", ["--snapshot"]):
        sys.exit(1)
    results.append(("build labels", True))

    # 4. Rebuild the operational analyst lead queue.
    results.append(("build lead queue", run_step("Build lead queue", "anomaly.validation.build_lead_queue", [], critical=False)))

    # 5. Re-run the (retrospective) temporal backtest. Critical (feeds self-eval).
    if not run_step("Run validation backtest", "anomaly.validation.run_validation", []):
        sys.exit(1)
    results.append(("backtest", True))

    # 5b. Operational scorecard + the leakage-corrected point-in-time evaluation
    #     (point-in-time ranking, dev/test holdout, volume stratification, BH-FDR).
    results.append(("scorecard", run_step("Agent scorecard", "anomaly.validation.scorecard", [], critical=False)))
    results.append(("point-in-time", run_step("Point-in-time validation (honest)", "anomaly.validation.point_in_time", [], critical=False)))

    # 6. Record self-eval + dashboard.
    results.append(("self-eval", run_step("Self-evaluation", "anomaly.validation.self_eval", [])))

    dur = time.time() - t_start
    print(f"\n{'='*70}\n[pipeline] DONE in {dur:.1f}s")
    for name, ok in results:
        print(f"  {'OK ' if ok else 'ERR'}  {name}")
    # Non-zero exit if any non-critical refresh failed, so a scheduler surfaces it.
    if any(not ok for _, ok in results):
        print("[pipeline] WARNING: one or more steps reported a problem (see above).")
        sys.exit(2)


if __name__ == "__main__":
    main()
