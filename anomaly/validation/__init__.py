"""
anomaly.validation -- Phase 4: backtest the fraud-lead flags against the
government ground-truth labels (anomaly.groundtruth).

Because the positive label (LEIE exclusion / CMS revocation after the flagged
conduct year) is sparse (<0.1% of flagged providers), lagged (events trail
conduct by ~2-6 years), and right-censored (recent flags have not had time to
mature), this package deliberately does NOT report accuracy/ROC-AUC. It reports:

  * lift@k and precision@k  -- base-rate-normalized ranking quality (headline),
                               always with the absolute true-positive count and
                               a Fisher-exact significance test.
  * lead-time               -- months from flag to sanction (the product value).
  * PR-AUC                  -- precision-recall area (positive-unlabeled aware).
  * a fee-schedule ablation -- does lift survive when fee-schedule-derived flags
                               are removed? (guards against the falsification
                               finding that Stage 1 largely re-derives the PFS.)

Output is framed as INVESTIGATIVE LEADS for a human analyst, never accusations.
"""
