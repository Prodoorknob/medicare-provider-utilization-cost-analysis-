"""
anomaly.groundtruth -- external government ground-truth label sources for
validating the fraud-lead agent (Phase 4: autonomous validation).

Two NPI-keyed, periodically-released government sources are ingested here and
unified into a single labels table (`labels.parquet`) that the validation
harness backtests flags against:

  - OIG LEIE          (monthly)   -- exclusions under 42 USC 1320a-7
                                     (loaded by anomaly/external/leie_loader.py)
  - CMS Revoked       (quarterly) -- Medicare enrollment revocations under
    Providers &                      42 CFR 424.535 (cms_revoked.py, new 2026)
    Suppliers

These are SPARSE, LAGGED, RIGHT-CENSORED positive-only labels. They are used to
measure *lift* and *lead-time*, not classification accuracy. See
anomaly/validation/ for the metric design and the documented caveats.
"""
