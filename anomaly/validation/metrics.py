"""
metrics.py -- ranking-quality metrics for sparse, positive-only fraud labels.

All functions take a ranking already sorted best-first (highest suspicion at
index 0) and a parallel boolean `is_positive` array (True = the provider later
appears in the ground-truth labels under a qualifying, post-conduct sanction).

Design choices (see anomaly/validation/__init__.py for why):
  - precision@k / lift@k are the headline; lift = precision@k / base_rate.
  - Every metric is reported WITH the absolute TP count so a ratio built on a
    handful of hits is not over-read.
  - Significance via Fisher's exact test (top-k vs the rest of the cohort).
  - Bootstrap CIs on precision@k (they will be wide; that is honest).
  - PR-AUC (average precision), never ROC-AUC, given the base rate.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

try:
    from scipy.stats import fisher_exact
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

try:
    from sklearn.metrics import average_precision_score
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    _HAVE_SKLEARN = False


def base_rate(is_positive: np.ndarray) -> float:
    is_positive = np.asarray(is_positive, dtype=bool)
    return float(is_positive.mean()) if is_positive.size else float("nan")


def precision_at_k(is_positive_sorted: np.ndarray, k: int) -> tuple[float, int, int]:
    """Return (precision@k, tp@k, k_effective)."""
    arr = np.asarray(is_positive_sorted, dtype=bool)
    k_eff = min(k, arr.size)
    if k_eff == 0:
        return float("nan"), 0, 0
    tp = int(arr[:k_eff].sum())
    return tp / k_eff, tp, k_eff


def lift_at_k(is_positive_sorted: np.ndarray, k: int, br: float | None = None) -> float:
    arr = np.asarray(is_positive_sorted, dtype=bool)
    br = base_rate(arr) if br is None else br
    p, _, _ = precision_at_k(arr, k)
    if not br or math.isnan(br) or br == 0:
        return float("nan")
    return p / br


def fisher_topk(is_positive_sorted: np.ndarray, k: int) -> tuple[float, float]:
    """Fisher exact (one-sided, enrichment) of positives in top-k vs the rest.

    Returns (odds_ratio, p_value). NaN if scipy unavailable or degenerate.
    """
    arr = np.asarray(is_positive_sorted, dtype=bool)
    n = arr.size
    k_eff = min(k, n)
    if k_eff == 0 or k_eff == n or not _HAVE_SCIPY:
        return float("nan"), float("nan")
    tp_top = int(arr[:k_eff].sum())
    fp_top = k_eff - tp_top
    pos_rest = int(arr[k_eff:].sum())
    neg_rest = (n - k_eff) - pos_rest
    table = [[tp_top, fp_top], [pos_rest, neg_rest]]
    try:
        odds, p = fisher_exact(table, alternative="greater")
    except Exception:
        return float("nan"), float("nan")
    return float(odds), float(p)


def bootstrap_precision_ci(is_positive_sorted: np.ndarray, k: int,
                           n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for precision@k (resampling the top-k labels)."""
    arr = np.asarray(is_positive_sorted, dtype=bool)
    k_eff = min(k, arr.size)
    if k_eff == 0:
        return float("nan"), float("nan")
    top = arr[:k_eff].astype(float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(top, size=(n_boot, k_eff), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def pr_auc(scores: np.ndarray, is_positive: np.ndarray) -> float:
    """Average precision (PR-AUC). Positive-unlabeled: treat unlabeled as negative."""
    if not _HAVE_SKLEARN:
        return float("nan")
    y = np.asarray(is_positive, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(average_precision_score(y, s))


def lead_time_stats(lead_days: Iterable[float]) -> dict:
    arr = np.asarray([d for d in lead_days if d is not None and not (isinstance(d, float) and math.isnan(d))],
                     dtype=float)
    arr = arr[arr > 0]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n":         int(arr.size),
        "median_mo": round(float(np.median(arr)) / 30.44, 1),
        "p25_mo":    round(float(np.percentile(arr, 25)) / 30.44, 1),
        "p75_mo":    round(float(np.percentile(arr, 75)) / 30.44, 1),
        "min_mo":    round(float(arr.min()) / 30.44, 1),
        "max_mo":    round(float(arr.max()) / 30.44, 1),
    }


def ks_table(is_positive_sorted: np.ndarray, ks: Iterable[int]) -> list[dict]:
    """Compute the full @k table once; returns a list of per-k dicts."""
    arr = np.asarray(is_positive_sorted, dtype=bool)
    br = base_rate(arr)
    rows = []
    for k in ks:
        p, tp, k_eff = precision_at_k(arr, k)
        lift = (p / br) if (br and not math.isnan(br) and br > 0) else float("nan")
        odds, pval = fisher_topk(arr, k)
        lo, hi = bootstrap_precision_ci(arr, k)
        rows.append({
            "k": k_eff, "tp": tp, "precision": p, "lift": lift,
            "ci_lo": lo, "ci_hi": hi, "fisher_p": pval,
        })
    return rows
