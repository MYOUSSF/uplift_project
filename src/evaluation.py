"""
evaluation.py
=============
Evaluation metrics specific to uplift / causal modelling.

Metrics
-------
qini_curve          – (treatment gains) vs (population treated fraction)
auqc                – Area Under the Qini Curve (higher is better)
uplift_curve        – Expected response rate lift vs random targeting
cumulative_gain     – Cumulative incremental conversions
kendall_rank_corr   – Rank-order agreement between predicted & true CATE
                      (only valid when ground-truth CATE is known, e.g. synthetic)

All curve functions return (x_axis, y_axis) arrays suitable for plotting.

References
----------
Radcliffe, N. J. (2007). Using control groups to target on predicted lift.
    Direct Marketing Analytics Journal, 1, 14-21.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple


def _check_inputs(y, treatment, scores):
    y         = np.asarray(y,         dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    scores    = np.asarray(scores,    dtype=float)
    assert len(y) == len(treatment) == len(scores), "Length mismatch."
    return y, treatment, scores


def qini_curve(
    y: np.ndarray,
    treatment: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Qini curve.

    Returns
    -------
    frac_treated : proportion of population targeted (x-axis, 0→1)
    qini_value   : cumulative net gain (y-axis)
    """
    y, treatment, scores = _check_inputs(y, treatment, scores)
    order = np.argsort(-scores)
    y_sorted  = y[order]
    t_sorted  = treatment[order]

    n  = len(y)
    n1 = treatment.sum()
    n0 = n - n1

    cum_t_resp = np.cumsum(y_sorted * t_sorted)
    cum_c_resp = np.cumsum(y_sorted * (1 - t_sorted))
    cum_t_n    = np.cumsum(t_sorted)
    cum_c_n    = np.cumsum(1 - t_sorted)

    # Avoid division by zero
    safe_n0 = np.where(n0 > 0, n0, 1)
    qini_vals = cum_t_resp - cum_c_resp * (cum_t_n / safe_n0)

    frac = np.arange(1, n + 1) / n
    # Subsample to n_bins points for speed
    idx = np.linspace(0, n - 1, n_bins, dtype=int)
    return np.concatenate([[0], frac[idx]]), np.concatenate([[0], qini_vals[idx]])


def auqc(
    y: np.ndarray,
    treatment: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 100,
) -> float:
    """Area under the Qini curve (trapezoidal rule, normalised by n)."""
    x, q = qini_curve(y, treatment, scores, n_bins)
    return float(np.trapezoid(q, x) if hasattr(np, "trapezoid") else np.trapz(q, x))


def uplift_curve(
    y: np.ndarray,
    treatment: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uplift curve: expected incremental response rate above random.

    Returns
    -------
    frac_treated : proportion targeted (x-axis)
    uplift_vals  : incremental response rate vs random (y-axis)
    """
    y, treatment, scores = _check_inputs(y, treatment, scores)
    order = np.argsort(-scores)
    y_sorted = y[order]
    t_sorted = treatment[order]

    n  = len(y)
    n1 = treatment.sum()
    n0 = n - n1

    cum_t  = np.cumsum(t_sorted)
    cum_c  = np.cumsum(1 - t_sorted)

    rate_t = np.where(cum_t  > 0, np.cumsum(y_sorted * t_sorted)     / cum_t,  0)
    rate_c = np.where(cum_c  > 0, np.cumsum(y_sorted * (1-t_sorted)) / cum_c, 0)
    uplift_vals = rate_t - rate_c

    frac = np.arange(1, n + 1) / n
    idx  = np.linspace(0, n - 1, n_bins, dtype=int)
    return np.concatenate([[0], frac[idx]]), np.concatenate([[0], uplift_vals[idx]])


def cumulative_gain_curve(
    y: np.ndarray,
    treatment: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cumulative incremental conversions vs fraction targeted.
    Shows how many extra conversions result from targeting top-k%.
    """
    y, treatment, scores = _check_inputs(y, treatment, scores)
    order     = np.argsort(-scores)
    y_sorted  = y[order]
    t_sorted  = treatment[order]

    n1 = treatment.sum()
    n0 = len(y) - n1

    cum_t_resp = np.cumsum(y_sorted * t_sorted)
    cum_c_resp = np.cumsum(y_sorted * (1 - t_sorted))
    safe_n0    = max(n0, 1)

    gain = cum_t_resp - cum_c_resp * (n1 / safe_n0)

    frac = np.arange(1, len(y) + 1) / len(y)
    idx  = np.linspace(0, len(y) - 1, n_bins, dtype=int)
    return np.concatenate([[0], frac[idx]]), np.concatenate([[0], gain[idx]])


def random_baseline_qini(
    y: np.ndarray,
    treatment: np.ndarray,
    n_bins: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonal random-targeting Qini baseline."""
    x    = np.linspace(0, 1, n_bins)
    n1   = treatment.sum()
    n0   = len(treatment) - n1
    safe = max(n0, 1)
    # Expected gain at 100% targeting
    y    = np.asarray(y)
    t    = np.asarray(treatment)
    max_gain = y[t == 1].sum() - y[t == 0].sum() * n1 / safe
    return x, x * max_gain


def summarise_metrics(results: dict) -> pd.DataFrame:
    """
    Build a summary DataFrame from a dict of {model_name: auqc_score}.
    """
    rows = [{"Model": name, "AUQC": score} for name, score in results.items()]
    return pd.DataFrame(rows).sort_values("AUQC", ascending=False).reset_index(drop=True)
