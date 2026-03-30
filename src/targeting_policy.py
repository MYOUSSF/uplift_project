"""
targeting_policy.py
===================
Translates CATE estimates into an actionable targeting policy.

Core idea
---------
Given a budget constraint (e.g. "we can email at most 30 % of customers"),
rank customers by their predicted uplift score and target the top-k.

This module implements:
    optimal_threshold  – choose the uplift score cutoff that maximises
                          expected incremental conversions under a budget.
    policy_summary     – describe the policy in plain-language numbers.
    budget_sweep       – compute incremental gain across budget levels
                          (used to generate the budget allocation chart).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def optimal_threshold(
    scores: np.ndarray,
    budget_frac: float = 0.30,
) -> float:
    """
    Return the minimum uplift score needed to stay within budget_frac.

    Parameters
    ----------
    scores       : predicted CATE for each customer
    budget_frac  : fraction of population to target (0 < budget_frac ≤ 1)

    Returns
    -------
    threshold : float — target customers with scores ≥ threshold
    """
    if budget_frac >= 1.0:
        return float(np.min(scores))
    k = max(1, int(np.ceil(len(scores) * budget_frac)))
    return float(np.partition(scores, -k)[-k])


def apply_policy(
    scores: np.ndarray,
    budget_frac: float = 0.30,
) -> np.ndarray:
    """Return binary targeting decision (1 = send email) for each customer."""
    threshold = optimal_threshold(scores, budget_frac)
    return (scores >= threshold).astype(int)


def policy_summary(
    scores: np.ndarray,
    y: np.ndarray,
    treatment: np.ndarray,
    budget_frac: float = 0.30,
    cost_per_email: float = 0.50,
    revenue_per_conversion: float = 50.0,
) -> dict:
    """
    Compute financial summary of a targeting policy vs random targeting.

    Returns
    -------
    dict with keys:
        n_targeted, budget_used, targeted_conversion_rate,
        random_conversion_rate, incremental_conversions,
        expected_revenue_lift, roi
    """
    y         = np.asarray(y,         dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    scores    = np.asarray(scores,    dtype=float)

    decision = apply_policy(scores, budget_frac)
    n = len(scores)

    n_targeted = int(decision.sum())

    # Estimate conversion rates using treated observations as proxy
    treated_mask   = treatment == 1
    targeted_obs   = decision == 1
    control_obs    = decision == 0

    # In-sample estimates (real experiment data used as oracle)
    rate_treated = float(y[treated_mask & targeted_obs].mean()) if (treated_mask & targeted_obs).sum() > 0 else 0.0
    rate_control = float(y[~treated_mask].mean()) if (~treated_mask).sum() > 0 else 0.0
    rate_random  = float(y[treated_mask].mean())  if treated_mask.sum() > 0 else 0.0

    incremental_conversions = (rate_treated - rate_control) * n_targeted
    email_cost              = n_targeted * cost_per_email
    revenue_lift            = incremental_conversions * revenue_per_conversion
    profit_lift             = revenue_lift - email_cost
    roi                     = (profit_lift / email_cost * 100) if email_cost > 0 else 0.0

    return {
        "n_total":                    n,
        "n_targeted":                 n_targeted,
        "budget_frac":                budget_frac,
        "email_cost_usd":             round(email_cost, 2),
        "targeted_conversion_rate":   round(rate_treated, 4),
        "control_conversion_rate":    round(rate_control, 4),
        "random_conversion_rate":     round(rate_random,  4),
        "incremental_conversions":    round(incremental_conversions, 1),
        "revenue_lift_usd":           round(revenue_lift, 2),
        "profit_lift_usd":            round(profit_lift,  2),
        "roi_pct":                    round(roi, 1),
    }


def budget_sweep(
    scores: np.ndarray,
    y: np.ndarray,
    treatment: np.ndarray,
    budgets: Optional[np.ndarray] = None,
    cost_per_email: float = 0.50,
    revenue_per_conversion: float = 50.0,
) -> pd.DataFrame:
    """
    Compute profit lift across a range of budget fractions.
    Used to plot the optimal budget allocation curve.

    Returns
    -------
    DataFrame with columns:
        budget_frac, n_targeted, profit_lift_usd, roi_pct
    """
    if budgets is None:
        budgets = np.linspace(0.01, 1.0, 50)

    rows = []
    for b in budgets:
        s = policy_summary(scores, y, treatment, b, cost_per_email, revenue_per_conversion)
        rows.append({
            "budget_frac":     b,
            "n_targeted":      s["n_targeted"],
            "profit_lift_usd": s["profit_lift_usd"],
            "roi_pct":         s["roi_pct"],
        })
    return pd.DataFrame(rows)
