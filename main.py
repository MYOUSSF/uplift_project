"""
main.py
=======
End-to-end pipeline for the Hillstrom Uplift Modelling project.

Usage
-----
    python main.py                  # full run
    python main.py --skip-eda       # skip EDA charts
    python main.py --budget 0.25    # policy at 25% budget

Steps
-----
1. Download & preprocess Hillstrom dataset
2. EDA visualisations
3. Fit S-Learner, T-Learner, X-Learner
4. Evaluate with Qini / AUQC
5. Plot all comparison charts
6. Compute optimal targeting policy
7. Budget sensitivity analysis
8. Print final report
"""

import sys
import os
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Make src importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader     import load_processed, get_feature_cols
from uplift_models   import get_all_models
from evaluation      import auqc, summarise_metrics
from targeting_policy import policy_summary, budget_sweep
from visualizations  import (
    plot_eda_overview,
    plot_uplift_distribution,
    plot_qini_curves,
    plot_cumulative_gain,
    plot_feature_importance,
    plot_budget_curve,
    plot_cate_heatmap,
    plot_model_comparison,
)


# ── Config ────────────────────────────────────────────────────────────────────

SEED         = 42
TEST_SIZE    = 0.30
OUTCOME_COL  = "conversion"   # or "visit"
BUDGET_FRAC  = 0.30           # default budget: target top 30%


# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_summary(summary: dict):
    print("\n  Policy Summary")
    print(f"  {'─'*40}")
    for k, v in summary.items():
        label = k.replace("_", " ").title()
        if isinstance(v, float):
            print(f"  {label:<35} {v:>10.2f}")
        else:
            print(f"  {label:<35} {v!s:>10}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main(skip_eda: bool = False, budget_frac: float = BUDGET_FRAC):
    t0 = time.time()

    # ── 1. Data ───────────────────────────────────────────────────────────────
    section("Step 1: Load & Preprocess Data")
    df      = load_processed()
    feature_cols = get_feature_cols()

    print(f"\n  Dataset shape : {df.shape}")
    print(f"  Treatment arms:\n{df['segment'].value_counts().to_string()}")
    print(f"\n  Outcome rates:")
    for col in ["visit", "conversion"]:
        print(f"    {col}: {df[col].mean():.4f}")

    # Binary treatment (any email vs none)
    df_model = df[df["treatment"] != -1].copy()   # all rows valid after preprocessing

    X         = df_model[feature_cols]
    treatment = df_model["treatment_binary"].values
    y         = df_model[OUTCOME_COL].values

    X_tr, X_te, t_tr, t_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, treatment, y, df_model.index,
        test_size=TEST_SIZE, random_state=SEED, stratify=treatment
    )
    df_test = df_model.loc[idx_te].reset_index(drop=True)

    print(f"\n  Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

    # ── 2. EDA ────────────────────────────────────────────────────────────────
    if not skip_eda:
        section("Step 2: Exploratory Data Analysis")
        p = plot_eda_overview(df)
        print(f"  EDA chart → {p}")

    # ── 3. Fit models ─────────────────────────────────────────────────────────
    section("Step 3: Fit Meta-Learners")
    models = get_all_models(outcome="binary")
    cate_dict = {}

    for model in models:
        t_fit = time.time()
        print(f"\n  Fitting {model.name} …", end="", flush=True)
        model.fit(X_tr, t_tr, y_tr)
        cate   = model.predict(X_te)
        cate_dict[model.name] = cate
        print(f"  done ({time.time() - t_fit:.1f}s)  "
              f"mean CATE = {cate.mean():.5f}  "
              f"std = {cate.std():.5f}")

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    section("Step 4: Evaluation — AUQC Scores")
    auqc_dict = {}
    for name, cate in cate_dict.items():
        score = auqc(y_te, t_te, cate)
        auqc_dict[name] = score
        print(f"  {name:<15}  AUQC = {score:.6f}")

    summary_df = summarise_metrics(auqc_dict)
    best_model_name = summary_df.iloc[0]["Model"]
    best_cate       = cate_dict[best_model_name]
    print(f"\n  Best model: {best_model_name}")

    # ── 5. Visualisations ────────────────────────────────────────────────────
    section("Step 5: Generating Visualisations")

    plot_uplift_distribution(cate_dict)
    plot_qini_curves(cate_dict, y_te, t_te)
    plot_cumulative_gain(cate_dict, y_te, t_te)

    # Feature importance — use T-Learner
    t_learner = next(m for m in models if m.name == "T-Learner")
    plot_feature_importance(t_learner, feature_cols)

    plot_cate_heatmap(df_test, best_cate, model_name=best_model_name)
    plot_model_comparison(auqc_dict)

    # ── 6. Targeting policy ──────────────────────────────────────────────────
    section(f"Step 6: Optimal Targeting Policy (budget = {budget_frac*100:.0f}%)")
    pol = policy_summary(best_cate, y_te, t_te, budget_frac=budget_frac)
    print_summary(pol)

    # ── 7. Budget sweep ───────────────────────────────────────────────────────
    section("Step 7: Budget Sensitivity Analysis")
    bdf = budget_sweep(best_cate, y_te, t_te)
    plot_budget_curve(bdf)

    opt_idx = bdf["profit_lift_usd"].idxmax()
    opt_budget = bdf.loc[opt_idx, "budget_frac"]
    opt_profit = bdf.loc[opt_idx, "profit_lift_usd"]
    print(f"\n  Optimal budget fraction : {opt_budget*100:.1f}%")
    print(f"  Max incremental profit  : ${opt_profit:,.2f}")

    # ── 8. Final summary ─────────────────────────────────────────────────────
    section("Step 8: Results Summary")
    print(f"""
  Dataset       : Hillstrom MineThatData (64,000 customers)
  Outcome       : {OUTCOME_COL}
  Best model    : {best_model_name}  (AUQC = {auqc_dict[best_model_name]:.6f})
  Budget policy : Target top {budget_frac*100:.0f}% by predicted uplift
  Incremental   : ~{pol['incremental_conversions']:.0f} extra conversions
  Profit lift   : ${pol['profit_lift_usd']:,.2f}
  ROI           : {pol['roi_pct']:.1f}%
  Optimal tgt   : {opt_budget*100:.1f}% of customers
  Elapsed time  : {time.time() - t0:.1f}s
    """)

    # Save metrics to CSV
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(out_dir, "model_comparison.csv"), index=False)
    bdf.to_csv(os.path.join(out_dir, "budget_sweep.csv"), index=False)
    print(f"  Outputs saved to ./outputs/")
    print("\n  All done ✓")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uplift Modelling Pipeline")
    parser.add_argument("--skip-eda", action="store_true",
                        help="Skip EDA visualisations")
    parser.add_argument("--budget", type=float, default=BUDGET_FRAC,
                        help="Budget fraction for targeting policy (default: 0.30)")
    args = parser.parse_args()
    main(skip_eda=args.skip_eda, budget_frac=args.budget)
