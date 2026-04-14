"""
visualizations.py
=================
All publication-quality charts for the uplift modelling project.

Charts
------
plot_eda_overview       – EDA: treatment arm sizes, conversion rates, spend dist.
plot_uplift_distribution – Histogram of predicted CATE per model
plot_qini_curves        – Qini curves + random baseline for all models
plot_cumulative_gain    – Cumulative gain curves
plot_feature_importance – Top-15 features by XGBoost gain (T-Learner treated arm)
plot_budget_curve       – Profit lift vs budget fraction
plot_cate_heatmap       – Mean predicted CATE by recency × history segment
plot_model_comparison   – Bar chart of AUQC scores
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Optional

from evaluation import qini_curve, cumulative_gain_curve, random_baseline_qini

# ── Style ─────────────────────────────────────────────────────────────────────

PALETTE = {
    # Baselines — visually distant from each other and from X variants
    "S-Learner":            "#5C6BC0",  # indigo
    "T-Learner":            "#26A69A",  # teal
    # X-Learner family — orange/red hue range, clearly grouped but distinct
    "X-Learner":            "#EF6C00",  # dark orange  (no calibration)
    "X-Learner [platt]":    "#E53935",  # red          (Platt scaling)
    "X-Learner [isotonic]": "#AD1457",  # deep pink    (isotonic)
    # Misc
    "Random":               "#90A4AE",  # grey
    "control":              "#B0BEC5",
    "treated":              "#5C6BC0",
}

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")

def _savefig(fig, name: str, dpi: int = 150):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[viz] Saved → outputs/figures/{name}")
    return path


def _style():
    plt.rcParams.update({
        "figure.facecolor":  "#FAFAFA",
        "axes.facecolor":    "#FFFFFF",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.35,
        "grid.linestyle":    "--",
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
    })


# ── EDA ───────────────────────────────────────────────────────────────────────

def plot_eda_overview(df: pd.DataFrame) -> str:
    _style()
    fig = plt.figure(figsize=(16, 10), facecolor="#FAFAFA")
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Treatment arm sizes
    ax1 = fig.add_subplot(gs[0, 0])
    arms  = df["segment"].value_counts()
    colors = ["#B0BEC5", "#5C6BC0", "#26A69A"]
    bars = ax1.bar(arms.index, arms.values, color=colors, edgecolor="white", width=0.6)
    ax1.set_title("Treatment arm sizes")
    ax1.set_ylabel("Customers")
    ax1.set_xticks(range(len(arms.index)))
    ax1.set_xticklabels(arms.index, rotation=12, ha="right")
    for bar, v in zip(bars, arms.values):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 200,
                 f"{v:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 2. Conversion rate by arm
    ax2 = fig.add_subplot(gs[0, 1])
    conv = df.groupby("segment")["conversion"].mean() * 100
    conv = conv.reindex(arms.index)
    bars2 = ax2.bar(conv.index, conv.values, color=colors, edgecolor="white", width=0.6)
    ax2.set_title("Conversion rate by arm")
    ax2.set_ylabel("Conversion rate (%)")
    ax2.set_xticks(range(len(conv.index)))
    ax2.set_xticklabels(conv.index, rotation=12, ha="right")
    for bar, v in zip(bars2, conv.values):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                 f"{v:.2f}%", ha="center", va="bottom", fontsize=9)

    # 3. Spend distribution (log-scale)
    ax3 = fig.add_subplot(gs[0, 2])
    seg_colors = {"No E-Mail": "#B0BEC5", "Mens E-Mail": "#5C6BC0",
                  "Womens E-Mail": "#26A69A"}
    for seg, col in seg_colors.items():
        sub = df[(df["segment"] == seg) & (df["spend"] > 0)]["spend"]
        if len(sub) > 0:
            ax3.hist(np.log1p(sub), bins=30, alpha=0.5, color=col, label=seg, density=True)
    ax3.set_title("Spend distribution (log1p)")
    ax3.set_xlabel("log(1 + spend)")
    ax3.set_ylabel("Density")
    ax3.legend(fontsize=8)

    # 4. Recency distribution
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(df["recency"], bins=12, color="#5C6BC0", alpha=0.8, edgecolor="white")
    ax4.set_title("Customer recency (months)")
    ax4.set_xlabel("Months since last purchase")
    ax4.set_ylabel("Count")

    # 5. Visit rate by arm
    ax5 = fig.add_subplot(gs[1, 1])
    visit = df.groupby("segment")["visit"].mean() * 100
    visit = visit.reindex(arms.index)
    bars5 = ax5.bar(visit.index, visit.values, color=colors, edgecolor="white", width=0.6)
    ax5.set_title("Website visit rate by arm")
    ax5.set_ylabel("Visit rate (%)")
    ax5.set_xticks(range(len(visit.index)))
    ax5.set_xticklabels(visit.index, rotation=12, ha="right")
    for bar, v in zip(bars5, visit.values):
        ax5.text(bar.get_x() + bar.get_width()/2, v + 0.1,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    # 6. History spend box
    ax6 = fig.add_subplot(gs[1, 2])
    seg_data = [df[df["segment"] == s]["history"].values for s in arms.index]
    bp = ax6.boxplot(seg_data, patch_artist=True, widths=0.5,
                     medianprops=dict(color="white", linewidth=2))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax6.set_xticks([1, 2, 3])
    ax6.set_xticklabels(arms.index, rotation=12, ha="right")
    ax6.set_title("Historical spend by arm")
    ax6.set_ylabel("Spend ($)")

    fig.suptitle("Hillstrom Dataset — Exploratory Data Analysis",
                 fontsize=15, fontweight="bold", y=1.01)
    return _savefig(fig, "01_eda_overview.png")


# ── CATE distribution ─────────────────────────────────────────────────────────

def plot_uplift_distribution(cate_dict: dict) -> str:
    _style()
    n_models = len(cate_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), facecolor="#FAFAFA")
    if n_models == 1:
        axes = [axes]

    for ax, (name, cate) in zip(axes, cate_dict.items()):
        color = PALETTE.get(name, "#607D8B")
        ax.hist(cate, bins=60, color=color, alpha=0.85, edgecolor="white", density=True)
        ax.axvline(0, color="#E53935", linewidth=1.5, linestyle="--", label="Zero lift")
        ax.axvline(np.mean(cate), color="#333", linewidth=1.5, linestyle="-",
                   label=f"Mean = {np.mean(cate):.4f}")
        ax.set_title(f"{name}\nCATE distribution")
        ax.set_xlabel("Predicted uplift (CATE)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        pct_pos = 100 * (cate > 0).mean()
        ax.text(0.97, 0.95, f"{pct_pos:.1f}% positive",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="#43A047",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", alpha=0.8))

    fig.suptitle("Predicted CATE Distribution — All Meta-Learners",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _savefig(fig, "02_cate_distributions.png")


# ── Qini curves ───────────────────────────────────────────────────────────────

def plot_qini_curves(cate_dict: dict, y: np.ndarray,
                     treatment: np.ndarray) -> str:
    _style()
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#FAFAFA")

    x_rand, y_rand = random_baseline_qini(y, treatment)
    ax.plot(x_rand, y_rand, color=PALETTE["Random"], linewidth=1.5,
            linestyle="--", label="Random targeting", zorder=1)

    for name, cate in cate_dict.items():
        x, q = qini_curve(y, treatment, cate)
        ax.plot(x, q, color=PALETTE.get(name, "#607D8B"),
                linewidth=2.5, label=name, zorder=3)

    ax.fill_between(x_rand, y_rand, alpha=0.04, color="gray")
    ax.set_xlabel("Fraction of population targeted")
    ax.set_ylabel("Cumulative net gain (Qini)")
    ax.set_title("Qini Curves — Uplift Model Comparison")
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    return _savefig(fig, "03_qini_curves.png")


# ── Cumulative gain ───────────────────────────────────────────────────────────

def plot_cumulative_gain(cate_dict: dict, y: np.ndarray,
                         treatment: np.ndarray) -> str:
    _style()
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#FAFAFA")

    for name, cate in cate_dict.items():
        x, g = cumulative_gain_curve(y, treatment, cate)
        ax.plot(x, g, color=PALETTE.get(name, "#607D8B"),
                linewidth=2.5, label=name)

    ax.axhline(0, color="#B0BEC5", linewidth=1, linestyle="--")
    ax.set_xlabel("Fraction of population targeted")
    ax.set_ylabel("Cumulative incremental conversions")
    ax.set_title("Cumulative Gain Curves")
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    return _savefig(fig, "04_cumulative_gain.png")


# ── Feature importance ────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list, n_top: int = 10) -> str:
    """Works for T-Learner or S-Learner (which expose _m1 or _model)."""
    _style()

    if hasattr(model, "_m1"):
        xgb_model = model._m1
    elif hasattr(model, "_model"):
        xgb_model = model._model
    else:
        print("[viz] Cannot extract feature importance from this model type.")
        return ""

    gain = xgb_model.get_booster().get_score(importance_type="gain")
    # Map f0, f1, … back to feature names
    fi = {}
    for k, v in gain.items():
        if k.startswith("f"):
            idx = int(k[1:])
            fname = feature_names[idx] if idx < len(feature_names) else k
        else:
            fname = k
        fi[fname] = v

    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:n_top])

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#FAFAFA")
    bars = ax.barh(list(fi_sorted.keys())[::-1],
                   list(fi_sorted.values())[::-1],
                   color="#5C6BC0", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"Top-{n_top} Feature Importances ({model.name})")
    plt.tight_layout()
    return _savefig(fig, "05_feature_importance.png")


# ── Budget curve ──────────────────────────────────────────────────────────────

def plot_budget_curve(budget_df: pd.DataFrame) -> str:
    _style()
    fig, ax1 = plt.subplots(figsize=(10, 5), facecolor="#FAFAFA")

    color_profit = "#5C6BC0"
    color_roi    = "#EF6C00"

    ax1.plot(budget_df["budget_frac"] * 100, budget_df["profit_lift_usd"],
             color=color_profit, linewidth=2.5, label="Profit lift ($)")
    ax1.fill_between(budget_df["budget_frac"] * 100,
                     budget_df["profit_lift_usd"], alpha=0.12, color=color_profit)
    ax1.set_xlabel("Budget (% of customers targeted)")
    ax1.set_ylabel("Incremental profit lift ($)", color=color_profit)
    ax1.tick_params(axis="y", labelcolor=color_profit)

    ax2 = ax1.twinx()
    ax2.plot(budget_df["budget_frac"] * 100, budget_df["roi_pct"],
             color=color_roi, linewidth=2.5, linestyle="--", label="ROI (%)")
    ax2.set_ylabel("Return on investment (%)", color=color_roi)
    ax2.tick_params(axis="y", labelcolor=color_roi)
    ax2.spines["top"].set_visible(False)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10)

    ax1.set_title("Budget Allocation Curve — Profit Lift vs Targeting Fraction")
    plt.tight_layout()
    return _savefig(fig, "06_budget_curve.png")


# ── CATE heatmap ──────────────────────────────────────────────────────────────

def plot_cate_heatmap(df: pd.DataFrame, cate: np.ndarray,
                      model_name: str = "X-Learner") -> str:
    _style()
    tmp = df.copy()
    tmp["cate"] = cate

    # Bin recency into 3 groups
    tmp["recency_bin"] = pd.cut(tmp["recency"],
                                bins=[0, 3, 7, 12],
                                labels=["1-3 mo", "4-7 mo", "8-12 mo"])

    pivot = tmp.groupby(["recency_bin", "history_segment"], observed=False)["cate"].mean().unstack()

    fig, ax = plt.subplots(figsize=(11, 4), facecolor="#FAFAFA")
    import matplotlib.cm as cm
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=-0.02, vmax=0.08)
    plt.colorbar(im, ax=ax, label="Mean predicted CATE")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_xlabel("History segment")
    ax.set_ylabel("Recency")
    ax.set_title(f"Mean Predicted CATE by Recency × History Segment ({model_name})")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color="black",
                        fontweight="bold" if abs(val) > 0.04 else "normal")

    plt.tight_layout()
    return _savefig(fig, "07_cate_heatmap.png")


# ── Model comparison bar ──────────────────────────────────────────────────────

def plot_model_comparison(auqc_dict: dict) -> str:
    _style()
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#FAFAFA")
    names  = list(auqc_dict.keys())
    scores = list(auqc_dict.values())
    colors = [PALETTE.get(n, "#607D8B") for n in names]

    bars = ax.bar(names, scores, color=colors, edgecolor="white", width=0.5)
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.0002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("AUQC (higher is better)")
    ax.set_title("Model Comparison — Area Under Qini Curve")
    plt.tight_layout()
    return _savefig(fig, "08_model_comparison.png")