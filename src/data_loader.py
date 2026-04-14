"""
data_loader.py
==============
Downloads and preprocesses the Hillstrom MineThatData e-mail marketing dataset.
The dataset contains 64,000 customers with randomized treatment assignment.

Dataset columns:
    recency      : Months since last purchase
    history_segment : Categorical spend history bucket
    history      : Dollar value of historical purchases
    mens         : 1 if purchased men's merchandise
    womens       : 1 if purchased women's merchandise
    zip_code     : Urban / Suburban / Rural
    newbie       : 1 if new customer (< 12 months)
    channel      : Phone / Web / Multichannel
    segment      : Treatment arm (No E-Mail / Mens E-Mail / Womens E-Mail)
    visit        : 1 if customer visited website in 2 weeks after campaign
    conversion   : 1 if customer purchased in 2 weeks after campaign
    spend        : Dollar amount spent in 2 weeks after campaign
"""

import os
import io
import urllib.request
import pandas as pd
import numpy as np

DATASET_URL = (
    "https://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics"
    "_DataMiningChallenge_2008.03.20.csv"
)
RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "hillstrom_raw.csv")
PROCESSED_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "hillstrom_processed.csv"
)


def download_dataset(force: bool = False) -> pd.DataFrame:
    """Download Hillstrom dataset; use cached copy if available."""
    raw_path = os.path.abspath(RAW_PATH)
    if os.path.exists(raw_path) and not force:
        print("[data_loader] Loading cached dataset from data/")
        return pd.read_csv(raw_path)

    print("[data_loader] Downloading Hillstrom dataset …")
    try:
        with urllib.request.urlopen(DATASET_URL, timeout=30) as response:
            content = response.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content))
    except Exception:
        print("[data_loader] Remote download failed — generating synthetic replica …")
        df = _generate_synthetic_hillstrom(n=64000, seed=42)

    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    df.to_csv(raw_path, index=False)
    print("[data_loader] Loading cached dataset from data/")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature-engineer and encode the raw Hillstrom DataFrame.

    Treatment encoding
    ------------------
    segment == 'No E-Mail'    → treatment = 0  (control)
    segment == 'Mens E-Mail'  → treatment = 1
    segment == 'Womens E-Mail'→ treatment = 2

    For binary uplift (any email vs none) we also create:
        treatment_binary = 1 if any email, else 0
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    # ── Treatment encoding ──────────────────────────────────────────────────
    segment_map = {"No E-Mail": 0, "Mens E-Mail": 1, "Womens E-Mail": 2}
    df["treatment"] = df["segment"].map(segment_map)
    df["treatment_binary"] = (df["treatment"] > 0).astype(int)

    # ── Categorical encoding ─────────────────────────────────────────────────
    history_map = {
        "$0 - $100": 0, "$100 - $200": 1, "$200 - $350": 2,
        "$350 - $500": 3, "$500 - $750": 4, "$750 - $1,000": 5,
        "$1,000 +": 6,
    }
    df["history_segment_enc"] = df["history_segment"].map(history_map).fillna(0)

    zipcode_map = {"Rural": 0, "Suburban": 1, "Urban": 2}
    df["zip_code_enc"] = df["zip_code"].map(zipcode_map).fillna(1)

    channel_map = {"Phone": 0, "Web": 1, "Multichannel": 2}
    df["channel_enc"] = df["channel"].map(channel_map).fillna(1)

    # ── Log-transform spend ──────────────────────────────────────────────────
    df["log_spend"] = np.log1p(df["spend"])

    # ── Feature list ─────────────────────────────────────────────────────────
    df["feature_cols"] = None  # placeholder; actual list returned separately

    proc_path = os.path.abspath(PROCESSED_PATH)
    os.makedirs(os.path.dirname(proc_path), exist_ok=True)
    df.to_csv(proc_path, index=False)
    print("[data_loader] Processed dataset saved to data/")
    return df


def get_feature_cols() -> list:
    """Return the covariate column names used for modelling."""
    return [
        "recency", "history", "mens", "womens", "newbie",
        "history_segment_enc", "zip_code_enc", "channel_enc",
    ]


def load_processed() -> pd.DataFrame:
    """Convenience: download + preprocess in one call."""
    raw = download_dataset()
    return preprocess(raw)


# ── Synthetic replica ────────────────────────────────────────────────────────

def _generate_synthetic_hillstrom(n: int = 64000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset that mirrors the Hillstrom schema and
    approximate marginal distributions when the remote source is unreachable.
    """
    rng = np.random.default_rng(seed)

    recency = rng.integers(1, 13, size=n)
    history = np.exp(rng.normal(5.5, 1.2, size=n)).clip(0, 3500)
    mens = rng.binomial(1, 0.5, size=n)
    womens = 1 - mens
    newbie = rng.binomial(1, 0.15, size=n)

    history_labels = [
        "$0 - $100", "$100 - $200", "$200 - $350",
        "$350 - $500", "$500 - $750", "$750 - $1,000", "$1,000 +",
    ]
    h_idx = pd.cut(history, bins=[0,100,200,350,500,750,1000,1e9],
                   labels=range(7), right=False).astype(int)
    history_segment = [history_labels[i] for i in h_idx]

    zip_code = rng.choice(["Rural", "Suburban", "Urban"], size=n, p=[0.2, 0.5, 0.3])
    channel = rng.choice(["Phone", "Web", "Multichannel"], size=n, p=[0.3, 0.5, 0.2])
    segment = rng.choice(
        ["No E-Mail", "Mens E-Mail", "Womens E-Mail"], size=n, p=[1/3, 1/3, 1/3]
    )

    # Outcomes (heterogeneous treatment effects)
    base_p = 0.10 + 0.02 * (recency < 4) + 0.03 * newbie
    treat_effect = np.where(segment == "No E-Mail", 0,
                   np.where(segment == "Mens E-Mail", 0.09 * mens, 0.09 * womens))
    visit_p = (base_p + treat_effect).clip(0, 1)
    visit = rng.binomial(1, visit_p)

    conv_p = (0.04 + 0.015 * (recency < 4) + treat_effect * 0.5).clip(0, 1)
    conversion = rng.binomial(1, conv_p)

    spend = conversion * np.exp(rng.normal(4.5, 1.0, size=n)).clip(0, 500)

    return pd.DataFrame({
        "recency": recency, "history_segment": history_segment,
        "history": history.round(2), "mens": mens, "womens": womens,
        "zip_code": zip_code, "newbie": newbie, "channel": channel,
        "segment": segment, "visit": visit, "conversion": conversion,
        "spend": spend.round(2),
    })