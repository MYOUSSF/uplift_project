"""
tests/test_pipeline.py
======================
Unit and integration tests for the uplift modelling pipeline.

Run with:
    pytest tests/ -v
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader     import _generate_synthetic_hillstrom, preprocess, get_feature_cols
from uplift_models   import SLearner, TLearner, XLearner, get_all_models
from evaluation      import qini_curve, auqc, uplift_curve, cumulative_gain_curve
from targeting_policy import optimal_threshold, apply_policy, policy_summary, budget_sweep


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_dataset():
    """Small processed dataset (2,000 rows) for fast tests."""
    raw = _generate_synthetic_hillstrom(n=2000, seed=0)
    df  = preprocess(raw)
    return df


@pytest.fixture(scope="module")
def xy(small_dataset):
    """Feature matrix, treatment, and outcome arrays."""
    df   = small_dataset
    cols = get_feature_cols()
    X    = df[cols]
    t    = df["treatment_binary"].values
    y    = df["conversion"].values
    return X, t, y


# ── Data loader tests ─────────────────────────────────────────────────────────

class TestDataLoader:
    def test_shape(self, small_dataset):
        assert small_dataset.shape[0] == 2000

    def test_columns_present(self, small_dataset):
        required = ["treatment", "treatment_binary", "conversion", "visit",
                    "history_segment_enc", "zip_code_enc", "channel_enc"]
        for col in required:
            assert col in small_dataset.columns, f"Missing column: {col}"

    def test_treatment_binary_values(self, small_dataset):
        assert set(small_dataset["treatment_binary"].unique()).issubset({0, 1})

    def test_no_nulls_in_features(self, small_dataset):
        cols = get_feature_cols()
        assert small_dataset[cols].isnull().sum().sum() == 0

    def test_treatment_balance(self, small_dataset):
        """Each arm should be roughly 1/3 of the data."""
        counts = small_dataset["segment"].value_counts(normalize=True)
        for arm in ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]:
            assert abs(counts.get(arm, 0) - 1/3) < 0.10


# ── Model tests ───────────────────────────────────────────────────────────────

class TestModels:
    @pytest.mark.parametrize("ModelClass", [SLearner, TLearner, XLearner])
    def test_fit_predict_shape(self, ModelClass, xy):
        X, t, y = xy
        model   = ModelClass(outcome="binary")
        model.fit(X, t, y)
        cate = model.predict(X)
        assert cate.shape == (len(X),), f"{ModelClass.__name__}: wrong output shape"

    @pytest.mark.parametrize("ModelClass", [SLearner, TLearner, XLearner])
    def test_cate_is_float(self, ModelClass, xy):
        X, t, y = xy
        model   = ModelClass(outcome="binary")
        model.fit(X, t, y)
        cate = model.predict(X)
        assert np.issubdtype(cate.dtype, np.floating)

    @pytest.mark.parametrize("ModelClass", [SLearner, TLearner, XLearner])
    def test_mean_cate_near_ate(self, ModelClass, xy):
        """
        Mean predicted CATE should be in the same ballpark as the naive
        ATE estimate (difference in mean outcomes by treatment arm).
        Not exact — just a sanity check.
        """
        X, t, y = xy
        ate_naive = y[t == 1].mean() - y[t == 0].mean()
        model = ModelClass(outcome="binary")
        model.fit(X, t, y)
        cate = model.predict(X)
        assert abs(cate.mean() - ate_naive) < 0.10, (
            f"{ModelClass.__name__}: mean CATE {cate.mean():.4f} "
            f"far from naive ATE {ate_naive:.4f}"
        )

    def test_get_all_models_length(self):
        models = get_all_models()
        assert len(models) == 3

    def test_model_names_unique(self):
        names = [m.name for m in get_all_models()]
        assert len(names) == len(set(names))


# ── Evaluation tests ──────────────────────────────────────────────────────────

class TestEvaluation:
    def _random_scores(self, n=500):
        rng = np.random.default_rng(1)
        y   = rng.binomial(1, 0.1, n)
        t   = rng.binomial(1, 0.5, n)
        s   = rng.uniform(-0.05, 0.15, n)
        return y, t, s

    def test_qini_curve_shape(self):
        y, t, s = self._random_scores()
        x, q    = qini_curve(y, t, s, n_bins=50)
        assert len(x) == len(q) == 51  # n_bins + 1 (prepended 0)
        assert x[0] == 0.0
        assert q[0] == 0.0

    def test_auqc_returns_float(self):
        y, t, s = self._random_scores()
        score   = auqc(y, t, s)
        assert isinstance(score, float)

    def test_perfect_model_beats_random(self):
        """
        A model that predicts high scores for truly positive treated customers
        should score higher than random noise scores.
        """
        rng  = np.random.default_rng(2)
        n    = 2000
        t    = rng.binomial(1, 0.5, n)
        y    = rng.binomial(1, 0.05 + 0.1 * t, n)

        # Oracle: perfectly separates high-effect from low-effect customers
        oracle_scores = y * t * 1.0 + rng.uniform(0, 0.01, n)
        noise_scores  = rng.uniform(0, 1, n)

        auqc_oracle = auqc(y, t, oracle_scores)
        auqc_random = auqc(y, t, noise_scores)
        assert auqc_oracle > auqc_random, "Oracle should beat random scores"

    def test_uplift_curve_bounds(self):
        y, t, s = self._random_scores()
        x, u    = uplift_curve(y, t, s)
        assert 0.0 <= x[0] <= 1.0
        assert 0.0 <= x[-1] <= 1.0

    def test_cumulative_gain_starts_at_zero(self):
        y, t, s = self._random_scores()
        x, g    = cumulative_gain_curve(y, t, s)
        assert x[0] == 0.0
        assert g[0] == 0.0


# ── Policy tests ──────────────────────────────────────────────────────────────

class TestPolicy:
    def _scores_and_data(self, n=1000):
        rng = np.random.default_rng(3)
        t   = rng.binomial(1, 0.5, n)
        y   = rng.binomial(1, 0.05 + 0.05 * t, n)
        s   = rng.uniform(-0.02, 0.12, n)
        return s, y, t

    def test_threshold_budget_respected(self):
        s, y, t = self._scores_and_data()
        for b in [0.10, 0.25, 0.50, 0.80]:
            decision = apply_policy(s, b)
            actual_frac = decision.mean()
            assert actual_frac <= b + 0.02, (
                f"Budget {b}: targeted {actual_frac:.3f} > allowed {b}"
            )

    def test_summary_keys_present(self):
        s, y, t = self._scores_and_data()
        result  = policy_summary(s, y, t, budget_frac=0.30)
        required_keys = [
            "n_total", "n_targeted", "budget_frac",
            "incremental_conversions", "profit_lift_usd", "roi_pct"
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_budget_sweep_monotone_coverage(self):
        s, y, t = self._scores_and_data()
        bdf = budget_sweep(s, y, t, budgets=np.linspace(0.1, 1.0, 10))
        # n_targeted should be non-decreasing with budget
        assert (np.diff(bdf["n_targeted"].values) >= 0).all()

    def test_optimal_threshold_full_budget(self):
        s = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        thresh = optimal_threshold(s, budget_frac=1.0)
        assert thresh <= s.min()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
