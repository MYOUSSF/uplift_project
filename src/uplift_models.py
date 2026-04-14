"""
uplift_models.py
================
Implements three meta-learner approaches for heterogeneous treatment effect
estimation (CATE / uplift):

    S-Learner  – Single model, treatment is just another feature.
    T-Learner  – Two separate outcome models, one per arm.
    X-Learner  – Improvement on T-Learner that reduces bias with unequal
                  treatment proportions (Künzel et al. 2019).

Class imbalance
---------------
All classifiers use scale_pos_weight computed directly from the training
labels passed to fit() as n_negative / n_positive (~16 for the Hillstrom
dataset). For T-Learner and X-Learner the ratio is computed per arm, since
the conversion rate differs between control (~4.6%) and treated (~6.6%).

Probability calibration (X-Learner only)
-----------------------------------------
Calibration is applied exclusively to the X-Learner because that is where
miscalibration causes the most damage: μ₀ and μ₁ feed directly into the
Stage 2 residuals D₁ = y − μ₀(x) and D₀ = μ₁(x) − y, where the error is
one-sided and does not cancel out. For S-Learner and T-Learner the
probabilities are differenced (μ₁ − μ₀), so calibration errors partially
cancel and the gain is smaller.

Two calibration methods are available via the `calibration` argument:
    None         – raw probabilities (scale_pos_weight correction only)
    "platt"      – Platt scaling (logistic regression on held-out predictions)
    "isotonic"   – Isotonic regression (non-parametric, more flexible)

Both use CalibratedClassifierCV with cv="prefit", which wraps the
already-fitted XGBoost and fits only the calibration layer.

All models expose a common interface:
    fit(X, treatment, y)  → self
    predict(X)            → np.ndarray of CATE estimates (same length as X)
    name                  → str

References
----------
Künzel, S. R., Sekhon, J. S., Wager, S., & Yu, B. (2019).
    Metalearners for estimating heterogeneous treatment effects using machine
    learning. PNAS, 116(10), 4156-4165.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from typing import Optional, Literal


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scale_pos_weight(y: np.ndarray) -> float:
    """Compute n_negative / n_positive from labels."""
    n_pos = float(y.sum())
    n_neg = len(y) - n_pos
    return n_neg / n_pos if n_pos > 0 else 1.0


def _make_clf(spw: float) -> XGBClassifier:
    """XGBClassifier with per-arm class-imbalance correction."""
    return XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def _default_reg() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def _calibrate(
    clf: XGBClassifier,
    X: pd.DataFrame,
    y: np.ndarray,
    method: str,
) -> object:
    """
    Wrap an already-fitted clf with a calibration layer.
    cv="prefit" fits only the calibration layer; XGBoost is not re-trained.
    """
    sklearn_method = "sigmoid" if method == "platt" else "isotonic"
    cal = CalibratedClassifierCV(clf, method=sklearn_method, cv="prefit")
    cal.fit(X, y)
    return cal


def _proba1(model, X: pd.DataFrame) -> np.ndarray:
    """Return P(y=1 | X) for raw or calibrated classifiers."""
    return model.predict_proba(X)[:, 1]


# ── S-Learner ─────────────────────────────────────────────────────────────────

class SLearner:
    """
    Single-model meta-learner.

    Trains one model on [X, T] and estimates CATE as:
        τ(x) = μ(x, 1) − μ(x, 0)

    Uses scale_pos_weight computed from y. No probability calibration —
    miscalibration errors cancel in the μ₁ − μ₀ difference.
    """

    name = "S-Learner"

    def __init__(self, outcome: str = "binary"):
        self.outcome = outcome

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray):
        Xt = X.copy()
        Xt["__treatment__"] = treatment

        if self.outcome == "binary":
            clf = _make_clf(_scale_pos_weight(y))
            clf.fit(Xt, y)
            self._model = clf
        else:
            self._model = _default_reg()
            self._model.fit(Xt, y)

        self._cols = Xt.columns.tolist()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X0 = X.copy(); X0["__treatment__"] = 0
        X1 = X.copy(); X1["__treatment__"] = 1

        if self.outcome == "binary":
            p0 = _proba1(self._model, X0[self._cols])
            p1 = _proba1(self._model, X1[self._cols])
        else:
            p0 = self._model.predict(X0[self._cols])
            p1 = self._model.predict(X1[self._cols])

        return p1 - p0


# ── T-Learner ─────────────────────────────────────────────────────────────────

class TLearner:
    """
    Two-model meta-learner.

    Trains μ₀ on control and μ₁ on treated, then:
        τ(x) = μ₁(x) − μ₀(x)

    scale_pos_weight computed per arm. No probability calibration —
    miscalibration errors partially cancel in the μ₁ − μ₀ difference.
    """

    name = "T-Learner"

    def __init__(self, outcome: str = "binary"):
        self.outcome = outcome

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray):
        mask = treatment == 1
        X0, y0 = X[~mask], y[~mask]
        X1, y1 = X[mask],  y[mask]

        if self.outcome == "binary":
            self._m0 = _make_clf(_scale_pos_weight(y0)); self._m0.fit(X0, y0)
            self._m1 = _make_clf(_scale_pos_weight(y1)); self._m1.fit(X1, y1)
        else:
            self._m0 = _default_reg(); self._m0.fit(X0, y0)
            self._m1 = _default_reg(); self._m1.fit(X1, y1)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.outcome == "binary":
            return _proba1(self._m1, X) - _proba1(self._m0, X)
        return self._m1.predict(X) - self._m0.predict(X)


# ── X-Learner ─────────────────────────────────────────────────────────────────

class XLearner:
    """
    Cross-fitting meta-learner (Künzel et al. 2019).

    Stage 1 : Fit μ₀ (control) and μ₁ (treated) with per-arm scale_pos_weight.
              Optionally calibrate both with Platt scaling or isotonic regression.
    Stage 2 : Impute individual treatment effects using calibrated probabilities:
                  D₁ = y₁ − μ₀(x₁)
                  D₀ = μ₁(x₀) − y₀
              Fit second-stage regressors τ₀ on D₀, τ₁ on D₁.
    Stage 3 : Blend with propensity score:
                  τ(x) = g(x) τ₀(x) + (1 − g(x)) τ₁(x)

    Calibration is applied here (not in S/T-Learner) because the Stage 2
    residuals involve a one-sided difference between a hard label (0/1) and a
    model probability. Miscalibration shrinks residuals toward zero and biases
    the second-stage regressors.
    """

    def __init__(
        self,
        outcome: str = "binary",
        calibration: Optional[Literal["platt", "isotonic"]] = None,
    ):
        self.outcome     = outcome
        self.calibration = calibration
        cal_tag          = f" [{calibration}]" if calibration else ""
        self.name        = f"X-Learner{cal_tag}"
        self._tau0       = _default_reg()
        self._tau1       = _default_reg()
        self._propensity = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray):
        mask = treatment == 1
        X0, y0 = X[~mask], y[~mask]
        X1, y1 = X[mask],  y[mask]

        # ── Stage 1 ──────────────────────────────────────────────────────────
        if self.outcome == "binary":
            m0 = _make_clf(_scale_pos_weight(y0)); m0.fit(X0, y0)
            m1 = _make_clf(_scale_pos_weight(y1)); m1.fit(X1, y1)
            if self.calibration:
                self._m0 = _calibrate(m0, X0, y0, self.calibration)
                self._m1 = _calibrate(m1, X1, y1, self.calibration)
            else:
                self._m0, self._m1 = m0, m1
        else:
            self._m0 = _default_reg(); self._m0.fit(X0, y0)
            self._m1 = _default_reg(); self._m1.fit(X1, y1)

        # ── Stage 2 ──────────────────────────────────────────────────────────
        if self.outcome == "binary":
            mu0_on_treated = _proba1(self._m0, X1)
            mu1_on_control = _proba1(self._m1, X0)
        else:
            mu0_on_treated = self._m0.predict(X1)
            mu1_on_control = self._m1.predict(X0)

        D1 = y1.astype(float) - mu0_on_treated
        D0 = mu1_on_control   - y0.astype(float)

        self._tau1.fit(X1, D1)
        self._tau0.fit(X0, D0)

        # ── Propensity ────────────────────────────────────────────────────────
        self._propensity.fit(X, treatment)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        g    = self._propensity.predict_proba(X)[:, 1]
        tau0 = self._tau0.predict(X)
        tau1 = self._tau1.predict(X)
        return g * tau0 + (1 - g) * tau1


# ── Convenience factory ───────────────────────────────────────────────────────

def get_all_variants(outcome: str = "binary") -> list:
    """
    Return the 5 models used for the comparison:
        S-Learner              (scale_pos_weight, no calibration)
        T-Learner              (scale_pos_weight, no calibration)
        X-Learner              (scale_pos_weight, no calibration)
        X-Learner [platt]      (scale_pos_weight + Platt scaling)
        X-Learner [isotonic]   (scale_pos_weight + isotonic regression)
    """
    return [
        SLearner(outcome=outcome),
        TLearner(outcome=outcome),
        XLearner(outcome=outcome, calibration=None),
        XLearner(outcome=outcome, calibration="platt"),
        XLearner(outcome=outcome, calibration="isotonic"),
    ]