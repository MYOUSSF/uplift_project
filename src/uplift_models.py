"""
uplift_models.py
================
Implements three meta-learner approaches for heterogeneous treatment effect
estimation (CATE / uplift):

    S-Learner  – Single model, treatment is just another feature.
    T-Learner  – Two separate outcome models, one per arm.
    X-Learner  – Improvement on T-Learner that reduces bias with unequal
                  treatment proportions (Künzel et al. 2019).

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
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from typing import Optional


# ── Default base learner ──────────────────────────────────────────────────────

def _default_clf():
    return XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def _default_reg():
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


# ── S-Learner ─────────────────────────────────────────────────────────────────

class SLearner:
    """
    Single-model meta-learner.

    Trains one model on [X, T] and estimates CATE as:
        τ(x) = μ(x, 1) − μ(x, 0)
    """

    name = "S-Learner"

    def __init__(self, base_learner=None, outcome: str = "binary"):
        self.outcome = outcome
        self._model = base_learner or (_default_clf() if outcome == "binary" else _default_reg())

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray):
        Xt = X.copy()
        Xt["__treatment__"] = treatment
        self._model.fit(Xt, y)
        self._cols = Xt.columns.tolist()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X0 = X.copy(); X0["__treatment__"] = 0
        X1 = X.copy(); X1["__treatment__"] = 1

        if self.outcome == "binary":
            p0 = self._model.predict_proba(X0[self._cols])[:, 1]
            p1 = self._model.predict_proba(X1[self._cols])[:, 1]
        else:
            p0 = self._model.predict(X0[self._cols])
            p1 = self._model.predict(X1[self._cols])

        return p1 - p0


# ── T-Learner ─────────────────────────────────────────────────────────────────

class TLearner:
    """
    Two-model meta-learner.

    Trains μ₀ on control group and μ₁ on treated group, then:
        τ(x) = μ₁(x) − μ₀(x)
    """

    name = "T-Learner"

    def __init__(self, base_learner=None, outcome: str = "binary"):
        self.outcome = outcome
        self._m0 = base_learner or (_default_clf() if outcome == "binary" else _default_reg())
        self._m1 = clone(self._m0)

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray):
        mask = treatment == 1
        self._m0.fit(X[~mask], y[~mask])
        self._m1.fit(X[mask],  y[mask])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.outcome == "binary":
            p0 = self._m0.predict_proba(X)[:, 1]
            p1 = self._m1.predict_proba(X)[:, 1]
        else:
            p0 = self._m0.predict(X)
            p1 = self._m1.predict(X)
        return p1 - p0


# ── X-Learner ─────────────────────────────────────────────────────────────────

class XLearner:
    """
    Cross-fitting meta-learner (Künzel et al. 2019).

    Stage 1: Fit outcome models μ₀, μ₁ (same as T-Learner).
    Stage 2: Compute imputed treatment effects:
        D₁ = y₁ − μ₀(x₁)   (in treated)
        D₀ = μ₁(x₀) − y₀   (in control)
             Fit τ₁(x) on D₁, τ₀(x) on D₀.
    Stage 3: Blend with propensity score g(x):
        τ(x) = g(x) τ₀(x) + (1 − g(x)) τ₁(x)
    """

    name = "X-Learner"

    def __init__(self, base_learner=None, outcome: str = "binary"):
        self.outcome = outcome
        self._m0 = base_learner or (_default_clf() if outcome == "binary" else _default_reg())
        self._m1 = clone(self._m0)
        self._tau0 = _default_reg()
        self._tau1 = _default_reg()
        self._propensity = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray):
        mask = treatment == 1

        # Stage 1
        self._m0.fit(X[~mask], y[~mask])
        self._m1.fit(X[mask],  y[mask])

        # Stage 2
        if self.outcome == "binary":
            mu0_on_treated  = self._m0.predict_proba(X[mask])[:, 1]
            mu1_on_control  = self._m1.predict_proba(X[~mask])[:, 1]
        else:
            mu0_on_treated  = self._m0.predict(X[mask])
            mu1_on_control  = self._m1.predict(X[~mask])

        D1 = y[mask].astype(float)  - mu0_on_treated
        D0 = mu1_on_control         - y[~mask].astype(float)

        self._tau1.fit(X[mask],  D1)
        self._tau0.fit(X[~mask], D0)

        # Propensity
        self._propensity.fit(X, treatment)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        g    = self._propensity.predict_proba(X)[:, 1]
        tau0 = self._tau0.predict(X)
        tau1 = self._tau1.predict(X)
        return g * tau0 + (1 - g) * tau1


# ── Convenience factory ───────────────────────────────────────────────────────

def get_all_models(outcome: str = "binary") -> list:
    """Return one instance of each meta-learner."""
    return [
        SLearner(outcome=outcome),
        TLearner(outcome=outcome),
        XLearner(outcome=outcome),
    ]
