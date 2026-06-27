"""Calibration and operating-point analysis.

`class_weight='balanced'` improves recall by pushing predicted probabilities toward
the positive class, which makes them poorly calibrated. This module compares the
balanced model against an unweighted model with and without post-hoc calibration, and
shows how to choose the decision threshold from the cost of errors rather than the
default 0.5. Everything is evaluated on a held-out test split.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .data import load_and_clean
from .evaluation import SEED, build_preprocessor


def split(seed: int = SEED):
    X, y = load_and_clean()
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)


def _tree(balanced: bool) -> Pipeline:
    """Regularized reference tree (depth 5, leaf 20) in the leak-safe pipeline."""
    return Pipeline(
        [
            ("prep", build_preprocessor()),
            (
                "clf",
                DecisionTreeClassifier(
                    max_depth=5,
                    min_samples_leaf=20,
                    class_weight=("balanced" if balanced else None),
                    random_state=SEED,
                ),
            ),
        ]
    )


def fit_probabilities(seed: int = SEED):
    """Fit four probability sources on train, return (y_test, {name: proba})."""
    Xtr, Xte, ytr, yte = split(seed)
    out = {}
    out["balanced_uncal"] = _tree(True).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    out["unweighted_uncal"] = _tree(False).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    out["unweighted_isotonic"] = (
        CalibratedClassifierCV(_tree(False), method="isotonic", cv=5)
        .fit(Xtr, ytr)
        .predict_proba(Xte)[:, 1]
    )
    out["unweighted_sigmoid"] = (
        CalibratedClassifierCV(_tree(False), method="sigmoid", cv=5)
        .fit(Xtr, ytr)
        .predict_proba(Xte)[:, 1]
    )
    return yte.reset_index(drop=True), out


def brier_table(yte, probs: dict) -> pd.DataFrame:
    """Brier (calibration) and ROC AUC (discrimination) per probability source."""
    rows = [
        {"model": k, "brier": brier_score_loss(yte, p), "roc_auc": roc_auc_score(yte, p)}
        for k, p in probs.items()
    ]
    return pd.DataFrame(rows).round(4)


def reliability(yte, probs: dict, n_bins: int = 10) -> dict:
    """{name: (mean_predicted, observed_fraction)} for reliability curves."""
    out = {}
    for k, p in probs.items():
        frac, mean = calibration_curve(yte, p, n_bins=n_bins, strategy="quantile")
        out[k] = (mean, frac)
    return out


def threshold_sweep(yte, proba) -> pd.DataFrame:
    rows = []
    for t in np.linspace(0.05, 0.95, 19):
        pred = (proba >= t).astype(int)
        rows.append(
            {
                "threshold": round(float(t), 2),
                "precision": precision_score(yte, pred, zero_division=0),
                "recall": recall_score(yte, pred, zero_division=0),
                "f1": f1_score(yte, pred, zero_division=0),
                "flagged": int(pred.sum()),
            }
        )
    return pd.DataFrame(rows).round(3)


def threshold_for_recall(yte, proba, target: float = 0.90) -> dict | None:
    """Highest threshold (most precise) that still achieves at least `target` recall."""
    prec, rec, thr = precision_recall_curve(yte, proba)
    ok = np.where(rec[:-1] >= target)[0]
    if len(ok) == 0:
        return None
    i = ok[-1]
    return {
        "threshold": round(float(thr[i]), 3),
        "precision": round(float(prec[i]), 3),
        "recall": round(float(rec[i]), 3),
    }


def threshold_by_cost(yte, proba, fn_cost: float = 5, fp_cost: float = 1) -> dict:
    """Threshold minimizing expected cost given false-negative vs false-positive costs."""
    best = None
    for t in np.linspace(0.01, 0.99, 99):
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        cost = fn * fn_cost + fp * fp_cost
        if best is None or cost < best["cost"]:
            best = {
                "threshold": round(float(t), 3),
                "cost": float(cost),
                "fn": int(fn),
                "fp": int(fp),
                "recall": round(recall_score(yte, pred, zero_division=0), 3),
                "precision": round(precision_score(yte, pred, zero_division=0), 3),
            }
    return best
