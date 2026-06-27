"""Leak-safe evaluation: per-fold preprocessing, feature selection, and tuning.

The original notebook encoded, imputed, and selected features on the full dataset
*before* the train/test split, leaking test information. Everything here is built
inside an sklearn Pipeline so it is fit only on each training fold.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data import CONTINUOUS, PASSTHROUGH

SEED = 42


def build_preprocessor() -> ColumnTransformer:
    """Scale continuous features, pass binary/ordinal through. Fit per fold."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), CONTINUOUS),
            ("pass", "passthrough", PASSTHROUGH),
        ]
    )


def _pipe(clf) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_preprocessor()),
            ("select", SelectKBest(f_classif)),
            ("clf", clf),
        ]
    )


def models() -> dict:
    """Each entry: (pipeline, hyperparameter grid for the inner search)."""
    return {
        "logistic_regression": (
            _pipe(LogisticRegression(class_weight="balanced", max_iter=1000, solver="liblinear")),
            {"select__k": [6, 8, 11], "clf__C": [0.01, 0.1, 1, 10]},
        ),
        "decision_tree": (
            _pipe(DecisionTreeClassifier(class_weight="balanced", random_state=SEED)),
            {
                "select__k": [8, 11],
                "clf__max_depth": [3, 5, 7, None],
                "clf__min_samples_leaf": [1, 5, 20],
            },
        ),
    }


def _metrics(y_true, proba) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "recall": recall_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "brier": brier_score_loss(y_true, proba),
    }


def nested_cv(pipe, grid, X, y, scoring="roc_auc", seed=SEED) -> pd.DataFrame:
    """Nested CV: 5-fold outer estimate, 3-fold inner tuning. No leakage."""
    outer = StratifiedKFold(5, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(outer.split(X, y), 1):
        inner = StratifiedKFold(3, shuffle=True, random_state=seed)
        gs = GridSearchCV(pipe, grid, scoring=scoring, cv=inner, n_jobs=-1)
        gs.fit(X.iloc[tr], y.iloc[tr])
        proba = gs.best_estimator_.predict_proba(X.iloc[te])[:, 1]
        m = _metrics(y.iloc[te], proba)
        m["fold"] = fold
        m["best_params"] = gs.best_params_
        rows.append(m)
    return pd.DataFrame(rows)


def run_all(X, y) -> pd.DataFrame:
    """Nested CV for every model; returns tidy per-fold results."""
    frames = []
    for name, (pipe, grid) in models().items():
        df = nested_cv(pipe, grid, X, y)
        df.insert(0, "model", name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summary(results: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std per model across outer folds."""
    metric_cols = ["roc_auc", "pr_auc", "recall", "precision", "f1", "brier"]
    agg = results.groupby("model")[metric_cols].agg(["mean", "std"])
    return agg.round(3)


def leak_demo(X, y, k=8, seed=SEED) -> dict:
    """Quantify the feature-selection leak: selecting k features inside each fold
    (safe) vs once on the full dataset using the target (leaky). Logistic model,
    5-fold outer, ROC AUC."""
    outer = StratifiedKFold(5, shuffle=True, random_state=seed)

    def lr():
        return LogisticRegression(class_weight="balanced", max_iter=1000, solver="liblinear")

    safe = Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_classif, k=k)), ("clf", lr())])
    safe_auc = []
    for tr, te in outer.split(X, y):
        safe.fit(X.iloc[tr], y.iloc[tr])
        safe_auc.append(roc_auc_score(y.iloc[te], safe.predict_proba(X.iloc[te])[:, 1]))

    # Leaky: pick the k best features using the entire dataset (incl. future test rows).
    sel_all = SelectKBest(f_classif, k=k).fit(StandardScaler().fit_transform(X), y)
    cols = X.columns[sel_all.get_support()]
    leaky = Pipeline([("sc", StandardScaler()), ("clf", lr())])
    leaky_auc = []
    for tr, te in outer.split(X[cols], y):
        leaky.fit(X[cols].iloc[tr], y.iloc[tr])
        leaky_auc.append(roc_auc_score(y.iloc[te], leaky.predict_proba(X[cols].iloc[te])[:, 1]))

    return {
        "k": k,
        "safe_auc_mean": float(np.mean(safe_auc)),
        "leaky_auc_mean": float(np.mean(leaky_auc)),
        "inflation": float(np.mean(leaky_auc) - np.mean(safe_auc)),
        "leaky_selected": list(cols),
    }
