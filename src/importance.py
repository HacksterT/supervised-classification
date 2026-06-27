"""Feature importance, done honestly.

Tree impurity importance (`feature_importances_`) is biased toward high-cardinality
continuous features. Permutation importance, measured on held-out test data, asks the
real question: how much does performance drop when this feature is shuffled? This module
computes both and compares them, using the unweighted reference tree from pass 2.
"""

import pandas as pd
from sklearn.inspection import permutation_importance

from .calibration import _tree, split
from .evaluation import SEED


def feature_importance(seed: int = SEED, n_repeats: int = 30, scoring: str = "roc_auc") -> pd.DataFrame:
    """Permutation importance (on test) vs impurity importance, aligned and sorted."""
    Xtr, Xte, ytr, yte = split(seed)
    pipe = _tree(balanced=False).fit(Xtr, ytr)

    perm = permutation_importance(
        pipe, Xte, yte, scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=-1
    )
    impurity = pipe.named_steps["clf"].feature_importances_

    df = pd.DataFrame(
        {
            "feature": Xte.columns,  # ColumnTransformer preserves CONTINUOUS + PASSTHROUGH order
            "perm_importance": perm.importances_mean,
            "perm_std": perm.importances_std,
            "impurity_importance": impurity,
        }
    )
    return df.sort_values("perm_importance", ascending=False).reset_index(drop=True).round(4)
