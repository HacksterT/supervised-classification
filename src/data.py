"""Data loading and preparation for the personal-loan classification task."""

from pathlib import Path

import pandas as pd

# Resolve from either the notebook dir or the repo root.
_CANDIDATES = (Path("../data/Loan_Modelling.csv"), Path("data/Loan_Modelling.csv"))
DATA_PATH = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[-1])

TARGET = "Personal_Loan"

# ID is an identifier. ZIPCode collapses to a single region (all California 9xxxx),
# so its first-digit "region" feature is constant and carries no signal. Drop both.
DROP = ["ID", "ZIPCode"]

# Continuous features get scaled (matters for logistic regression); the rest are
# binary or low-cardinality ordinals passed through untouched.
CONTINUOUS = ["Age", "Experience", "Income", "CCAvg", "Mortgage"]
PASSTHROUGH = ["Family", "Education", "Securities_Account", "CD_Account", "Online", "CreditCard"]
FEATURES = CONTINUOUS + PASSTHROUGH


def load_and_clean(path: Path = DATA_PATH):
    """Load the dataset, fix known quirks, return (X, y).

    The only cleaning is fixing the 52 negative `Experience` values (data-entry
    sign errors) by taking the absolute value. No target-based transformation
    happens here; all of that lives inside the per-fold pipeline to avoid leakage.
    """
    df = pd.read_csv(path)
    df["Experience"] = df["Experience"].abs()
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y
