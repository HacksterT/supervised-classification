# Supervised Classification

A supervised classification project: predicting which existing bank customers will accept a personal loan, using logistic regression and decision trees evaluated to a clinical-research standard.

As with the rest of the portfolio, the point is not the loan domain. It is the **evaluation discipline** on an imbalanced binary target: honest validation, calibration, and threshold choice driven by the cost of errors rather than raw accuracy. The logistic and tree baselines here also feed the Clinical-QML Benchmark (forthcoming) as classical reference models.

## Status

Migrated to Python 3.13 + uv and **runs end-to-end** (23/23 code cells, 0 errors; scikit-learn 1.9, pandas 3.0.3, numpy 2.5). Port fixes applied: dropped the phantom `sklearn_pandas` version-print import (it was never used for modeling, encoding is `get_dummies` in cell 64), resolved the hardcoded Windows `read_csv` path to `data/`, and reset the kernelspec. The one `!pip` cell was already commented out and is inert.

Rigor pass 1 is done: `notebooks/02-leak-safe-pipeline.ipynb` rebuilds the evaluation as a leak-safe `ColumnTransformer` + `Pipeline` with nested cross-validation (logic in `src/`), and `docs/learnings.md` records the findings. Honest headline: the tuned decision tree reaches ROC AUC 0.991 / PR AUC 0.921, and the feature-selection leak turned out negligible (-0.001 AUC) on this low-dimensional data, so the refactor's value here was correctness and reproducibility, not a changed number.

## Dataset

5,000 customers, 14 attributes, binary target (`Personal_Loan`). Features include age, income, family size, education, mortgage, and existing-account flags. The positive class is rare (~9.6%), which is the whole reason calibration and threshold selection matter here. Source: a public loan-modelling dataset; included in `data/`.

## Approach and roadmap

- [x] Baseline models: logistic regression and decision tree.
- [x] Stack migration: runs end-to-end on Python 3.13 + uv.
- [x] Fix the leakage: encoding, scaling, and feature selection now run inside a per-fold `ColumnTransformer` + `Pipeline`.
- [x] Nested cross-validation: 5-fold outer estimate, 3-fold inner tuning.
- [x] Reproducible evaluation packaged in `src/` (`data.py`, `evaluation.py`).
- [x] Imbalance handling: class weights + PR AUC/recall, plus cost-based threshold selection (`notebooks/03`).
- [x] Calibration: reliability curves + Brier across balanced/unweighted/isotonic/sigmoid. Finding: the unweighted tree is already calibrated (Brier 0.012); balanced weights caused the miscalibration.
- [x] Interpretability: permutation importance (on test) vs impurity (`notebooks/04`). Four features (Income, Education, Family, CCAvg) carry all the signal. SHAP optional later (would reintroduce the `numba>=0.61` floor).

## Structure

```
supervised-classification/
  data/        Loan_Modelling.csv
  notebooks/   01-supervised-classification.ipynb   (baseline; upgrade notebooks added as 02-, 03-)
  src/         reusable data + evaluation code (the rigor layer)
  results/     figures and metrics written by the notebooks
```

## Environment

Managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run jupyter lab
```

## Provenance

This project originated as coursework and has been rebuilt as an independent, rigor-upgraded portfolio piece. All third-party course materials (slides, templates, provided low-code scaffolds, vendor branding) have been removed; the work here is my own.
