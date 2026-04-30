"""
Emma Gogic
501179657
CPS 844 - Data Mining
Assignment 1

Dataset:
- ObesityDataSet_raw_and_data_sinthetic.csv
- Target column: NObeyesdad (7 classes)

What this script does
1) Loads the CSV and prints basic dataset info
2) Preprocesses:
   - Numeric: median imputation + standardization
   - Categorical: most-frequent imputation + one-hot encoding
3) Trains & compares >= 5 different classifiers on the SAME split:
   - Multinomial Logistic Regression
   - Linear SVM
   - KNN
   - Decision Tree
   - Random Forest
   - Gradient Boosting (extra)
4) Reports:
   - Accuracy, Macro-F1, Weighted-F1
   - Confusion matrix + classification report for best model
5) Feature importance:
   - Permutation importance (model-agnostic) for the best model

How to run
    python obesity_classification.py --data "ObesityDataSet_raw_and_data_sinthetic.csv" --outdir outputs --cv

Outputs (in --outdir)
- metrics_table.csv
- classification_report_<best_model>.txt
- confusion_matrix_<best_model>.csv
- permutation_importance_<best_model>.csv
- (optional) cv_summary_<best_model>.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


TARGET_COL = "NObeyesdad"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object


def build_preprocess(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    """Impute + scale numerics; impute + one-hot categoricals."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_models(seed: int) -> List[ModelSpec]:
    """At least 5 distinct classification methods."""
    return [
        ModelSpec(
            "LogisticRegression",
            LogisticRegression(
                max_iter=4000,
                solver="saga",
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        ModelSpec(
            "LinearSVM",
            LinearSVC(C=1.0, max_iter=10000, random_state=seed),
        ),
        ModelSpec(
            "KNN",
            KNeighborsClassifier(n_neighbors=11, weights="distance"),
        ),
        ModelSpec(
            "DecisionTree",
            DecisionTreeClassifier(max_depth=10, random_state=seed),
        ),
        ModelSpec(
            "RandomForest",
            RandomForestClassifier(n_estimators=250, random_state=seed, n_jobs=-1),
        ),
        ModelSpec(
            "GradientBoosting",
            GradientBoostingClassifier(random_state=seed),
        ),
    ]


def evaluate_holdout(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocess: ColumnTransformer,
    models: List[ModelSpec],
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """Train all models on the same split and compare metrics."""
    rows = []
    fitted: Dict[str, Pipeline] = {}

    for spec in models:
        pipe = Pipeline([("preprocess", preprocess), ("model", spec.estimator)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rows.append(
            {
                "Model": spec.name,
                "Accuracy": float(accuracy_score(y_test, y_pred)),
                "F1_macro": float(f1_score(y_test, y_pred, average="macro")),
                "F1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
            }
        )
        fitted[spec.name] = pipe

    metrics_df = pd.DataFrame(rows).sort_values("F1_weighted", ascending=False).reset_index(drop=True)
    return metrics_df, fitted


def write_best_model_artifacts(
    *,
    outdir: Path,
    best_name: str,
    best_pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Write classification report and confusion matrix for the best model."""
    y_pred = best_pipe.predict(X_test)
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / f"classification_report_{best_name}.txt").write_text(
        classification_report(y_test, y_pred, digits=4),
        encoding="utf-8",
    )

    labels = sorted(y_test.unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(outdir / f"confusion_matrix_{best_name}.csv", index=True)


def compute_perm_importance(
    *,
    best_pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """
    Permutation importance (model-agnostic).
    We compute importance on the transformed (one-hot) feature space so feature names match.
    """
    preprocess: ColumnTransformer = best_pipe.named_steps["preprocess"]
    feature_names = preprocess.get_feature_names_out()

    X_test_t = preprocess.transform(X_test)
    model = best_pipe.named_steps["model"]

    # scoring chosen to match "best model" selection metric
    result = permutation_importance(
        model,
        X_test_t,
        y_test,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="f1_weighted",
    )

    imp_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return imp_df


def run_cv_summary(*, pipe: Pipeline, X: pd.DataFrame, y: pd.Series, seed: int, folds: int = 10) -> pd.DataFrame:
    """Optional: stratified k-fold CV for more stable estimates."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro", "f1_weighted": "f1_weighted"}
    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    return pd.DataFrame(
        {
            "metric": ["accuracy", "f1_macro", "f1_weighted"],
            "mean": [scores["test_accuracy"].mean(), scores["test_f1_macro"].mean(), scores["test_f1_weighted"].mean()],
            "std": [scores["test_accuracy"].std(), scores["test_f1_macro"].std(), scores["test_f1_weighted"].std()],
            "folds": [folds, folds, folds],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ObesityDataSet_raw_and_data_sinthetic.csv")
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv", action="store_true", help="Also run 10-fold CV on the best holdout model (slower).")
    args = parser.parse_args()

    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {df.columns.tolist()}")

    print("Dataset shape:", df.shape)
    print("Target distribution:\n", df[TARGET_COL].value_counts(), "\n")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Numeric vs categorical detection:
    # Try to coerce object columns to numeric. If most values convert, treat as numeric.
    num_cols = []
    cat_cols = []
    
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            num_cols.append(c)
            continue
    
        # Try converting to numeric
        coerced = pd.to_numeric(X[c], errors="coerce")
        convert_ratio = coerced.notna().mean()  # fraction that successfully converted
    
        # If 95%+ values look numeric, treat as numeric (and store coerced version)
        if convert_ratio >= 0.95:
            X[c] = coerced
            num_cols.append(c)
        else:
            cat_cols.append(c)
    
    print("Categorical columns:", cat_cols)
    print("Numeric columns:", num_cols)


    preprocess = build_preprocess(cat_cols, num_cols)
    models = get_models(args.seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    metrics_df, fitted = evaluate_holdout(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, preprocess=preprocess, models=models
    )
    metrics_df.to_csv(outdir / "metrics_table.csv", index=False)

    best_name = metrics_df.loc[0, "Model"]
    best_pipe = fitted[best_name]

    print("=== Holdout comparison (sorted by weighted F1) ===")
    print(metrics_df.to_string(index=False))
    print("\nBest model:", best_name)

    # Best model artifacts
    write_best_model_artifacts(outdir=outdir, best_name=best_name, best_pipe=best_pipe, X_test=X_test, y_test=y_test)

    # Feature importance
    imp_df = compute_perm_importance(best_pipe=best_pipe, X_test=X_test, y_test=y_test, seed=args.seed)
    imp_df.to_csv(outdir / f"permutation_importance_{best_name}.csv", index=False)
    print("\nTop 15 important (permutation) features:")
    print(imp_df.head(15).to_string(index=False))

    # Optional CV on best model
    if args.cv:
        cv_df = run_cv_summary(pipe=best_pipe, X=X, y=y, seed=args.seed, folds=10)
        cv_df.to_csv(outdir / f"cv_summary_{best_name}.csv", index=False)
        print("\n=== 10-fold CV summary for best model ===")
        print(cv_df.to_string(index=False))

    print(f"\nAll outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()


