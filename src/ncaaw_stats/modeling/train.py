"""Training routines for production-ready models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DEFAULT_TARGET = "ws_48_pro"

# Columns that are not available at prediction time or leak future info.
PRO_FEATURE_BLOCKLIST = {
    "ws_48_pro",
    "per_pro",
    "adv_ws_48",
    "adv_per",
    "debut_year",
}


@dataclass
class TrainingResult:
    model: Pipeline
    metrics: Dict[str, float]
    features: List[str]
    categorical_features: List[str]
    numeric_features: List[str]
    best_params: Dict[str, Any]
    model_name: str
    split_strategy: str
    time_column: Optional[str]
    time_cutoff: Optional[int]
    train_size: int
    test_size: int
    selected_features: List[str]
    feature_selection: str
    feature_selection_params: Dict[str, Any]


def _binary_target(series: pd.Series) -> pd.Series:
    series = series.fillna(0)
    return (series > 0).astype(int)


def prepare_training_frame(
    df: pd.DataFrame,
    target: str = DEFAULT_TARGET,
    drop_columns: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    df = df.copy()
    y = _binary_target(df[target])

    X = df.drop(columns=[target])

    drop_set = set(drop_columns or []).union(PRO_FEATURE_BLOCKLIST)
    drop_cols = [c for c in X.columns if c in drop_set]
    if drop_cols:
        X = X.drop(columns=drop_cols)

    categorical_cols = [c for c in ["conference", "college_team", "position"] if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    return X, y, categorical_cols, numeric_cols


def build_preprocessor(categorical_cols: List[str], numeric_cols: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _max_combinations(param_grid: Dict[str, Iterable[Any]]) -> int:
    total = 1
    for values in param_grid.values():
        total *= len(list(values))
    return total


def build_feature_selector(
    strategy: str,
    k_best: int = 30,
) -> Optional[Any]:
    strategy = strategy.lower()
    if strategy == "none":
        return None

    if strategy == "kbest":
        return SelectKBest(score_func=mutual_info_classif, k=k_best)

    if strategy == "l1":
        return SelectFromModel(
            LogisticRegression(
                penalty="l1",
                solver="liblinear",
                max_iter=5000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        )

    if strategy == "rf":
        return SelectFromModel(
            RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
            )
        )

    raise ValueError(f"Unknown feature selection strategy: {strategy}")


def _model_candidates() -> List[Tuple[str, Any, Dict[str, Any]]]:
    candidates: List[Tuple[str, Any, Dict[str, Any]]] = []

    candidates.append(
        (
            "logistic_regression",
            LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
            {
                "model__C": np.logspace(-3, 2, 6),
                "model__penalty": ["l2"],
                "model__class_weight": [None, "balanced"],
                "model__solver": ["lbfgs"],
            },
        )
    )

    candidates.append(
        (
            "random_forest",
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [None, 8, 16, 24],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__class_weight": [None, "balanced"],
            },
        )
    )

    candidates.append(
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__max_depth": [None, 6, 12],
                "model__learning_rate": [0.02, 0.05, 0.1],
                "model__max_iter": [200, 400, 600],
                "model__l2_regularization": [0.0, 0.1, 1.0],
            },
        )
    )

    try:
        from xgboost import XGBClassifier  # type: ignore

        candidates.append(
            (
                "xgboost",
                XGBClassifier(
                    objective="binary:logistic",
                    random_state=RANDOM_STATE,
                    eval_metric="auc",
                    tree_method="hist",
                ),
                {
                    "model__n_estimators": [300, 600, 900],
                    "model__max_depth": [3, 5, 7],
                    "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__subsample": [0.7, 0.9, 1.0],
                    "model__colsample_bytree": [0.7, 0.9, 1.0],
                },
            )
        )
    except Exception:
        pass

    return candidates


def train_best_model(
    df: pd.DataFrame,
    target: str = DEFAULT_TARGET,
    drop_columns: Optional[Iterable[str]] = None,
    search_iterations: int = 25,
    cv_splits: int = 5,
    time_column: Optional[str] = None,
    time_cutoff: Optional[int] = None,
    feature_selection: str = "none",
    k_best: int = 30,
) -> TrainingResult:
    X, y, categorical_cols, numeric_cols = prepare_training_frame(df, target, drop_columns)

    split_strategy = "random"
    if time_column and time_column in df.columns and time_cutoff is not None:
        time_values = pd.to_numeric(df[time_column], errors="coerce")
        train_mask = time_values <= time_cutoff
        test_mask = time_values > time_cutoff

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            raise ValueError(
                f"Time split failed. Check {time_column} and cutoff {time_cutoff}."
            )

        X_train = X.loc[train_mask].copy()
        y_train = y.loc[train_mask].copy()
        X_test = X.loc[test_mask].copy()
        y_test = y.loc[test_mask].copy()
        split_strategy = "time"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    selector = build_feature_selector(feature_selection, k_best=k_best)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    best_model = None
    best_params: Dict[str, Any] = {}
    best_score = -np.inf
    best_name = ""

    for name, estimator, param_grid in _model_candidates():
        pipeline_steps = [("preprocess", preprocessor)]
        if selector is not None:
            pipeline_steps.append(("feature_select", selector))
        pipeline_steps.append(("model", estimator))

        pipeline = Pipeline(steps=pipeline_steps)

        max_iter = min(search_iterations, _max_combinations(param_grid))
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=max_iter,
            scoring="roc_auc",
            n_jobs=-1,
            cv=cv,
            verbose=0,
            random_state=RANDOM_STATE,
        )
        search.fit(X_train, y_train)

        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_model = search.best_estimator_
            best_params = search.best_params_
            best_name = name

    if best_model is None:
        raise RuntimeError("No model was trained. Check candidate configuration.")

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }

    feature_names = list(
        best_model.named_steps["preprocess"].get_feature_names_out()
    )
    selected_features = feature_names
    if "feature_select" in best_model.named_steps:
        selector_step = best_model.named_steps["feature_select"]
        if hasattr(selector_step, "get_support"):
            mask = selector_step.get_support()
            selected_features = list(np.array(feature_names)[mask])

    result = TrainingResult(
        model=best_model,
        metrics=metrics,
        features=list(X.columns),
        categorical_features=categorical_cols,
        numeric_features=numeric_cols,
        best_params=best_params,
        model_name=best_name,
        split_strategy=split_strategy,
        time_column=time_column,
        time_cutoff=time_cutoff,
        train_size=int(len(X_train)),
        test_size=int(len(X_test)),
        selected_features=selected_features,
        feature_selection=feature_selection,
        feature_selection_params={"k_best": k_best},
    )
    return result


def save_training_artifacts(result: TrainingResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(result.model, output_dir / "wnba_success_model.joblib")

    metrics_df = pd.DataFrame([result.metrics])
    metrics_df.to_csv(output_dir / "training_metrics.csv", index=False)

    schema = {
        "features": result.features,
        "categorical_features": result.categorical_features,
        "numeric_features": result.numeric_features,
        "selected_features": result.selected_features,
        "feature_selection": result.feature_selection,
        "feature_selection_params": result.feature_selection_params,
        "model_name": result.model_name,
        "best_params": result.best_params,
    }
    (output_dir / "feature_schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8"
    )

    report = {
        "metrics": result.metrics,
        "model_name": result.model_name,
        "best_params": result.best_params,
        "split_strategy": result.split_strategy,
        "time_column": result.time_column,
        "time_cutoff": result.time_cutoff,
        "train_size": result.train_size,
        "test_size": result.test_size,
        "feature_selection": result.feature_selection,
        "feature_selection_params": result.feature_selection_params,
        "selected_feature_count": len(result.selected_features),
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
