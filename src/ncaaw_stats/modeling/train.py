"""Training routines for production-ready models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, SelectPercentile, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DEFAULT_TARGET = "success"

# Columns that are not available at prediction time or leak future info.
PRO_FEATURE_BLOCKLIST = {
    "success",
    "career_win_shares",
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
    target: str
    target_metadata: Dict[str, Any]


def _find_player_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("player_name", "name", "player"):
        if col in df.columns:
            return col
    return None


def _resolve_career_ws_source_column(
    df: pd.DataFrame,
    preferred_column: Optional[str] = None,
) -> str:
    preferred = [preferred_column] if preferred_column else []
    explicit_candidates = [
        "career_win_shares",
        "career_ws",
        "wnba_career_ws",
        "pro_career_ws",
        "ws_pro",
        "tot_ws",
        "adv_ws",
    ]

    candidates: List[str] = []
    for col in preferred + explicit_candidates:
        if col and col in df.columns and col not in candidates:
            candidates.append(col)

    # Heuristic fallback: any WS-like numeric column except WS/48 rate columns.
    ws_like_cols = [c for c in df.columns if "ws" in c.lower()]
    for col in ws_like_cols:
        col_lower = col.lower()
        if "ws_48" in col_lower or "/48" in col_lower:
            continue
        if col not in candidates:
            candidates.append(col)

    for col in candidates:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            return col

    raise ValueError(
        "Could not identify a usable career Win Shares column. "
        "Pass --career-ws-col with a valid numeric column."
    )


def _build_pro_exposure_mask(df: pd.DataFrame) -> Tuple[Optional[pd.Series], List[str]]:
    exposure_cols: List[str] = []
    for col in df.columns:
        col_lower = col.lower()
        is_pro_col = ("pro" in col_lower) or ("wnba" in col_lower)
        is_minutes_or_games = any(
            token in col_lower for token in ("minutes", "minute", "min", "mp", "games", "gp", "_g", "g_")
        )
        if is_pro_col and is_minutes_or_games:
            exposure_cols.append(col)

    if not exposure_cols:
        return None, []

    numeric_view = df[exposure_cols].apply(pd.to_numeric, errors="coerce")
    exposure_mask = numeric_view.fillna(0).gt(0).any(axis=1)
    return exposure_mask, exposure_cols


def add_success_target(
    df: pd.DataFrame,
    career_ws_column: Optional[str] = None,
    threshold: Optional[float] = None,
    target_column: str = DEFAULT_TARGET,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()

    # Career Win Shares comes from the first usable WS-like column (or explicit override).
    source_col = _resolve_career_ws_source_column(df, preferred_column=career_ws_column)
    career_win_shares = pd.to_numeric(df[source_col], errors="coerce")

    player_col = _find_player_column(df)
    aggregated = False
    if player_col and df[player_col].duplicated(keep=False).any():
        # If the input has one row per season, aggregate to career totals per player.
        career_win_shares = career_win_shares.groupby(df[player_col]).transform("sum")
        aggregated = True

    df["career_win_shares"] = career_win_shares

    exposure_mask, exposure_cols = _build_pro_exposure_mask(df)
    if exposure_mask is not None and exposure_mask.any() and career_win_shares[exposure_mask].notna().any():
        threshold_population = career_win_shares[exposure_mask].dropna()
        threshold_population_desc = "players with recorded WNBA minutes/games"
    else:
        threshold_population = career_win_shares.dropna()
        threshold_population_desc = "players with non-null career_win_shares"

    if threshold_population.empty:
        raise ValueError("No non-null career Win Shares values available to define success threshold.")

    # Default X uses the 75th percentile (top quartile) of the threshold population.
    chosen_threshold = float(threshold) if threshold is not None else float(threshold_population.quantile(0.50))

    success = pd.Series(np.nan, index=df.index, dtype="float")
    valid_ws = career_win_shares.notna()
    success.loc[valid_ws] = (career_win_shares.loc[valid_ws] >= chosen_threshold).astype(int)
    df[target_column] = success

    labeled_ws = career_win_shares[valid_ws]
    positive_count = int((labeled_ws >= chosen_threshold).sum())
    negative_count = int((labeled_ws < chosen_threshold).sum())
    labeled_count = positive_count + negative_count

    ws_stats = {
        "count": int(threshold_population.shape[0]),
        "missing_rate": float(career_win_shares.isna().mean()),
        "mean": float(threshold_population.mean()),
        "median": float(threshold_population.quantile(0.5)),
        "std": float(threshold_population.std(ddof=1)),
        "p25": float(threshold_population.quantile(0.25)),
        "p50": float(threshold_population.quantile(0.50)),
        "p75": float(threshold_population.quantile(0.75)),
    }

    ws_family_drop = [
        c
        for c in df.columns
        if ("ws" in c.lower()) or ("win_share" in c.lower()) or ("winshare" in c.lower())
    ]
    drop_columns = sorted(set(ws_family_drop).union({source_col, "career_win_shares"}))
    metadata = {
        "target_column": target_column,
        "career_win_shares_source_column": source_col,
        "career_win_shares_aggregated_from_seasons": aggregated,
        "career_win_shares_aggregation_key": player_col if aggregated else None,
        "threshold_selection_population": threshold_population_desc,
        "pro_exposure_columns": exposure_cols,
        "threshold": chosen_threshold,
        "career_win_shares_stats": ws_stats,
        "class_balance": {
            "labeled_count": labeled_count,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_rate": float(positive_count / labeled_count) if labeled_count else 0.0,
        },
        "drop_columns": drop_columns,
    }
    return df, metadata


def _binary_target(series: pd.Series) -> pd.Series:
    return (series > 0).astype(int)


def prepare_training_frame(
    df: pd.DataFrame,
    target: str = DEFAULT_TARGET,
    drop_columns: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    df = df.copy()
    target_values = pd.to_numeric(df[target], errors="coerce")
    valid_rows = target_values.notna()
    if not valid_rows.any():
        raise ValueError(f"Target column '{target}' does not contain any valid labels.")
    if valid_rows.sum() < len(df):
        df = df.loc[valid_rows].copy()
        target_values = target_values.loc[valid_rows]

    y = _binary_target(target_values)
    if y.nunique() < 2:
        raise ValueError(
            f"Target column '{target}' has only one class after preprocessing. "
            "Adjust threshold or provide more data."
        )

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
    max_selected_features: Optional[int] = None,
) -> Optional[Any]:
    strategy = strategy.lower()
    if strategy == "none":
        return None

    if strategy == "kbest":
        initial_k = k_best
        if max_selected_features is not None:
            initial_k = min(initial_k, max_selected_features)
        return SelectKBest(score_func=mutual_info_classif, k=max(1, initial_k))

    if strategy in {"auto", "percentile"}:
        if strategy == "auto":
            initial_k = 30
            if max_selected_features is not None:
                initial_k = min(initial_k, max_selected_features)
            return SelectKBest(score_func=mutual_info_classif, k=max(1, initial_k))
        # Cross-validated percentile tuning avoids target leakage and adapts complexity.
        return SelectPercentile(score_func=mutual_info_classif, percentile=50)

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
                n_jobs=1,
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
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
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


def _selected_feature_count(model: Pipeline) -> int:
    feature_names = list(model.named_steps["preprocess"].get_feature_names_out())
    if "feature_select" not in model.named_steps:
        return len(feature_names)

    selector_step = model.named_steps["feature_select"]
    if hasattr(selector_step, "get_support"):
        return int(selector_step.get_support().sum())
    return len(feature_names)


def _build_k_candidates(k_best: int, upper_bound: int) -> List[int]:
    if upper_bound <= 0:
        return [1]

    base_candidates = {
        5,
        10,
        max(1, k_best // 2),
        max(1, k_best),
        max(1, k_best + 10),
        max(1, k_best + 20),
        max(1, upper_bound // 3),
        max(1, (2 * upper_bound) // 3),
        upper_bound,
    }
    bounded = sorted({v for v in base_candidates if 1 <= v <= upper_bound})
    return bounded or [min(upper_bound, 1)]


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
    target_metadata: Optional[Dict[str, Any]] = None,
    max_selected_features: Optional[int] = None,
) -> TrainingResult:
    np.random.seed(RANDOM_STATE)
    if max_selected_features is not None and max_selected_features <= 0:
        max_selected_features = None

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
    selector = build_feature_selector(
        feature_selection,
        k_best=k_best,
        max_selected_features=max_selected_features,
    )
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    preprocessed_feature_count: Optional[int] = None
    if selector is not None:
        preprocess_probe = clone(preprocessor)
        preprocess_probe.fit(X_train)
        preprocessed_feature_count = len(preprocess_probe.get_feature_names_out())

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

        current_param_grid = dict(param_grid)
        if selector is not None:
            strategy = feature_selection.lower()
            effective_upper = preprocessed_feature_count or len(X.columns)
            if max_selected_features is not None:
                effective_upper = min(effective_upper, max_selected_features)
            effective_upper = max(1, effective_upper)

            if strategy in {"auto", "kbest"}:
                current_param_grid["feature_select__k"] = _build_k_candidates(
                    k_best=k_best,
                    upper_bound=effective_upper,
                )
            elif strategy == "percentile":
                percentiles = [20, 30, 40, 50, 60, 70, 80, 90, 100]
                if max_selected_features is not None and preprocessed_feature_count:
                    max_pct = int(
                        np.floor(
                            100.0 * float(effective_upper) / float(preprocessed_feature_count)
                        )
                    )
                    max_pct = max(1, min(100, max_pct))
                    percentiles = [p for p in percentiles if p <= max_pct]
                    if not percentiles:
                        percentiles = [max_pct]
                current_param_grid["feature_select__percentile"] = percentiles

        max_iter = min(search_iterations, _max_combinations(current_param_grid))
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=current_param_grid,
            n_iter=max_iter,
            scoring="roc_auc",
            n_jobs=1,
            cv=cv,
            verbose=0,
            random_state=RANDOM_STATE,
        )
        search.fit(X_train, y_train)
        candidate_model = search.best_estimator_
        if (
            max_selected_features is not None
            and _selected_feature_count(candidate_model) > max_selected_features
        ):
            continue

        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_model = candidate_model
            best_params = search.best_params_
            best_name = name

    if best_model is None:
        raise RuntimeError(
            "No model was trained under the current feature-selection constraints. "
            "Try increasing --max-selected-features or adjusting --feature-selection."
        )

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else float("nan"),
        "pr_auc": average_precision_score(y_test, y_proba) if y_test.nunique() > 1 else float("nan"),
        "confusion_matrix_tn": float(tn),
        "confusion_matrix_fp": float(fp),
        "confusion_matrix_fn": float(fn),
        "confusion_matrix_tp": float(tp),
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
        feature_selection_params={
            "k_best": k_best,
            "max_selected_features": max_selected_features,
        },
        target=target,
        target_metadata=target_metadata or {},
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
        "target": result.target,
        "target_metadata": result.target_metadata,
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
        "target": result.target,
        "target_metadata": result.target_metadata,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
