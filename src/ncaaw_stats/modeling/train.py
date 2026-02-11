"""Training routines for baseline models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass
class TrainingResult:
    model: RandomForestClassifier
    metrics: Dict[str, float]
    features: list[str]


def prepare_features(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df[target] = df[target].fillna(0)
    df[target] = (df[target] > 0).astype(int)

    X = df.drop(columns=[target])
    y = df[target]

    categorical_cols = [c for c in ["conference", "college_team", "position"] if c in X.columns]
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    numeric_cols = X.columns
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=numeric_cols)

    return X_imputed, y


def train_random_forest(df: pd.DataFrame, target: str = "ws_48_pro") -> TrainingResult:
    X, y = prepare_features(df, target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
    }

    return TrainingResult(model=model, metrics=metrics, features=list(X.columns))


def save_training_artifacts(result: TrainingResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, output_dir / "wnba_success_model.joblib")

    metrics_df = pd.DataFrame([result.metrics])
    metrics_df.to_csv(output_dir / "training_metrics.csv", index=False)

    features_df = pd.DataFrame({"feature": result.features})
    features_df.to_csv(output_dir / "model_features.csv", index=False)
