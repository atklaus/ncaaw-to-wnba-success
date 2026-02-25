#!/usr/bin/env python
"""Train a tuned classifier and save artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ncaaw_stats.config import ARTIFACTS_DIR, PROCESSED_DIR
from ncaaw_stats.modeling.train import save_training_artifacts, train_best_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the WNBA success classifier.")
    parser.add_argument("--model-df", type=Path, default=PROCESSED_DIR / "model_df.csv")
    parser.add_argument("--target", type=str, default="ws_48_pro")
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument(
        "--drop-cols",
        type=str,
        default="",
        help="Comma-separated list of columns to exclude from training.",
    )
    parser.add_argument("--search-iters", type=int, default=25)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument(
        "--time-col",
        type=str,
        default="",
        help="Column name to use for time-based holdout (e.g., last_season).",
    )
    parser.add_argument(
        "--time-cutoff",
        type=int,
        default=None,
        help="Train on <= cutoff year, test on > cutoff year.",
    )
    parser.add_argument(
        "--feature-selection",
        type=str,
        default="none",
        help="Feature selection strategy: none, kbest, l1, rf",
    )
    parser.add_argument(
        "--k-best",
        type=int,
        default=10,
        help="Number of top features to keep when using kbest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_df
    if not model_path.exists():
        legacy_path = PROCESSED_DIR / "legacy" / "model_df.xlsx"
        if legacy_path.exists():
            model_path = legacy_path
        else:
            raise FileNotFoundError(
                f"Model dataframe not found at {args.model_df}. "
                f"Also checked {legacy_path}."
            )

    if model_path.suffix.lower() in {".xlsx", ".xls"}:
        model_df = pd.read_excel(model_path)
    else:
        model_df = pd.read_csv(model_path)

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]

    result = train_best_model(
        model_df,
        target=args.target,
        drop_columns=drop_cols,
        search_iterations=args.search_iters,
        cv_splits=args.cv_splits,
        time_column=args.time_col or None,
        time_cutoff=args.time_cutoff,
        feature_selection=args.feature_selection,
        k_best=args.k_best,
    )
    save_training_artifacts(result, args.out_dir)


if __name__ == "__main__":
    main()
