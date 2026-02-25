#!/usr/bin/env python
"""Train a tuned classifier and save artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ncaaw_stats.config import ARTIFACTS_DIR, PROCESSED_DIR
from ncaaw_stats.modeling.train import add_success_target, save_training_artifacts, train_best_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the WNBA success classifier.")
    parser.add_argument("--model-df", type=Path, default=PROCESSED_DIR / "model_df.csv")
    parser.add_argument("--target", type=str, default="success")
    parser.add_argument(
        "--career-ws-col",
        type=str,
        default="",
        help="Optional explicit source column for career Win Shares.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=None,
        help="Optional fixed threshold X for success = career_win_shares >= X.",
    )
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
        default="auto",
        help="Feature selection strategy: auto, none, percentile, kbest, l1, rf",
    )
    parser.add_argument(
        "--k-best",
        type=int,
        default=10,
        help="Number of top features to keep when using kbest.",
    )
    parser.add_argument(
        "--max-selected-features",
        type=int,
        default=40,
        help="Upper bound on selected features (<=0 disables cap).",
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
    target_metadata = {}
    if args.target == "success":
        model_df, target_metadata = add_success_target(
            model_df,
            career_ws_column=args.career_ws_col or None,
            threshold=args.success_threshold,
            target_column=args.target,
        )
        for col in target_metadata.get("drop_columns", []):
            if col not in drop_cols:
                drop_cols.append(col)

        print("Career Win Shares summary:")
        print(json.dumps(target_metadata.get("career_win_shares_stats", {}), indent=2, sort_keys=True))
        print(f"Career Win Shares source column: {target_metadata.get('career_win_shares_source_column')}")
        print(
            "Threshold population:",
            target_metadata.get("threshold_selection_population"),
        )
        print(f"Chosen threshold X: {target_metadata.get('threshold')}")
        print(
            "Class balance:",
            json.dumps(target_metadata.get("class_balance", {}), indent=2, sort_keys=True),
        )

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
        target_metadata=target_metadata,
        max_selected_features=args.max_selected_features,
    )
    save_training_artifacts(result, args.out_dir)

    print(f"Selected feature count: {len(result.selected_features)}")
    print("Training metrics:")
    print(json.dumps(result.metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
