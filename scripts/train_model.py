#!/usr/bin/env python
"""Train a baseline classifier and save artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ncaaw_stats.config import ARTIFACTS_DIR, PROCESSED_DIR
from ncaaw_stats.modeling.train import save_training_artifacts, train_random_forest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the WNBA success classifier.")
    parser.add_argument("--model-df", type=Path, default=PROCESSED_DIR / "model_df.csv")
    parser.add_argument("--target", type=str, default="ws_48_pro")
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_df = pd.read_csv(args.model_df)

    result = train_random_forest(model_df, target=args.target)
    save_training_artifacts(result, args.out_dir)


if __name__ == "__main__":
    main()
