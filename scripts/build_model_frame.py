#!/usr/bin/env python
"""Build the modeling dataframe by merging NCAA and WNBA datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ncaaw_stats.config import PROCESSED_DIR
from ncaaw_stats.modeling.dataset import build_model_frame, save_model_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build modeling dataframe.")
    parser.add_argument("--ncaa", type=Path, default=PROCESSED_DIR / "ncaa_processed.csv")
    parser.add_argument("--wnba", type=Path, default=PROCESSED_DIR / "wnba_processed.csv")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "model_df.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ncaa_df = pd.read_csv(args.ncaa)
    wnba_df = pd.read_csv(args.wnba)

    model_df = build_model_frame(ncaa_df, wnba_df)
    save_model_frame(model_df, args.out)


if __name__ == "__main__":
    main()
