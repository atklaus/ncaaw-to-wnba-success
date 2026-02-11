#!/usr/bin/env python
"""Build processed datasets from raw reference CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

from ncaaw_stats.config import PROCESSED_DIR, RAW_DIR, ensure_dirs
from ncaaw_stats.pipelines.build_datasets import (
    build_case_study_dataset,
    build_ncaa_dataset,
    build_wnba_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed datasets.")
    parser.add_argument("--ncaa-ref", type=Path, default=RAW_DIR / "ncaa_ref")
    parser.add_argument("--wnba-ref", type=Path, default=RAW_DIR / "wnba_ref")
    parser.add_argument("--case-study", type=Path, default=RAW_DIR / "case_study")
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    return parser.parse_args()


def main() -> None:
    ensure_dirs()
    args = parse_args()

    if args.ncaa_ref.exists():
        build_ncaa_dataset(args.ncaa_ref, args.out_dir / "ncaa_processed.csv")

    if args.wnba_ref.exists():
        build_wnba_dataset(args.wnba_ref, args.out_dir / "wnba_processed.csv")

    if args.case_study.exists():
        build_case_study_dataset(args.case_study, args.out_dir / "ncaa_case_study.csv")


if __name__ == "__main__":
    main()
