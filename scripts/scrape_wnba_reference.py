#!/usr/bin/env python
"""Scrape WNBA player reference data."""

from __future__ import annotations

import argparse
import string
from pathlib import Path

from ncaaw_stats.config import RAW_DIR
from ncaaw_stats.scrapers.wnba_reference import scrape_wnba_players


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Basketball Reference WNBA players.")
    parser.add_argument("--letters", type=str, default=string.ascii_lowercase)
    parser.add_argument("--out-dir", type=Path, default=RAW_DIR / "wnba_ref")
    parser.add_argument("--sleep", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scrape_wnba_players(args.letters, args.out_dir, sleep_s=args.sleep)


if __name__ == "__main__":
    main()
