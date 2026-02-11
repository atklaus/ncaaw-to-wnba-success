#!/usr/bin/env python
"""Scrape NCAA player reference tables given a URL mapping file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ncaaw_stats.config import RAW_DIR
from ncaaw_stats.scrapers.sports_reference import scrape_ncaa_players


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Sports Reference NCAA players.")
    parser.add_argument(
        "--url-map",
        type=Path,
        default=RAW_DIR / "ncaa_player_urls.csv",
        help="CSV with columns: player_name, sports_reference_url",
    )
    parser.add_argument("--out-dir", type=Path, default=RAW_DIR / "ncaa_ref")
    parser.add_argument("--url-column", type=str, default="sports_reference_url")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    players_df = pd.read_csv(args.url_map)
    scrape_ncaa_players(players_df, args.out_dir, url_column=args.url_column)


if __name__ == "__main__":
    main()
