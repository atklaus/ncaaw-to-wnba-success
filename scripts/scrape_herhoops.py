#!/usr/bin/env python
"""Fetch a table from HerHoopsStats (requires account)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ncaaw_stats.config import EXTERNAL_DIR, get_secret, load_env
from ncaaw_stats.scrapers.herhoops import HerHoopsClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape HerHoopsStats data.")
    parser.add_argument("--username", type=str, default=None)
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument(
        "--url",
        type=str,
        default="https://herhoopstats.com/stats/ncaa/research/team_total_games/?division=1&min_season=2010&max_season=2023&result=both&loc_h=1&loc_a=1&loc_n=1&submit=true",
    )
    parser.add_argument("--out", type=Path, default=EXTERNAL_DIR / "herhoops_team_totals.csv")
    return parser.parse_args()


def main() -> None:
    load_env()
    args = parse_args()

    username = args.username or get_secret("HERHOOPS_USERNAME")
    password = args.password or get_secret("HERHOOPS_PASSWORD")
    if not username or not password:
        raise SystemExit("Missing HERHOOPS credentials. Set env vars or pass --username/--password.")

    client = HerHoopsClient(username=username, password=password)
    page_html = client.get_html(args.url)
    df = pd.read_html(str(page_html))[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
