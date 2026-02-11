"""Scrapers for Sports Reference NCAA player pages."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup

from ncaaw_stats.utils.http import create_session
from ncaaw_stats.utils.pandas import standardize_columns

TABLES = {
    "advanced": "Advanced",
    "per_game": "Per Game",
    "totals": "Totals",
}


def _extract_table(soup: BeautifulSoup, heading: str) -> Optional[pd.DataFrame]:
    h2_tag = soup.find("h2", string=heading)
    if not h2_tag:
        return None
    table = h2_tag.find_next("table")
    if table is None:
        return None
    return pd.read_html(str(table))[0]


def fetch_player_stats(url: str, sleep_s: float = 2.0) -> pd.DataFrame:
    session = create_session()
    response = session.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "lxml")

    dataframes: dict[str, pd.DataFrame] = {}
    for prefix, heading in TABLES.items():
        table_df = _extract_table(soup, heading)
        if table_df is not None:
            dataframes[prefix] = table_df.add_prefix(f"{prefix}_")

    if "per_game" not in dataframes:
        raise ValueError(f"Missing per-game table for {url}")

    base_df = dataframes["per_game"]
    if "advanced" in dataframes:
        base_df = base_df.merge(
            dataframes["advanced"],
            how="left",
            left_on="per_game_Season",
            right_on="advanced_Season",
        )
    if "totals" in dataframes:
        base_df = base_df.merge(
            dataframes["totals"],
            how="left",
            left_on="per_game_Season",
            right_on="totals_Season",
        )

    time.sleep(sleep_s)
    return base_df


def scrape_ncaa_players(players_df: pd.DataFrame, output_dir: Path, url_column: str) -> None:
    """Scrape NCAA player tables given a dataframe that includes player URLs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in players_df.iterrows():
        url = row.get(url_column)
        player_name = row.get("player_name") or row.get("name")
        if not isinstance(url, str) or not url:
            continue

        try:
            stats_df = fetch_player_stats(url)
            stats_df = standardize_columns(stats_df)
            stats_df["player_name"] = player_name
            stats_df.to_csv(output_dir / f"{player_name}.csv", index=False)
        except Exception as exc:
            print(f"Failed to scrape {player_name}: {exc}")
