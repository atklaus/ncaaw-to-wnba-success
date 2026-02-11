"""Scrapers for WNBA player pages on Basketball Reference."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import pandas as pd
from bs4 import BeautifulSoup

from ncaaw_stats.utils.http import create_session
from ncaaw_stats.utils.pandas import standardize_columns

BASE_URL = "https://www.basketball-reference.com/wnba/players/{letter}/"
REF_HOME = "https://www.basketball-reference.com"


def scrape_wnba_letter(letter: str, output_dir: Path, sleep_s: float = 2.0) -> None:
    session = create_session()
    response = session.get(BASE_URL.format(letter=letter))
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html5lib")
    url_dict = {a.text.strip(): a.get("href") for a in soup.find_all("a") if a.get("href")}

    player_urls = [
        href
        for href in url_dict.values()
        if f"/wnba/players/{letter}/" in href and href != f"/wnba/players/{letter}/"
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    for player_url in player_urls:
        time.sleep(sleep_s)
        player_page = session.get(REF_HOME + player_url)
        player_page.raise_for_status()

        player_html = BeautifulSoup(player_page.text, "html5lib")
        player_name = player_html.find("h1").find("span").get_text(strip=True)

        soup = BeautifulSoup(player_page.content, "lxml")
        adv_table = soup.find("h2", string="Advanced").find_next("table")
        pg_table = soup.find("h2", string="Per Game").find_next("table")

        adv_df = pd.read_html(str(adv_table))[0].add_prefix("adv_")
        pg_df = pd.read_html(str(pg_table))[0].add_prefix("pg_")

        base_df = pg_df.merge(adv_df, how="left", left_on="pg_Year", right_on="adv_Year")
        base_df["player_name"] = player_name

        url_dict = {a.text.strip(): a.get("href") for a in player_html.find_all("a") if a.get("href")}
        for key, value in url_dict.items():
            if value and "college=" in value:
                base_df["college_team"] = key

        base_df = standardize_columns(base_df)
        base_df.to_csv(output_dir / f"{player_name}.csv", index=False)


def scrape_wnba_players(letters: Iterable[str], output_dir: Path, sleep_s: float = 2.0) -> None:
    for letter in letters:
        scrape_wnba_letter(letter, output_dir=output_dir, sleep_s=sleep_s)
