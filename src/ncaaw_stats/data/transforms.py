"""Dataset-specific transformations."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ncaaw_stats.utils.pandas import standardize_columns


def filter_year_or_career(value: str, year: int) -> bool:
    """Keep rows with a season year greater than the threshold or Career rows."""
    try:
        return int(value) > year
    except ValueError:
        return value == "Career"


def fill_pg_conf(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill pg_conf by season within each player."""
    df = standardize_columns(df)
    grouped_df = df.groupby(["name", "pg_season"])["pg_conf"].last().reset_index()
    pivot_table = grouped_df.pivot(index="name", columns="pg_season", values="pg_conf")
    pivot_table = pivot_table.ffill(axis=1)
    return pivot_table.unstack().reset_index(name="pg_conf")


def add_first_year_column(df: pd.DataFrame, column_name: str = "first_year") -> pd.DataFrame:
    """Add each player's first year column based on pg_year_num."""
    df = df.copy()
    df_filtered = df.dropna(subset=["pg_year_num"])
    first_year = df_filtered.groupby("player_name")["pg_year_num"].min()
    df[column_name] = df["player_name"].map(first_year)
    return df


def add_last_year_column(df: pd.DataFrame, column_name: str = "last_year") -> pd.DataFrame:
    """Add each player's last year column based on pg_year_num."""
    df = df.copy()
    df_filtered = df.dropna(subset=["pg_year_num"])
    last_year = df_filtered.groupby("player_name")["pg_year_num"].max()
    df[column_name] = df["player_name"].map(last_year)
    return df
