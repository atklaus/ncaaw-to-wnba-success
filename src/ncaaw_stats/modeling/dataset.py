"""Construct modeling-ready datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ncaaw_stats.data.io import write_csv
from ncaaw_stats.utils.pandas import standardize_columns


def build_model_frame(ncaa_df: pd.DataFrame, wnba_df: pd.DataFrame) -> pd.DataFrame:
    """Merge NCAA and WNBA datasets into a single modeling frame."""
    ncaa_df = standardize_columns(ncaa_df)
    wnba_df = standardize_columns(wnba_df)

    merged = ncaa_df.merge(
        wnba_df,
        left_on="name",
        right_on="player_name",
        how="left",
        suffixes=("_college", "_pro"),
    )

    merged = merged.rename(
        columns={
            "adv_ws_48": "ws_48_pro",
            "adv_per": "per_pro",
            "player_name_college": "player_name",
        }
    )

    if "player_name" not in merged.columns and "name" in merged.columns:
        merged["player_name"] = merged["name"]
    elif "name" in merged.columns:
        merged["player_name"] = merged["player_name"].fillna(merged["name"])

    if "pg_school" in merged.columns:
        merged["college_team"] = merged["pg_school"]

    drop_cols = {
        "player_name_pro",
        "name",
        "pg_school",
        "pg_season",
        "adv_class",
        "pg_class",
        "adv_school",
        "pg_conf",
        "tot_season",
        "tot_school",
        "tot_class",
    }
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    merged = merged.dropna(how="all", axis=1)
    merged = merged.reset_index(drop=True)

    return merged


def save_model_frame(df: pd.DataFrame, path: Path) -> None:
    write_csv(df, path)
