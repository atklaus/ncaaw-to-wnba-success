"""Build processed datasets from raw reference CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ncaaw_stats.data.io import read_csv_folder, write_csv
from ncaaw_stats.data.transforms import add_first_year_column, fill_pg_conf, filter_year_or_career
from ncaaw_stats.utils.pandas import standardize_columns


def build_ncaa_dataset(ncaa_ref_dir: Path, output_path: Path) -> pd.DataFrame:
    """Build NCAA player-level dataset from per-player reference CSVs."""
    df = read_csv_folder(ncaa_ref_dir, add_name_from_filename=True)
    df = standardize_columns(df)

    conf_df = fill_pg_conf(df)
    conf_df = conf_df[conf_df["pg_season"] == "Career"].copy()
    conf_df["conference"] = conf_df["pg_conf"]
    df = df.merge(conf_df[["name", "conference"]], on="name", how="left")

    df["pg_year_num"] = pd.to_numeric(df["pg_season"].str[:4], errors="coerce")
    df_filtered = df.dropna(subset=["pg_year_num"])
    last_year = df_filtered.groupby(["name", "pg_school"])["pg_year_num"].max()

    df["_key"] = list(zip(df["name"], df["pg_school"]))
    df["last_season"] = df["_key"].map(last_year)
    df.drop(columns=["_key"], inplace=True)

    df = df.loc[df["pg_season"].apply(lambda x: filter_year_or_career(str(x), 1997))]
    df = df[df["pg_season"] == "Career"]
    df.dropna(subset=["tot_pts"], inplace=True)

    write_csv(df, output_path)
    return df


def build_wnba_dataset(wnba_ref_dir: Path, output_path: Path) -> pd.DataFrame:
    """Build WNBA player-level dataset from per-player reference CSVs."""
    df = read_csv_folder(wnba_ref_dir, add_name_from_filename=True)
    df = standardize_columns(df)

    df["pg_year_num"] = pd.to_numeric(df["pg_year"], errors="coerce")
    df = add_first_year_column(df, column_name="debut_year")
    df = df.loc[df["pg_year"].apply(lambda x: filter_year_or_career(str(x), 2002))]
    df = df.dropna(subset=["college_team"])

    df = df[df["pg_year"] == "Career"]
    keep_cols = ["player_name", "college_team", "adv_ws_48", "adv_per", "debut_year"]
    df = df[keep_cols]

    write_csv(df, output_path)
    return df


def build_case_study_dataset(case_study_dir: Path, output_path: Path) -> pd.DataFrame:
    """Build a case study NCAA dataset from per-player reference CSVs."""
    df = read_csv_folder(case_study_dir, add_name_from_filename=True)
    df = standardize_columns(df)

    conf_df = fill_pg_conf(df)
    conf_df = conf_df[conf_df["pg_season"] == "Career"].copy()
    conf_df["conference"] = conf_df["pg_conf"]
    df = df.merge(conf_df[["name", "conference"]], on="name", how="left")
    df = df[df["pg_school"] != "Overall"]

    df["pg_year_num"] = pd.to_numeric(df["pg_season"].str[:4], errors="coerce")
    df_filtered = df.dropna(subset=["pg_year_num"])
    last_year = df_filtered.groupby(["name", "pg_school"])["pg_year_num"].max()

    df["_key"] = list(zip(df["name"], df["pg_school"]))
    df["last_season"] = df["_key"].map(last_year)

    most_recent_class = df_filtered.groupby(["name", "pg_school"])["pg_class"].last()
    df["most_recent_class"] = df["_key"].map(most_recent_class)

    df.drop(columns=["_key"], inplace=True)

    df = df[df["pg_season"] == "Career"]
    df.dropna(subset=["tot_pts"], inplace=True)
    df.drop(columns=["pg_year_num"], inplace=True)

    write_csv(df, output_path)
    return df
