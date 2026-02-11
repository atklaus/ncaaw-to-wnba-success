"""Pandas helpers used across pipelines."""

from __future__ import annotations

import re
from typing import Iterable, Optional

import pandas as pd


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase snake_case and drop unnamed columns."""
    df = df.copy()
    clean_cols = []
    for col in df.columns:
        col_str = str(col).strip().lower()
        col_str = re.sub(r"%", "_pct", col_str)
        col_str = re.sub(r"[/]", "_", col_str)
        col_str = re.sub(r"[^0-9a-zA-Z]+", "_", col_str)
        col_str = col_str.strip("_")
        clean_cols.append(col_str)

    df.columns = clean_cols
    df = df.loc[:, ~df.columns.str.contains(r"^unnamed", case=False, regex=True)]
    return df


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Ensure a DataFrame contains each column, adding missing columns as NA."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def coerce_numeric(df: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Coerce columns to numeric where possible."""
    df = df.copy()
    cols = columns or df.columns
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df
