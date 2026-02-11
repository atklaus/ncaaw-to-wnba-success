"""File IO helpers for datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ncaaw_stats.utils.pandas import ensure_columns


def read_csv_folder(
    folder: Path,
    add_name_from_filename: bool = True,
    name_column: str = "name",
    pattern: str = "*.csv",
) -> pd.DataFrame:
    """Read all CSVs in a folder and return a concatenated dataframe."""
    paths = sorted(Path(folder).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    all_columns: set[str] = set()
    for path in paths:
        all_columns.update(pd.read_csv(path, nrows=0).columns)

    ordered_columns = sorted(all_columns)
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df = ensure_columns(df, ordered_columns)
        if add_name_from_filename:
            df[name_column] = path.stem
        if add_name_from_filename:
            df = df[ordered_columns + [name_column]]
        else:
            df = df[ordered_columns]
        frames.append(df)

    return pd.concat(frames, axis=0, ignore_index=True)


def write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Write dataframe to disk with parent dirs created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
