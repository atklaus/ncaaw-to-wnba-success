"""Configuration and common paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("NCAAW_STATS_DATA_DIR", ROOT_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
SAMPLES_DIR = DATA_DIR / "samples"

ARTIFACTS_DIR = Path(os.getenv("NCAAW_STATS_ARTIFACTS_DIR", ROOT_DIR / "artifacts"))
REPORTS_DIR = Path(os.getenv("NCAAW_STATS_REPORTS_DIR", ROOT_DIR / "reports"))
FIGURES_DIR = REPORTS_DIR / "figures"
SLIDES_DIR = REPORTS_DIR / "slides"


def ensure_dirs() -> None:
    """Create expected directory structure if missing."""
    for path in [
        DATA_DIR,
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        EXTERNAL_DIR,
        SAMPLES_DIR,
        ARTIFACTS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        SLIDES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_env(dotenv_path: Optional[Path] = None) -> bool:
    """Load environment variables from a .env file if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return False

    path = dotenv_path or (ROOT_DIR / ".env")
    if path.exists():
        load_dotenv(path)
        return True
    return False


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read a secret from environment variables."""
    return os.getenv(name, default)
