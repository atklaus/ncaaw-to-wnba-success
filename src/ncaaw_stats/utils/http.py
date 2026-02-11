"""HTTP helpers for scraping."""

from __future__ import annotations

import random
from typing import Iterable, Optional

import requests

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
]


def random_headers(user_agents: Optional[Iterable[str]] = None) -> dict:
    """Return a randomized User-Agent header."""
    agents = list(user_agents or USER_AGENTS)
    return {"User-Agent": random.choice(agents)}


def create_session(user_agents: Optional[Iterable[str]] = None) -> requests.Session:
    """Create a requests session with a randomized User-Agent."""
    session = requests.Session()
    session.headers.update(random_headers(user_agents))
    return session
