"""Scraper helpers for HerHoopsStats (requires login)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup
import lxml.html
import pandas as pd

LOGIN_URL = "https://herhoopstats.com/accounts/login/?return_url=/"
HOME_URL = "https://herhoopstats.com"


@dataclass
class HerHoopsClient:
    username: str
    password: str

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self._login()

    def _login(self) -> None:
        login = self.session.get(LOGIN_URL)
        login_html = lxml.html.fromstring(login.text)
        hidden_inputs = login_html.xpath(r"//form//input[@type='hidden']")
        form = {x.attrib["name"]: x.attrib.get("value", "") for x in hidden_inputs}
        form["email"] = self.username
        form["password"] = self.password
        self.session.post(LOGIN_URL, data=form, headers=dict(referer=LOGIN_URL))

    def get_html(self, url: str) -> BeautifulSoup:
        response = self.session.get(url, headers=dict(referer=url))
        response.raise_for_status()
        return BeautifulSoup(response.text, "html5lib")

    @staticmethod
    def get_url_dict(page_html: BeautifulSoup) -> dict:
        href_dict = {}
        for item in page_html.find_all("a"):
            text = item.text.strip()
            href = item.get("href")
            if text and href:
                href_dict[text] = href
        return href_dict

    @staticmethod
    def get_table_by_heading(page_html: BeautifulSoup, heading_text: str) -> pd.DataFrame:
        divs = page_html.find_all("div", class_="card mb-3")
        for div in divs:
            heading = div.find("h2")
            if heading and heading.text.strip() == heading_text:
                return pd.read_html(str(div))[0]
        raise ValueError(f"Table not found: {heading_text}")
