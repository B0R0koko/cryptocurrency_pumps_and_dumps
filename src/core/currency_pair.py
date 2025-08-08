import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests

from core.time_utils import Bounds


@dataclass
class CurrencyPair:
    base: str
    term: str

    @classmethod
    def from_string(cls, symbol: str):
        """Parse CurrencyPair from string formatted like this: ADA-USDT"""
        base, term = symbol.split("-")
        return cls(base=base, term=term)  # type:ignore

    def __str__(self) -> str:
        return f"{self.base}-{self.term}"

    @property
    def name(self) -> str:
        return f"{self.base}-{self.term}"

    @property
    def binance_name(self) -> str:
        return f"{self.base}{self.term}"


def collect_all_spot_currency_pairs() -> List[CurrencyPair]:
    """Collect a set of all CurrencyPairs traded on Binance"""
    resp = requests.get("https://api.binance.com/api/v3/exchangeInfo")
    data: Dict[str, Any] = resp.json()
    return [
        CurrencyPair(base=entry["baseAsset"], term=entry["quoteAsset"]) for entry in data["symbols"]
    ]


def collect_all_usdm_currency_pairs() -> List[CurrencyPair]:
    """Collect a set of all CurrencyPairs traded on BINANCE_USDM"""
    resp = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo")
    data: Dict[str, Any] = resp.json()
    return [
        CurrencyPair(base=entry["baseAsset"], term=entry["quoteAsset"]) for entry in data["symbols"]
    ]


def get_cross_section_currencies(hive_dir: Path, bounds: Bounds) -> List[CurrencyPair]:
    matched_dirs: List[str] = []

    for directory in os.listdir(hive_dir):
        match: Optional[re.Match[str]] = re.search(string=directory, pattern=r"(\d{4}-\d{2}-\d{2})")
        date_matched: Optional[str] = match.group(1) if match else None
        if date_matched is None:
            continue

        dir_date: date = datetime.strptime(date_matched, "%Y-%m-%d").date()

        if bounds.contain_days(day=dir_date):
            matched_dirs.append(directory)

    all_currency_pairs: set[CurrencyPair] = set()

    for directory in matched_dirs:
        symbol_directories: List[str] = os.listdir(hive_dir.joinpath(directory))
        all_currency_pairs |= set(
            CurrencyPair.from_string(symbol=directory.split("=")[1]) for directory in symbol_directories
        )

    return list(all_currency_pairs)
