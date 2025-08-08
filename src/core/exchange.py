from enum import Enum
from pathlib import Path
from typing import Dict, List

from core.paths import BINANCE_SPOT_HIVE_TRADES, BINANCE_USDM_HIVE_TRADES, OKX_SPOT_HIVE_TRADES


class Exchange(Enum):
    BINANCE_SPOT = "binance_spot"

    def get_hive_location(self) -> Path:
        hives: Dict[Exchange, Path] = {
            Exchange.BINANCE_SPOT: BINANCE_SPOT_HIVE_TRADES,
        }
        return hives[self]

    @staticmethod
    def parse_from_lower(exchange_str: str) -> "Exchange":
        return Exchange[exchange_str.upper()]
