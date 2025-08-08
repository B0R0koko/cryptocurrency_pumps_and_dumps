from dataclasses import dataclass
from datetime import datetime
from typing import Dict

from core.currency_pair import CurrencyPair
from core.exchange import Exchange


@dataclass
class PumpEvent:
    currency_pair: CurrencyPair
    time: datetime
    exchange: Exchange

    def __str__(self) -> str:
        formatted_time: str = self.time.strftime("%Y-%m-%dT%H:%M:%S")
        return f"{self.currency_pair.name}-{self.exchange.name}-{formatted_time}"

    def as_dict(self) -> Dict[str, str]:
        return {
            "currency_pair": self.currency_pair.name,
            "time": self.time.isoformat(),
            "exchange": self.exchange.name,
        }
