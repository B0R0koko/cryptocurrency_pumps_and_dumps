import functools
from dataclasses import dataclass
from datetime import datetime
from typing import Dict

from core.currency_pair import CurrencyPair
from core.exchange import Exchange


@functools.total_ordering
@dataclass
class PumpEvent:
    currency_pair: CurrencyPair
    time: datetime
    exchange: Exchange

    def __str__(self) -> str:
        formatted_time: str = self.time.strftime("%Y-%m-%dT%H-%M-%S")
        return f"{self.currency_pair.name}:{self.exchange.name}:{formatted_time}"

    def as_dict(self) -> Dict[str, str]:
        return {
            "currency_pair": self.currency_pair.name,
            "time": self.time.isoformat(),
            "exchange": self.exchange.name,
        }

    def is_manipulated(self, cp: CurrencyPair) -> bool:
        return self.currency_pair == cp

    def as_pump_hash(self) -> str:
        return str(self)

    @classmethod
    def from_pump_hash(cls, pump_hash: str) -> "PumpEvent":
        cp_str, exchange_str, time_str = pump_hash.split(":")

        return cls(
            currency_pair=CurrencyPair.from_string(symbol=cp_str),
            exchange=Exchange[exchange_str.upper()],
            time=datetime.strptime(time_str, "%Y-%m-%dT%H-%M-%S"),
        )

    def __lt__(self, other) -> bool:
        if not isinstance(other, PumpEvent):
            return NotImplemented
        return (self.time, self.currency_pair.name) < (other.time, other.currency_pair.name)
