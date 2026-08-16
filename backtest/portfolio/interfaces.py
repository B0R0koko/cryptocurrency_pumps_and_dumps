from datetime import datetime
from typing import Protocol

from core.currency_pair import CurrencyPair
from core.pump_event import PumpEvent


class QuoteToUSDTProvider(Protocol):
    """Maps a quote asset and timestamp to an indicative USDT conversion rate."""

    def get_quote_to_usdt_indicative_price(self, quote_asset: str, ts: datetime) -> float: ...


class ExecutionImpactModel(Protocol):
    """Execution-impact abstraction used by the simulation engine."""

    def estimate_vwap_price(self, base_price: float, side: int, notional_quote: float) -> tuple[float, float]: ...


class ImpactModelProvider(Protocol):
    """Loads or constructs execution-impact models for a pump/currency context."""

    def get_impact_model(
        self,
        pump: PumpEvent,
        currency_pair: CurrencyPair,
        end_exclusive: datetime | None = None,
    ) -> ExecutionImpactModel: ...
