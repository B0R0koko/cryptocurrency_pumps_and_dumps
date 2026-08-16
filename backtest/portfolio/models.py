from dataclasses import dataclass
from datetime import datetime

from core.currency_pair import CurrencyPair
from core.pump_event import PumpEvent


@dataclass(frozen=True)
class OrderIntent:
    """
    Defines what the strategy intends to execute for a single asset trade.
    """

    currency_pair: CurrencyPair
    pump: PumpEvent
    entry_price: float
    exit_price: float
    entry_ts: datetime
    exit_ts: datetime
    intended_notional_quote: float
    is_manipulated_asset: bool = False


@dataclass(frozen=True)
class ExecutionResult:
    """
    Result of simulating execution for an OrderIntent.
    """

    entry_price: float
    exit_price: float
    entry_filled_notional_quote: float
    exit_filled_notional_quote: float
    entry_filled_notional_usdt: float
    exit_filled_notional_usdt: float
    entry_impact_bps: float
    exit_impact_bps: float
