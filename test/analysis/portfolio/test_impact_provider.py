"""Past-only regression tests for the ordinary execution-impact provider."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from backtest.portfolio.impact_provider import LookbackImpactModelProvider
from core.columns import IS_BUYER_MAKER, PRICE, QUANTITY, TRADE_TIME
from core.currency_pair import CurrencyPair
from core.exchange import Exchange
from core.pump_event import PumpEvent
from core.time_utils import Bounds


def test_entry_impact_fit_ends_at_entry_and_excludes_preannouncement_future() -> None:
    cp = CurrencyPair.from_string("AAA-BTC")
    pump_time = datetime(2021, 6, 1, 12, 0)
    entry_time = pump_time - timedelta(minutes=15)
    pump = PumpEvent(currency_pair=cp, time=pump_time, exchange=Exchange.BINANCE_SPOT)
    observed_bounds: list[Bounds] = []

    def load_trades(bounds: Bounds, currency_pair: CurrencyPair) -> pd.DataFrame:
        observed_bounds.append(bounds)
        return pd.DataFrame(
            {
                TRADE_TIME: [entry_time - timedelta(minutes=5)],
                PRICE: [100.0],
                QUANTITY: [1.0],
                IS_BUYER_MAKER: [False],
            }
        )

    provider = LookbackImpactModelProvider(load_trades=load_trades, lookback_days=14)
    provider.get_impact_model(pump=pump, currency_pair=cp, end_exclusive=entry_time)

    assert observed_bounds[-1] == Bounds(
        start_inclusive=entry_time - timedelta(days=14),
        end_exclusive=entry_time,
    )


def test_pre_pump_impact_provider_rejects_post_announcement_cutoff() -> None:
    cp = CurrencyPair.from_string("AAA-BTC")
    pump_time = datetime(2021, 6, 1, 12, 0)
    pump = PumpEvent(currency_pair=cp, time=pump_time, exchange=Exchange.BINANCE_SPOT)
    provider = LookbackImpactModelProvider(
        load_trades=lambda bounds, currency_pair: pd.DataFrame(),
        lookback_days=14,
    )

    with pytest.raises(ValueError, match="cannot be fitted past"):
        provider.get_impact_model(
            pump=pump,
            currency_pair=cp,
            end_exclusive=pump_time + timedelta(seconds=1),
        )
