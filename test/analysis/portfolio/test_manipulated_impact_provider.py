"""Regression tests for the manipulated exit-impact provider.

Covers two audit findings:
1. ``end_exclusive`` must actually truncate the fit window; otherwise the exit
   impact model peeks at post-exit data.
2. The cache key must include the effective end so two exits at different times
   receive different models. The previous ``(symbol, pump.time)`` key returned
   the model fitted from the first call regardless of the requested end.
"""

from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd

from backtest.portfolio.manipulated_impact_provider import ManipulatedImpactModelProvider
from core.columns import IS_BUYER_MAKER, PRICE, QUANTITY, TRADE_TIME
from core.currency_pair import CurrencyPair
from core.exchange import Exchange
from core.pump_event import PumpEvent
from core.time_utils import Bounds


def _build_manipulation_window_trades(pump_time: datetime) -> pd.DataFrame:
    """Two synthetic sell-dominated candles: an early quiet one at t+5s and a
    LATE aggressive one at t+9m30s that would blow up the fitted beta if the
    provider forgets to honor ``end_exclusive``.
    """
    rows: List[dict] = []
    # Small sell candle near the start of the manipulation window.
    for i in range(5):
        rows.append(
            {
                TRADE_TIME: pump_time + timedelta(seconds=5 + i),
                PRICE: 100.0 - 0.01 * i,
                QUANTITY: 1.0,
                IS_BUYER_MAKER: True,
            }
        )
    # Massive sell candle near the tail of the manipulation window.
    late_start: datetime = pump_time + timedelta(minutes=9, seconds=30)
    for i in range(20):
        rows.append(
            {
                TRADE_TIME: late_start + timedelta(seconds=i),
                PRICE: 90.0 - 0.5 * i,
                QUANTITY: 100.0,
                IS_BUYER_MAKER: True,
            }
        )
    return pd.DataFrame(rows)


def test_end_exclusive_truncates_fit_window_and_excludes_post_exit_trades() -> None:
    cp = CurrencyPair.from_string("AAA-BTC")
    pump_time = datetime(2021, 6, 1, 12, 0, 0)
    pump = PumpEvent(currency_pair=cp, time=pump_time, exchange=Exchange.BINANCE_SPOT)
    all_trades: pd.DataFrame = _build_manipulation_window_trades(pump_time=pump_time)

    received_bounds: list[Bounds] = []

    def load_trades(bounds: Bounds, currency_pair: CurrencyPair) -> pd.DataFrame:
        received_bounds.append(bounds)
        return all_trades[
            (all_trades[TRADE_TIME] >= bounds.start_inclusive) & (all_trades[TRADE_TIME] < bounds.end_exclusive)
        ].reset_index(drop=True)

    provider = ManipulatedImpactModelProvider(load_trades=load_trades)

    early_exit: datetime = pump_time + timedelta(minutes=1)
    provider.get_impact_model(pump=pump, currency_pair=cp, end_exclusive=early_exit)

    assert received_bounds[-1].start_inclusive == pump_time
    assert received_bounds[-1].end_exclusive == early_exit
    # No trade at or after early_exit should have made it in.
    loaded: pd.DataFrame = load_trades(received_bounds[-1], cp)
    assert (loaded[TRADE_TIME] < early_exit).all()
    assert (loaded[TRADE_TIME] >= pump_time).all()


def test_end_exclusive_caps_at_manipulation_window_when_exit_is_later() -> None:
    cp = CurrencyPair.from_string("AAA-BTC")
    pump_time = datetime(2021, 6, 1, 12, 0, 0)
    pump = PumpEvent(currency_pair=cp, time=pump_time, exchange=Exchange.BINANCE_SPOT)

    received_bounds: list[Bounds] = []

    def load_trades(bounds: Bounds, currency_pair: CurrencyPair) -> pd.DataFrame:
        received_bounds.append(bounds)
        return pd.DataFrame(columns=[TRADE_TIME, PRICE, QUANTITY, IS_BUYER_MAKER])

    provider = ManipulatedImpactModelProvider(load_trades=load_trades)
    exit_after_window: datetime = pump_time + timedelta(minutes=60)
    provider.get_impact_model(pump=pump, currency_pair=cp, end_exclusive=exit_after_window)

    # The manipulation-window cap wins.
    assert received_bounds[-1].end_exclusive == pump_time + ManipulatedImpactModelProvider.MANIPULATION_WINDOW


def test_cache_key_includes_effective_end_and_returns_distinct_models() -> None:
    """Two calls with different ``end_exclusive`` values must not collide in the
    provider cache. Previously, the second call would return a stale model."""
    cp = CurrencyPair.from_string("AAA-BTC")
    pump_time = datetime(2021, 6, 1, 12, 0, 0)
    pump = PumpEvent(currency_pair=cp, time=pump_time, exchange=Exchange.BINANCE_SPOT)
    all_trades: pd.DataFrame = _build_manipulation_window_trades(pump_time=pump_time)

    load_calls: list[Bounds] = []

    def load_trades(bounds: Bounds, currency_pair: CurrencyPair) -> pd.DataFrame:
        load_calls.append(bounds)
        return all_trades[
            (all_trades[TRADE_TIME] >= bounds.start_inclusive) & (all_trades[TRADE_TIME] < bounds.end_exclusive)
        ].reset_index(drop=True)

    provider = ManipulatedImpactModelProvider(load_trades=load_trades)

    early_exit: datetime = pump_time + timedelta(minutes=1)
    late_exit: datetime = pump_time + timedelta(minutes=10)

    model_early = provider.get_impact_model(pump=pump, currency_pair=cp, end_exclusive=early_exit)
    model_late = provider.get_impact_model(pump=pump, currency_pair=cp, end_exclusive=late_exit)

    # Two distinct fit windows must have triggered two distinct loader calls.
    assert len(load_calls) == 2
    # The late-exit fit sees the aggressive tail candle and must produce a
    # strictly larger beta (or, at minimum, a different one) than the early-exit
    # fit which only saw the small early candle.
    assert model_late.beta > model_early.beta
    assert not np.isclose(model_late.beta, model_early.beta)


def test_cache_hits_when_end_exclusive_matches() -> None:
    cp = CurrencyPair.from_string("AAA-BTC")
    pump_time = datetime(2021, 6, 1, 12, 0, 0)
    pump = PumpEvent(currency_pair=cp, time=pump_time, exchange=Exchange.BINANCE_SPOT)
    all_trades: pd.DataFrame = _build_manipulation_window_trades(pump_time=pump_time)

    load_calls: list[Bounds] = []

    def load_trades(bounds: Bounds, currency_pair: CurrencyPair) -> pd.DataFrame:
        load_calls.append(bounds)
        return all_trades[
            (all_trades[TRADE_TIME] >= bounds.start_inclusive) & (all_trades[TRADE_TIME] < bounds.end_exclusive)
        ].reset_index(drop=True)

    provider = ManipulatedImpactModelProvider(load_trades=load_trades)
    exit_ts: datetime = pump_time + timedelta(minutes=3)

    first = provider.get_impact_model(pump=pump, currency_pair=cp, end_exclusive=exit_ts)
    second = provider.get_impact_model(pump=pump, currency_pair=cp, end_exclusive=exit_ts)

    assert first is second
    assert len(load_calls) == 1
