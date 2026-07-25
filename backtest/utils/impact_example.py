from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl

from backtest.portfolio.PriceImpact import (
    PriceImpactFitResult,
    fit_price_impact_model_from_klines_with_diagnostics,
    trades_to_klines,
)
from backtest.utils.IndicativePriceProvider import IndicativePriceProvider
from backtest.utils.sample import Dataset
from core.columns import DATE, IS_BUYER_MAKER, PRICE, QUANTITY, SYMBOL, TRADE_TIME
from core.currency_pair import CurrencyPair
from core.exchange import Exchange
from core.pump_event import PumpEvent
from core.time_utils import Bounds


@dataclass
class ImpactExample:
    pump: PumpEvent
    currency_pair: CurrencyPair
    fit_result: PriceImpactFitResult


@dataclass
class ExitImpactExample:
    pump: PumpEvent
    currency_pair: CurrencyPair
    fit_result: PriceImpactFitResult


def _load_trades_from_hive(
    hive: pl.LazyFrame,
    bounds: Bounds,
    currency_pair: CurrencyPair,
) -> pd.DataFrame:
    return (
        hive.filter(
            (pl.col(SYMBOL) == currency_pair.name)
            & (pl.col(DATE).is_between(bounds.day0, bounds.day1))
            & (pl.col(TRADE_TIME) >= bounds.start_inclusive)
            & (pl.col(TRADE_TIME) < bounds.end_exclusive)
        )
        .collect()
        .sort(by=TRADE_TIME)
        .select([TRADE_TIME, PRICE, QUANTITY, IS_BUYER_MAKER])
        .to_pandas()
    )


def find_best_impact_example(
    dataset: Dataset,
    lookback_days: int = 14,
    min_candles: int = 500,
    min_side_samples: int = 25,
    min_positive_impacts: int = 50,
) -> ImpactExample:
    """
    Search pump events in *dataset* for the best asset to illustrate the sqrt impact model.

    Loads trades from hive, resamples into 1-minute candles, and fits using
    net volume as the order flow proxy.
    """
    hive = pl.scan_parquet(Exchange.BINANCE_SPOT.get_hive_location(), hive_partitioning=True)
    price_provider = IndicativePriceProvider()

    best: Optional[ImpactExample] = None
    best_score: Optional[tuple] = None

    for pump in dataset.get_pumps():
        cp = pump.currency_pair
        bounds = Bounds(start_inclusive=pump.time - timedelta(days=lookback_days), end_exclusive=pump.time)
        trades = _load_trades_from_hive(hive, bounds, cp)
        if len(trades) < min_candles:
            continue

        klines = trades_to_klines(trades, freq="5min")
        if len(klines) < min_candles:
            continue

        try:
            q2u = price_provider.get_quote_to_usdt_indicative_price(cp.term, pump.time)
        except FileNotFoundError:
            q2u = 1.0

        result = fit_price_impact_model_from_klines_with_diagnostics(
            klines=klines,
            quote_to_usdt=q2u,
            sample_frequency="5min",
        )
        diag = result.diagnostics.set_index("side")
        buy_n = int(diag.loc["buy", "num_samples"]) if "buy" in diag.index else 0
        sell_n = int(diag.loc["sell", "num_samples"]) if "sell" in diag.index else 0
        pos = int((result.samples["impact_bps"] > 0).sum()) if not result.samples.empty else 0

        if min(buy_n, sell_n) < min_side_samples or pos < min_positive_impacts:
            continue

        score = (min(buy_n, sell_n), pos)
        if best_score is None or score > best_score:
            best = ImpactExample(pump=pump, currency_pair=cp, fit_result=result)
            best_score = score

    if best is None:
        raise ValueError("No pump asset with usable kline-based impact samples found")
    return best


def find_best_exit_impact_example(
    dataset: Dataset,
    manipulation_window_minutes: int = 10,
    min_sell_samples: int = 20,
) -> ExitImpactExample:
    """
    Search pump events for the best asset to illustrate the exit impact model.

    Loads trades from the manipulation window (10 min after pump), resamples
    into 5-second candles, and fits using only sell-dominated candles.
    """
    hive = pl.scan_parquet(Exchange.BINANCE_SPOT.get_hive_location(), hive_partitioning=True)
    price_provider = IndicativePriceProvider()

    best: Optional[ExitImpactExample] = None
    best_score: int = 0

    for pump in dataset.get_pumps():
        cp = pump.currency_pair
        bounds = Bounds(
            start_inclusive=pump.time,
            end_exclusive=pump.time + timedelta(minutes=manipulation_window_minutes),
        )
        trades = _load_trades_from_hive(hive, bounds, cp)
        if len(trades) < 50:
            continue

        klines = trades_to_klines(trades, freq="5s")
        if len(klines) < 10:
            continue

        try:
            q2u = price_provider.get_quote_to_usdt_indicative_price(cp.term, pump.time)
        except FileNotFoundError:
            q2u = 1.0

        result = fit_price_impact_model_from_klines_with_diagnostics(
            klines=klines,
            quote_to_usdt=q2u,
            sell_only=True,
            sample_frequency="5s",
        )

        n_samples = len(result.samples)
        if n_samples < min_sell_samples:
            continue

        if n_samples > best_score:
            best = ExitImpactExample(pump=pump, currency_pair=cp, fit_result=result)
            best_score = n_samples

    if best is None:
        raise ValueError("No pump asset with usable exit impact samples found")
    return best
