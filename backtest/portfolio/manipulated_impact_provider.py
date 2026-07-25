import logging
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional, Tuple

import pandas as pd

from backtest.portfolio.PriceImpact import (
    PriceImpactModel,
    fit_price_impact_model_from_klines,
    trades_to_klines,
)
from backtest.portfolio.interfaces import QuoteToUSDTProvider
from core.currency_pair import CurrencyPair
from core.pump_event import PumpEvent
from core.time_utils import Bounds

LoadTradesFn = Callable[[Bounds, CurrencyPair], pd.DataFrame]


class ManipulatedImpactModelProvider:
    """
    Fit impact models on the manipulation window using sell-side data only.

    Loads trade-level data from the manipulation window starting at the pump time,
    resamples into 5-second candles, and fits using only sell-dominated candles
    (negative net buying volume). This captures the liquidity regime during
    position exit, where buying and selling are not balanced.

    The upper bound of the fit window is capped by an optional ``end_exclusive``
    passed to :meth:`get_impact_model`: the fit window is
    ``[pump.time, min(pump.time + MANIPULATION_WINDOW, end_exclusive))``. This
    prevents leaking post-exit data into the model that estimates the exit impact.
    """

    MANIPULATION_WINDOW = timedelta(minutes=10)

    def __init__(
        self,
        load_trades: LoadTradesFn,
        indicative_price_provider: Optional[QuoteToUSDTProvider] = None,
    ):
        self._load_trades: LoadTradesFn = load_trades
        self._indicative_price_provider: Optional[QuoteToUSDTProvider] = indicative_price_provider
        self._cache: Dict[Tuple[str, datetime, datetime], PriceImpactModel] = {}

    def _get_quote_to_usdt(self, currency_pair: CurrencyPair, ts: datetime) -> float:
        if self._indicative_price_provider is None:
            return 1.0
        try:
            return self._indicative_price_provider.get_quote_to_usdt_indicative_price(
                quote_asset=currency_pair.term,
                ts=ts,
            )
        except Exception:
            logging.warning(
                "Falling back to 1.0 quote-to-USDT rate for %s at %s (indicative price lookup failed)",
                currency_pair.name,
                ts,
            )
            return 1.0

    def get_impact_model(
        self,
        pump: PumpEvent,
        currency_pair: CurrencyPair,
        end_exclusive: datetime | None = None,
    ) -> PriceImpactModel:
        """
        Fit an impact model on sell-side data from the manipulation window.

        When ``end_exclusive`` is provided, the fit window is truncated at
        ``min(pump.time + MANIPULATION_WINDOW, end_exclusive)`` so no data at or
        after the actual exit timestamp is used. The effective end is part of the
        cache key to keep exits at different times from returning stale models.
        """
        window_end: datetime = pump.time + self.MANIPULATION_WINDOW
        if end_exclusive is not None:
            window_end = min(window_end, end_exclusive)

        cache_key: Tuple[str, datetime, datetime] = (currency_pair.name, pump.time, window_end)
        if cache_key in self._cache:
            return self._cache[cache_key]

        bounds = Bounds(
            start_inclusive=pump.time,
            end_exclusive=window_end,
        )

        trades = self._load_trades(bounds, currency_pair)
        klines = trades_to_klines(trades, freq="5s")
        quote_to_usdt = self._get_quote_to_usdt(currency_pair=currency_pair, ts=pump.time)
        model = fit_price_impact_model_from_klines(
            klines=klines,
            quote_to_usdt=quote_to_usdt,
            sell_only=True,
            sample_frequency="5s",
        )

        self._cache[cache_key] = model
        return model
