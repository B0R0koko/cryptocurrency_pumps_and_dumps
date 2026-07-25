import logging
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional, Tuple

import pandas as pd

from backtest.portfolio.PriceImpact import (
    PriceImpactModel,
    fit_price_impact_model_from_klines,
    trades_to_klines,
)
from backtest.portfolio.interfaces import ImpactModelProvider, QuoteToUSDTProvider
from core.currency_pair import CurrencyPair
from core.pump_event import PumpEvent
from core.time_utils import Bounds

LoadTradesFn = Callable[[Bounds, CurrencyPair], pd.DataFrame]


class LookbackImpactModelProvider(ImpactModelProvider):
    """
    Cache and provide impact models fit on a fixed lookback per asset/pump.

    Loads trade-level data, resamples into 5-minute candles, and fits using
    absolute net volume as the order flow proxy. Notionals normalised to USDT
    via a QuoteToUSDTProvider.
    """

    def __init__(
        self,
        load_trades: LoadTradesFn,
        lookback_days: int,
        indicative_price_provider: Optional[QuoteToUSDTProvider] = None,
    ):
        self._load_trades: LoadTradesFn = load_trades
        self.lookback_days: int = lookback_days
        self._indicative_price_provider: Optional[QuoteToUSDTProvider] = indicative_price_provider
        self._cache: Dict[Tuple[str, datetime], PriceImpactModel] = {}

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

    def get_impact_model(self, pump: PumpEvent, currency_pair: CurrencyPair) -> PriceImpactModel:
        """
        Return cached or newly-fitted impact model for `(currency_pair, pump.time)`.
        """
        cache_key: Tuple[str, datetime] = (currency_pair.name, pump.time)
        if cache_key in self._cache:
            return self._cache[cache_key]

        bounds = Bounds(
            start_inclusive=pump.time - timedelta(days=self.lookback_days),
            end_exclusive=pump.time,
        )

        trades = self._load_trades(bounds, currency_pair)
        klines = trades_to_klines(trades, freq="5min")
        quote_to_usdt = self._get_quote_to_usdt(currency_pair=currency_pair, ts=pump.time)
        model = fit_price_impact_model_from_klines(
            klines=klines,
            quote_to_usdt=quote_to_usdt,
            sample_frequency="5min",
        )

        self._cache[cache_key] = model
        return model
