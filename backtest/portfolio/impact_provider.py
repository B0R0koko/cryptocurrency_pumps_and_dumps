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

    def get_impact_model(
        self,
        pump: PumpEvent,
        currency_pair: CurrencyPair,
        end_exclusive: Optional[datetime] = None,
    ) -> PriceImpactModel:
        """
        Return a cached or newly fitted past-only impact model.

        ``end_exclusive`` is the decision/execution timestamp whose impact is
        being estimated. Defaulting to the pump time preserves the standalone
        provider API, while callers simulating an earlier entry must pass the
        actual entry timestamp so trades observed after entry cannot leak into
        its execution model.
        """
        effective_end: datetime = end_exclusive or pump.time
        if effective_end > pump.time:
            raise ValueError(
                "Pre-pump impact model cannot be fitted past the pump time; "
                f"got end_exclusive={effective_end}, pump_time={pump.time}"
            )

        cache_key: Tuple[str, datetime] = (currency_pair.name, effective_end)
        if cache_key in self._cache:
            return self._cache[cache_key]

        bounds = Bounds(
            start_inclusive=effective_end - timedelta(days=self.lookback_days),
            end_exclusive=effective_end,
        )

        trades = self._load_trades(bounds, currency_pair)
        klines = trades_to_klines(trades, freq="5min")
        quote_to_usdt = self._get_quote_to_usdt(currency_pair=currency_pair, ts=effective_end)
        model = fit_price_impact_model_from_klines(
            klines=klines,
            quote_to_usdt=quote_to_usdt,
            sample_frequency="5min",
        )

        self._cache[cache_key] = model
        return model
