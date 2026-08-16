from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from core.columns import (
    BINANCE_KLINES_COLS,
    CLOSE_PRICE,
    OPEN_TIME,
    QUOTE_ASSET_VOLUME,
    VOLUME,
)

from core.paths import BINANCE_SPOT_RAW_KLINES


class IndicativePriceProvider:
    """
    Provides indicative prices from Binance 1m kline VWAP.
    """

    def __init__(self, raw_klines_dir: Path = BINANCE_SPOT_RAW_KLINES):
        self.raw_klines_dir: Path = raw_klines_dir
        self._cache: Dict[Tuple[str, date], Optional[pd.Series]] = {}

    def _kline_zip_path(self, symbol: str, day: date) -> Path:
        return self.raw_klines_dir / symbol / f"klines@1m@{str(day)}.zip"

    def _load_symbol_day_vwap_series(self, symbol: str, day: date) -> Optional[pd.Series]:
        """
        Load (or fetch from cache) a 1-minute indicative VWAP series for one day.

        VWAP is computed as `quote_asset_volume / volume` per kline, with close
        price fallback for zero-volume bars.
        """
        cache_key: Tuple[str, date] = (symbol, day)
        if cache_key in self._cache:
            return self._cache[cache_key]

        path: Path = self._kline_zip_path(symbol=symbol, day=day)
        if not path.exists():
            self._cache[cache_key] = None
            return None

        df: pd.DataFrame = pd.read_csv(
            path,
            compression="zip",
            header=None,
            names=BINANCE_KLINES_COLS,
            usecols=[OPEN_TIME, VOLUME, QUOTE_ASSET_VOLUME, CLOSE_PRICE],
        )

        if df.empty:
            self._cache[cache_key] = None
            return None

        df[OPEN_TIME] = pd.to_datetime(df[OPEN_TIME], unit="ms")

        volume = df[VOLUME].astype(float).to_numpy()
        quote_volume = df[QUOTE_ASSET_VOLUME].astype(float).to_numpy()
        close_price = df[CLOSE_PRICE].astype(float).to_numpy()

        with np.errstate(divide="ignore", invalid="ignore"):
            vwap = np.where(volume > 0.0, quote_volume / volume, close_price)

        series = pd.Series(vwap.astype(float), index=df[OPEN_TIME]).sort_index()
        series = series[~series.index.duplicated(keep="last")]
        self._cache[cache_key] = series
        return series

    def get_indicative_price(self, symbol: str, ts: datetime) -> Optional[float]:
        """
        Return the most recent *completed* minute-level VWAP at ``ts``.

        A kline labelled 10:00 contains trades through 10:00:59.999; using it
        at 10:00:05 would leak the rest of that minute. We therefore require
        the bar's right edge to be no later than ``ts`` and never fall forward
        to a later bar.
        """
        # At the beginning of a UTC day there may be no completed current-day
        # bar yet. Look back one date so 00:00:xx can use the completed 23:59
        # bar instead of failing or falling forward.
        for day in (ts.date(), ts.date() - timedelta(days=1)):
            series: Optional[pd.Series] = self._load_symbol_day_vwap_series(
                symbol=symbol,
                day=day,
            )
            if series is None or series.empty:
                continue

            last_completed_open = pd.Timestamp(ts) - pd.Timedelta(minutes=1)
            completed = series.loc[:last_completed_open]
            if not completed.empty:
                return float(completed.iloc[-1])
        return None

    def get_quote_to_usdt_indicative_price(self, quote_asset: str, ts: datetime) -> float:
        """
        Returns how many USDT one unit of `quote_asset` is worth at timestamp `ts`.
        """
        quote_asset = quote_asset.upper()
        if quote_asset == "USDT":
            return 1.0

        direct_symbol = f"{quote_asset}-USDT"
        direct_price: Optional[float] = self.get_indicative_price(
            symbol=direct_symbol,
            ts=ts,
        )
        if direct_price is not None and np.isfinite(direct_price) and direct_price > 0:
            return float(direct_price)

        inverse_symbol = f"USDT-{quote_asset}"
        inverse_price: Optional[float] = self.get_indicative_price(
            symbol=inverse_symbol,
            ts=ts,
        )
        if inverse_price is not None and np.isfinite(inverse_price) and inverse_price > 0:
            return float(1.0 / inverse_price)

        raise FileNotFoundError(
            f"Unable to find 1m kline indicative price for quote asset "
            f"{quote_asset} at {ts.isoformat()} in {self.raw_klines_dir}"
        )
