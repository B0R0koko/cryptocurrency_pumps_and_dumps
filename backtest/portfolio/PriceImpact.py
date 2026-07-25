from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from core.columns import (
    CLOSE_PRICE,
    IS_BUYER_MAKER,
    OPEN_TIME,
    PRICE,
    QUANTITY,
    QUOTE_ASSET_VOLUME,
    TAKER_BUY_QUOTE_ASSET_VOLUME,
    TRADE_TIME,
)


def _fit_sqrt_regression(notionals: np.ndarray, impacts: np.ndarray) -> float:
    """Constrained OLS for I(Q) = beta * sqrt(Q), no intercept.

    beta >= 0 is enforced so impact is non-negative and monotonically increasing.
    """
    if len(notionals) == 0:
        return 0.0
    X = np.sqrt(notionals).reshape(-1, 1)
    result = lsq_linear(X, impacts, bounds=([0.0], [np.inf]))
    return float(result.x[0])


@dataclass(frozen=True)
class PriceImpactModel:
    """
    Square-root market-impact model: I(Q) = beta * sqrt(Q).

    A single beta is fitted on pooled absolute impacts from both buy and sell
    candles (no intercept, no side separation). Regression is fitted in
    USDT-normalised notional space so the impact curve is comparable across
    time (BTC price regimes). The public interface accepts notionals in quote
    currency; the model converts internally using the stored ``quote_to_usdt``
    rate from fitting time.

    ``num_samples`` and ``sample_frequency`` describe the regression inputs:
    the number of candle samples (or aggregated meta-orders) used to fit beta,
    and the candle frequency those samples were aggregated at (e.g. ``"5min"``
    for pre-pump lookback or ``"5s"`` for manipulation-window exits). Trade-
    level fits record ``"trade"`` as the frequency.
    """

    beta: float
    quote_to_usdt: float
    num_samples: int
    sample_frequency: str = "unknown"

    def predict_impact_bps(self, side: int, notional_quote: float) -> float:
        """Terminal impact: I(Q) = beta * sqrt(Q_usdt)."""
        if notional_quote <= 0:
            return 0.0
        q_usdt = notional_quote * self.quote_to_usdt
        return self.beta * np.sqrt(q_usdt)

    def predict_vwap_impact_bps(self, side: int, notional_quote: float) -> float:
        """
        VWAP impact in USDT space.

        I_vwap(Q) = (1/Q) * integral from 0 to Q of beta*sqrt(q) dq
                   = 2/3 * beta * sqrt(Q)
        """
        if notional_quote <= 0:
            return 0.0
        q_usdt = notional_quote * self.quote_to_usdt
        return (2.0 / 3.0) * self.beta * np.sqrt(q_usdt)

    def estimate_vwap_price(self, base_price: float, side: int, notional_quote: float) -> tuple[float, float]:
        """
        VWAP execution price.

        Entry (side=1):  p_vwap = p * (1 + I_vwap / 1e4)
        Exit  (side=-1): p_vwap = p * (1 - I_vwap / 1e4)
        """
        if notional_quote <= 0:
            return max(1e-12, float(base_price)), 0.0
        vwap_impact_bps = self.predict_vwap_impact_bps(side=side, notional_quote=notional_quote)
        vwap_price = max(1e-12, float(base_price) * (1.0 + side * vwap_impact_bps / 1e4))
        return vwap_price, vwap_impact_bps


@dataclass(frozen=True)
class PriceImpactFitResult:
    model: PriceImpactModel
    samples: pd.DataFrame
    diagnostics: pd.DataFrame


def _empty_price_impact_model(quote_to_usdt: float = 1.0, sample_frequency: str = "unknown") -> PriceImpactModel:
    return PriceImpactModel(
        beta=0.0,
        quote_to_usdt=quote_to_usdt,
        num_samples=0,
        sample_frequency=sample_frequency,
    )


_IMPACT_SAMPLE_COLUMNS = [
    "trade_time",
    "price_first",
    "price_last",
    "notional_quote",
    "notional_usdt",
    "side",
    "impact_bps",
]


def _empty_impact_samples() -> pd.DataFrame:
    return pd.DataFrame(columns=_IMPACT_SAMPLE_COLUMNS)


def aggregate_trades_to_orders(trades: pd.DataFrame, quote_to_usdt: float = 1.0) -> pd.DataFrame:
    """
    Aggregate fills sharing the same trade_time into meta-orders.

    Each group represents fills from a single market order walking through
    the order book.  We record the first and last fill price, total notional
    (in both quote and USDT), and net side (buy vs sell initiated).
    """
    if trades.empty:
        return _empty_impact_samples()

    required = {TRADE_TIME, PRICE, QUANTITY, IS_BUYER_MAKER}
    if not required.issubset(trades.columns):
        return _empty_impact_samples()

    df = trades.copy()
    df[PRICE] = pd.to_numeric(df[PRICE], errors="coerce")
    df[QUANTITY] = pd.to_numeric(df[QUANTITY], errors="coerce")
    df = df.dropna(subset=[TRADE_TIME, PRICE, QUANTITY])
    if df.empty:
        return _empty_impact_samples()

    df["quote_notional"] = df[PRICE] * df[QUANTITY]
    # is_buyer_maker=True means taker sell, False means taker buy
    df["signed_notional"] = np.where(df[IS_BUYER_MAKER], -df["quote_notional"], df["quote_notional"])

    grouped = df.groupby(TRADE_TIME, sort=True).agg(
        price_first=(PRICE, "first"),
        price_last=(PRICE, "last"),
        total_quote_notional=("quote_notional", "sum"),
        net_signed_notional=("signed_notional", "sum"),
    )
    grouped = grouped[grouped["total_quote_notional"] > 0].copy()
    if grouped.empty:
        return _empty_impact_samples()

    grouped["side"] = np.where(grouped["net_signed_notional"] >= 0, 1, -1)
    grouped["notional_quote"] = grouped["total_quote_notional"]
    grouped["notional_usdt"] = grouped["notional_quote"] * quote_to_usdt
    # Absolute impact: take abs of price return regardless of side
    grouped["impact_bps"] = np.abs(grouped["price_last"] / grouped["price_first"] - 1.0) * 1e4

    result = grouped.reset_index().rename(columns={TRADE_TIME: "trade_time"})
    return result[_IMPACT_SAMPLE_COLUMNS].reset_index(drop=True)


def trades_to_klines(trades: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """Resample tick-level trades into candles with buy/sell volume split.

    Parameters
    ----------
    freq : str
        Pandas resample frequency string, e.g. ``"5min"``, ``"1min"``, ``"5s"``.
    """
    if trades.empty:
        return pd.DataFrame(columns=[OPEN_TIME, CLOSE_PRICE, QUOTE_ASSET_VOLUME, TAKER_BUY_QUOTE_ASSET_VOLUME])

    df = trades.copy()
    df[PRICE] = pd.to_numeric(df[PRICE], errors="coerce")
    df[QUANTITY] = pd.to_numeric(df[QUANTITY], errors="coerce")
    df = df.dropna(subset=[TRADE_TIME, PRICE, QUANTITY])
    if df.empty:
        return pd.DataFrame(columns=[OPEN_TIME, CLOSE_PRICE, QUOTE_ASSET_VOLUME, TAKER_BUY_QUOTE_ASSET_VOLUME])

    df["quote_notional"] = df[PRICE] * df[QUANTITY]
    # is_buyer_maker=True → taker sell; False → taker buy
    df["buy_notional"] = np.where(df[IS_BUYER_MAKER], 0.0, df["quote_notional"])

    df = df.set_index(pd.DatetimeIndex(df[TRADE_TIME]))
    resampled = df.resample(freq).agg(
        close_price=(PRICE, "last"),
        quote_asset_volume=("quote_notional", "sum"),
        taker_buy_quote_asset_volume=("buy_notional", "sum"),
    )
    resampled = resampled.dropna(subset=["close_price"])
    resampled = resampled[resampled["quote_asset_volume"] > 0]
    resampled.index.name = OPEN_TIME
    return resampled.reset_index()


def aggregate_klines_to_samples(
    klines: pd.DataFrame,
    quote_to_usdt: float = 1.0,
    sell_only: bool = False,
) -> pd.DataFrame:
    """
    Build impact samples from candles using absolute net volume and absolute impact.

    Parameters
    ----------
    sell_only : bool
        If True, keep only candles with net selling pressure (negative net buy volume).
    """
    if klines.empty:
        return _empty_impact_samples()

    required = {OPEN_TIME, CLOSE_PRICE, QUOTE_ASSET_VOLUME, TAKER_BUY_QUOTE_ASSET_VOLUME}
    if not required.issubset(klines.columns):
        return _empty_impact_samples()

    df = klines.copy()
    for col in [CLOSE_PRICE, QUOTE_ASSET_VOLUME, TAKER_BUY_QUOTE_ASSET_VOLUME]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[OPEN_TIME, CLOSE_PRICE, QUOTE_ASSET_VOLUME, TAKER_BUY_QUOTE_ASSET_VOLUME])
    df = df.sort_values(OPEN_TIME).reset_index(drop=True)

    if len(df) < 2:
        return _empty_impact_samples()

    prev_close = df[CLOSE_PRICE].shift(1)
    net_buy_volume = 2 * df[TAKER_BUY_QUOTE_ASSET_VOLUME] - df[QUOTE_ASSET_VOLUME]
    price_return_bps = (df[CLOSE_PRICE] / prev_close - 1.0) * 1e4

    # Drop first row (no previous close) and zero net volume rows.
    valid = prev_close.notna() & (prev_close > 0) & (net_buy_volume != 0)
    df = df[valid].copy()
    net_buy_volume = net_buy_volume[valid]
    price_return_bps = price_return_bps[valid]
    prev_close = prev_close[valid]

    df["side"] = np.where(net_buy_volume > 0, 1, -1)

    if sell_only:
        sell_mask = df["side"] == -1
        df = df[sell_mask].copy()
        net_buy_volume = net_buy_volume[sell_mask]
        price_return_bps = price_return_bps[sell_mask]
        prev_close = prev_close[sell_mask]

    if df.empty:
        return _empty_impact_samples()

    df["notional_quote"] = np.abs(net_buy_volume.values)
    df["notional_usdt"] = df["notional_quote"] * quote_to_usdt
    # Absolute impact: take abs of price return regardless of side.
    df["impact_bps"] = np.abs(price_return_bps.values)
    df["trade_time"] = df[OPEN_TIME]
    df["price_first"] = prev_close.values
    df["price_last"] = df[CLOSE_PRICE].values

    return df[_IMPACT_SAMPLE_COLUMNS].reset_index(drop=True)


def _side_diagnostics(
    side_name: str,
    num_samples: int,
    median_impact_bps: float,
    max_impact_bps: float,
) -> dict:
    return {
        "side": side_name,
        "num_samples": num_samples,
        "median_impact_bps": median_impact_bps,
        "max_impact_bps": max_impact_bps,
    }


def fit_price_impact_model(
    trades: pd.DataFrame,
    quote_to_usdt: float = 1.0,
) -> PriceImpactModel:
    """
    Fit a sqrt market-impact model from trade-level data.

    Trades sharing the same execution timestamp are aggregated into meta-orders.
    Absolute impacts from both buy and sell sides are pooled and fitted as
    I(Q) = beta * sqrt(Q_usdt), with no intercept.
    """
    return fit_price_impact_model_with_diagnostics(
        trades=trades,
        quote_to_usdt=quote_to_usdt,
    ).model


def fit_price_impact_model_with_diagnostics(
    trades: pd.DataFrame,
    quote_to_usdt: float = 1.0,
) -> PriceImpactFitResult:
    samples = aggregate_trades_to_orders(trades=trades, quote_to_usdt=quote_to_usdt)
    return _fit_from_samples(samples, quote_to_usdt, sample_frequency="trade")


def _fit_from_samples(
    samples: pd.DataFrame,
    quote_to_usdt: float,
    sample_frequency: str,
) -> PriceImpactFitResult:
    if samples.empty:
        empty = _empty_price_impact_model(quote_to_usdt=quote_to_usdt, sample_frequency=sample_frequency)
        diagnostics = pd.DataFrame(
            [
                _side_diagnostics("buy", 0, 0.0, 0.0),
                _side_diagnostics("sell", 0, 0.0, 0.0),
            ]
        )
        return PriceImpactFitResult(model=empty, samples=samples, diagnostics=diagnostics)

    # Pool both sides and fit a single regression: I(Q) = beta * sqrt(Q)
    notionals_usdt = samples["notional_usdt"].to_numpy(dtype=float)
    valid = notionals_usdt > 0
    impacts = samples["impact_bps"].to_numpy(dtype=float)
    beta = _fit_sqrt_regression(notionals_usdt[valid], impacts[valid]) if valid.any() else 0.0

    buy_df = samples[samples["side"] == 1]
    sell_df = samples[samples["side"] == -1]

    buy_diag = _side_diagnostics(
        "buy",
        len(buy_df),
        float(buy_df["impact_bps"].median()) if len(buy_df) > 0 else 0.0,
        float(buy_df["impact_bps"].max()) if len(buy_df) > 0 else 0.0,
    )
    sell_diag = _side_diagnostics(
        "sell",
        len(sell_df),
        float(sell_df["impact_bps"].median()) if len(sell_df) > 0 else 0.0,
        float(sell_df["impact_bps"].max()) if len(sell_df) > 0 else 0.0,
    )
    buy_diag["beta"] = beta
    sell_diag["beta"] = beta

    model = PriceImpactModel(
        beta=beta,
        quote_to_usdt=quote_to_usdt,
        num_samples=int(samples.shape[0]),
        sample_frequency=sample_frequency,
    )
    diagnostics = pd.DataFrame([buy_diag, sell_diag])
    return PriceImpactFitResult(model=model, samples=samples, diagnostics=diagnostics)


def fit_price_impact_model_from_klines(
    klines: pd.DataFrame,
    quote_to_usdt: float = 1.0,
    sell_only: bool = False,
    sample_frequency: str = "unknown",
) -> PriceImpactModel:
    """
    Fit a sqrt market-impact model from klines (any frequency).

    Uses absolute net volume per candle as Q and absolute close-to-close
    price change as the impact signal. Both sides are pooled into a single
    regression: I(Q) = beta * sqrt(Q_usdt), no intercept.

    Parameters
    ----------
    sell_only : bool
        If True, use only sell-dominated candles for fitting.
    sample_frequency : str
        Frequency string describing the candle aggregation used to build
        ``klines`` (e.g. ``"5min"``, ``"5s"``). Stored on the fitted model.
    """
    return fit_price_impact_model_from_klines_with_diagnostics(
        klines=klines,
        quote_to_usdt=quote_to_usdt,
        sell_only=sell_only,
        sample_frequency=sample_frequency,
    ).model


def fit_price_impact_model_from_klines_with_diagnostics(
    klines: pd.DataFrame,
    quote_to_usdt: float = 1.0,
    sell_only: bool = False,
    sample_frequency: str = "unknown",
) -> PriceImpactFitResult:
    samples = aggregate_klines_to_samples(klines=klines, quote_to_usdt=quote_to_usdt, sell_only=sell_only)
    return _fit_from_samples(samples, quote_to_usdt, sample_frequency=sample_frequency)
