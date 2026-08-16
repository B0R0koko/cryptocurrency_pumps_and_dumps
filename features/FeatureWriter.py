import os
from bisect import bisect_left
from datetime import datetime, timedelta
from functools import partial
from multiprocessing import Pool, RLock
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import List, Dict, Optional, Any

from tqdm import tqdm

from core.columns import SYMBOL, TRADE_TIME, DATE, IS_BUYER_MAKER, PRICE, QUANTITY
from core.currency_pair import CurrencyPair, get_cross_section_currencies
from core.exchange import Exchange
from core.paths import FEATURE_DIR, get_root_dir
from core.pump_event import PumpEvent
from core.time_utils import Bounds, NamedTimeDelta
from core.utils import configure_logging
from features.feature_exprs import *
from features.utils import load_pumps, aggregate_into_trades

# Offsets used to compute features
REGRESSOR_OFFSETS: List[NamedTimeDelta] = [
    NamedTimeDelta.FIVE_MINUTES,
    NamedTimeDelta.FIFTEEN_MINUTES,
    NamedTimeDelta.ONE_HOUR,
    NamedTimeDelta.TWO_HOURS,
    NamedTimeDelta.FOUR_HOURS,
    NamedTimeDelta.TWELVE_HOURS,
    NamedTimeDelta.ONE_DAY,
    NamedTimeDelta.TWO_DAYS,
    NamedTimeDelta.ONE_WEEK,
    NamedTimeDelta.TWO_WEEKS,
]

# Offsets to compute decay returns
DECAY_OFFSETS: List[NamedTimeDelta] = [
    NamedTimeDelta.ONE_MINUTE,
    NamedTimeDelta.TWO_MINUTES,
    NamedTimeDelta.THREE_MINUTES,
    NamedTimeDelta.FOUR_MINUTES,
    NamedTimeDelta.FIVE_MINUTES,
]

# Decision lag between the last trade a feature is allowed to see and the pump
# announcement time. The portfolio submits its entry at ``T - DECISION_LAG``
# and fills on the first subsequent pre-announcement trade (see
# ``TOPKPortfolio.buy_before``); features must reflect only information
# available when that order is submitted. The interval
# ``[T - DECISION_LAG, T]`` must never contribute to any feature. Changing this
# constant requires regenerating all feature parquets from raw trades.
DECISION_LAG: timedelta = timedelta(minutes=15)

# Normalizer statistics are estimated from a fixed, past-only history whose
# right edge is the decision time. Candidate assets must also have traded
# recently enough that the backtest can form a contemporaneous entry price.
NORMALIZER_LOOKBACK: timedelta = timedelta(days=30)
ELIGIBILITY_LOOKBACK: timedelta = timedelta(days=1)


def get_currency_pairs(bounds: Bounds) -> List[CurrencyPair]:
    # Reading partition directories is much faster than scanning parquet data to list symbols.
    return [
        currency_pair
        for currency_pair in get_cross_section_currencies(
            hive_dir=Exchange.BINANCE_SPOT.get_hive_location(),
            bounds=bounds,
        )
        if currency_pair.term == "BTC"
    ]


class PumpsFeatureWriter:

    def __init__(self, pump_events: List[PumpEvent]):
        self._pump_events: List[PumpEvent] = pump_events
        self._pump_times_by_currency: Dict[str, List[datetime]] = {}
        for pump_event in sorted(self._pump_events, key=lambda event: event.time):
            self._pump_times_by_currency.setdefault(pump_event.currency_pair.name, []).append(pump_event.time)

        self._hive: pl.LazyFrame = pl.scan_parquet(Exchange.BINANCE_SPOT.get_hive_location(), hive_partitioning=True)

    def load_data_for_currency_pair(self, bounds: Bounds, currency_pair: CurrencyPair) -> pl.DataFrame:
        """Load data for currency from HiveDataset.

        ``bounds.end_exclusive`` is treated as exclusive on both the date and
        trade-time predicates via ``closed="left"``, matching the contract of
        :class:`Bounds`.
        """
        return (
            self._hive.filter(
                (pl.col(SYMBOL) == currency_pair.name)
                & (pl.col(DATE).is_between(bounds.day0, bounds.day1))
                & (pl.col(TRADE_TIME).is_between(bounds.start_inclusive, bounds.end_exclusive, closed="left"))
            )
            .collect()
            .sort(by=TRADE_TIME)
        )

    @staticmethod
    def side_expr() -> pl.Expr:
        """
        Overwrite the way we compute side sign. For Binance, we do it with IS_BUYER_MAKER
        """
        return 1 - 2 * pl.col(IS_BUYER_MAKER)

    def preprocess_data_for_currency(self, df: pl.DataFrame) -> pl.DataFrame:
        """Preprocess data loaded from the hive"""
        df = df.with_columns(
            quote_abs=pl.col(PRICE) * pl.col(QUANTITY),
            side=self.side_expr(),
        )
        df = df.with_columns(
            quote_sign=pl.col("quote_abs") * pl.col("side"),
            quantity_sign=pl.col(QUANTITY) * pl.col("side"),
        )
        # Aggregate into trades
        df_trades: pl.DataFrame = aggregate_into_trades(df_ticks=df)

        assert df_trades[TRADE_TIME].is_sorted(descending=False), "Data must be in ascending order by TRADE_TIME"

        # Compute slippages
        df_trades = df_trades.with_columns(
            quote_slippage_abs=(pl.col("quote_abs") - pl.col("price_first") * pl.col("quantity_abs")).abs()
        )
        df_trades = df_trades.with_columns(
            quote_slippage_sign=pl.col("quote_slippage_abs") * pl.col("quantity_sign").sign(),
            # Add lags of price_last and trade_time
            price_last_prev=pl.col("price_last").shift(1),
            trade_time_prev=pl.col(TRADE_TIME).shift(1),
        )
        return df_trades

    @staticmethod
    def has_pre_decision_activity(df_ticks: pl.DataFrame, rb: datetime) -> bool:
        """Return whether the asset traded in the exact past-only eligibility window."""
        return not df_ticks.filter(
            pl.col(TRADE_TIME).is_between(
                rb - ELIGIBILITY_LOOKBACK,
                rb,
                closed="left",
            )
        ).is_empty()

    def _num_prev_pumps(self, currency_pair: CurrencyPair, pump_event: PumpEvent) -> int:
        pump_times: List[datetime] = self._pump_times_by_currency.get(currency_pair.name, [])
        return bisect_left(pump_times, pump_event.time)

    def compute_features(self, df: pl.DataFrame, currency_pair: CurrencyPair, pump_event: PumpEvent) -> Dict[str, Any]:
        """Compute the regressor and target features for one currency pair.

        Feature-window semantics
        ------------------------
        All regressor features are computed from data strictly available at the
        portfolio entry time ``rb = pump_event.time - DECISION_LAG``. No trade
        with ``TRADE_TIME >= rb`` contributes to any feature, so the interval
        ``[T - DECISION_LAG, T]`` is by construction excluded. The in-window
        features (return, flow imbalance, slippage, powerlaw alpha, etc.) run
        on trades filtered with ``TRADE_TIME in [lb, rb)`` (left-closed).

        Z-score normaliser geometry (rb-anchored hourly bars)
        -----------------------------------------------------
        The z-score normalisers are computed on hourly bars anchored at ``rb``
        rather than at calendar hour boundaries. Bar ``k`` (k >= 0) covers
        ``[rb - (k+1)*1h, rb - k*1h)``; the newest bar ``k = 0`` ends exactly at
        ``rb``. By construction every bar is a full hour of pre-rb data and no
        bar can overlap ``[rb, T]``.

        This anchoring is implemented via ``group_by_dynamic(every=1h,
        period=1h, offset=<rb-mod-1h>)`` on the strictly-pre-rb trades, with the
        offset chosen so the polars bin boundaries fall on ``..., rb - 2h,
        rb - 1h, rb``. When ``rb`` is on the calendar hour the offset is zero;
        when ``rb`` is at ``HH:MM:SS.uuu`` the offset is ``MM*60 + SS + uuu/1e6``
        seconds so bar edges land on the ``:MM:SS.uuu`` grid.

        For windows of at least one hour, z-score numerators average the exact
        union of rb-anchored hourly bars in ``[rb-X, rb)``. For the 5- and
        15-minute windows, the exact raw-trade window is converted to an hourly
        rate before comparison with the historical hourly moments. Consequently
        ``@5MIN``, ``@15MIN`` and ``@1H`` retain distinct information while every
        numerator still ends strictly before ``rb``.

        Note: on-disk features from earlier runs (calendar-anchored bars) are
        NOT equivalent and must be regenerated end-to-end.
        """
        features: Dict[str, Any] = {}
        window: NamedTimeDelta
        one_hour: timedelta = timedelta(hours=1)

        rb: datetime = pump_event.time - DECISION_LAG

        # Truncate trades to strictly-pre-rb, then aggregate into hourly bars
        # anchored so the newest bar ends exactly at rb.
        df_pre_rb: pl.DataFrame = df.filter(pl.col(TRADE_TIME) < rb)
        offset_us: int = (rb.minute * 60 + rb.second) * 1_000_000 + rb.microsecond
        df_hourly: pl.DataFrame = df_pre_rb.group_by_dynamic(
            index_column=TRADE_TIME,
            period=one_hour,
            every=one_hour,
            offset=f"{offset_us}us",
        ).agg(
            asset_return_pips=(pl.col("price_last").last() / pl.col("price_first").first() - 1) * 1e4,
            quote_abs=pl.col("quote_abs").sum(),
        )
        # Belt-and-braces: drop any bar whose right edge extends past rb.
        # After the strict pre-rb truncation this should be empty, but the
        # invariant is cheap and makes downstream code auditable.
        df_hourly_full: pl.DataFrame = df_hourly.filter(pl.col(TRADE_TIME) + pl.duration(hours=1) <= rb)
        asset_return_mean: float = df_hourly_full.select(pl.col("asset_return_pips").mean()).item()
        asset_return_std: float = df_hourly_full.select(pl.col("asset_return_pips").std()).item()
        quote_abs_mean: float = df_hourly_full.select(pl.col("quote_abs").mean()).item()
        quote_abs_std: float = df_hourly_full.select(pl.col("quote_abs").std()).item()

        for window in REGRESSOR_OFFSETS:
            # Feature window ``[lb, rb)`` — left-closed / right-open so trades
            # at exactly ``rb`` (the portfolio entry time) do not leak into any
            # feature.
            lb: datetime = rb - window.get_td()
            df_filtered: pl.DataFrame = df.filter(pl.col(TRADE_TIME).is_between(lb, rb, closed="left"))

            window_hours: float = window.get_td().total_seconds() / one_hour.total_seconds()
            df_hourly_filtered: pl.DataFrame = df_hourly_full.filter(pl.col(TRADE_TIME) >= lb)

            window_values: Dict[str, Any] = df_filtered.select(
                compute_return().alias("asset_return"),
                compute_share_of_long_trades().alias("share_of_long_trades"),
                compute_powerlaw_alpha().alias("powerlaw_alpha"),
                compute_slippage_imbalance().alias("slippage_imbalance"),
                compute_flow_imbalance().alias("flow_imbalance"),
                compute_num_trades().alias("num_trades"),
            ).to_dicts()[0]
            if window_hours < 1.0:
                # Use the exact short window rather than silently reusing the
                # same full one-hour bar for 5MIN, 15MIN and 1H.
                window_return: Any = window_values["asset_return"]
                window_quote_abs: Any = df_filtered.select(pl.col("quote_abs").sum()).item()
                asset_return_hourly: Any = window_return / window_hours if window_return is not None else None
                quote_abs_hourly: Any = window_quote_abs / window_hours if window_quote_abs is not None else None
                hourly_values: Dict[str, Any] = {
                    "asset_return_zscore": (
                        (asset_return_hourly - asset_return_mean) / asset_return_std
                        if asset_return_hourly is not None and asset_return_std not in (None, 0)
                        else None
                    ),
                    "quote_abs_zscore": (
                        (quote_abs_hourly - quote_abs_mean) / quote_abs_std
                        if quote_abs_hourly is not None and quote_abs_std not in (None, 0)
                        else None
                    ),
                }
            else:
                hourly_values = df_hourly_filtered.select(
                    compute_asset_return_zscore(
                        asset_return_mean=asset_return_mean,
                        asset_return_std=asset_return_std,
                    ).alias("asset_return_zscore"),
                    compute_quote_abs_zscore(quote_abs_mean=quote_abs_mean, quote_abs_std=quote_abs_std).alias(
                        "quote_abs_zscore"
                    ),
                ).to_dicts()[0]

            values: Dict[str, float] = {
                FeatureType.ASSET_RETURN.col_name(offset=window): window_values["asset_return"],
                FeatureType.ASSET_RETURN_ZSCORE.col_name(offset=window): hourly_values["asset_return_zscore"],
                FeatureType.QUOTE_ABS_ZSCORE.col_name(offset=window): hourly_values["quote_abs_zscore"],
                FeatureType.SHARE_OF_LONG_TRADES.col_name(offset=window): window_values["share_of_long_trades"],
                FeatureType.POWERLAW_ALPHA.col_name(offset=window): window_values["powerlaw_alpha"],
                FeatureType.SLIPPAGE_IMBALANCE.col_name(offset=window): window_values["slippage_imbalance"],
                FeatureType.FLOW_IMBALANCE.col_name(offset=window): window_values["flow_imbalance"],
                FeatureType.NUM_TRADES.col_name(offset=window): window_values["num_trades"],
            }
            features |= values

        features[FeatureType.NUM_PREV_PUMP.lower()] = self._num_prev_pumps(
            currency_pair=currency_pair, pump_event=pump_event
        )

        # Price decay
        for decay_window in DECAY_OFFSETS:
            features[f"target_return@{decay_window.get_slug()}"] = (
                df.filter(
                    pl.col(TRADE_TIME).is_between(
                        pump_event.time,
                        pump_event.time + decay_window.get_td(),
                        closed="left",
                    )
                )
                .select(compute_return())
                .item()
            )

        return features

    def create_cross_section(self, pump_event: PumpEvent, position: int) -> Optional[pl.DataFrame]:
        rb: datetime = pump_event.time - DECISION_LAG
        feature_bounds: Bounds = Bounds(
            start_inclusive=rb - NORMALIZER_LOOKBACK,
            end_exclusive=pump_event.time + max(window.get_td() for window in DECAY_OFFSETS),
        )
        # Partition discovery is intentionally restricted to the past. Because
        # the boundary day's partition can contain later trades, exact timestamp
        # eligibility is checked after loading each candidate below.
        discovery_bounds: Bounds = Bounds(
            start_inclusive=rb - ELIGIBILITY_LOOKBACK,
            end_exclusive=rb,
        )
        pbar = tqdm(desc=f"Loading currency_pairs", position=2 + position, leave=False)
        currency_pairs: List[CurrencyPair] = get_currency_pairs(bounds=discovery_bounds)

        if len(currency_pairs) == 0:
            pbar.set_description(f"Error: no currencies in the cross-section of the pump {str(pump_event)}")
            return None

        cross_section_features: List[Dict[str, float]] = []

        pbar.set_description("Iterating over currency_pairs")
        pbar.total = len(currency_pairs)

        for currency_pair in currency_pairs:
            df_ticks: pl.DataFrame = self.load_data_for_currency_pair(
                bounds=feature_bounds,
                currency_pair=currency_pair,
            )
            eligible: bool = self.has_pre_decision_activity(df_ticks=df_ticks, rb=rb)
            if not eligible:
                pbar.update(1)
                continue

            df: pl.DataFrame = self.preprocess_data_for_currency(df=df_ticks)
            features: Dict[str, Any] = self.compute_features(df=df, currency_pair=currency_pair, pump_event=pump_event)
            features["currency_pair"] = currency_pair.name
            cross_section_features.append(features)
            pbar.update(1)

        if not any(row["currency_pair"] == pump_event.currency_pair.name for row in cross_section_features):
            pbar.set_description(f"Error: no pre-decision data found for target currency {str(pump_event)}")
            return None

        return pl.DataFrame(data=cross_section_features)

    def _write_cross_section(self, pump_event: PumpEvent, position: int = 0) -> None:
        features: Optional[pl.DataFrame] = self.create_cross_section(pump_event=pump_event, position=position)
        if features is not None:
            path: Path = FEATURE_DIR / "pumps" / f"{str(pump_event)}.parquet"
            os.makedirs(path.parent, exist_ok=True)
            features.write_parquet(file=path)

    def run(self, pump_events: List[PumpEvent]) -> None:
        for pump_event in tqdm(pump_events):
            self._write_cross_section(pump_event=pump_event)

    def run_parallel(self, cpu_count: int) -> None:
        tqdm.set_lock(RLock())  # for managing output contention

        with Pool(
            processes=cpu_count,
            initializer=tqdm.set_lock,
            initargs=(tqdm.get_lock(),),
        ) as pool:
            promises: List[AsyncResult] = []
            i: int = 0

            for pump_event in self._pump_events:
                promises.append(
                    pool.apply_async(
                        partial(
                            self._write_cross_section,
                            pump_event=pump_event,
                            position=i % cpu_count,
                        )
                    )
                )
                i += 1

            for p in tqdm(promises, desc="Overall progress", position=0):
                p.get()

    @property
    def pump_times_by_currency(self):
        return self._pump_times_by_currency


def main():
    import argparse

    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-count", type=int, default=16)
    args = parser.parse_args()
    pump_events: List[PumpEvent] = load_pumps(path=get_root_dir() / "resources/pumps.json")
    writer = PumpsFeatureWriter(pump_events=pump_events)
    writer.run_parallel(cpu_count=args.cpu_count)


if __name__ == "__main__":
    main()
