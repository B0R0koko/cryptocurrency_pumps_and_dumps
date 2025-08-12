import logging
import os
from datetime import datetime
from functools import partial
from multiprocessing import Pool, freeze_support, RLock
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import List, Dict, Optional, Any

from tqdm import tqdm

from core.columns import SYMBOL, TRADE_TIME, DATE, IS_BUYER_MAKER, PRICE, QUANTITY
from core.currency_pair import CurrencyPair
from core.exchange import Exchange
from core.paths import FEATURE_DIR, get_root_dir
from core.pump_event import PumpEvent
from core.time_utils import Bounds, NamedTimeDelta
from core.utils import configure_logging
from feature_writer.feature_exprs import *
from feature_writer.utils import load_pumps, aggregate_into_trades

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
    NamedTimeDelta.TWO_WEEKS
]

# Offsets to compute decay returns
DECAY_OFFSETS: List[NamedTimeDelta] = [
    NamedTimeDelta.ONE_MINUTE,
    NamedTimeDelta.TWO_MINUTES,
    NamedTimeDelta.THREE_MINUTES,
    NamedTimeDelta.FOUR_MINUTES,
    NamedTimeDelta.FIVE_MINUTES,
]


class PumpsFeatureWriter:

    def __init__(self):
        self._hive: pl.LazyFrame = pl.scan_parquet(
            Exchange.BINANCE_SPOT.get_hive_location(), hive_partitioning=True
        )

    def load_data_for_currency_pair(self, bounds: Bounds, currency_pair: CurrencyPair) -> pl.DataFrame:
        """Load data for currency from HiveDataset"""
        return (
            self._hive
            .filter(
                (pl.col(SYMBOL) == currency_pair.name) &
                (pl.col(DATE).is_between(bounds.day0, bounds.day1)) &
                (pl.col(TRADE_TIME).is_between(bounds.start_inclusive, bounds.end_exclusive))
            )
            .collect()
            .sort(by=TRADE_TIME)
        )

    @staticmethod
    def side_expr() -> pl.Expr:
        """
        Overwrite the way we compute side sign. For Binance we do it with IS_BUYER_MAKER
        """
        return 1 - 2 * pl.col(IS_BUYER_MAKER)

    def preprocess_data_for_currency(self, df: pl.DataFrame) -> pl.DataFrame:
        """Preprocess data loaded from the hive"""
        df = df.sort(by=TRADE_TIME, descending=False)
        df = df.with_columns(
            quote_abs=pl.col(PRICE) * pl.col(QUANTITY),
            side=self.side_expr(),
        )
        df = df.with_columns(
            quote_sign=pl.col("quote_abs") * pl.col("side"),
            quantity_sign=pl.col(QUANTITY) * pl.col("side")
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
            trade_time_prev=pl.col(TRADE_TIME).shift(1)
        )
        return df_trades

    def get_currency_pairs(self, bounds: Bounds) -> List[CurrencyPair]:
        # Get the set of symbols that have data
        unique_symbols: List[str] = pl.Series(
            self._hive.filter(
                pl.col(DATE).is_between(bounds.day0, bounds.day1) &
                pl.col(SYMBOL).str.ends_with("BTC")
            )
            .select(SYMBOL).unique().collect()
        ).to_list()
        return [
            CurrencyPair.from_string(symbol=symbol) for symbol in unique_symbols
        ]

    @staticmethod
    def compute_features(df: pl.DataFrame, pump_event: PumpEvent) -> Dict[str, Any]:
        features: Dict[str, float] = {}
        window: NamedTimeDelta

        df_hourly: pl.DataFrame = (
            df.group_by_dynamic(
                index_column=TRADE_TIME,
                period=timedelta(hours=1),
                every=timedelta(hours=1),
            )
            .agg(
                asset_return_pips=(pl.col("price_last").last() / pl.col("price_first").first() - 1) * 1e4,
                quote_abs=pl.col("quote_abs").sum()
            )
        )
        asset_return_std: float = df_hourly.select(pl.col("asset_return_pips").std()).item()
        quote_abs_std: float = df_hourly.select(pl.col("quote_abs").std()).item()

        rb: datetime = pump_event.time - timedelta(hours=1)

        for window in REGRESSOR_OFFSETS:
            # Compute using data 1 hour prior to the pump
            df_filtered: pl.DataFrame = df.filter(pl.col(TRADE_TIME).is_between(rb - window.get_td(), rb))
            df_hourly_filtered: pl.DataFrame = df_hourly.filter(pl.col(TRADE_TIME).is_between(rb - window.get_td(), rb))

            values: Dict[str, float] = {
                # Asset return
                FeatureType.ASSET_RETURN.col_name(offset=window): df_filtered.select(compute_return()).item(),
                # Asset return zscore
                FeatureType.ASSET_RETURN_ZSCORE.col_name(offset=window): df_hourly_filtered.select(
                    compute_asset_return_zscore(asset_return_std=asset_return_std)
                ).item(),
                # Quote abs zscore
                FeatureType.QUOTE_ABS_ZSCORE.col_name(offset=window): df_hourly_filtered.select(
                    compute_quote_abs_zscore(quote_abs_std=quote_abs_std)
                ).item(),
                # Share of long trades
                FeatureType.SHARE_OF_LONG_TRADES.col_name(offset=window): df_filtered.select(
                    compute_share_of_long_trades()).item(),
                # Powerlaw alpha
                FeatureType.POWERLAW_ALPHA.col_name(offset=window): df_filtered.select(
                    compute_powerlaw_alpha()).item(),
                # Slippage imbalance
                FeatureType.SLIPPAGE_IMBALANCE.col_name(offset=window): df_filtered.select(
                    compute_slippage_imbalance()).item(),
                # Flow imbalance
                FeatureType.FLOW_IMBALANCE.col_name(offset=window): df_filtered.select(
                    compute_flow_imbalance()).item(),
                # Num trades
                FeatureType.NUM_TRADES.col_name(offset=window): df_filtered.select(compute_num_trades()).item(),
            }
            features |= values

        # Price decay
        for decay_window in DECAY_OFFSETS:
            features[f"target_return@{decay_window.get_slug()}"] = (
                df.filter(
                    pl.col(TRADE_TIME).is_between(pump_event.time, pump_event.time + decay_window.get_td())
                )
                .select(compute_return()).item()
            )

        return features

    def create_cross_section(self, pump_event: PumpEvent, position: int) -> Optional[pl.DataFrame]:
        logging.info("Creating cross section")
        bounds: Bounds = Bounds(
            start_inclusive=pump_event.time - timedelta(days=30),
            end_exclusive=pump_event.time + timedelta(hours=1),
        )
        pbar = tqdm(desc=f"Loading currency_pairs", position=2 + position, leave=False)
        currency_pairs: List[CurrencyPair] = self.get_currency_pairs(bounds=bounds)

        if len(currency_pairs) == 0:
            pbar.set_description(f"Error: no currencies in the cross-section of the pump {str(pump_event)}")
            return None

        if pump_event.currency_pair not in currency_pairs:
            pbar.set_description(f"Error: no data found for target currency {str(pump_event)}")
            return None

        cross_section_features: List[Dict[str, float]] = []

        pbar.set_description("Iterating over currency_pairs")
        pbar.total = len(currency_pairs)

        for currency_pair in currency_pairs:
            df: pl.DataFrame = self.load_data_for_currency_pair(bounds=bounds, currency_pair=currency_pair)
            df = self.preprocess_data_for_currency(df=df)
            features: Dict[str, Any] = self.compute_features(df=df, pump_event=pump_event)
            features["currency_pair"] = currency_pair.name
            cross_section_features.append(features)
            pbar.update(1)

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

    def run_parallel(self, pump_events: List[PumpEvent], cpu_count: int) -> None:
        freeze_support()  # for Windows support
        tqdm.set_lock(RLock())  # for managing output contention

        with Pool(
                processes=cpu_count, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),), maxtasksperchild=1
        ) as pool:
            promises: List[AsyncResult] = []
            i: int = 0

            for pump_event in pump_events:
                promises.append(
                    pool.apply_async(
                        partial(
                            self._write_cross_section,
                            pump_event=pump_event,
                            position=i % cpu_count
                        )
                    )
                )
                i += 1

            for p in tqdm(promises, desc="Overall progress", position=0):
                p.get()


def main():
    configure_logging()
    pump_events: List[PumpEvent] = load_pumps(
        path=get_root_dir() / "src/resources/pumps.json"
    )
    writer = PumpsFeatureWriter()
    writer.run_parallel(pump_events=pump_events, cpu_count=10)


if __name__ == "__main__":
    main()
