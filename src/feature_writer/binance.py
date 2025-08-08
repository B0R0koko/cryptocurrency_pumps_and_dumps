import logging
import os
from datetime import timedelta, datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Any

import polars as pl
from tqdm import tqdm

from core.columns import SYMBOL, TRADE_TIME, DATE, IS_BUYER_MAKER, PRICE, QUANTITY
from core.currency_pair import CurrencyPair
from core.exchange import Exchange
from core.paths import FEATURE_DIR, get_root_dir
from core.time_utils import Bounds
from core.utils import configure_logging
from feature_writer.feature_exprs import compute_return, compute_share_of_long_trades, \
    compute_powerlaw_alpha, compute_slippage_imbalance, compute_flow_imbalance, compute_asset_return_zscore, \
    compute_quote_abs_zscore, compute_num_trades
from feature_writer.enums import PumpEvent
from feature_writer.utils import load_pumps


class NamedTimeDelta(Enum):
    ONE_MINUTE = (timedelta(minutes=1), "1MIN")
    TWO_MINUTES = (timedelta(minutes=2), "2MIN")
    THREE_MINUTES = (timedelta(minutes=3), "3MIN")
    FOUR_MINUTES = (timedelta(minutes=4), "4MIN")
    FIVE_MINUTES = (timedelta(minutes=5), "5MIN")
    FIFTEEN_MINUTES = (timedelta(minutes=15), "15MIN")
    ONE_HOUR = (timedelta(hours=1), "1H")
    TWO_HOURS = (timedelta(hours=2), "2H")
    FOUR_HOURS = (timedelta(hours=4), "4H")
    TWELVE_HOURS = (timedelta(hours=12), "12H")
    ONE_DAY = (timedelta(days=1), "1D")
    TWO_DAYS = (timedelta(days=2), "2D")
    ONE_WEEK = (timedelta(weeks=1), "7D")
    TWO_WEEKS = (timedelta(weeks=2), "14D")

    def get_td(self) -> timedelta:
        return self.value[0]

    def get_slug(self) -> str:
        return self.value[1]


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
        Overwrite the way we compute side sign. For Binance we do it with IS_BUYER_MAKER field
        for OKX we use simply use Side Literal string
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

        for window in (
                NamedTimeDelta.ONE_MINUTE,
                NamedTimeDelta.TWO_MINUTES,
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
        ):
            # Compute using data 1 hour prior to the pump
            df_filtered: pl.DataFrame = df.filter(pl.col(TRADE_TIME).is_between(rb - window.get_td(), rb))
            df_hourly_filtered: pl.DataFrame = df_hourly.filter(pl.col(TRADE_TIME).is_between(rb - window.get_td(), rb))

            values: Dict[str, float] = {
                f"asset_return@{window.get_slug()}": df_filtered.select(compute_return()).item(),
                f"asset_return_zscore@{window.get_slug()}": df_hourly_filtered.select(
                    compute_asset_return_zscore(asset_return_std=asset_return_std)
                ).item(),
                f"quote_abs_zscore@{window.get_slug()}": df_hourly_filtered.select(
                    compute_quote_abs_zscore(quote_abs_std=quote_abs_std)
                ).item(),
                f"share_of_long_trades@{window.get_slug()}": df_filtered.select(compute_share_of_long_trades()).item(),
                f"powerlaw_alpha@{window.get_slug()}": df_filtered.select(compute_powerlaw_alpha()).item(),
                f"slippage_imbalance@{window.get_slug()}": df_filtered.select(compute_slippage_imbalance()).item(),
                f"flow_imbalance@{window.get_slug()}": df_filtered.select(compute_flow_imbalance()).item(),
                f"num_trades@{window.get_slug()}": df_filtered.select(compute_num_trades()).item(),
            }
            features |= values

        # Price decay
        for decay_window in (
                NamedTimeDelta.ONE_MINUTE,
                NamedTimeDelta.TWO_MINUTES,
                NamedTimeDelta.THREE_MINUTES,
                NamedTimeDelta.FOUR_MINUTES,
                NamedTimeDelta.FIVE_MINUTES,
        ):
            features[f"target_return@{decay_window.get_slug()}"] = (
                df.filter(
                    pl.col(TRADE_TIME).is_between(pump_event.time, pump_event.time + decay_window.get_td())
                )
                .select(compute_return()).item()
            )

        return features

    def create_cross_section(self, pump_event: PumpEvent) -> Optional[pl.DataFrame]:
        logging.info("Creating cross section")
        bounds: Bounds = Bounds(
            start_inclusive=pump_event.time - timedelta(days=30),
            end_exclusive=pump_event.time + timedelta(hours=1),
        )
        currency_pairs: List[CurrencyPair] = self.get_currency_pairs(bounds=bounds)

        if len(currency_pairs) == 0:
            logging.error("No currencies in the cross-section of the pump %s", pump_event)
            return None

        if pump_event.currency_pair not in currency_pairs:
            logging.error("No data found for target currency %s", pump_event)
            return None

        cross_section_features: List[Dict[str, float]] = []

        for currency_pair in tqdm(currency_pairs):
            df: pl.DataFrame = self.load_data_for_currency_pair(bounds=bounds, currency_pair=currency_pair)
            df = self.preprocess_data_for_currency(df=df)
            features: Dict[str, Any] = self.compute_features(df=df, pump_event=pump_event)
            features["currency_pair"] = currency_pair.name
            cross_section_features.append(features)

        return pl.DataFrame(data=cross_section_features)

    @staticmethod
    def write_cross_section(features: pl.DataFrame, pump_event: PumpEvent) -> None:
        path: Path = FEATURE_DIR / "pumps" / f"{str(pump_event)}.parquet"
        os.makedirs(path.parent, exist_ok=True)
        logging.info("Writing cross section to %s", path)
        features.write_parquet(file=path)

    def run(self, pump_events: List[PumpEvent]) -> None:
        for pump_event in tqdm(pump_events):
            try:
                features: Optional[pl.DataFrame] = self.create_cross_section(pump_event=pump_event)
                if features is not None:
                    self.write_cross_section(features=features, pump_event=pump_event)

            except Exception as e:
                logging.error("%s", e)


def main():
    configure_logging()
    pump_events: List[PumpEvent] = load_pumps(
        path=get_root_dir() / "src/feature_writer/Pumps/resources/pumps.json"
    )
    writer = PumpsFeatureWriter()
    writer.run(pump_events=pump_events[304:])


if __name__ == "__main__":
    main()
