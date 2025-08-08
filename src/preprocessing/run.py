import logging
from datetime import date

import polars as pl

from core.columns import TRADE_TIME
from core.exchange import Exchange
from core.time_utils import Bounds
from core.utils import configure_logging


def run_main():
    """Check if the data uploaded to HIVE is correct"""
    configure_logging()
    bounds: Bounds = Bounds.for_days(
        date(2025, 5, 1), date(2025, 5, 15)
    )
    logging.info("Collecting data for %s", str(bounds))
    df: pl.DataFrame = (
        pl.scan_parquet(Exchange.BINANCE_SPOT.get_hive_location(), hive_partitioning=True)
        .filter(
            (pl.col("symbol") == "BTC-USDT") &
            (pl.col("date").is_between(bounds.day0, bounds.day1)) &
            (pl.col(TRADE_TIME).is_between(bounds.start_inclusive, bounds.end_exclusive))
        )
        .collect()
    )

    print(df)


if __name__ == "__main__":
    run_main()
