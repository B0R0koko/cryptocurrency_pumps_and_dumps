import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import polars as pl

from core.columns import TRADE_TIME, PRICE
from core.currency_pair import CurrencyPair
from core.exchange import Exchange
from core.paths import FEATURE_DIR
from core.paths import get_root_dir
from core.pump_event import PumpEvent
from core.utils import configure_logging


def aggregate_into_trades(df_ticks: pl.DataFrame) -> pl.DataFrame:
    """Aggregate ticks into trades by TRADE_TIME"""
    df_trades: pl.DataFrame = (
        df_ticks
        .group_by(TRADE_TIME, maintain_order=True)
        .agg(
            price_first=pl.col(PRICE).first(),  # if someone placed a trade with price impact, then price_first
            price_last=pl.col(PRICE).last(),  # and price_last will differ
            # Amount spent in quote asset for the trade
            quote_abs=pl.col("quote_abs").sum(),
            quote_sign=pl.col("quote_sign").sum(),
            quantity_sign=pl.col("quantity_sign").sum(),
            # Amount of base asset transacted
            quantity_abs=pl.col("quantity").sum(),
            num_ticks=pl.col("price").count(),  # number of ticks for each trade
        )
    )
    df_trades = df_trades.with_columns(is_long=pl.col("quantity_sign") >= 0)
    return df_trades


def load_pumps(path: Path) -> List[PumpEvent]:
    """
    path: Path - path to the JSON file with labeled known pump events
    returns: List[PumpEvent]
    """
    pump_events: List[PumpEvent] = []
    with open(path) as file:
        for event in json.load(file):
            pump_events.append(
                PumpEvent(
                    currency_pair=CurrencyPair.from_string(symbol=event["symbol"]),
                    time=datetime.strptime(event["time"], "%Y-%m-%d %H:%M:%S"),
                    exchange=Exchange.parse_from_lower(event["exchange"]),
                )
            )
    return pump_events


def create_dataset() -> pd.DataFrame:
    configure_logging()
    path: Path = get_root_dir() / "src/resources/pumps.json"
    pump_events: List[PumpEvent] = load_pumps(path=path)

    dfs: List[pd.DataFrame] = []

    for pump in pump_events:
        cross_section_path: Path = FEATURE_DIR / "pumps" / f"{str(pump)}.parquet"

        if not cross_section_path.exists():
            logging.info("No cross section found for pump %s", pump)
            continue

        df_cross_section: pd.DataFrame = pd.read_parquet(cross_section_path)

        # Add additional columns
        df_cross_section["pump_hash"] = str(pump)
        df_cross_section["pump_time"] = pump.time
        df_cross_section["pumped_currency_pair"] = pump.currency_pair.name

        dfs.append(df_cross_section)

    df: pd.DataFrame = pd.concat(dfs)
    df = df.reset_index(drop=True)
    return df


def main():
    configure_logging()
    df: pd.DataFrame = create_dataset()
    logging.info("%s", df["pump_hash"].nunique())


if __name__ == "__main__":
    main()
