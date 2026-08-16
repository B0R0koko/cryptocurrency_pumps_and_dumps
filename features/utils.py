import json
from datetime import datetime
from pathlib import Path
from typing import List

import polars as pl

from core.columns import TRADE_TIME, PRICE
from core.currency_pair import CurrencyPair
from core.exchange import Exchange
from core.pump_event import PumpEvent


def aggregate_into_trades(df_ticks: pl.DataFrame) -> pl.DataFrame:
    """Aggregate ticks into trades by TRADE_TIME"""
    df_trades: pl.DataFrame = df_ticks.group_by(TRADE_TIME, maintain_order=True).agg(
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
    # A zero-net-side timestamp is neutral, not a buy. Counting it as long
    # biases the share-of-long-trades feature upward in mixed-fill groups.
    df_trades = df_trades.with_columns(is_long=pl.col("quantity_sign") > 0)
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
