from datetime import datetime, timedelta

import pandas as pd

from backtest.pipelines.BaseModel import ImplementsRank
from backtest.portfolio.TOPKPortfolio import TOPKPortfolio
from backtest.utils.sample import Dataset
from core.currency_pair import CurrencyPair
from core.exchange import Exchange
from core.pump_event import PumpEvent


class _DummyModel(ImplementsRank):
    def rank(self, dataset: Dataset) -> pd.Series:
        return pd.Series(dtype=float)


def _manager() -> TOPKPortfolio:
    return TOPKPortfolio(
        model=_DummyModel(),
        portfolio_size=1,
        buy_before=timedelta(minutes=15),
        sell_after=timedelta(minutes=1),
        order_notional_quote=0.0,
        order_notional_usdt=0.0,
    )


def _pump() -> tuple[PumpEvent, CurrencyPair]:
    cp = CurrencyPair.from_string("AAA-BTC")
    return (
        PumpEvent(
            currency_pair=cp,
            time=datetime(2021, 1, 2, 12, 0, 0),
            exchange=Exchange.BINANCE_SPOT,
        ),
        cp,
    )


def test_regular_leg_uses_first_trade_at_decision_and_first_trade_at_announcement() -> None:
    pump, cp = _pump()
    decision = pump.time - timedelta(minutes=15)
    prices = pd.Series(
        [90.0, 100.0, 101.0, 110.0, 111.0],
        index=[
            decision - timedelta(seconds=1),
            decision,
            decision + timedelta(seconds=1),
            pump.time,
            pump.time + timedelta(seconds=1),
        ],
    )

    tx = _manager().regular_transaction(ts_price=prices, pump=pump, cp=cp)

    assert tx.entry_ts == decision
    assert tx.entry_price == 100.0
    assert tx.exit_ts == pump.time
    assert tx.exit_price == 110.0


def test_pumped_leg_uses_first_trade_at_or_after_one_minute() -> None:
    pump, cp = _pump()
    decision = pump.time - timedelta(minutes=15)
    exit_threshold = pump.time + timedelta(minutes=1)
    prices = pd.Series(
        [100.0, 125.0, 130.0, 131.0],
        index=[
            decision,
            pump.time,
            exit_threshold,
            exit_threshold + timedelta(seconds=1),
        ],
    )

    tx = _manager().pumped_transaction(ts_price=prices, pump=pump, cp=cp)

    assert tx.entry_ts == decision
    assert tx.entry_price == 100.0
    assert tx.exit_ts == exit_threshold
    assert tx.exit_price == 130.0


def test_leg_is_empty_when_no_preannouncement_entry_exists() -> None:
    pump, cp = _pump()
    prices = pd.Series(
        [90.0, 110.0],
        index=[pump.time - timedelta(minutes=16), pump.time],
    )

    tx = _manager().regular_transaction(ts_price=prices, pump=pump, cp=cp)

    assert tx.is_empty()
