from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta, datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import polars as pl

from backtest.pipelines.BaseModel import ImplementsRank
from backtest.portfolio.pnl import PnLCalculator, USDTPnLCalculator
from backtest.utils.sample import Dataset
from core.columns import SYMBOL, DATE, TRADE_TIME, PRICE, QUANTITY, IS_BUYER_MAKER
from core.currency_pair import CurrencyPair
from core.exchange import Exchange
from core.pump_event import PumpEvent
from core.time_utils import Bounds


@dataclass
class Portfolio:
    currency_pairs: List[CurrencyPair]
    weights: np.ndarray[float]

    def __repr__(self) -> str:
        return f"Portfolio: {dict(zip(self.currency_pairs, self.weights.round(4)))}"

    def get_weight(self, cp: CurrencyPair) -> float:
        """Return weight of the asset in the portfolio"""
        return self.weights[self.currency_pairs.index(cp)]


@dataclass
class Transaction:
    currency_pair: CurrencyPair
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    entry_ts: Optional[datetime] = None
    exit_ts: Optional[datetime] = None
    intended_notional_quote: Optional[float] = None
    entry_filled_notional_quote: Optional[float] = None
    exit_filled_notional_quote: Optional[float] = None
    entry_filled_notional_usdt: Optional[float] = None
    exit_filled_notional_usdt: Optional[float] = None
    entry_impact_bps: float = 0.0
    exit_impact_bps: float = 0.0
    entry_impact_num_bars: int = 0
    exit_impact_num_bars: int = 0

    @property
    def transaction_return(self) -> float:
        """
        Net fractional return for one round-trip transaction.

        Includes a fixed 25 bps round-trip trading cost.
        """
        assert self.entry_price is not None and self.exit_price is not None
        return self.exit_price / self.entry_price - 1 - 0.0025

    @classmethod
    def empty(cls, currency_pair: CurrencyPair) -> "Transaction":
        return Transaction(currency_pair=currency_pair)

    def is_empty(self) -> bool:
        return self.entry_price is None and self.exit_price is None


class PortfolioStats:

    def __init__(
        self,
        portfolio: Portfolio,
        txs: List[Transaction],
        pump: PumpEvent,
        pnl_calculator: Optional[PnLCalculator] = None,
    ):
        self.portfolio: Portfolio = portfolio
        self.txs: List[Transaction] = txs
        self.pump: PumpEvent = pump
        self._pnl_calculator: PnLCalculator = pnl_calculator if pnl_calculator is not None else USDTPnLCalculator()

    @property
    def pnl(self) -> float:
        """
        :return: PnL of the portfolio in USDT. Skips assets with no price data.
        """
        return self._pnl_calculator.calculate_portfolio_pnl(portfolio=self.portfolio, txs=self.txs)

    def _weighted_transaction_metric(self, attr_name: str) -> float:
        """
        Return the portfolio-weighted average of a transaction metric.
        """
        total_weighted_value: float = 0.0
        total_weight: float = 0.0
        for tx in self.txs:
            if tx.is_empty():
                continue
            value = getattr(tx, attr_name)
            if value is None:
                continue
            weight = float(self.portfolio.get_weight(tx.currency_pair))
            total_weighted_value += weight * float(value)
            total_weight += weight
        if np.isclose(total_weight, 0.0):
            return 0.0
        return total_weighted_value / total_weight

    @property
    def mean_entry_impact_bps(self) -> float:
        """
        Portfolio-weighted average entry impact in basis points.
        """
        return self._weighted_transaction_metric("entry_impact_bps")

    @property
    def mean_exit_impact_bps(self) -> float:
        """
        Portfolio-weighted average exit impact in basis points.
        """
        return self._weighted_transaction_metric("exit_impact_bps")

    @property
    def mean_entry_impact_num_bars(self) -> float:
        """
        Portfolio-weighted average number of bars behind the entry impact model.
        """
        return self._weighted_transaction_metric("entry_impact_num_bars")

    @property
    def mean_exit_impact_num_bars(self) -> float:
        """
        Portfolio-weighted average number of bars behind the exit impact model.
        """
        return self._weighted_transaction_metric("exit_impact_num_bars")

    @property
    def executed_notional_usdt(self) -> float:
        """
        Portfolio-weighted executed notional in USDT at entry.
        """
        total_executed_notional_usdt: float = 0.0
        for tx in self.txs:
            if tx.is_empty() or tx.entry_filled_notional_usdt is None:
                continue
            total_executed_notional_usdt += float(self.portfolio.get_weight(tx.currency_pair)) * float(
                tx.entry_filled_notional_usdt
            )
        return total_executed_notional_usdt

    def has_pump(self) -> bool:
        """
        :return: True if the portfolio has a pump
        """
        return self.pump.currency_pair in self.portfolio.currency_pairs

    def __repr__(self) -> str:
        return (
            f"PortfolioStats:\nCorresponding pump:{self.pump.as_pump_hash()}\nPnL:{self.pnl}\nHasPump:{self.has_pump()}"
        )


class ImplementsPortfolio(ABC):

    def __init__(self, model: ImplementsRank, buy_before: timedelta, sell_after: timedelta):
        self.model: ImplementsRank = model
        self._hive: pl.LazyFrame = pl.scan_parquet(Exchange.BINANCE_SPOT.get_hive_location(), hive_partitioning=True)
        self._buy_before: timedelta = buy_before
        self._sell_after: timedelta = sell_after
        self._pnl_calculator: PnLCalculator = USDTPnLCalculator()

    @abstractmethod
    def create_portfolio(self, cross_section: Dataset) -> Portfolio: ...

    def load_price_ts(self, bounds: Bounds, currency_pair: CurrencyPair) -> pd.Series:
        """Loads price time series for a given currency pair"""
        data: pd.DataFrame = self.load_trades(bounds=bounds, currency_pair=currency_pair)
        return pd.Series(data=data[PRICE].values, index=data[TRADE_TIME])

    def load_trades(self, bounds: Bounds, currency_pair: CurrencyPair) -> pd.DataFrame:
        """Loads raw trades for a given currency pair and time bounds"""
        return (
            self._hive.filter(
                (pl.col(SYMBOL) == currency_pair.name)
                & (pl.col(DATE).is_between(bounds.day0, bounds.day1))
                & (pl.col(TRADE_TIME) >= bounds.start_inclusive)
                & (pl.col(TRADE_TIME) < bounds.end_exclusive)
            )
            .collect()
            .sort(by=TRADE_TIME)
            .select([TRADE_TIME, PRICE, QUANTITY, IS_BUYER_MAKER])
            .to_pandas()
        )

    @abstractmethod
    def regular_transaction(self, ts_price: pd.Series, pump: PumpEvent, cp: CurrencyPair) -> Transaction:
        """
        Define when we entry and exit for the regular asset which is not manipulated
        """

    @abstractmethod
    def pumped_transaction(self, ts_price: pd.Series, pump: PumpEvent, cp: CurrencyPair) -> Transaction:
        """
        Define when we exit and when we can enter for the manipulated asset
        """

    def _create_cross_section_transactions(self, pump: PumpEvent, portfolio: Portfolio) -> List[Transaction]:
        """
        Returns a list of transactions for cross-section
        """
        bounds: Bounds = Bounds(pump.time - timedelta(days=1), pump.time + timedelta(days=1))
        txs: List[Transaction] = []

        for cp in portfolio.currency_pairs:
            ts_price: pd.Series = self.load_price_ts(bounds=bounds, currency_pair=cp)
            tx: Transaction
            if not pump.is_manipulated(cp):
                tx = self.regular_transaction(ts_price=ts_price, pump=pump, cp=cp)
            else:
                tx = self.pumped_transaction(ts_price=ts_price, pump=pump, cp=cp)
            txs.append(tx)

        return txs

    def evaluate_for_pump(self, dataset: Dataset, pump: PumpEvent) -> PortfolioStats:
        """
        :params cross_section: a cross-section dataframe containing all features needed for model to make predictions
        :params pump: pump event of the current cross-section
        :returns: return of the portfolio selected by the model and corresponding portfolio
        """
        cross_section: Dataset = dataset.get_cross_section(pump=pump)
        portfolio: Portfolio = self.create_portfolio(cross_section=cross_section)
        txs: List[Transaction] = self._create_cross_section_transactions(pump=pump, portfolio=portfolio)
        return PortfolioStats(
            portfolio=portfolio,
            txs=txs,
            pump=pump,
            pnl_calculator=self._pnl_calculator,
        )

    def compute_overall_pnl(self, dataset: Dataset) -> float:
        """Compute overall pnl for self.model on provided dataset: Dataset"""
        pnls: List[float] = []
        for pump in dataset.get_pumps():
            stats: PortfolioStats = self.evaluate_for_pump(dataset=dataset, pump=pump)
            pnls.append(stats.pnl)
        return np.array(pnls).sum()
