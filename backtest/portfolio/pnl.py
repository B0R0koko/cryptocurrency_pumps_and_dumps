from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from backtest.portfolio.BasePortfolio import Portfolio, Transaction


class PnLCalculator(ABC):
    @abstractmethod
    def calculate_transaction_pnl(self, tx: "Transaction") -> float: ...

    def calculate_portfolio_pnl(self, portfolio: "Portfolio", txs: List["Transaction"]) -> float:
        """Aggregate weighted transaction PnL for non-empty legs in a portfolio."""
        pnl: float = 0.0
        for tx in txs:
            if tx.is_empty():
                continue
            pnl += self.calculate_transaction_pnl(tx) * portfolio.get_weight(tx.currency_pair)
        return pnl


class USDTPnLCalculator(PnLCalculator):
    """
    Computes transaction PnL in USDT when conversion metadata is available.
    """

    def calculate_transaction_pnl(self, tx: "Transaction") -> float:
        """
        Compute transaction PnL in USDT.

        When execution metadata is present, PnL is the USDT value of the fully
        liquidated exit proceeds minus the entry investment and a 25 bp fee on
        entry capital. This includes the BTC/USDT move over the holding period
        instead of mixing a BTC return with an exit-time USDT conversion.
        """
        if (
            tx.entry_filled_notional_usdt is not None
            and tx.exit_filled_notional_usdt is not None
            and tx.entry_filled_notional_usdt > 0
        ):
            fee_usdt: float = 0.0025 * tx.entry_filled_notional_usdt
            return tx.exit_filled_notional_usdt - tx.entry_filled_notional_usdt - fee_usdt
        return tx.transaction_return
