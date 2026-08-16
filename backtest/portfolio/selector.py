from typing import List

import numpy as np
import pandas as pd

from backtest.pipelines.BaseModel import ImplementsRank
from backtest.portfolio.BasePortfolio import Portfolio
from backtest.utils.columns import COL_CURRENCY_PAIR, COL_PROBAS_PRED
from backtest.utils.sample import Dataset
from core.currency_pair import CurrencyPair


class TopKPortfolioSelector:
    """
    Builds an equal-weight Top@k portfolio from model scores.
    """

    def __init__(self, portfolio_size: int):
        if portfolio_size <= 0:
            raise ValueError("portfolio_size must be positive")
        self.portfolio_size: int = portfolio_size

    def select_portfolio(self, model: ImplementsRank, cross_section: Dataset) -> Portfolio:
        """
        Rank the cross-section and pick the Top@k assets with equal weights.
        """
        scores: np.ndarray = model.rank(dataset=cross_section)
        scored_df = cross_section.all_data().copy()
        scored_df[COL_PROBAS_PRED] = scores

        df_portfolio: pd.DataFrame = scored_df.sort_values(
            by=[COL_PROBAS_PRED, COL_CURRENCY_PAIR],
            ascending=[False, True],
            kind="stable",
        ).iloc[: self.portfolio_size]
        weights = np.full(shape=(len(df_portfolio),), fill_value=1.0 / len(df_portfolio))
        currency_pairs: List[CurrencyPair] = [
            CurrencyPair.from_string(symbol=symbol) for symbol in df_portfolio[COL_CURRENCY_PAIR]
        ]
        return Portfolio(currency_pairs=currency_pairs, weights=weights)
