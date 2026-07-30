from datetime import timedelta
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from backtest.pipelines.BaseModel import ImplementsRank
from backtest.portfolio.BasePortfolio import PortfolioStats
from backtest.portfolio.TOPKPortfolio import TOPKPortfolio
from backtest.utils.experiment import Experiment
from backtest.utils.IndicativePriceProvider import IndicativePriceProvider
from backtest.utils.metrics import (
    calculate_balanced_accuracy,
    calculate_f1,
    calculate_pr_auc,
    calculate_topk,
    calculate_topk_percent,
    calculate_topk_percent_auc,
)
from backtest.utils.sample import Dataset, DatasetType


def highlight_max(s: pd.Series) -> List[str]:
    """Pandas Styler helper: highlight the maximum value in a Series."""
    is_max = s == s.max()
    return ["background-color: yellow" if v else "" for v in is_max]


def random_topk_baseline(dataset: Dataset, bins: Sequence[int] = (1, 2, 3, 5, 10, 20, 30)) -> List[float]:
    """Expected Top@k accuracy of a random classifier."""
    df_all = dataset.all_data()
    topks = []
    for k in bins:
        p = k / df_all.groupby("pump_hash")["pump_hash"].count()
        topks.append(p.sum() / len(p))
    return topks


def evaluate_experiments_topk(
    experiments: List[Experiment],
    bins: Sequence[int] = (1, 2, 3, 5, 10, 20, 30),
) -> pd.DataFrame:
    """Compute Top@k for every experiment on its TEST split. Returns (bins x experiments) DataFrame."""
    topk_vals: Dict[str, List[float]] = {}
    for experiment in tqdm(experiments, desc="Top@k"):
        name = experiment.get_experiment_name()
        sample = experiment.get_sample()
        model = experiment.get_model()
        vals = calculate_topk(model=model, dataset=sample.get_dataset(DatasetType.TEST), bins=list(bins))
        topk_vals[name] = vals
    return pd.DataFrame(topk_vals)


def evaluate_experiments_topk_percent(
    experiments: List[Experiment],
    bins: Sequence[float] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
) -> pd.DataFrame:
    """Compute Top@k% for every experiment on its TEST split."""
    topkp_vals: Dict[str, List[float]] = {}
    for experiment in tqdm(experiments, desc="Top@k%"):
        name = experiment.get_experiment_name()
        sample = experiment.get_sample()
        model = experiment.get_model()
        vals = calculate_topk_percent(model=model, dataset=sample.get_dataset(DatasetType.TEST), bins=list(bins))
        topkp_vals[name] = vals
    return pd.DataFrame(topkp_vals)


def evaluate_experiments_topk_percent_curves(
    experiments: List[Experiment],
    bins: np.ndarray | None = None,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compute dense Top@k% curves and Top@k%AUC for every experiment.

    Returns (DataFrame of curves, dict of experiment_name -> Top@k%AUC).
    """
    if bins is None:
        bins = np.arange(0, 0.21, 0.01)
    topkp_vals: Dict[str, List[float]] = {}
    auc_scores: Dict[str, float] = {}
    for experiment in tqdm(experiments, desc="Top@k% curves"):
        name = experiment.get_experiment_name()
        sample = experiment.get_sample()
        model = experiment.get_model()
        dataset = sample.get_dataset(DatasetType.TEST)
        vals = calculate_topk_percent(model=model, dataset=dataset, bins=bins)
        topkp_vals[name] = vals
        auc_scores[name] = calculate_topk_percent_auc(model=model, dataset=dataset)
    return pd.DataFrame(topkp_vals), auc_scores


def evaluate_experiments_classification(
    experiments: List[Experiment],
) -> pd.DataFrame:
    """Compute PR-AUC, F1 and Balanced Accuracy for every experiment on TEST."""
    rows: List[Dict[str, float | str]] = []
    for experiment in tqdm(experiments, desc="Classification metrics"):
        name = experiment.get_experiment_name()
        sample = experiment.get_sample()
        model = experiment.get_model()
        dataset = sample.get_dataset(DatasetType.TEST)
        rows.append(
            {
                "Model": name,
                "PR-AUC": calculate_pr_auc(model=model, dataset=dataset),
                "F1 (Top@1 per cross-section)": calculate_f1(
                    model=model, dataset=dataset, decision_rule="top1_per_cross_section"
                ),
                "Balanced Accuracy (Top@1 per cross-section)": calculate_balanced_accuracy(
                    model=model, dataset=dataset, decision_rule="top1_per_cross_section"
                ),
            }
        )
    return pd.DataFrame(rows).set_index("Model")


def get_equity_curve_for_experiment(
    experiment: Experiment,
    portfolio_size: int = 5,
    use_price_impact: bool = False,
    order_notional_usdt: float = 1.0,
) -> pd.DataFrame:
    """Run top-k portfolio backtest for one experiment and return time-indexed PnL series."""
    sample = experiment.get_sample()
    manager = TOPKPortfolio(
        model=experiment.get_model(),
        portfolio_size=portfolio_size,
        use_price_impact=use_price_impact,
        order_notional_usdt=order_notional_usdt,
    )
    dataset = sample.get_dataset(ds_type=DatasetType.TEST)
    results: List[Dict[str, Any]] = []

    for pump in tqdm(
        sorted(dataset.get_pumps()),
        desc=f"Equity curve: {experiment.get_experiment_name()} K={portfolio_size}",
    ):
        stat: PortfolioStats = manager.evaluate_for_pump(dataset=dataset, pump=pump)
        results.append({"time": pump.time, "portfolio_return": stat.pnl})

    return pd.DataFrame(results).set_index("time")


def compute_equity_curves(
    experiments: List[Experiment],
    portfolio_size: int = 5,
    use_price_impact: bool = False,
    order_notional_usdt: float = 1.0,
) -> pd.DataFrame:
    """Compute equity curves for multiple experiments and return a combined DataFrame."""
    curves: Dict[str, pd.Series] = {}
    for experiment in experiments:
        curves[experiment.experiment_name] = get_equity_curve_for_experiment(
            experiment=experiment,
            portfolio_size=portfolio_size,
            use_price_impact=use_price_impact,
            order_notional_usdt=order_notional_usdt,
        )
    df = pd.concat(curves.values(), axis=1)
    df.columns = list(curves.keys())
    return df


def compute_portfolio_statistics(equity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annualized return, volatility, and Sharpe ratio from daily equity returns.

    Parameters
    ----------
    equity_df : pd.DataFrame
        Time-indexed DataFrame where each column is a strategy's per-trade returns.
        Index should be date-like (will be grouped by date).
    """
    df = equity_df.copy()
    df.index = df.index.date
    df = df.groupby(df.index).sum()

    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df_full = df.reindex(full, fill_value=0)

    annualized_returns = df_full.mean() * 365
    annualized_std = df_full.std() * np.sqrt(365)
    mean_trade_return = df.mean()
    sharpe_ratio = annualized_returns / annualized_std

    return pd.DataFrame(
        {
            "trading avg return": mean_trade_return,
            "annualized return": annualized_returns,
            "annualized volatility": annualized_std,
            "Sharpe ratio": sharpe_ratio,
        }
    )


def get_btc_buy_and_hold_baseline(
    dataset: Dataset,
) -> pd.DataFrame:
    """
    Compute BTC buy-and-hold cumulative return over the test set period.

    Buys BTC/USDT at the first pump event time and holds until the last.
    Returns are allocated to each pump event timestamp as the incremental
    BTC return since the previous event, so the cumulative sum matches
    the total BTC return over the period — directly comparable with the
    strategy equity curves.
    """
    price_provider = IndicativePriceProvider()
    pumps = sorted(dataset.get_pumps())

    if len(pumps) < 2:
        return pd.DataFrame(columns=["portfolio_return"])

    # Get BTC price at each pump time
    prices: List[Dict[str, Any]] = []
    for pump in pumps:
        price: Optional[float] = price_provider.get_indicative_price("BTC-USDT", pump.time)
        if price is not None and price > 0:
            prices.append({"time": pump.time, "btc_price": price})

    if len(prices) < 2:
        return pd.DataFrame(columns=["portfolio_return"])

    df = pd.DataFrame(prices).set_index("time").sort_index()

    # Incremental return between consecutive pump events
    # cumsum of these = total BTC return from first to last event
    df["portfolio_return"] = df["btc_price"].pct_change().fillna(0.0)

    return df[["portfolio_return"]]
