from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import backtest.utils.evaluation as evaluation
from backtest.utils.evaluation import (
    align_portfolio_returns_with_btc,
    compute_btc_buy_and_hold_statistics,
    compute_portfolio_statistics,
    get_btc_buy_and_hold_baseline,
    select_best_experiment_on_validation,
)
from backtest.utils.sample import DatasetType
from core.currency_pair import CurrencyPair
from core.exchange import Exchange
from core.pump_event import PumpEvent


class _Dataset:
    def __init__(self, pumps: list[PumpEvent]):
        self._pumps = pumps

    def get_pumps(self) -> list[PumpEvent]:
        return self._pumps


def test_btc_buy_and_hold_increments_sum_to_simple_full_period_return(monkeypatch) -> None:
    start = datetime(2022, 1, 1, 12)
    pumps = [
        PumpEvent(
            currency_pair=CurrencyPair.from_string("AAA-BTC"),
            time=start + timedelta(days=i),
            exchange=Exchange.BINANCE_SPOT,
        )
        for i in range(3)
    ]
    benchmark_start = pumps[0].time - timedelta(minutes=15)
    benchmark_end = pumps[-1].time + timedelta(minutes=1)
    prices = {
        benchmark_start: 100.0,
        datetime(2022, 1, 2): 110.0,
        datetime(2022, 1, 3): 90.0,
        benchmark_end: 95.0,
    }
    requested_timestamps: list[datetime] = []

    class _PriceProvider:
        def get_indicative_price(self, symbol: str, ts: datetime) -> float:
            assert symbol == "BTC-USDT"
            requested_timestamps.append(ts)
            return prices[ts]

    monkeypatch.setattr(evaluation, "IndicativePriceProvider", _PriceProvider)

    result = get_btc_buy_and_hold_baseline(_Dataset(pumps))  # type: ignore[arg-type]

    assert requested_timestamps[0] == benchmark_start
    assert requested_timestamps[-1] == benchmark_end
    assert np.allclose(result["portfolio_return"].to_numpy(), [0.1, -0.2, 0.05])
    assert np.isclose(result["portfolio_return"].sum(), 95.0 / 100.0 - 1.0)
    assert result.attrs["start_time"] == benchmark_start
    assert result.attrs["end_time"] == benchmark_end


def test_portfolio_and_btc_returns_are_aligned_on_one_calendar() -> None:
    portfolio = pd.DataFrame(
        {"K=1": [0.1, 0.2, -0.1]},
        index=pd.to_datetime(["2022-01-01 12:00", "2022-01-01 18:00", "2022-01-03 12:00"]),
    )
    btc = pd.DataFrame(
        {
            "portfolio_return": [0.03, -0.01],
            "market_return": [0.03, -0.01 / 1.03],
            "btc_price": [103.0, 102.0],
            "period_days": [1.0, 1.0],
        },
        index=pd.to_datetime(["2022-01-02", "2022-01-03"]),
    )
    btc.attrs.update({"start_time": datetime(2022, 1, 1, 12), "end_time": datetime(2022, 1, 3, 12)})

    aligned = align_portfolio_returns_with_btc(portfolio, btc)

    assert aligned.index.tolist() == list(pd.date_range("2022-01-01", "2022-01-03"))
    assert np.allclose(aligned["K=1"].to_numpy(), [0.3, 0.0, -0.1])
    assert np.allclose(aligned["BTCUSDT buy-and-hold"].to_numpy(), [0.0, 0.03, -0.01])


def test_btc_statistics_use_exact_period_return_and_full_day_volatility() -> None:
    btc = pd.DataFrame(
        {
            "portfolio_return": [0.01, -0.02, 0.005],
            "market_return": [0.01, -0.02 / 1.01, 0.005 / 0.99],
            "btc_price": [101.0, 99.0, 99.5],
            "period_days": [0.5, 1.0, 0.5],
        }
    )
    btc.attrs.update({"start_time": datetime(2022, 1, 1), "end_time": datetime(2022, 1, 3)})

    stats = compute_btc_buy_and_hold_statistics(btc)

    assert np.isclose(float(stats["cumulative return"]), -0.005)
    assert np.isclose(float(stats["annualized return"]), -0.005 * 365 / 2)
    assert np.isnan(float(stats["annualized volatility"]))
    assert np.isnan(float(stats["Sharpe ratio"]))


def test_portfolio_average_return_is_computed_per_event_not_per_active_day() -> None:
    index = pd.to_datetime(["2022-01-01 10:00", "2022-01-01 12:00", "2022-01-02 10:00"])
    equity = pd.DataFrame({"strategy": [0.1, 0.2, -0.1]}, index=index)

    stats = compute_portfolio_statistics(equity)

    assert np.isclose(stats.loc["strategy", "average event return"], (0.1 + 0.2 - 0.1) / 3)


def test_model_selection_uses_validation_only(monkeypatch) -> None:
    requested_splits: list[DatasetType] = []

    class _Sample:
        def get_dataset(self, ds_type: DatasetType) -> object:
            requested_splits.append(ds_type)
            return object()

    class _Experiment:
        def __init__(self, name: str, score: float):
            self.name = name
            self.score = score
            self.sample = _Sample()

        def get_experiment_name(self) -> str:
            return self.name

        def get_sample(self) -> _Sample:
            return self.sample

        def get_model(self) -> float:
            return self.score

    experiments = [_Experiment("first", 0.4), _Experiment("best", 0.8), _Experiment("third", 0.6)]
    monkeypatch.setattr(evaluation, "calculate_topk_percent_auc", lambda model, dataset: model)

    selected, ranking = select_best_experiment_on_validation(experiments)  # type: ignore[arg-type]

    assert selected is experiments[1]
    assert ranking.index.tolist() == ["best", "third", "first"]
    assert requested_splits == [DatasetType.VALIDATION] * len(experiments)
