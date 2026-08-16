from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import backtest.utils.evaluation as evaluation
from backtest.utils.evaluation import (
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
    start = datetime(2022, 1, 1)
    pumps = [
        PumpEvent(
            currency_pair=CurrencyPair.from_string("AAA-BTC"),
            time=start + timedelta(days=i),
            exchange=Exchange.BINANCE_SPOT,
        )
        for i in range(3)
    ]
    prices = dict(zip((pump.time for pump in pumps), [100.0, 110.0, 90.0]))

    class _PriceProvider:
        def get_indicative_price(self, symbol: str, ts: datetime) -> float:
            return prices[ts]

    monkeypatch.setattr(evaluation, "IndicativePriceProvider", _PriceProvider)

    result = get_btc_buy_and_hold_baseline(_Dataset(pumps))  # type: ignore[arg-type]

    assert np.allclose(result["portfolio_return"].to_numpy(), [0.0, 0.1, -0.2])
    assert np.isclose(result["portfolio_return"].sum(), 90.0 / 100.0 - 1.0)


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
