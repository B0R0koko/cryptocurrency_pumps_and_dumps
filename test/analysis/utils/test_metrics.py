from typing import List

import numpy as np
import pandas as pd

from backtest.pipelines.BaseModel import ImplementsRank
from backtest.utils.columns import COL_PROBAS_PRED, COL_PUMP_HASH, COL_IS_PUMPED
from backtest.utils.feature_set import FeatureSet
from backtest.utils.metrics import (
    calculate_topk_percent,
    calculate_topk,
    calculate_f1,
    calculate_pr_auc,
    calculate_balanced_accuracy,
)
from backtest.utils.sample import Dataset, DatasetType
from core.utils import configure_logging

_test_data: pd.DataFrame = pd.DataFrame(
    data=[
        # Cross-section 1
        (0.07, "pump#1", 0),
        (0.08, "pump#1", 0),
        (0.09, "pump#1", 0),
        (0.1, "pump#1", 1),
        (0.2, "pump#1", 0),
        (0.3, "pump#1", 0),
        # Cross-section 2
        (0.4, "pump#2", 0),
        (0.5, "pump#2", 0),
        (0.6, "pump#2", 0),
        (0.7, "pump#2", 0),
        (0.8, "pump#2", 1),
        (0.9, "pump#2", 0),
    ],
    columns=[COL_PROBAS_PRED, COL_PUMP_HASH, COL_IS_PUMPED],
)

_perfect_data: pd.DataFrame = pd.DataFrame(
    data=[
        # Cross-section 1
        (0.9, "pump#1", 1),
        (0.8, "pump#1", 0),
        (0.7, "pump#1", 0),
        # Cross-section 2
        (0.95, "pump#2", 1),
        (0.4, "pump#2", 0),
        (0.3, "pump#2", 0),
    ],
    columns=[COL_PROBAS_PRED, COL_PUMP_HASH, COL_IS_PUMPED],
)

feature_set: FeatureSet = FeatureSet(
    numeric_features=[],
    target=COL_IS_PUMPED,
    categorical_features=None,
    eval_fields=[COL_PUMP_HASH],
)


class DummyModel(ImplementsRank):

    def rank(self, dataset: Dataset) -> pd.Series:
        return _test_data[COL_PROBAS_PRED].values


class DummyPerfectModel(ImplementsRank):

    def rank(self, dataset: Dataset) -> pd.Series:
        return _perfect_data[COL_PROBAS_PRED].values


def test_calculate_topk():
    """
    Given the data above
    TOP1 = 0, TOP2 = 1/2, TOP5 = 1
    """
    configure_logging()
    topk_bins: List[int] = [1, 2, 5]

    dataset: Dataset = Dataset(data=_test_data, ds_type=DatasetType.TEST, feature_set=feature_set)
    model: DummyModel = DummyModel()
    topks: pd.Series = calculate_topk(model=model, dataset=dataset, bins=topk_bins)

    assert np.allclose(topks.values, [0, 0.5, 1], atol=1e-5)


def test_calculate_topk_percent():
    """
    Given the data above
    TOP10% = 10% of the data is at least 1 element => TOP10% = TOP1 = 0
    TOP20% = 20% of the data is 2 elements => TOP20% = 0.5
    TOP50% = 50% of the data is 3 elements => TOP50% = 1
    """
    topk_bins: List[float] = [0.1, 0.2, 0.5]

    dataset: Dataset = Dataset(data=_test_data, ds_type=DatasetType.TEST, feature_set=feature_set)
    model: DummyModel = DummyModel()
    topks: pd.Series = calculate_topk_percent(model=model, dataset=dataset, bins=topk_bins)

    assert np.allclose(topks.values, [0, 0.5, 1], atol=1e-5)


def test_calculate_f1_and_balanced_accuracy_with_top1_rule() -> None:
    dataset: Dataset = Dataset(data=_test_data, ds_type=DatasetType.TEST, feature_set=feature_set)
    model: DummyModel = DummyModel()

    f1: float = calculate_f1(model=model, dataset=dataset, decision_rule="top1_per_cross_section")
    balanced_acc: float = calculate_balanced_accuracy(
        model=model, dataset=dataset, decision_rule="top1_per_cross_section"
    )

    assert np.isclose(f1, 0.0)
    assert np.isclose(balanced_acc, 0.4)


def test_calculate_f1_with_threshold_rule() -> None:
    dataset: Dataset = Dataset(data=_test_data, ds_type=DatasetType.TEST, feature_set=feature_set)
    model: DummyModel = DummyModel()

    f1: float = calculate_f1(model=model, dataset=dataset, decision_rule="threshold", threshold=0.5)
    balanced_acc: float = calculate_balanced_accuracy(
        model=model, dataset=dataset, decision_rule="threshold", threshold=0.5
    )

    assert np.isclose(f1, 2 / 7)
    assert np.isclose(balanced_acc, 0.55)


def test_calculate_pr_auc_and_classification_metrics_for_perfect_ranking() -> None:
    dataset: Dataset = Dataset(data=_perfect_data, ds_type=DatasetType.TEST, feature_set=feature_set)
    model: DummyPerfectModel = DummyPerfectModel()

    pr_auc: float = calculate_pr_auc(model=model, dataset=dataset)
    f1: float = calculate_f1(model=model, dataset=dataset, decision_rule="top1_per_cross_section")
    balanced_acc: float = calculate_balanced_accuracy(
        model=model, dataset=dataset, decision_rule="top1_per_cross_section"
    )

    assert np.isclose(pr_auc, 1.0)
    assert np.isclose(f1, 1.0)
    assert np.isclose(balanced_acc, 1.0)


def test_calculate_f1_with_top1_rule_handles_non_contiguous_index() -> None:
    df_non_contiguous: pd.DataFrame = _test_data.copy()
    df_non_contiguous.index = np.arange(1000, 1000 + len(df_non_contiguous))

    dataset: Dataset = Dataset(data=df_non_contiguous, ds_type=DatasetType.TEST, feature_set=feature_set)
    model: DummyModel = DummyModel()
    f1: float = calculate_f1(model=model, dataset=dataset, decision_rule="top1_per_cross_section")

    assert np.isclose(f1, 0.0)


def test_calculate_topk_does_not_mutate_source_dataframe() -> None:
    """Regression: previously assigned COL_PROBAS_PRED onto ``dataset.all_data()``,
    mutating the caller's frame. The metric must operate on a copy."""
    # Build a source frame that does NOT already contain COL_PROBAS_PRED so we can
    # observe whether the metric leaks a prediction column onto the caller.
    source: pd.DataFrame = _test_data.copy().rename(columns={COL_PROBAS_PRED: "raw_score"})
    dataset: Dataset = Dataset(data=source, ds_type=DatasetType.TEST, feature_set=feature_set)

    class ExternalScoreModel(ImplementsRank):
        def rank(self, dataset: Dataset) -> pd.Series:
            return dataset.all_data()["raw_score"].to_numpy()

    _ = calculate_topk(model=ExternalScoreModel(), dataset=dataset, bins=[1, 2, 5])
    _ = calculate_topk_percent(model=ExternalScoreModel(), dataset=dataset, bins=[0.1, 0.2, 0.5])

    assert COL_PROBAS_PRED not in source.columns
    assert COL_PROBAS_PRED not in dataset.all_data().columns


def _make_dataset_with_missing_pump_cross_section() -> Dataset:
    """Two cross-sections, one with the pumped row, one without any pump at all."""
    frame: pd.DataFrame = pd.DataFrame(
        data=[
            (0.9, "pump#has-pump", 1),
            (0.5, "pump#has-pump", 0),
            (0.4, "pump#has-pump", 0),
            (0.8, "pump#no-pump", 0),
            (0.6, "pump#no-pump", 0),
            (0.3, "pump#no-pump", 0),
        ],
        columns=[COL_PROBAS_PRED, COL_PUMP_HASH, COL_IS_PUMPED],
    )
    return Dataset(data=frame, ds_type=DatasetType.TEST, feature_set=feature_set)


def test_calculate_topk_denominator_uses_cross_sections_with_pump_not_pumped_row_count() -> None:
    """Regression: TOP1 should be 1.0 when the only cross-section containing a pump
    ranks it first, regardless of how many additional pump-free cross-sections
    are appended."""
    dataset: Dataset = _make_dataset_with_missing_pump_cross_section()

    class TopScoreModel(ImplementsRank):
        def rank(self, dataset: Dataset) -> pd.Series:
            return dataset.all_data()[COL_PROBAS_PRED].to_numpy()

    topk: pd.Series = calculate_topk(model=TopScoreModel(), dataset=dataset, bins=[1, 2])
    assert np.isclose(topk.loc[1], 1.0)
    assert np.isclose(topk.loc[2], 1.0)


def test_calculate_topk_percent_denominator_uses_cross_sections_with_pump() -> None:
    dataset: Dataset = _make_dataset_with_missing_pump_cross_section()

    class TopScoreModel(ImplementsRank):
        def rank(self, dataset: Dataset) -> pd.Series:
            return dataset.all_data()[COL_PROBAS_PRED].to_numpy()

    topk_pct: pd.Series = calculate_topk_percent(model=TopScoreModel(), dataset=dataset, bins=[0.34, 0.67])
    # 34% of 3 rows = 2 rows; top-2 in the pumped cross-section contains the pump.
    assert np.isclose(topk_pct.loc[0.34], 1.0)
    assert np.isclose(topk_pct.loc[0.67], 1.0)
