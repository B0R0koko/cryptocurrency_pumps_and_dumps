"""Regression tests for the CatBoost custom TOPKAUC metric.

The kernel used inside catboost's early-stopping loop must agree with
:func:`backtest.utils.metrics.calculate_topk_percent_auc` and must correctly
disambiguate which pool (train vs val) is being scored on any given call.
"""

from typing import List

import numpy as np
import pandas as pd

from backtest.pipelines.BaseModel import ImplementsRank
from backtest.pipelines.CatboostClassifierTOPKAUC.pipe import (
    _BINS,
    _MAX_K_PERCENT,
    TOPKAUCMetric,
    _precompute_groups,
    _topkpauc_kernel,
)
from backtest.utils.columns import COL_IS_PUMPED, COL_PROBAS_PRED, COL_PUMP_HASH
from backtest.utils.feature_set import FeatureSet
from backtest.utils.metrics import calculate_topk_percent_auc
from backtest.utils.sample import Dataset, DatasetType

_FEATURE_SET: FeatureSet = FeatureSet(
    numeric_features=["score"],
    target=COL_IS_PUMPED,
    categorical_features=None,
    eval_fields=[COL_PUMP_HASH],
)


class _ScoreColumnModel(ImplementsRank):
    def rank(self, dataset: Dataset) -> np.ndarray:
        return dataset.all_data()["score"].to_numpy()


def _make_cross_sections(
    n_cross_sections: int,
    rows_per_cross_section: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Random scores per cross-section with the pumped row placed at rank 1."""
    rng = np.random.default_rng(seed)
    rows: List[dict] = []
    for cs_idx in range(n_cross_sections):
        scores = rng.uniform(size=rows_per_cross_section)
        pumped_row = int(rng.integers(0, rows_per_cross_section))
        # Give the pumped asset a mid-to-high score so bins > 0% hit at some cutoff.
        scores[pumped_row] = 0.5 + 0.5 * rng.random()
        for i in range(rows_per_cross_section):
            rows.append(
                {
                    COL_PUMP_HASH: f"pump-{cs_idx}",
                    COL_IS_PUMPED: i == pumped_row,
                    "score": float(scores[i]),
                }
            )
    return pd.DataFrame(rows)


def test_topkpauc_kernel_matches_calculate_topk_percent_auc() -> None:
    """The numba kernel used by catboost must produce the exact same value as
    the pure-python metric helper on the same cross-sections."""
    df = _make_cross_sections(n_cross_sections=25, rows_per_cross_section=100, seed=17)

    sort_idx, is_pumped_sorted, group_starts, num_cs_with_pump = _precompute_groups(df)
    scores_sorted = df["score"].to_numpy()[sort_idx]
    raw_auc = _topkpauc_kernel(
        scores_by_group=scores_sorted,
        is_pumped_by_group=is_pumped_sorted,
        group_starts=group_starts,
        bins=_BINS,
        num_cross_sections_with_pump=num_cs_with_pump,
    )
    kernel_metric = raw_auc / _MAX_K_PERCENT

    dataset = Dataset(data=df, feature_set=_FEATURE_SET, ds_type=DatasetType.TEST)
    reference_metric = calculate_topk_percent_auc(model=_ScoreColumnModel(), dataset=dataset)

    assert np.isclose(kernel_metric, reference_metric)


def test_topkpauc_kernel_denominator_uses_cross_sections_with_pump() -> None:
    """Regression: previously the kernel divided by the number of pumped rows.

    When a cross-section carries two pumped rows the two denominators diverge:
    ``calculate_topk_percent`` uses the number of pump-carrying cross-sections
    (in this data: 2) while the buggy kernel used ``is_pumped.sum()`` (which is
    3 here). The correct denominator is now 2 and matches the metrics helper.
    """
    # 3 cross-sections: A has one pumped row, B has NO pumped row, C has TWO
    # pumped rows (deliberately breaking the one-per-cross-section invariant so
    # the denominator semantics matter).
    df = pd.DataFrame(
        [
            {COL_PUMP_HASH: "A", COL_IS_PUMPED: True, "score": 0.9},
            {COL_PUMP_HASH: "A", COL_IS_PUMPED: False, "score": 0.2},
            {COL_PUMP_HASH: "A", COL_IS_PUMPED: False, "score": 0.1},
            {COL_PUMP_HASH: "B", COL_IS_PUMPED: False, "score": 0.8},
            {COL_PUMP_HASH: "B", COL_IS_PUMPED: False, "score": 0.4},
            {COL_PUMP_HASH: "B", COL_IS_PUMPED: False, "score": 0.3},
            {COL_PUMP_HASH: "C", COL_IS_PUMPED: True, "score": 0.95},
            {COL_PUMP_HASH: "C", COL_IS_PUMPED: True, "score": 0.7},
            {COL_PUMP_HASH: "C", COL_IS_PUMPED: False, "score": 0.05},
        ]
    )
    _, _, _, num_cs_with_pump = _precompute_groups(df)
    # Only cross-sections A and C carry a pump, so denominator must be 2
    # (the previous implementation would have set it to 3 = is_pumped.sum()).
    assert num_cs_with_pump == 2

    # The two implementations must return the same value now.
    dataset = Dataset(data=df, feature_set=_FEATURE_SET, ds_type=DatasetType.TEST)
    reference_metric = calculate_topk_percent_auc(model=_ScoreColumnModel(), dataset=dataset)

    sort_idx, is_pumped_sorted, group_starts, num_cs = _precompute_groups(df)
    scores_sorted = df["score"].to_numpy()[sort_idx]
    raw_auc = _topkpauc_kernel(
        scores_by_group=scores_sorted,
        is_pumped_by_group=is_pumped_sorted,
        group_starts=group_starts,
        bins=_BINS,
        num_cross_sections_with_pump=num_cs,
    )
    kernel_metric = raw_auc / _MAX_K_PERCENT
    assert np.isclose(kernel_metric, reference_metric)


def test_topkpauc_metric_dispatches_correctly_when_train_and_val_have_equal_length() -> None:
    """Regression: the previous implementation checked
    ``probas_pred.shape[0] == self._val_len`` and, when train_len == val_len,
    silently evaluated train predictions against val labels. The dispatcher
    must fall back to comparing the label vector."""
    df_train = pd.DataFrame(
        [
            {COL_PUMP_HASH: "train-A", COL_IS_PUMPED: True, COL_PROBAS_PRED: 0.9},
            {COL_PUMP_HASH: "train-A", COL_IS_PUMPED: False, COL_PROBAS_PRED: 0.4},
            {COL_PUMP_HASH: "train-B", COL_IS_PUMPED: True, COL_PROBAS_PRED: 0.95},
            {COL_PUMP_HASH: "train-B", COL_IS_PUMPED: False, COL_PROBAS_PRED: 0.5},
        ]
    )
    df_val = pd.DataFrame(
        [
            {COL_PUMP_HASH: "val-A", COL_IS_PUMPED: False, COL_PROBAS_PRED: 0.6},
            {COL_PUMP_HASH: "val-A", COL_IS_PUMPED: True, COL_PROBAS_PRED: 0.3},
            {COL_PUMP_HASH: "val-B", COL_IS_PUMPED: False, COL_PROBAS_PRED: 0.7},
            {COL_PUMP_HASH: "val-B", COL_IS_PUMPED: True, COL_PROBAS_PRED: 0.2},
        ]
    )
    assert len(df_train) == len(df_val)

    metric = TOPKAUCMetric(df_train=df_train, df_val=df_val)

    # Train pool: pumped row ranks first in every cross-section. Correct
    # dispatch produces a strictly higher metric than dispatching the same
    # scores against the val labels (where the pumped rows sit at the bottom).
    train_scores = df_train[COL_PROBAS_PRED].to_numpy()
    train_targets = df_train[COL_IS_PUMPED].to_numpy()
    train_metric, _ = metric.evaluate(approxes=[train_scores], target=train_targets, weight=None)

    val_scores = df_val[COL_PROBAS_PRED].to_numpy()
    val_targets = df_val[COL_IS_PUMPED].to_numpy()
    val_metric, _ = metric.evaluate(approxes=[val_scores], target=val_targets, weight=None)

    # Train metric must be strictly better than the val metric, which is only
    # possible if the dispatcher uses the labels to disambiguate. The buggy
    # length-only dispatcher would evaluate both against the val labels and
    # return the val (low) score for both calls.
    assert train_metric > val_metric
    assert val_metric < 0.5


def test_topkpauc_metric_matches_calculate_topk_percent_auc_on_val_pool() -> None:
    df_train = _make_cross_sections(n_cross_sections=6, rows_per_cross_section=80, seed=3)
    df_val = _make_cross_sections(n_cross_sections=8, rows_per_cross_section=80, seed=5)

    metric = TOPKAUCMetric(df_train=df_train, df_val=df_val)
    val_scores = df_val["score"].to_numpy()
    val_targets = df_val[COL_IS_PUMPED].to_numpy()
    metric_value, _ = metric.evaluate(approxes=[val_scores], target=val_targets, weight=None)

    dataset = Dataset(data=df_val, feature_set=_FEATURE_SET, ds_type=DatasetType.TEST)
    reference = calculate_topk_percent_auc(model=_ScoreColumnModel(), dataset=dataset)
    assert np.isclose(metric_value, reference)
