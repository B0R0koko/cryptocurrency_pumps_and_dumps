from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

import backtest.pipelines.BasePipeline as base_pipeline_module
from backtest.pipelines.BasePipeline import (
    BasePipeline,
    TRAIN_END,
    VALIDATION_END,
    cross_section_standardisation,
    fillna_with_median_by_cross_section,
)
from backtest.utils.columns import (
    COL_ASSET_RETURN_RANK,
    COL_CURRENCY_PAIR,
    COL_IS_PUMPED,
    COL_PUMP_HASH,
    COL_PUMP_ID,
    COL_PUMP_TIME,
    COL_PUMPED_CURRENCY_PAIR,
)
from backtest.utils.feature_set import FeatureSet
from backtest.utils.sample import DatasetType, Sample
from core.feature_type import FeatureType
from core.time_utils import NamedTimeDelta
from features.FeatureWriter import REGRESSOR_OFFSETS


def _scale_columns() -> list[str]:
    return (
        FeatureType.ASSET_RETURN.col_names(offsets=REGRESSOR_OFFSETS)
        + FeatureType.ASSET_RETURN_ZSCORE.col_names(offsets=REGRESSOR_OFFSETS)
        + FeatureType.QUOTE_ABS_ZSCORE.col_names(offsets=REGRESSOR_OFFSETS)
        + FeatureType.POWERLAW_ALPHA.col_names(offsets=REGRESSOR_OFFSETS)
    )


def test_cross_section_standardisation_skips_constant_features_without_dividing_by_zero() -> None:
    cols = _scale_columns()
    rows: list[dict[str, float]] = []

    for row_idx in range(4):
        pump_id = 0 if row_idx < 2 else 1
        row: dict[str, float] = {COL_PUMP_ID: float(pump_id)}
        for col in cols:
            if pump_id == 0:
                row[col] = 5.0
            else:
                row[col] = 1.0 if row_idx == 2 else 3.0
        rows.append(row)

    df = pd.DataFrame(rows)

    scaled = cross_section_standardisation(df=df)
    representative_col = cols[0]

    # Constant cross-section should be preserved exactly instead of raising.
    assert np.allclose(
        scaled.loc[scaled[COL_PUMP_ID] == 0, representative_col].to_numpy(),
        df.loc[df[COL_PUMP_ID] == 0, representative_col].to_numpy(),
    )

    # Variable cross-section should still be standardized.
    varying_values = scaled.loc[scaled[COL_PUMP_ID] == 1, representative_col].to_numpy()
    assert np.isclose(varying_values.mean(), 0.0)
    assert np.isclose(varying_values[0], -varying_values[1])


class _DummyPipeline(BasePipeline):
    """Minimal pipeline used to drive :meth:`BasePipeline.build_datasets` in tests."""

    def create_sample(self) -> Sample:  # pragma: no cover - not needed for these tests
        raise NotImplementedError

    def train(self, sample: Sample, tuned: bool = True):  # pragma: no cover
        raise NotImplementedError

    def build_model(self):  # pragma: no cover
        raise NotImplementedError

    def optimize_parameters(self):  # pragma: no cover
        raise NotImplementedError


def _make_raw_dataset() -> pd.DataFrame:
    """Three cross-sections: one FAILED train pump, one healthy validation pump,
    one FAILED test pump. Failed means the pumped asset lands at rank >= 10 by
    target return, which the survivorship filter would drop.
    """
    target_col: str = f"target_return@{NamedTimeDelta.ONE_MINUTE.get_slug()}"
    rows: List[Dict[str, object]] = []
    num_assets: int = 12

    def push_cross_section(hash_id: str, pump_time: datetime, pumped_symbol: str, pumped_rank: int) -> None:
        # Rank 1 = highest return. Assign returns 100, 99, ..., 89 to the other 11
        # assets and override the pumped asset's return so it lands at the
        # requested rank.
        pumped_return: float = 100.0 - (pumped_rank - 1)
        # Non-pumped assets fill returns 100, 99, ... skipping the value used
        # for the pumped asset.
        non_pumped_returns: List[float] = [100.0 - i for i in range(num_assets) if i != (pumped_rank - 1)]
        return_idx: int = 0
        for i in range(num_assets):
            symbol: str = f"AST{i:02d}-BTC"
            if symbol == pumped_symbol:
                target_return: float = pumped_return
            else:
                target_return = non_pumped_returns[return_idx]
                return_idx += 1
            rows.append(
                {
                    COL_PUMP_HASH: hash_id,
                    COL_PUMP_TIME: pump_time,
                    COL_CURRENCY_PAIR: symbol,
                    COL_PUMPED_CURRENCY_PAIR: pumped_symbol,
                    COL_IS_PUMPED: symbol == pumped_symbol,
                    target_col: target_return,
                    "feature_a": float(i),
                }
            )

    # Train pump: pumped asset ranked 11 (failed, would be dropped).
    push_cross_section(
        hash_id="pump-train-failed",
        pump_time=TRAIN_END - pd.Timedelta(days=30),
        pumped_symbol="AST05-BTC",
        pumped_rank=11,
    )
    # Validation pump: pumped asset ranked 1 (healthy).
    push_cross_section(
        hash_id="pump-val-good",
        pump_time=TRAIN_END + pd.Timedelta(days=10),
        pumped_symbol="AST00-BTC",
        pumped_rank=1,
    )
    # Test pump: pumped asset ranked 12 (failed) — must stay in TEST.
    push_cross_section(
        hash_id="pump-test-failed",
        pump_time=VALIDATION_END + pd.Timedelta(days=15),
        pumped_symbol="AST07-BTC",
        pumped_rank=12,
    )

    df: pd.DataFrame = pd.DataFrame(rows)
    df[COL_PUMP_ID] = df.groupby(COL_PUMP_HASH, sort=False).ngroup()
    return df


def test_survivorship_filter_drops_failed_train_pump_but_keeps_failed_test_pump(monkeypatch) -> None:
    """Regression: ``remove_failed_pump_cross_sections`` must run on the TRAIN
    slice only, not on the raw frame before ``split_by_time``. A "failed" pump
    that lands in the TEST window must remain visible to reported metrics."""
    raw_df: pd.DataFrame = _make_raw_dataset()

    # Force the raw-dataset cache to return our synthetic frame and bypass the
    # preprocessed cache so we exercise the real ``build_datasets`` path.
    monkeypatch.setattr(base_pipeline_module, "_get_raw_dataset_cached", lambda: raw_df.copy(deep=True))
    monkeypatch.setattr(base_pipeline_module, "_PREPROCESSED_DATASETS_CACHE", {}, raising=False)
    # Avoid expensive cross-section standardisation for this test (features irrelevant).
    monkeypatch.setattr(
        base_pipeline_module,
        "cross_section_standardisation",
        lambda df: df,
    )
    # ``BasePipeline.preprocess_data`` also clips powerlaw columns; supply them.
    for col in FeatureType.POWERLAW_ALPHA.col_names(offsets=REGRESSOR_OFFSETS):
        raw_df[col] = 1.5

    pipeline: _DummyPipeline = _DummyPipeline()
    datasets = pipeline.build_datasets()

    train_hashes = set(datasets[DatasetType.TRAIN][COL_PUMP_HASH].unique())
    validation_hashes = set(datasets[DatasetType.VALIDATION][COL_PUMP_HASH].unique())
    test_hashes = set(datasets[DatasetType.TEST][COL_PUMP_HASH].unique())

    # Failed train pump gets dropped by the survivorship filter.
    assert "pump-train-failed" not in train_hashes
    # Validation pump is healthy so it stays regardless.
    assert "pump-val-good" in validation_hashes
    # Failed TEST pump MUST remain: this is the whole point of the fix.
    assert "pump-test-failed" in test_hashes

    # The transient ranking column added by ``remove_failed_pump_cross_sections``
    # is not left behind on any split.
    for ds in datasets.values():
        assert COL_ASSET_RETURN_RANK not in ds.columns


def test_survivorship_filter_preserves_preexisting_ranker_label_column() -> None:
    """Regression: CatboostRanker's label column shares the name
    ``COL_ASSET_RETURN_RANK``. Since the filter now runs after
    ``preprocess_data`` (which creates that label), it must neither
    overwrite nor drop a pre-existing column of that name."""
    from backtest.pipelines.BasePipeline import remove_failed_pump_cross_sections

    df: pd.DataFrame = _make_raw_dataset()
    label_values: pd.Series = pd.Series(np.linspace(0.0, 1.0, len(df)), index=df.index)
    df[COL_ASSET_RETURN_RANK] = label_values

    filtered: pd.DataFrame = remove_failed_pump_cross_sections(df=df)

    assert COL_ASSET_RETURN_RANK in filtered.columns
    # Surviving rows keep their original label values (index was reset by the filter).
    surviving_mask = df[COL_PUMP_HASH].isin(filtered[COL_PUMP_HASH].unique())
    expected = label_values[surviving_mask].reset_index(drop=True)
    pd.testing.assert_series_equal(filtered[COL_ASSET_RETURN_RANK], expected, check_names=False)


def test_fillna_with_median_by_cross_section_uses_train_only_global_fallback() -> None:
    """Regression: the global median fallback used when an entire cross-section
    is NaN must be computed on TRAIN rows only (``COL_PUMP_TIME < TRAIN_END``).
    Otherwise validation/test rows leak into the imputation prior."""
    train_time: datetime = TRAIN_END - pd.Timedelta(days=30)
    val_time: datetime = TRAIN_END + pd.Timedelta(days=1)
    test_time: datetime = VALIDATION_END + pd.Timedelta(days=1)

    feature_set = FeatureSet(numeric_features=["feature_a"], target=COL_IS_PUMPED)

    df: pd.DataFrame = pd.DataFrame(
        [
            # Train cross-section with observed feature values 1 and 3 (median = 2).
            {COL_PUMP_HASH: "train-obs", COL_PUMP_TIME: train_time, "feature_a": 1.0},
            {COL_PUMP_HASH: "train-obs", COL_PUMP_TIME: train_time, "feature_a": 3.0},
            # Validation cross-section with observed feature values 100 and 200
            # (median = 150). If this leaks into the global prior, the all-NaN
            # cross-sections below would be filled with ~50, not 2.
            {COL_PUMP_HASH: "val-obs", COL_PUMP_TIME: val_time, "feature_a": 100.0},
            {COL_PUMP_HASH: "val-obs", COL_PUMP_TIME: val_time, "feature_a": 200.0},
            # All-NaN validation cross-section: must be filled with the TRAIN median.
            {COL_PUMP_HASH: "val-nan", COL_PUMP_TIME: val_time, "feature_a": np.nan},
            # All-NaN test cross-section: same.
            {COL_PUMP_HASH: "test-nan", COL_PUMP_TIME: test_time, "feature_a": np.nan},
        ]
    )
    filled: pd.DataFrame = fillna_with_median_by_cross_section(df=df, feature_set=feature_set)

    val_nan_value = filled.loc[filled[COL_PUMP_HASH] == "val-nan", "feature_a"].iloc[0]
    test_nan_value = filled.loc[filled[COL_PUMP_HASH] == "test-nan", "feature_a"].iloc[0]
    # Train-only median is (1 + 3) / 2 = 2.0; the leaked cross-frame median
    # would be much higher because of the validation values 100 and 200.
    assert np.isclose(val_nan_value, 2.0)
    assert np.isclose(test_nan_value, 2.0)
