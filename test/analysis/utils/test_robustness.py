from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from backtest.pipelines.BaseModel import ImplementsRank
from backtest.robust.robustness import (
    run_cross_section_subset_robustness,
    subset_cross_sections,
    summarise_feature_importance_distribution,
    summarise_robustness_distribution,
)
from backtest.utils.columns import COL_IS_PUMPED, COL_PUMP_HASH
from backtest.utils.feature_set import FeatureSet
from backtest.utils.sample import Dataset, DatasetType, Sample

FEATURE_SET = FeatureSet(
    numeric_features=["score"],
    target=COL_IS_PUMPED,
    categorical_features=None,
    eval_fields=[COL_PUMP_HASH],
)


class DummyModel(ImplementsRank):

    def rank(self, dataset: Dataset) -> np.ndarray:
        return dataset.get_data()["score"].to_numpy()


class DummyPipeline:

    def __init__(
        self,
        datasets: Dict[DatasetType, pd.DataFrame],
        seen_train_cross_sections: List[int],
    ):
        self._datasets = datasets
        self._seen_train_cross_sections = seen_train_cross_sections
        self.feature_set = FEATURE_SET

    def build_datasets(self) -> Dict[DatasetType, pd.DataFrame]:
        return {ds_type: df.copy(deep=True) for ds_type, df in self._datasets.items()}

    def create_sample(self) -> Sample:
        datasets: Dict[DatasetType, pd.DataFrame] = self.build_datasets()
        self._seen_train_cross_sections.append(int(datasets[DatasetType.TRAIN][COL_PUMP_HASH].nunique()))
        return Sample.from_pandas(datasets=datasets, feature_set=self.feature_set)

    def train(self, sample: Sample, tuned: bool = False) -> DummyModel:  # noqa: ARG002
        return DummyModel()


class _DummyEstimator:
    """Mimics a fitted CatBoost estimator: exposes feature importances + names."""

    def __init__(self, feature_names: List[str], feature_importances: np.ndarray):
        self.feature_names_ = feature_names
        self.feature_importances_ = feature_importances


class DummyFIModel(ImplementsRank):
    """Pipeline model that wraps an estimator in ``_model`` like CatboostClassifierModel."""

    def __init__(self, estimator: _DummyEstimator):
        self._model = estimator

    def rank(self, dataset: Dataset) -> np.ndarray:
        return dataset.get_data()["score"].to_numpy()


class DummyFIPipeline(DummyPipeline):

    def __init__(
        self,
        datasets: Dict[DatasetType, pd.DataFrame],
        seen_train_cross_sections: List[int],
        run_counter: List[int],
    ):
        super().__init__(datasets=datasets, seen_train_cross_sections=seen_train_cross_sections)
        self._run_counter = run_counter

    def train(self, sample: Sample, tuned: bool = False) -> DummyFIModel:  # noqa: ARG002
        run_idx: int = len(self._run_counter)
        self._run_counter.append(run_idx)
        # Importances jitter per run so the bootstrap distribution is non-degenerate;
        # median ordering stays c > b > a.
        feature_importances: np.ndarray = np.array([10.0 + run_idx, 30.0 + run_idx, 60.0 - 2 * run_idx], dtype=float)
        return DummyFIModel(_DummyEstimator(feature_names=["a", "b", "c"], feature_importances=feature_importances))


def _build_cross_sections(prefix: str, n_cross_sections: int, rows_per_cross_section: int) -> pd.DataFrame:
    rows = []
    for cs_idx in range(n_cross_sections):
        pump_hash = f"{prefix}-{cs_idx}"
        for row_idx in range(rows_per_cross_section):
            rows.append(
                {
                    COL_PUMP_HASH: pump_hash,
                    COL_IS_PUMPED: row_idx == 0,
                    "score": float(rows_per_cross_section - row_idx),
                }
            )

    return pd.DataFrame(rows)


def test_subset_cross_sections_returns_complete_groups() -> None:
    df: pd.DataFrame = _build_cross_sections(prefix="train", n_cross_sections=6, rows_per_cross_section=3)

    subset_a: pd.DataFrame = subset_cross_sections(df=df, subset_fraction=0.5, random_state=17)
    subset_b: pd.DataFrame = subset_cross_sections(df=df, subset_fraction=0.5, random_state=17)

    assert subset_a[COL_PUMP_HASH].nunique() == 3
    assert subset_b[COL_PUMP_HASH].nunique() == 3
    assert set(subset_a[COL_PUMP_HASH].unique()) == set(subset_b[COL_PUMP_HASH].unique())
    assert subset_a.groupby(COL_PUMP_HASH).size().nunique() == 1
    assert subset_a.groupby(COL_PUMP_HASH).size().iloc[0] == 3


def test_run_cross_section_subset_robustness_saves_distribution(tmp_path: Path) -> None:
    datasets: Dict[DatasetType, pd.DataFrame] = {
        DatasetType.TRAIN: _build_cross_sections(prefix="train", n_cross_sections=6, rows_per_cross_section=3),
        DatasetType.VALIDATION: _build_cross_sections(prefix="val", n_cross_sections=2, rows_per_cross_section=3),
        DatasetType.TEST: _build_cross_sections(prefix="test", n_cross_sections=4, rows_per_cross_section=3),
    }
    seen_train_cross_sections: List[int] = []

    def factory() -> DummyPipeline:
        return DummyPipeline(datasets=datasets, seen_train_cross_sections=seen_train_cross_sections)

    output_path: Path = tmp_path / "robustness.csv"
    results: pd.DataFrame = run_cross_section_subset_robustness(
        pipeline_factory=factory,
        subset_fraction=0.5,
        n_runs=4,
        tuned=False,
        topk_bins=[0.1, 0.5],
        base_seed=10,
        output_path=output_path,
    )

    assert results.shape[0] == 4
    assert {"run_idx", "seed", "train_cross_sections", "topk_percent_auc"}.issubset(results.columns)
    assert (results["train_cross_sections"] == 3).all()
    assert len(seen_train_cross_sections) == 4
    assert all(cs == 3 for cs in seen_train_cross_sections)
    assert (results["topk_percent_auc"] > 0.9).all()
    assert results["topk_percent_auc"].nunique() == 1
    assert np.allclose(results["topk_percent@0.1"].to_numpy(), np.ones(4))
    assert np.allclose(results["topk_percent@0.5"].to_numpy(), np.ones(4))
    assert output_path.exists()

    loaded: pd.DataFrame = pd.read_csv(output_path)
    assert loaded.shape == results.shape


def test_summarise_robustness_distribution_uses_topk_columns() -> None:
    results: pd.DataFrame = pd.DataFrame(
        {
            "run_idx": [0, 1, 2],
            "topk_percent_auc": [0.6, 0.7, 0.8],
            "topk_percent@0.1": [0.2, 0.3, 0.4],
        }
    )
    summary: pd.DataFrame = summarise_robustness_distribution(results=results)

    assert "topk_percent_auc" in summary.index
    assert "topk_percent@0.1" in summary.index
    assert "mean" in summary.columns
    assert "50%" in summary.columns


def test_run_cross_section_subset_robustness_collects_feature_importances(tmp_path: Path) -> None:
    datasets: Dict[DatasetType, pd.DataFrame] = {
        DatasetType.TRAIN: _build_cross_sections(prefix="train", n_cross_sections=6, rows_per_cross_section=3),
        DatasetType.VALIDATION: _build_cross_sections(prefix="val", n_cross_sections=2, rows_per_cross_section=3),
        DatasetType.TEST: _build_cross_sections(prefix="test", n_cross_sections=4, rows_per_cross_section=3),
    }
    seen_train_cross_sections: List[int] = []
    run_counter: List[int] = []

    def factory() -> DummyFIPipeline:
        return DummyFIPipeline(
            datasets=datasets,
            seen_train_cross_sections=seen_train_cross_sections,
            run_counter=run_counter,
        )

    fi_path: Path = tmp_path / "feature_importances.csv"
    results: pd.DataFrame = run_cross_section_subset_robustness(
        pipeline_factory=factory,
        subset_fraction=0.5,
        n_runs=4,
        tuned=False,
        topk_bins=[0.1, 0.5],
        base_seed=10,
        collect_feature_importances=True,
        feature_importance_output_path=fi_path,
    )

    feature_importances: pd.DataFrame = results.attrs["feature_importances"]
    assert set(feature_importances.columns) == {"run_idx", "seed", "feature", "feature_importance"}
    assert feature_importances.shape[0] == 4 * 3  # n_runs x n_features
    assert sorted(feature_importances["feature"].unique()) == ["a", "b", "c"]
    assert feature_importances["run_idx"].nunique() == 4

    # Values must be collected faithfully: DummyFIPipeline emits a=10+run, b=30+run, c=60-2*run.
    pivot = feature_importances.pivot(index="run_idx", columns="feature", values="feature_importance")
    np.testing.assert_array_equal(pivot["a"].to_numpy(), np.array([10.0, 11.0, 12.0, 13.0]))
    np.testing.assert_array_equal(pivot["b"].to_numpy(), np.array([30.0, 31.0, 32.0, 33.0]))
    np.testing.assert_array_equal(pivot["c"].to_numpy(), np.array([60.0, 58.0, 56.0, 54.0]))
    # Ranking by mean importance is preserved across runs: c > b > a.
    means = feature_importances.groupby("feature")["feature_importance"].mean()
    assert means["c"] > means["b"] > means["a"]

    assert fi_path.exists()
    loaded: pd.DataFrame = pd.read_csv(fi_path)
    assert loaded.shape == feature_importances.shape


def test_run_cross_section_subset_robustness_skips_feature_importances_by_default() -> None:
    datasets: Dict[DatasetType, pd.DataFrame] = {
        DatasetType.TRAIN: _build_cross_sections(prefix="train", n_cross_sections=4, rows_per_cross_section=3),
        DatasetType.VALIDATION: _build_cross_sections(prefix="val", n_cross_sections=2, rows_per_cross_section=3),
        DatasetType.TEST: _build_cross_sections(prefix="test", n_cross_sections=4, rows_per_cross_section=3),
    }
    seen_train_cross_sections: List[int] = []

    def factory() -> DummyPipeline:
        return DummyPipeline(datasets=datasets, seen_train_cross_sections=seen_train_cross_sections)

    # DummyModel has no `_model`; with the default flag we must not touch feature importances.
    results: pd.DataFrame = run_cross_section_subset_robustness(
        pipeline_factory=factory,
        subset_fraction=0.5,
        n_runs=2,
        tuned=False,
        topk_bins=[0.1, 0.5],
        base_seed=10,
    )
    assert results.attrs["feature_importances"].empty


def test_summarise_feature_importance_distribution_sorts_by_median() -> None:
    feature_importances: pd.DataFrame = pd.DataFrame(
        {
            "run_idx": [0, 0, 0, 1, 1, 1],
            "seed": [10, 10, 10, 11, 11, 11],
            "feature": ["a", "b", "c", "a", "b", "c"],
            "feature_importance": [10.0, 30.0, 60.0, 11.0, 31.0, 58.0],
        }
    )

    summary: pd.DataFrame = summarise_feature_importance_distribution(feature_importances)

    assert list(summary.index) == ["c", "b", "a"]
    assert {"mean", "std", "p10", "p25", "p50", "p75", "p90"}.issubset(summary.columns)
    assert summary.loc["c", "p50"] == 59.0
    assert summary.loc["a", "p50"] == 10.5

    top_one: pd.DataFrame = summarise_feature_importance_distribution(feature_importances, top_n=1)
    assert list(top_one.index) == ["c"]
