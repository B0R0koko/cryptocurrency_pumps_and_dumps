import logging
from functools import partial
from typing import Dict, Any, Tuple

import numpy as np
import optuna
import pandas as pd
from catboost import Pool
from numba import njit
from optuna import Study, Trial
from overrides import overrides

from backtest.pipelines.BaseModel import BaseModel
from backtest.pipelines.BasePipeline import BasePipeline
from backtest.pipelines.CatboostClassifier.model import CatboostClassifierModel
from backtest.pipelines.study import create_study
from backtest.utils.columns import COL_PUMP_HASH, COL_IS_PUMPED
from backtest.utils.feature_set import FeatureSet
from backtest.utils.metrics import calculate_topk_percent_auc, calculate_topk_percent
from backtest.utils.sample import DatasetType, Sample, Dataset
from core.paths import SQLITE_URL
from core.utils import configure_logging

_BASE_PARAMS: Dict[str, Any] = {
    "objective": "Logloss",
    "sampling_frequency": "PerTree",
    "num_boost_round": 1000,
    "auto_class_weights": "Balanced",
    "verbose": 10,
    "random_seed": 42,
}

_MAX_K_PERCENT: float = 0.20
_STEP: float = 0.005
_BINS: np.ndarray = np.arange(0, _MAX_K_PERCENT + _STEP, _STEP).astype(np.float64)


@njit(cache=True)
def _topkpauc_kernel(
    scores_by_group: np.ndarray,
    is_pumped_by_group: np.ndarray,
    group_starts: np.ndarray,
    bins: np.ndarray,
    num_cross_sections_with_pump: int,
) -> float:
    """
    Numba kernel: for each cross-section (contiguous slice defined by ``group_starts``)
    sort rows by ``scores`` desc, compute a cumulative "any pumped in top-i" indicator,
    and for each K% bin increment the count if the top-K% contains a pumped sample.
    Returns the trapezoidal AUC over ``bins`` of the cumulative hit rate (counts /
    num_cross_sections_with_pump). Caller divides by the bin range to obtain the
    normalised metric (TOPKAUC).

    The denominator ``num_cross_sections_with_pump`` matches
    :func:`backtest.utils.metrics.calculate_topk_percent`: it counts cross-sections
    that contain at least one pumped row, not pumped rows themselves. Under the
    project's "one pumped row per cross-section" invariant these coincide, but
    the semantics matter if that invariant ever breaks.
    """
    n_bins = bins.shape[0]
    counts = np.zeros(n_bins, dtype=np.int64)
    n_groups = group_starts.shape[0] - 1

    for g in range(n_groups):
        start = group_starts[g]
        end = group_starts[g + 1]
        n_rows = end - start
        if n_rows == 0:
            continue

        sub_scores = scores_by_group[start:end]
        sub_pumped = is_pumped_by_group[start:end]
        order = np.argsort(-sub_scores)

        any_so_far = False
        any_arr = np.empty(n_rows, dtype=np.bool_)
        for i in range(n_rows):
            if sub_pumped[order[i]]:
                any_so_far = True
            any_arr[i] = any_so_far

        for b in range(n_bins):
            k = int(np.ceil(n_rows * bins[b]))
            if k < 1:
                continue
            if k > n_rows:
                k = n_rows
            if any_arr[k - 1]:
                counts[b] += 1

    if num_cross_sections_with_pump == 0:
        return 0.0
    rates = counts.astype(np.float64) / num_cross_sections_with_pump
    auc_val = 0.0
    for i in range(n_bins - 1):
        auc_val += 0.5 * (rates[i] + rates[i + 1]) * (bins[i + 1] - bins[i])
    return auc_val


def _precompute_groups(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Build the permutation and group-boundary arrays consumed by
    :func:`_topkpauc_kernel`. Pandas is used here so the kernel itself can stay
    pure numpy/numba-compatible. The return tuple is ``(sort_idx, is_pumped_sorted,
    group_starts, num_cross_sections_with_pump)`` where rows sorted by ``sort_idx``
    are contiguous per cross-section, and ``num_cross_sections_with_pump`` is the
    denominator matching :func:`backtest.utils.metrics.calculate_topk_percent`.
    """
    pump_hashes: np.ndarray = df[COL_PUMP_HASH].to_numpy()
    is_pumped: np.ndarray = df[COL_IS_PUMPED].to_numpy(dtype=np.bool_)
    codes, _ = pd.factorize(pump_hashes, sort=False)
    sort_idx: np.ndarray = np.argsort(codes, kind="stable").astype(np.int64)
    sorted_codes: np.ndarray = codes[sort_idx]
    is_pumped_sorted: np.ndarray = is_pumped[sort_idx]
    if sorted_codes.size == 0:
        group_starts = np.array([0], dtype=np.int64)
    else:
        change_points: np.ndarray = np.where(np.diff(sorted_codes) != 0)[0] + 1
        group_starts = np.concatenate(([0], change_points, [sorted_codes.size])).astype(np.int64)

    n_groups: int = group_starts.shape[0] - 1
    num_cross_sections_with_pump: int = 0
    for g in range(n_groups):
        start = int(group_starts[g])
        end = int(group_starts[g + 1])
        if end > start and bool(is_pumped_sorted[start:end].any()):
            num_cross_sections_with_pump += 1
    return sort_idx, is_pumped_sorted, group_starts, num_cross_sections_with_pump


class TOPKAUCMetric:
    """
    CatBoost custom eval metric computing TOPKAUC over ``K% in (0, _MAX_K_PERCENT]``
    (normalised to ``(0, 1)``). Higher is better (``is_max_optimal → True``).

    Precomputes cross-section grouping and is-pumped labels in ``__init__`` so the
    per-iteration path is a single permutation and a numba-JIT'd kernel call.

    CatBoost calls ``evaluate`` once per pool per iteration; the pool being
    scored is disambiguated by row count first and, if the two pools happen to
    have the same length, by the label vector supplied in ``target``. This
    prevents train scores from being evaluated against validation labels (a
    silent bug in earlier versions when ``train_len == val_len``).
    """

    def __init__(self, df_train: pd.DataFrame, df_val: pd.DataFrame) -> None:
        self._train_ctx: Tuple[np.ndarray, np.ndarray, np.ndarray, int] = _precompute_groups(df_train)
        self._val_ctx: Tuple[np.ndarray, np.ndarray, np.ndarray, int] = _precompute_groups(df_val)
        self._train_len: int = len(df_train)
        self._val_len: int = len(df_val)
        self._train_labels: np.ndarray = df_train[COL_IS_PUMPED].to_numpy(dtype=np.bool_)
        self._val_labels: np.ndarray = df_val[COL_IS_PUMPED].to_numpy(dtype=np.bool_)

    def is_max_optimal(self) -> bool:
        return True

    def _select_context(self, probas_pred: np.ndarray, target: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        n_rows: int = probas_pred.shape[0]
        # Fast path when train/val have distinct row counts.
        if n_rows == self._train_len and n_rows != self._val_len:
            return self._train_ctx
        if n_rows == self._val_len and n_rows != self._train_len:
            return self._val_ctx
        # Ambiguous row count: fall back to comparing the labels catboost
        # passes to the metric against the precomputed ones.
        if target is None:
            return self._val_ctx
        target_arr: np.ndarray = np.asarray(target, dtype=np.float64).astype(np.bool_)
        if target_arr.shape[0] == self._train_labels.shape[0] and np.array_equal(target_arr, self._train_labels):
            return self._train_ctx
        return self._val_ctx

    def evaluate(self, approxes, target, weight) -> Tuple[float, float]:
        assert len(approxes) == 1
        probas_pred: np.ndarray = np.asarray(approxes[0], dtype=np.float64)

        sort_idx, is_pumped_sorted, group_starts, num_cross_sections_with_pump = self._select_context(
            probas_pred=probas_pred, target=target
        )

        if num_cross_sections_with_pump == 0:
            return 0.0, 1.0

        scores_sorted: np.ndarray = probas_pred[sort_idx]
        raw_auc: float = _topkpauc_kernel(
            scores_by_group=scores_sorted,
            is_pumped_by_group=is_pumped_sorted,
            group_starts=group_starts,
            bins=_BINS,
            num_cross_sections_with_pump=num_cross_sections_with_pump,
        )
        return raw_auc / _MAX_K_PERCENT, 1.0

    def get_final_error(self, error, weight) -> float:
        return error


def _objective(trial: Trial, sample: Sample) -> float:
    tuned_params: Dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.7, 1),
        "subsample": trial.suggest_float("subsample", 0.7, 1),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
    }
    df_train: pd.DataFrame = sample.get_dataset(ds_type=DatasetType.TRAIN).all_data()
    df_val: pd.DataFrame = sample.get_dataset(ds_type=DatasetType.VALIDATION).all_data()
    # Add custom evaluation metric that maximizes TOPKAUC
    base_params: Dict[str, Any] = _BASE_PARAMS | {"eval_metric": TOPKAUCMetric(df_train=df_train, df_val=df_val)}

    model: CatboostClassifierModel = CatboostClassifierModel(params=base_params | tuned_params)
    model.train(sample=sample)

    val: Dataset = sample.get_dataset(ds_type=DatasetType.VALIDATION)
    topkpauc: float = calculate_topk_percent_auc(model=model, dataset=val)
    return topkpauc


class CatboostClassifierTOPKAUCPipeline(BasePipeline):

    def __init__(self):
        self.feature_set: FeatureSet = FeatureSet.auto()

    @overrides
    def get_model_params(self, base_params: Dict[str, Any], study_name: str) -> Dict[str, Any]:
        study: Study = optuna.load_study(study_name=study_name, storage=SQLITE_URL)
        model_params: Dict[str, Any] = base_params | study.best_params
        return model_params

    def create_sample(self) -> Sample:
        # we also need to set_pools as Catboost uses Pool under the hood
        datasets: Dict[DatasetType, pd.DataFrame] = self.build_datasets()
        sample: Sample = Sample.from_pandas(datasets=datasets, feature_set=self.feature_set)
        for ds_type, dataset in sample.iter_datasets():
            dataset.set_pool(
                Pool(
                    data=dataset.get_data(),
                    label=dataset.get_label(),
                    cat_features=self.feature_set.categorical_features,
                )
            )

        return sample

    def optimize_parameters(self, n_trials: int = 500) -> Study:
        logging.info("Running <optimize_parameters> for CatboostClassifierTOPKAUCPipeline")
        sample: Sample = self.create_sample()
        study: Study = create_study(study_name="CatboostClassifierTOPKAUCPipeline", start_new=True)
        study.optimize(partial(_objective, sample=sample), n_trials=n_trials)
        return study

    def train(self, sample: Sample, tuned: bool = True) -> CatboostClassifierModel:
        df_train: pd.DataFrame = sample.get_dataset(ds_type=DatasetType.TRAIN).all_data()
        df_val: pd.DataFrame = sample.get_dataset(ds_type=DatasetType.VALIDATION).all_data()
        # Add custom evaluation metric that maximizes TOPKAUC
        model_params: Dict[str, Any] = _BASE_PARAMS | {"eval_metric": TOPKAUCMetric(df_train=df_train, df_val=df_val)}
        if tuned:
            model_params = self.get_model_params(
                base_params=model_params, study_name="CatboostClassifierTOPKAUCPipeline"
            )

        model: CatboostClassifierModel = CatboostClassifierModel(params=model_params)
        model.train(sample=sample)
        return model

    def build_model(self, tuned: bool = True) -> BaseModel:
        logging.info("Running <build_model> for CatboostClassifierTOPKAUCPipeline")
        sample: Sample = self.create_sample()
        model: CatboostClassifierModel = self.train(sample=sample, tuned=tuned)

        topk_vals: pd.Series = calculate_topk_percent(
            model=model,
            dataset=sample.get_dataset(ds_type=DatasetType.VALIDATION),
            bins=[0.01, 0.02, 0.05, 0.1, 0.2],
        )
        logging.info(f"TopK Accuracy:\n%s", topk_vals)

        return model


def main():
    configure_logging()
    pipe = CatboostClassifierTOPKAUCPipeline()
    pipe.optimize_parameters()


if __name__ == "__main__":
    main()
