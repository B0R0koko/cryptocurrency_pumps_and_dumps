import logging
from abc import ABC, abstractmethod
from datetime import datetime
from functools import partial
from typing import List, Dict, Any

import optuna
import pandas as pd
from optuna import Study

from backtest.pipelines.BaseModel import BaseModel, ImplementsRank
from backtest.pipelines.study import create_study
from backtest.portfolio.TOPKPortfolio import portfolio_pnl_objective
from backtest.utils.build_dataset import create_dataset
from backtest.utils.columns import (
    COL_IS_PUMPED,
    COL_CURRENCY_PAIR,
    COL_PUMPED_CURRENCY_PAIR,
    COL_PUMP_TIME,
    COL_PUMP_HASH,
    COL_PUMP_ID,
)
from backtest.utils.feature_set import FeatureSet
from backtest.utils.sample import split_by_time, DatasetType, Sample
from core.feature_type import FeatureType
from core.paths import SQLITE_URL
from core.time_utils import NamedTimeDelta
from features.FeatureWriter import REGRESSOR_OFFSETS

_RAW_DATASET_CACHE: pd.DataFrame | None = None
_PREPROCESSED_DATASETS_CACHE: Dict[str, Dict[DatasetType, pd.DataFrame]] = {}

# Time-split boundaries applied to ``COL_PUMP_TIME``: TRAIN < TRAIN_END,
# VALIDATION in [TRAIN_END, VALIDATION_END), TEST >= VALIDATION_END.
TRAIN_END: datetime = datetime(2020, 9, 1)
VALIDATION_END: datetime = datetime(2021, 5, 1)


def _copy_datasets(
    datasets: Dict[DatasetType, pd.DataFrame],
) -> Dict[DatasetType, pd.DataFrame]:
    return {ds_type: dataset.copy(deep=True) for ds_type, dataset in datasets.items()}


def _get_raw_dataset_cached() -> pd.DataFrame:
    global _RAW_DATASET_CACHE

    if _RAW_DATASET_CACHE is None:
        logging.info("Building raw dataset from feature files")
        _RAW_DATASET_CACHE = create_dataset()
        _RAW_DATASET_CACHE[COL_IS_PUMPED] = (
            _RAW_DATASET_CACHE[COL_CURRENCY_PAIR] == _RAW_DATASET_CACHE[COL_PUMPED_CURRENCY_PAIR]
        )
    else:
        logging.info("Using cached raw dataset")

    return _RAW_DATASET_CACHE.copy(deep=True)


def cross_section_standardisation(df: pd.DataFrame) -> pd.DataFrame:
    asset_return_cols: List[str] = FeatureType.ASSET_RETURN.col_names(offsets=REGRESSOR_OFFSETS)
    asset_return_zscore_cols: List[str] = FeatureType.ASSET_RETURN_ZSCORE.col_names(offsets=REGRESSOR_OFFSETS)
    quote_abs_zscore_cols: List[str] = FeatureType.QUOTE_ABS_ZSCORE.col_names(offsets=REGRESSOR_OFFSETS)
    powerlaw_cols: List[str] = FeatureType.POWERLAW_ALPHA.col_names(offsets=REGRESSOR_OFFSETS)

    cols_to_scale: List[str] = asset_return_cols + asset_return_zscore_cols + quote_abs_zscore_cols + powerlaw_cols
    grouped = df.groupby(COL_PUMP_ID, sort=False)[cols_to_scale]
    means: pd.DataFrame = grouped.transform("mean")
    stds: pd.DataFrame = grouped.transform("std")
    nuniques: pd.DataFrame = grouped.transform("nunique")
    safe_stds: pd.DataFrame = stds.mask(stds == 0)
    scaled: pd.DataFrame = (df[cols_to_scale] - means).div(safe_stds)
    apply_scaling: pd.DataFrame = nuniques != 1

    df_scaled: pd.DataFrame = df.copy()
    df_scaled[cols_to_scale] = df[cols_to_scale].where(~apply_scaling, scaled)
    return df_scaled


def remove_failed_pump_cross_sections(df: pd.DataFrame) -> pd.DataFrame:
    """Drop cross-sections whose pumped asset is not top-10 by 1-minute post-pump return.

    Ranks are kept in a standalone Series rather than a temporary column: this function
    now runs after ``preprocess_data``, where CatboostRanker has already created its
    label column with the same name as ``COL_ASSET_RETURN_RANK``.
    """
    logging.info("Removing failed pump cross sections")
    target_return_col: str = f"target_return@{NamedTimeDelta.ONE_MINUTE.get_slug()}"
    ranks: pd.Series = df.groupby(COL_PUMP_ID, sort=False)[target_return_col].rank(ascending=False, method="dense")
    pumped_mask: pd.Series = df[COL_IS_PUMPED]
    pumped_hashes: pd.Series = df.loc[pumped_mask, COL_PUMP_HASH]
    assert pumped_hashes.is_unique, "Found many pumps within one cross-section"

    # Keep NaN ranks for parity with the previous implementation:
    # the old `pump_rank >= 10` check did not remove NaN-ranked pumps.
    pumped_ranks: pd.Series = ranks[pumped_mask]
    valid_mask: pd.Series = pumped_ranks.isna() | (pumped_ranks < 10)
    valid_pumps: pd.Series = pumped_hashes[valid_mask]
    pumps_removed: int = pumped_hashes.shape[0] - valid_pumps.shape[0]

    df = df[df[COL_PUMP_HASH].isin(valid_pumps)].reset_index(drop=True)

    if pumps_removed > 0:
        logging.warning("Removed %s failed pumps", pumps_removed)

    return df


def fillna_with_median_by_cross_section(df: pd.DataFrame, feature_set: FeatureSet) -> pd.DataFrame:
    """Group by PUMP_HASH and fill missing values with cross-section median values.

    The global fallback medians used when an entire cross-section is NaN are computed
    from train-period rows only (``COL_PUMP_TIME < TRAIN_END``). Using train rows as
    the source of the global prior avoids leaking validation/test information into
    the imputation step.

    Some regressor columns can be entirely NaN across the whole panel (e.g. the
    5MIN / 15MIN z-scores under the T - 15min feature cutoff, where no full
    hourly bar fits inside the sub-hour window). When that happens both the
    cross-section median and the global-train median are NaN. To keep the
    downstream sklearn estimators happy (they refuse NaN inputs) we fall back
    to 0.0 for the residual, which is neutral under the downstream cross-
    sectional standardisation (Eq. ~cs\\_standardization) — a filled-with-mean
    column becomes a zero column after standardisation and cannot inject
    signal.
    """
    regressors: List[str] = feature_set.regressors
    train_mask: pd.Series = df[COL_PUMP_TIME] < TRAIN_END
    train_source: pd.DataFrame = df.loc[train_mask, regressors] if train_mask.any() else df[regressors]
    global_medians: pd.Series = train_source.median()
    cross_section_medians: pd.DataFrame = df.groupby(COL_PUMP_HASH, sort=False)[regressors].transform("median")

    df_nonans: pd.DataFrame = df.copy()
    df_nonans[regressors] = df_nonans[regressors].fillna(cross_section_medians).fillna(global_medians).fillna(0.0)

    logging.info("Nans\n%s", df_nonans[regressors].isna().sum().sort_values(ascending=False))
    return df_nonans


def add_col_pump_id(df: pd.DataFrame) -> pd.DataFrame:
    df[COL_PUMP_ID] = df.groupby(by=COL_PUMP_HASH, sort=False).ngroup()
    return df


class BasePipeline(ABC):

    @abstractmethod
    def create_sample(self) -> Sample: ...

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Define all data preprocessing steps here.

        Note on survivorship filtering: ``remove_failed_pump_cross_sections`` is
        applied AFTER the time split, and only to the TRAIN slice, inside
        :meth:`build_datasets`. This step relies on the post-pump ``target_return``
        label and dropping "failed" pumps from validation/test would silently
        overstate reported metrics.
        """
        df = add_col_pump_id(df=df)
        powerlaw_cols: List[str] = FeatureType.POWERLAW_ALPHA.col_names(offsets=REGRESSOR_OFFSETS)
        df[powerlaw_cols] = df[powerlaw_cols].clip(1, 2)
        df_scaled: pd.DataFrame = cross_section_standardisation(df=df)
        return df_scaled

    def build_datasets(self) -> Dict[DatasetType, pd.DataFrame]:
        cache_key: str = self.__class__.__name__
        if cache_key in _PREPROCESSED_DATASETS_CACHE:
            logging.info("Using cached preprocessed datasets for %s", cache_key)
            return _copy_datasets(_PREPROCESSED_DATASETS_CACHE[cache_key])

        logging.info("Building dataset and preprocessing data")
        df: pd.DataFrame = _get_raw_dataset_cached()
        df = self.preprocess_data(df=df)
        datasets: Dict[DatasetType, pd.DataFrame] = split_by_time(
            df=df,
            time_bins=[TRAIN_END, VALIDATION_END],
            names=[DatasetType.TRAIN, DatasetType.VALIDATION, DatasetType.TEST],
            time_col=COL_PUMP_TIME,
        )
        # Survivorship filter is applied to TRAIN only: it uses a post-pump label,
        # so dropping "failed" cross-sections from validation/test would inflate
        # reported metrics.
        train_df: pd.DataFrame = datasets[DatasetType.TRAIN]
        datasets[DatasetType.TRAIN] = remove_failed_pump_cross_sections(df=train_df).reset_index(drop=True)
        for ds_type, dataset in datasets.items():
            logging.info("Dataset %s. Shape %s", ds_type, dataset.shape)

        _PREPROCESSED_DATASETS_CACHE[cache_key] = _copy_datasets(datasets)
        return _copy_datasets(datasets)

    def get_model_params(self, base_params: Dict[str, Any], study_name: str) -> Dict[str, Any]:
        logging.info("Loading parameters from %s", study_name)
        study: Study = optuna.load_study(study_name=study_name, storage=SQLITE_URL)
        return base_params | study.best_params

    @abstractmethod
    def train(self, sample: Sample, tuned: bool = True): ...

    @abstractmethod
    def build_model(self) -> BaseModel: ...

    @abstractmethod
    def optimize_parameters(self, n_trials: int = 100): ...

    def optimize_portfolio_strategy(self) -> None:
        """Tune portfolio timing / top-k size against the VALIDATION split.

        The Optuna objective :func:`portfolio_pnl_objective` evaluates each candidate
        configuration on ``DatasetType.VALIDATION``. The TEST split is never touched
        during tuning; it is reserved for final reported PnL.
        """
        sample: Sample = self.create_sample()
        model: ImplementsRank = self.train(sample=sample)
        study: Study = create_study(study_name="TOPKPortfolioStrategy", start_new=False)
        study.optimize(partial(portfolio_pnl_objective, model=model, sample=sample), n_trials=20)
