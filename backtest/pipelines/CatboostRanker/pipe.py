import logging
from functools import partial
from typing import Dict, List, Any

import pandas as pd
from catboost import Pool
from optuna import Study, Trial
from overrides import overrides

from backtest.pipelines.BaseModel import BaseModel
from backtest.pipelines.BasePipeline import (
    BasePipeline,
    cross_section_standardisation,
    add_col_pump_id,
)
from backtest.pipelines.CatboostRanker.model import CatboostRankerModel
from backtest.pipelines.study import create_study
from backtest.utils.columns import *
from backtest.utils.feature_set import FeatureSet
from backtest.utils.metrics import calculate_topk_percent, calculate_topk_percent_auc
from backtest.utils.sample import DatasetType, Sample, Dataset
from core.feature_type import FeatureType
from core.utils import configure_logging
from features.FeatureWriter import REGRESSOR_OFFSETS

_BASE_PARAMS: Dict[str, Any] = {
    "objective": "YetiRank:mode=NDCG",
    "border_count": 255,
    "verbose": False,
    "random_seed": 42,
}


def _objective(trial: Trial, sample: Sample) -> float:
    tuned_params: Dict[str, Any] = {
        "iterations": trial.suggest_int("iterations", 10, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.7, 1),
    }

    model: CatboostRankerModel = CatboostRankerModel(params=_BASE_PARAMS | tuned_params)
    model.train(sample=sample)

    val: Dataset = sample.get_dataset(ds_type=DatasetType.VALIDATION)
    topkpauc: float = calculate_topk_percent_auc(model=model, dataset=val)
    return topkpauc


class CatboostRankerPipeline(BasePipeline):

    def __init__(self):
        self.feature_set: FeatureSet = FeatureSet.auto()
        self.feature_set.target = "asset_return_rank"  # from 1...N, first has the highest return within cross-section

    @overrides
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the supervised ranking label after past-only feature preprocessing."""
        df = add_col_pump_id(df=df)
        # Clip powerlaw alpha features to (1, 2)
        powerlaw_cols: List[str] = FeatureType.POWERLAW_ALPHA.col_names(offsets=REGRESSOR_OFFSETS)
        df[powerlaw_cols] = df[powerlaw_cols].clip(1, 2)
        # Fillna target for ranker which is target_return@5MIN
        df["target_return@5MIN"] = df["target_return@5MIN"].fillna(0)
        df_scaled: pd.DataFrame = cross_section_standardisation(df=df)
        # Create rankings
        df_scaled[self.feature_set.target] = df_scaled.groupby(COL_PUMP_ID, sort=False)["target_return@5MIN"].rank(
            pct=True, ascending=False
        )
        assert df_scaled[COL_PUMP_ID].is_monotonic_increasing, "GroupId must be monotonic increasing"
        return df_scaled

    def create_sample(self) -> Sample:
        datasets: Dict[DatasetType, pd.DataFrame] = self.build_datasets()
        sample: Sample = Sample.from_pandas(datasets=datasets, feature_set=self.feature_set)
        # we also need to set_pools as Catboost uses Pool under the hood
        for ds_type, dataset in sample.iter_datasets():
            dataset.set_pool(
                Pool(
                    data=dataset.get_data(),
                    label=dataset.get_label(),
                    cat_features=self.feature_set.categorical_features,
                    group_id=dataset.all_data()[COL_PUMP_ID],
                )
            )
        return sample

    def optimize_parameters(self, n_trials: int = 100) -> Study:
        logging.info("Running <optimize_parameters> for CatboostRankerPipeline")
        sample: Sample = self.create_sample()
        study: Study = create_study(study_name="CatboostRankerPipelineStudy", start_new=True)
        study.optimize(partial(_objective, sample=sample), n_trials=n_trials)
        return study

    def train(self, sample: Sample, tuned: bool = True) -> CatboostRankerModel:
        model_params: Dict[str, Any] = _BASE_PARAMS
        if tuned:
            model_params = self.get_model_params(base_params=_BASE_PARAMS, study_name="CatboostRankerPipelineStudy")

        model: CatboostRankerModel = CatboostRankerModel(params=model_params)
        model.train(sample=sample)
        return model

    def build_model(self, tuned: bool = True) -> BaseModel:
        logging.info("Running <build_model> for CatboostRankerPipeline")
        sample: Sample = self.create_sample()
        model: CatboostRankerModel = self.train(sample=sample, tuned=tuned)

        topk_vals: pd.Series = calculate_topk_percent(
            model=model,
            dataset=sample.get_dataset(ds_type=DatasetType.VALIDATION),
            bins=[0.01, 0.02, 0.05, 0.1, 0.2],
        )
        logging.info("Top@k%% accuracy:\n%s", topk_vals)
        return model


def main():
    configure_logging()
    pipe = CatboostRankerPipeline()
    pipe.optimize_parameters(n_trials=100)


if __name__ == "__main__":
    main()
