import logging
from functools import partial
from typing import Dict, Any

import optuna
import pandas as pd
from catboost import Pool
from optuna import Study, Trial
from overrides import overrides

from backtest.pipelines.BaseModel import BaseModel
from backtest.pipelines.BasePipeline import BasePipeline
from backtest.pipelines.CatboostClassifier.model import CatboostClassifierModel
from backtest.pipelines.study import create_study
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
    "verbose": False,
    "random_seed": 42,
}


def _objective(trial: Trial, sample: Sample) -> float:
    tuned_params: Dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.7, 1),
        "subsample": trial.suggest_float("subsample", 0.7, 1),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
    }

    model: CatboostClassifierModel = CatboostClassifierModel(params=_BASE_PARAMS | tuned_params)
    model.train(sample=sample)

    val: Dataset = sample.get_dataset(ds_type=DatasetType.VALIDATION)
    topkpauc: float = calculate_topk_percent_auc(model=model, dataset=val)
    return topkpauc


class CatboostClassifierPipeline(BasePipeline):

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

    def optimize_parameters(self, n_trials: int = 100) -> Study:
        logging.info("Running <optimize_parameters> for CatboostClassifierPipeline")
        sample: Sample = self.create_sample()
        study: Study = create_study(study_name="CatboostClassifierPipelineStudy", start_new=True)
        study.optimize(partial(_objective, sample=sample), n_trials=n_trials)
        return study

    def train(self, sample: Sample, tuned: bool = True) -> CatboostClassifierModel:
        model_params: Dict[str, Any] = _BASE_PARAMS
        if tuned:
            model_params = self.get_model_params(base_params=_BASE_PARAMS, study_name="CatboostClassifierPipelineStudy")

        model: CatboostClassifierModel = CatboostClassifierModel(params=model_params)
        model.train(sample=sample)
        return model

    def build_model(self, tuned: bool = True) -> BaseModel:
        logging.info("Running<build_model> for CatboostClassifierPipeline")
        sample: Sample = self.create_sample()
        model: CatboostClassifierModel = self.train(sample=sample, tuned=tuned)

        topk_vals: pd.Series = calculate_topk_percent(
            model=model,
            dataset=sample.get_dataset(ds_type=DatasetType.VALIDATION),
            bins=[0.01, 0.02, 0.05, 0.1, 0.2],
        )
        logging.info("Top@k%% accuracy:\n%s", topk_vals)

        return model


def main():
    configure_logging()
    pipe = CatboostClassifierPipeline()
    pipe.optimize_parameters(n_trials=100)


if __name__ == "__main__":
    main()
