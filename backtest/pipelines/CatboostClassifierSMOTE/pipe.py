import logging
from functools import partial
from typing import Dict, List, Any

import optuna
import pandas as pd
from catboost import Pool
from imblearn.over_sampling import SMOTE
from optuna import Trial, Study
from overrides import overrides

from backtest.pipelines.BaseModel import BaseModel
from backtest.pipelines.BasePipeline import (
    BasePipeline,
    cross_section_standardisation,
    fillna_with_median_by_cross_section,
    add_col_pump_id,
)
from backtest.pipelines.CatboostClassifier.model import CatboostClassifierModel
from backtest.pipelines.study import create_study
from backtest.utils.columns import *
from backtest.utils.feature_set import FeatureSet
from backtest.utils.metrics import calculate_topk_percent, calculate_topk_percent_auc
from backtest.utils.sample import DatasetType, Sample, Dataset
from core.feature_type import FeatureType
from core.paths import SQLITE_URL
from core.utils import configure_logging
from features.FeatureWriter import REGRESSOR_OFFSETS

_BASE_PARAMS: Dict[str, Any] = {
    "objective": "Logloss",
    "sampling_frequency": "PerTree",
    "num_boost_round": 1000,
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


class CatboostClassifierSMOTEPipeline(BasePipeline):

    def __init__(self):
        self.feature_set: FeatureSet = FeatureSet.auto()

    @overrides
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Define all data preprocessing steps here.

        Survivorship filtering is applied to the TRAIN split only in
        :meth:`BasePipeline.build_datasets`.
        """
        df = add_col_pump_id(df=df)
        df[COL_IS_PUMPED] = df[COL_CURRENCY_PAIR] == df[COL_PUMPED_CURRENCY_PAIR]  # attach binary target
        powerlaw_cols: List[str] = FeatureType.POWERLAW_ALPHA.col_names(offsets=REGRESSOR_OFFSETS)
        df[powerlaw_cols] = df[powerlaw_cols].clip(1, 2)
        df_scaled: pd.DataFrame = cross_section_standardisation(df=df)
        df_scaled = fillna_with_median_by_cross_section(df=df_scaled, feature_set=self.feature_set)
        return df_scaled

    def apply_smote(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the SMOTE algorithm to the Dataset
        """
        logging.info("Applying SMOTE")
        X, y = SMOTE(random_state=42).fit_resample(X=df[self.feature_set.regressors], y=df[self.feature_set.target])
        df_train: pd.DataFrame = X
        df_train[self.feature_set.target] = y
        return df_train

    @overrides
    def get_model_params(self, base_params: Dict[str, Any], study_name: str) -> Dict[str, Any]:
        study: Study = optuna.load_study(study_name=study_name, storage=SQLITE_URL)
        model_params: Dict[str, Any] = base_params | study.best_params
        return model_params

    def create_sample(self) -> Sample:
        datasets: Dict[DatasetType, pd.DataFrame] = self.build_datasets()
        df_train: pd.DataFrame = self.apply_smote(df=datasets.get(DatasetType.TRAIN))
        datasets[DatasetType.TRAIN] = df_train
        sample: Sample = Sample.from_pandas(datasets=datasets, feature_set=self.feature_set)
        # we also need to set_pools as Catboost uses Pool under the hood
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
        logging.info("Running <optimize_parameters> for CatboostClassifierSMOTEPipeline")
        sample: Sample = self.create_sample()
        study: Study = create_study(study_name="CatboostClassifierSMOTEPipelineStudy", start_new=True)
        study.optimize(partial(_objective, sample=sample), n_trials=n_trials)
        return study

    def train(self, sample: Sample, tuned: bool = True) -> CatboostClassifierModel:
        model_params: Dict[str, Any] = _BASE_PARAMS
        if tuned:
            model_params = self.get_model_params(
                base_params=_BASE_PARAMS,
                study_name="CatboostClassifierSMOTEPipelineStudy",
            )

        model: CatboostClassifierModel = CatboostClassifierModel(params=model_params)
        model.train(sample=sample)
        return model

    def build_model(self, tuned: bool = True) -> BaseModel:
        logging.info("Running <build_model> for CatboostClassifierSMOTEPipeline")
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
    pipe = CatboostClassifierSMOTEPipeline()
    pipe.optimize_parameters()


if __name__ == "__main__":
    main()
