from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from backtest.utils.columns import COL_IS_PUMPED
from core.feature_type import FeatureType
from features.FeatureWriter import REGRESSOR_OFFSETS


@dataclass
class FeatureSet:

    def __init__(
        self,
        numeric_features: List[str],
        target: str,
        categorical_features: Optional[List[str]] = None,
        eval_fields: Optional[List[str]] = None,
    ):
        self.numeric_features: List[str] = numeric_features
        self.categorical_features: Optional[List[str]] = categorical_features
        self.eval_fields: Optional[List[str]] = eval_fields
        self.target: str = target

    def check_against(self, df: pd.DataFrame) -> None:
        categorical_features = self.categorical_features or []
        assert set(self.numeric_features).issubset(set(df.columns)), "Some numeric features are missing"
        assert set(categorical_features).issubset(set(df.columns)), "Some categorical features are missing"
        assert self.target in df.columns, "Target column is missing"

    @property
    def regressors(self) -> List[str]:
        return self.numeric_features + (self.categorical_features or [])

    @property
    def all_columns(self) -> List[str]:
        return self.numeric_features + (self.categorical_features or []) + (self.eval_fields or [])

    @classmethod
    def auto(cls) -> "FeatureSet":
        """Enumerate every offset-aware feature type in :class:`FeatureType`
        declaration order and produce the flat list of numeric features that
        the pipelines consume.

        Determinism note: this iterates ``FeatureType`` directly (enum
        declaration order) instead of ``set(list(FeatureType))``. The set path
        is nondeterministic across processes because Python's set iteration
        depends on ``PYTHONHASHSEED``; two processes then hand CatBoost
        columns in different orders, and CatBoost's tie-breaking during greedy
        tree splits picks different splits — this was the top source of the
        cross-process training nondeterminism the paper hit.
        """
        numeric_features: List[str] = []

        features_with_offsets: List[FeatureType] = [
            feature_type for feature_type in FeatureType if feature_type != FeatureType.NUM_PREV_PUMP
        ]

        for feature_type in features_with_offsets:
            numeric_features.extend(feature_type.col_names(offsets=REGRESSOR_OFFSETS))

        numeric_features.append(FeatureType.NUM_PREV_PUMP.lower())

        return cls(
            numeric_features=numeric_features,
            target=COL_IS_PUMPED,
            categorical_features=None,
        )

    @classmethod
    def empty(cls) -> "FeatureSet":
        return cls(numeric_features=[], target=COL_IS_PUMPED, categorical_features=None)
