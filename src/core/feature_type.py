from enum import Enum, auto
from typing import List

from core.time_utils import NamedTimeDelta


class FeatureType(Enum):
    """
    This is the enum for features defined in feature_exprs.py
    """
    ASSET_RETURN = auto()
    ASSET_RETURN_ZSCORE = auto()
    QUOTE_ABS_ZSCORE = auto()
    SHARE_OF_LONG_TRADES = auto()
    POWERLAW_ALPHA = auto()
    SLIPPAGE_IMBALANCE = auto()
    FLOW_IMBALANCE = auto()
    NUM_TRADES = auto()

    def lower(self) -> str:
        return self.name.lower()

    def col_name(self, offset: NamedTimeDelta) -> str:
        """
        Returns names like asset_return@5MIN
        """
        return f"{self.name.lower()}@{offset.get_slug()}"

    def col_names(self, offsets: List[NamedTimeDelta]) -> List[str]:
        """Creates a list of names used in the final dataframe"""
        return [self.col_name(offset=offset) for offset in offsets]
