from typing import List

import pandas as pd


def cross_section_standardize(
    df: pd.DataFrame,
    cols_to_scale: List[str],
    group_col: str = "pump_hash",
) -> pd.DataFrame:
    """
    Apply cross-sectional z-score standardisation within each pump group.

    For every cross-section (identified by *group_col*) each column in
    *cols_to_scale* is replaced by (x - mean) / std computed over that group.

    Returns a new DataFrame with an additional ``pump_id`` column (integer
    index of the cross-section in iteration order).
    """
    if not cols_to_scale:
        result = df.copy().reset_index(drop=True)
        result["pump_id"] = result.groupby(group_col, sort=False).ngroup()
        return result

    result = df.copy()
    grouped = result.groupby(group_col, sort=False)[cols_to_scale]
    means = grouped.transform("mean")
    stds = grouped.transform("std").mask(lambda values: values == 0)
    nuniques = grouped.transform("nunique")
    scaled = (result[cols_to_scale] - means).div(stds)
    result[cols_to_scale] = scaled.mask(nuniques == 1, 0.0)
    result["pump_id"] = result.groupby(group_col, sort=False).ngroup()
    return result.reset_index(drop=True)
