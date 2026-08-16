import numpy as np
import pandas as pd

from backtest.utils.preprocessing import cross_section_standardize


def test_cross_section_standardize_handles_constant_and_missing_columns() -> None:
    df = pd.DataFrame(
        {
            "pump_hash": ["a", "a", "b", "b"],
            "feature": [5.0, 5.0, 1.0, 3.0],
            "missing": [np.nan, np.nan, np.nan, np.nan],
        }
    )

    result = cross_section_standardize(df, cols_to_scale=["feature", "missing"])

    assert result.loc[result["pump_hash"] == "a", "feature"].eq(0.0).all()
    varying = result.loc[result["pump_hash"] == "b", "feature"]
    assert np.isclose(varying.mean(), 0.0)
    assert result["missing"].isna().all()
    assert result["pump_id"].tolist() == [0, 0, 1, 1]


def test_cross_section_standardize_accepts_empty_column_list() -> None:
    df = pd.DataFrame({"pump_hash": ["b", "a", "b"], "value": [1.0, 2.0, 3.0]})

    result = cross_section_standardize(df, cols_to_scale=[])

    assert result["value"].tolist() == [1.0, 2.0, 3.0]
    assert result["pump_id"].tolist() == [0, 1, 0]
