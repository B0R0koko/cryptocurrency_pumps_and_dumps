import numpy as np
import pandas as pd

from backtest.pipelines.CatboostRanker.pipe import build_return_rank_labels
from backtest.utils.columns import COL_PUMP_ID


def test_highest_return_receives_first_percentile_rank() -> None:
    df = pd.DataFrame(
        {
            COL_PUMP_ID: [0, 0, 0, 1, 1],
            "target_return@5MIN": [10.0, 5.0, -2.0, -1.0, 4.0],
        }
    )

    labels = build_return_rank_labels(df)

    assert np.allclose(labels.iloc[:3].to_numpy(), [1.0 / 3.0, 2.0 / 3.0, 1.0])
    assert np.allclose(labels.iloc[3:].to_numpy(), [1.0, 0.5])
