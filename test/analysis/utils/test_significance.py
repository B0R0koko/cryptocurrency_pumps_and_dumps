import numpy as np
import pandas as pd

from backtest.utils.columns import COL_IS_PUMPED, COL_PROBAS_PRED, COL_PUMP_HASH
from backtest.robust.significance import (
    bootstrap_topk_ci,
    bootstrap_topk_percent_auc_ci,
    paired_bootstrap_topk_percent_auc_test,
)


def _build_scored_df(
    n_cross_sections: int,
    cross_section_size: int,
    pumped_rank_fn,
) -> pd.DataFrame:
    rows = []
    for cs_idx in range(n_cross_sections):
        pump_hash = f"pump-{cs_idx}"
        pumped_rank: int = pumped_rank_fn(cs_idx)

        # Higher score = higher rank.
        scores = np.linspace(1.0, 0.0, cross_section_size)
        is_pumped = np.zeros(cross_section_size, dtype=bool)
        is_pumped[pumped_rank] = True

        for row_idx in range(cross_section_size):
            rows.append(
                {
                    COL_PUMP_HASH: pump_hash,
                    COL_IS_PUMPED: bool(is_pumped[row_idx]),
                    COL_PROBAS_PRED: float(scores[row_idx]),
                }
            )

    return pd.DataFrame(rows)


def test_bootstrap_topk_percent_auc_ci_contains_point_estimate() -> None:
    # Half cross-sections have pumped asset at top, half at bottom.
    scored_df = _build_scored_df(
        n_cross_sections=40,
        cross_section_size=10,
        pumped_rank_fn=lambda i: 0 if i % 2 == 0 else 9,
    )

    ci = bootstrap_topk_percent_auc_ci(
        scored_df=scored_df,
        n_bootstrap=300,
        alpha=0.05,
        random_state=7,
    )

    assert ci.ci_lower <= ci.point_estimate <= ci.ci_upper
    assert ci.std_bootstrap > 0
    assert ci.n_bootstrap == 300


def test_bootstrap_topk_ci_returns_expected_shape() -> None:
    scored_df = _build_scored_df(
        n_cross_sections=25,
        cross_section_size=8,
        pumped_rank_fn=lambda _: 0,
    )

    ci_df = bootstrap_topk_ci(
        scored_df=scored_df,
        bins=[1, 2, 5],
        n_bootstrap=200,
        alpha=0.1,
        random_state=11,
    )

    assert list(ci_df.index) == [1, 2, 5]
    assert {"point_estimate", "ci_lower", "ci_upper", "n_bootstrap", "alpha"}.issubset(ci_df.columns)
    assert np.allclose(ci_df["point_estimate"].to_numpy(), np.ones(3))


def test_paired_bootstrap_topk_percent_auc_detects_significant_difference() -> None:
    scored_df_good = _build_scored_df(
        n_cross_sections=50,
        cross_section_size=12,
        pumped_rank_fn=lambda _: 0,
    )
    scored_df_bad = _build_scored_df(
        n_cross_sections=50,
        cross_section_size=12,
        pumped_rank_fn=lambda _: 11,
    )

    n_bootstrap: int = 400
    result = paired_bootstrap_topk_percent_auc_test(
        scored_df_a=scored_df_good,
        scored_df_b=scored_df_bad,
        n_bootstrap=n_bootstrap,
        alpha=0.05,
        random_state=13,
        alternative="greater",
    )

    assert result.observed_diff > 0
    assert result.ci_lower > 0
    # Add-one estimator: even when no bootstrap sample crosses zero, p is
    # 1 / (1 + n_bootstrap) > 0 rather than exactly 0.
    assert result.p_value > 0
    assert result.p_value <= 1.0 / (1.0 + n_bootstrap)


def test_paired_bootstrap_topk_percent_auc_p_value_is_strictly_positive() -> None:
    """The centred-null add-one test always returns a finite positive p-value."""
    scored_df_good = _build_scored_df(
        n_cross_sections=30,
        cross_section_size=12,
        pumped_rank_fn=lambda _: 0,
    )
    scored_df_bad = _build_scored_df(
        n_cross_sections=30,
        cross_section_size=12,
        pumped_rank_fn=lambda _: 11,
    )
    n_bootstrap: int = 200
    for alternative in ("greater", "less", "two-sided"):
        result = paired_bootstrap_topk_percent_auc_test(
            scored_df_a=scored_df_good,
            scored_df_b=scored_df_bad,
            n_bootstrap=n_bootstrap,
            alpha=0.05,
            random_state=13,
            alternative=alternative,
        )
        assert result.p_value > 0.0, f"p_value must be strictly positive for {alternative!r}"
