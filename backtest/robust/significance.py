from dataclasses import asdict, dataclass
from typing import Sequence, Literal, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import auc

from backtest.pipelines.BaseModel import ImplementsRank
from backtest.utils.columns import COL_IS_PUMPED, COL_PROBAS_PRED, COL_PUMP_HASH
from backtest.utils.sample import Dataset


@dataclass(frozen=True)
class BootstrapCI:
    metric: str
    point_estimate: float
    mean_bootstrap: float
    std_bootstrap: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int
    alpha: float

    def to_dict(self) -> Dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class PairedBootstrapTestResult:
    metric: str
    observed_diff: float
    mean_diff_bootstrap: float
    ci_lower: float
    ci_upper: float
    p_value: float
    alternative: str
    n_bootstrap: int
    alpha: float

    def to_dict(self) -> Dict[str, float | int | str]:
        return asdict(self)


def score_dataset(model: ImplementsRank, dataset: Dataset) -> pd.DataFrame:
    """
    Evaluate model scores once and keep only the columns required for grouped bootstrap.
    """
    scores: np.ndarray = model.rank(dataset=dataset)
    df: pd.DataFrame = dataset.all_data()[[COL_PUMP_HASH, COL_IS_PUMPED]].copy(deep=True)
    if len(scores) != len(df):
        raise ValueError(f"Model returned {len(scores)} scores, but dataset contains {len(df)} rows")
    df[COL_PROBAS_PRED] = scores
    return df


def _validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def _validate_n_bootstrap(n_bootstrap: int) -> None:
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")


def _cross_section_indicator_vectors_topk(scored_df: pd.DataFrame, bins: Sequence[int]) -> dict[str, np.ndarray]:
    if len(bins) == 0:
        raise ValueError("bins must not be empty")
    if any(k < 1 for k in bins):
        raise ValueError(f"all bins must be >= 1, got {bins}")

    vectors: dict[str, np.ndarray] = {}
    for pump_hash, df_cs in scored_df.groupby(COL_PUMP_HASH, sort=False):
        df_cs = df_cs.sort_values(by=COL_PROBAS_PRED, ascending=False, kind="stable")
        is_pumped: np.ndarray = df_cs[COL_IS_PUMPED].to_numpy(dtype=bool)
        if is_pumped.size == 0 or not is_pumped.any():
            continue
        cum_any: np.ndarray = np.logical_or.accumulate(is_pumped)
        k_idxs = np.clip(np.array(bins, dtype=int), 1, is_pumped.size) - 1
        vectors[str(pump_hash)] = cum_any[k_idxs].astype(float)

    if len(vectors) == 0:
        raise ValueError("scored_df has no non-empty cross-sections")
    return vectors


def _cross_section_indicator_vectors_topk_percent(
    scored_df: pd.DataFrame, bins: Sequence[float]
) -> dict[str, np.ndarray]:
    if len(bins) == 0:
        raise ValueError("bins must not be empty")
    if any(pct_bin < 0 for pct_bin in bins):
        raise ValueError(f"all bins must be >= 0, got {bins}")

    vectors: dict[str, np.ndarray] = {}
    bins_array: np.ndarray = np.array(bins, dtype=float)

    for pump_hash, df_cs in scored_df.groupby(COL_PUMP_HASH, sort=False):
        df_cs = df_cs.sort_values(by=COL_PROBAS_PRED, ascending=False, kind="stable")
        is_pumped: np.ndarray = df_cs[COL_IS_PUMPED].to_numpy(dtype=bool)
        if is_pumped.size == 0 or not is_pumped.any():
            continue
        cum_any: np.ndarray = np.logical_or.accumulate(is_pumped)
        k_raw: np.ndarray = np.ceil(is_pumped.size * bins_array).astype(int)
        positive: np.ndarray = k_raw > 0
        indicator: np.ndarray = np.zeros(bins_array.size, dtype=bool)
        if positive.any():
            k_idxs: np.ndarray = np.clip(k_raw[positive] - 1, 0, is_pumped.size - 1)
            indicator[positive] = cum_any[k_idxs]
        vectors[str(pump_hash)] = indicator.astype(float)

    if len(vectors) == 0:
        raise ValueError("scored_df has no non-empty cross-sections")
    return vectors


def _matrix_from_vectors(vectors: dict[str, np.ndarray], order: Sequence[str]) -> np.ndarray:
    return np.vstack([vectors[pump_hash] for pump_hash in order])


def _bootstrap_indices(n_cross_sections: int, n_bootstrap: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(seed=random_state)
    return rng.integers(0, n_cross_sections, size=(n_bootstrap, n_cross_sections))


def _curve_ci_from_matrix(
    matrix: np.ndarray,
    bins: Sequence[int | float],
    n_bootstrap: int,
    alpha: float,
    random_state: int,
) -> pd.DataFrame:
    _validate_alpha(alpha=alpha)
    _validate_n_bootstrap(n_bootstrap=n_bootstrap)

    n_cross_sections: int = matrix.shape[0]
    sampled_indices: np.ndarray = _bootstrap_indices(
        n_cross_sections=n_cross_sections,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    bootstrap_curves: np.ndarray = matrix[sampled_indices].mean(axis=1)
    point_curve: np.ndarray = matrix.mean(axis=0)

    return pd.DataFrame(
        {
            "point_estimate": point_curve,
            "mean_bootstrap": bootstrap_curves.mean(axis=0),
            "std_bootstrap": bootstrap_curves.std(axis=0, ddof=1),
            "ci_lower": np.quantile(bootstrap_curves, alpha / 2, axis=0),
            "ci_upper": np.quantile(bootstrap_curves, 1 - alpha / 2, axis=0),
            "n_bootstrap": n_bootstrap,
            "alpha": alpha,
        },
        index=bins,
    )


def bootstrap_topk_ci(
    scored_df: pd.DataFrame,
    bins: Sequence[int],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    vectors: dict[str, np.ndarray] = _cross_section_indicator_vectors_topk(scored_df=scored_df, bins=bins)
    order: list[str] = list(vectors.keys())
    matrix: np.ndarray = _matrix_from_vectors(vectors=vectors, order=order)
    return _curve_ci_from_matrix(
        matrix=matrix,
        bins=bins,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        random_state=random_state,
    )


def bootstrap_topk_percent_ci(
    scored_df: pd.DataFrame,
    bins: Sequence[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    vectors: dict[str, np.ndarray] = _cross_section_indicator_vectors_topk_percent(scored_df=scored_df, bins=bins)
    order: list[str] = list(vectors.keys())
    matrix: np.ndarray = _matrix_from_vectors(vectors=vectors, order=order)
    return _curve_ci_from_matrix(
        matrix=matrix,
        bins=bins,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        random_state=random_state,
    )


def _bootstrap_auc_distribution(
    matrix: np.ndarray,
    bins: Sequence[float],
    n_bootstrap: int,
    random_state: int,
    normaliser: float = 1.0,
) -> np.ndarray:
    sampled_indices: np.ndarray = _bootstrap_indices(
        n_cross_sections=matrix.shape[0],
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    bootstrap_curves: np.ndarray = matrix[sampled_indices].mean(axis=1)
    bins_array: np.ndarray = np.array(bins, dtype=float)
    return np.array([auc(x=bins_array, y=curve) / normaliser for curve in bootstrap_curves], dtype=float)


def bootstrap_topk_percent_auc_ci(
    scored_df: pd.DataFrame,
    bins: Sequence[float] | None = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
    max_k_percent: float = 0.20,
    step: float = 0.005,
) -> BootstrapCI:
    if bins is None:
        bins = np.arange(0, max_k_percent + step, step)

    _validate_alpha(alpha=alpha)
    _validate_n_bootstrap(n_bootstrap=n_bootstrap)

    vectors: dict[str, np.ndarray] = _cross_section_indicator_vectors_topk_percent(scored_df=scored_df, bins=bins)
    order: list[str] = list(vectors.keys())
    matrix: np.ndarray = _matrix_from_vectors(vectors=vectors, order=order)
    bins_array: np.ndarray = np.array(bins, dtype=float)
    normaliser: float = float(bins_array[-1] - bins_array[0])
    if normaliser <= 0:
        raise ValueError(f"bins range must be positive, got {bins_array[0]}..{bins_array[-1]}")

    point_curve: np.ndarray = matrix.mean(axis=0)
    point_estimate: float = float(auc(x=bins_array, y=point_curve)) / normaliser
    bootstrap_aucs: np.ndarray = _bootstrap_auc_distribution(
        matrix=matrix,
        bins=bins,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        normaliser=normaliser,
    )
    ci_lower, ci_upper = np.quantile(bootstrap_aucs, [alpha / 2, 1 - alpha / 2])

    return BootstrapCI(
        metric="topk_percent_auc",
        point_estimate=point_estimate,
        mean_bootstrap=float(np.mean(bootstrap_aucs)),
        std_bootstrap=float(np.std(bootstrap_aucs, ddof=1)),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )


def paired_bootstrap_topk_percent_auc_test(
    scored_df_a: pd.DataFrame,
    scored_df_b: pd.DataFrame,
    bins: Sequence[float] | None = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    max_k_percent: float = 0.20,
    step: float = 0.005,
) -> PairedBootstrapTestResult:
    if bins is None:
        bins = np.arange(0, max_k_percent + step, step)

    _validate_alpha(alpha=alpha)
    _validate_n_bootstrap(n_bootstrap=n_bootstrap)
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(f"Unknown alternative={alternative}")

    vectors_a: dict[str, np.ndarray] = _cross_section_indicator_vectors_topk_percent(scored_df=scored_df_a, bins=bins)
    vectors_b: dict[str, np.ndarray] = _cross_section_indicator_vectors_topk_percent(scored_df=scored_df_b, bins=bins)

    pump_hashes_a = set(vectors_a.keys())
    pump_hashes_b = set(vectors_b.keys())
    if pump_hashes_a != pump_hashes_b:
        missing_a = pump_hashes_b - pump_hashes_a
        missing_b = pump_hashes_a - pump_hashes_b
        raise ValueError(
            f"Cross-sections must match for paired bootstrap. Missing in A: {len(missing_a)}; Missing in B: {len(missing_b)}"
        )

    order: list[str] = list(vectors_a.keys())
    matrix_a: np.ndarray = _matrix_from_vectors(vectors=vectors_a, order=order)
    matrix_b: np.ndarray = _matrix_from_vectors(vectors=vectors_b, order=order)
    bins_array: np.ndarray = np.array(bins, dtype=float)
    normaliser: float = float(bins_array[-1] - bins_array[0])
    if normaliser <= 0:
        raise ValueError(f"bins range must be positive, got {bins_array[0]}..{bins_array[-1]}")

    observed_diff: float = (
        float(auc(x=bins_array, y=matrix_a.mean(axis=0)) - auc(x=bins_array, y=matrix_b.mean(axis=0))) / normaliser
    )

    sampled_indices: np.ndarray = _bootstrap_indices(
        n_cross_sections=matrix_a.shape[0],
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    bootstrap_curves_a: np.ndarray = matrix_a[sampled_indices].mean(axis=1)
    bootstrap_curves_b: np.ndarray = matrix_b[sampled_indices].mean(axis=1)
    auc_dist_a: np.ndarray = np.array(
        [auc(x=bins_array, y=curve) / normaliser for curve in bootstrap_curves_a], dtype=float
    )
    auc_dist_b: np.ndarray = np.array(
        [auc(x=bins_array, y=curve) / normaliser for curve in bootstrap_curves_b], dtype=float
    )
    diffs: np.ndarray = auc_dist_a - auc_dist_b

    ci_lower, ci_upper = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    # Build the bootstrap null distribution by centring the paired per-event
    # contributions at zero. Counting signs in the ordinary (uncentred)
    # confidence-interval distribution is not a valid test of H0: difference=0.
    per_cross_section_auc_a: np.ndarray = np.trapezoid(matrix_a, x=bins_array, axis=1) / normaliser
    per_cross_section_auc_b: np.ndarray = np.trapezoid(matrix_b, x=bins_array, axis=1) / normaliser
    paired_contributions: np.ndarray = per_cross_section_auc_a - per_cross_section_auc_b
    centred_contributions: np.ndarray = paired_contributions - paired_contributions.mean()
    null_diffs: np.ndarray = centred_contributions[sampled_indices].mean(axis=1)

    # Add-one Monte Carlo estimator: a finite resample cannot report p=0.
    if alternative == "greater":
        p_value: float = float((1 + int(np.sum(null_diffs >= observed_diff))) / (1 + n_bootstrap))
    elif alternative == "less":
        p_value = float((1 + int(np.sum(null_diffs <= observed_diff))) / (1 + n_bootstrap))
    else:
        p_value = float(
            (1 + int(np.sum(np.abs(null_diffs) >= abs(observed_diff)))) / (1 + n_bootstrap)
        )

    return PairedBootstrapTestResult(
        metric="topk_percent_auc",
        observed_diff=observed_diff,
        mean_diff_bootstrap=float(np.mean(diffs)),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=p_value,
        alternative=alternative,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )
