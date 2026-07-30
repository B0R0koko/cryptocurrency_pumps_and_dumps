from typing import Dict, Iterable, Literal

import numpy as np
import pandas as pd

from backtest.pipelines.BaseModel import ImplementsRank
from backtest.utils.columns import COL_PROBAS_PRED, COL_PUMP_HASH, COL_IS_PUMPED
from backtest.utils.sample import Dataset
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    f1_score,
    balanced_accuracy_score,
)


def calculate_topk(model: ImplementsRank, dataset: Dataset, bins: Iterable[float]) -> pd.Series:
    """
    :param bins: bins used to calculate topk
    :return: pd.Series with Top@k accuracy per bin. The value at ``k`` is the share of
        pump-carrying cross-sections whose top-K assets (ranked by model score) contain
        the pumped asset. The denominator is the number of cross-sections that contain
        at least one pumped row, not the number of pumped rows.
    """
    probas_pred: np.ndarray = model.rank(dataset=dataset)
    _df: pd.DataFrame = dataset.all_data().copy()
    _df[COL_PROBAS_PRED] = probas_pred

    count_by_bins: Dict[float, int] = {}
    num_cross_sections_with_pump: int = 0

    for pump_hash, df_cross_section in _df.groupby(COL_PUMP_HASH):
        if not df_cross_section[COL_IS_PUMPED].any():
            continue
        num_cross_sections_with_pump += 1
        df_cross_section = df_cross_section.sort_values(by=COL_PROBAS_PRED, ascending=False).reset_index(drop=True)
        for K in bins:
            contains_pump: bool = df_cross_section.iloc[:K][COL_IS_PUMPED].any()
            count_by_bins[K] = count_by_bins.get(K, 0) + int(contains_pump)

    counts = np.array(list(count_by_bins.values()))
    if num_cross_sections_with_pump == 0:
        return pd.Series(data=np.zeros(len(counts)), index=list(bins))

    return pd.Series(data=counts / num_cross_sections_with_pump, index=list(bins))


def calculate_topk_percent(model: ImplementsRank, dataset: Dataset, bins: Iterable[float]) -> pd.Series:
    """
    :param bins: bins used to calculate topk. K measures the share of cross-section taken as a portfolio
    :return: pd.Series with Top@k% accuracy per bin. The denominator is the number of
        cross-sections that contain at least one pumped row.
    """
    probas_pred: np.ndarray = model.rank(dataset=dataset)
    _df: pd.DataFrame = dataset.all_data().copy()
    _df[COL_PROBAS_PRED] = probas_pred

    count_by_bins: Dict[float, int] = {}
    num_cross_sections_with_pump: int = 0

    for pump_hash, df_cross_section in _df.groupby(COL_PUMP_HASH):
        if not df_cross_section[COL_IS_PUMPED].any():
            continue
        num_cross_sections_with_pump += 1
        df_cross_section = df_cross_section.sort_values(by=COL_PROBAS_PRED, ascending=False)
        n_rows = len(df_cross_section)

        for pct_bin in bins:
            k: int = int(np.ceil(n_rows * pct_bin))
            contains_pump: bool = df_cross_section.iloc[:k][COL_IS_PUMPED].any() if k > 0 else False
            count_by_bins[pct_bin] = count_by_bins.get(pct_bin, 0) + int(contains_pump)

    counts = np.array(list(count_by_bins.values()))
    if num_cross_sections_with_pump == 0:
        return pd.Series(data=np.zeros(len(counts)), index=list(bins))

    return pd.Series(data=counts / num_cross_sections_with_pump, index=list(bins))


def calculate_topk_percent_auc(
    model: ImplementsRank,
    dataset: Dataset,
    max_k_percent: float = 0.20,
    step: float = 0.005,
) -> float:
    """
    Compute the area under the Top@k% accuracy curve over ``k% in (0, max_k_percent]`` (Top@k%AUC) and
    normalise by the integration range so the result stays in ``(0, 1)``.

    Restricting the range to the steep, low-K% region (default 0-20%) makes the metric
    much more sensitive to differences between models and hyperparameters. At higher K%
    all reasonable models saturate near 1.0 and the AUC becomes flat.
    """
    bins: np.ndarray = np.arange(0, max_k_percent + step, step)
    topk_percentages: pd.Series = calculate_topk_percent(model=model, dataset=dataset, bins=bins)
    raw_auc: float = float(auc(x=topk_percentages.index, y=topk_percentages.values))
    return raw_auc / max_k_percent


def _with_scores(model: ImplementsRank, dataset: Dataset) -> pd.DataFrame:
    scores: np.ndarray = model.rank(dataset=dataset)
    # Ensure positional numpy indexing remains valid even when input dataframe has
    # non-contiguous or non-zero-based index values.
    df_scored: pd.DataFrame = dataset.all_data().copy().reset_index(drop=True)
    df_scored[COL_PROBAS_PRED] = scores
    return df_scored


def _predict_labels(
    df_scored: pd.DataFrame,
    decision_rule: Literal["top1_per_cross_section", "threshold"],
    threshold: float,
) -> np.ndarray:
    if decision_rule == "top1_per_cross_section":
        pred_labels = np.zeros(df_scored.shape[0], dtype=int)
        top_indices: np.ndarray = (
            df_scored.groupby(COL_PUMP_HASH, sort=False)[COL_PROBAS_PRED].idxmax().to_numpy(dtype=int)
        )
        pred_labels[top_indices] = 1
        return pred_labels

    if decision_rule == "threshold":
        return (df_scored[COL_PROBAS_PRED].to_numpy() >= threshold).astype(int)

    raise ValueError(f"Unknown decision_rule={decision_rule}")


def calculate_f1(
    model: ImplementsRank,
    dataset: Dataset,
    decision_rule: Literal["top1_per_cross_section", "threshold"] = "top1_per_cross_section",
    threshold: float = 0.5,
) -> float:
    df_scored: pd.DataFrame = _with_scores(model=model, dataset=dataset)
    y_true: np.ndarray = df_scored[COL_IS_PUMPED].to_numpy(dtype=int)
    y_pred: np.ndarray = _predict_labels(df_scored=df_scored, decision_rule=decision_rule, threshold=threshold)
    return float(f1_score(y_true=y_true, y_pred=y_pred, zero_division=0))


def calculate_balanced_accuracy(
    model: ImplementsRank,
    dataset: Dataset,
    decision_rule: Literal["top1_per_cross_section", "threshold"] = "top1_per_cross_section",
    threshold: float = 0.5,
) -> float:
    df_scored: pd.DataFrame = _with_scores(model=model, dataset=dataset)
    y_true: np.ndarray = df_scored[COL_IS_PUMPED].to_numpy(dtype=int)
    y_pred: np.ndarray = _predict_labels(df_scored=df_scored, decision_rule=decision_rule, threshold=threshold)
    return float(balanced_accuracy_score(y_true=y_true, y_pred=y_pred))


def calculate_pr_auc(model: ImplementsRank, dataset: Dataset) -> float:
    df_scored: pd.DataFrame = _with_scores(model=model, dataset=dataset)
    y_true: np.ndarray = df_scored[COL_IS_PUMPED].to_numpy(dtype=int)
    y_score: np.ndarray = df_scored[COL_PROBAS_PRED].to_numpy(dtype=float)
    precision: np.ndarray
    recall: np.ndarray
    precision, recall, _ = precision_recall_curve(y_true=y_true, y_score=y_score)
    return float(auc(x=recall, y=precision))
