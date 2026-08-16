import polars as pl

from core.feature_type import FeatureType


def compute_flow_imbalance() -> pl.Expr:
    return (pl.col("quote_sign").sum() / pl.col("quote_abs").sum()).alias(FeatureType.FLOW_IMBALANCE.name)


def compute_return() -> pl.Expr:
    """
    Compute return as simply the change in price between close_prices
    | p0, p1, p2 |t1| p3, p4, p5 |t2| -> return = (p5 / p2 - 1) * 1e4
    return =
    """
    return ((pl.col("price_last").last() / pl.col("price_last_prev").first() - 1) * 1e4).alias(
        FeatureType.ASSET_RETURN.name
    )


def compute_slippage_imbalance() -> pl.Expr:
    return (pl.col("quote_slippage_sign").sum() / pl.col("quote_slippage_abs").sum()).alias(
        FeatureType.SLIPPAGE_IMBALANCE.name
    )


def compute_powerlaw_alpha() -> pl.Expr:
    """
    1 + N / Sum(ln(Q_abs / Q_abs.min()))
    """
    return (1 + pl.len() / (pl.col("quote_abs") / pl.col("quote_abs").min()).log().sum()).alias(
        FeatureType.POWERLAW_ALPHA.name
    )


def compute_share_of_long_trades() -> pl.Expr:
    return (pl.col("is_long").sum() / pl.len()).alias(FeatureType.SHARE_OF_LONG_TRADES.name)


def compute_asset_return_zscore(asset_return_mean: float, asset_return_std: float) -> pl.Expr:
    return (pl.col("asset_return_pips").mean() - asset_return_mean) / asset_return_std


def compute_quote_abs_zscore(quote_abs_mean: float, quote_abs_std: float) -> pl.Expr:
    return (pl.col("quote_abs").mean() - quote_abs_mean) / quote_abs_std


def compute_num_trades() -> pl.Expr:
    return pl.len()
