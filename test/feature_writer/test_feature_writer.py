from datetime import datetime, timedelta

import numpy as np
import polars as pl

import features.FeatureWriter as feature_writer_module
from core.columns import TRADE_TIME, PRICE, QUANTITY, IS_BUYER_MAKER
from core.currency_pair import CurrencyPair
from core.exchange import Exchange
from core.feature_type import FeatureType
from core.pump_event import PumpEvent
from core.time_utils import NamedTimeDelta
from features.FeatureWriter import DECISION_LAG, PumpsFeatureWriter


def _build_writer(pump_events: list[PumpEvent]) -> PumpsFeatureWriter:
    writer = object.__new__(PumpsFeatureWriter)
    writer._pump_events = pump_events
    writer._pump_times_by_currency = {}
    for pump_event in sorted(pump_events, key=lambda event: event.time):
        writer.pump_times_by_currency.setdefault(pump_event.currency_pair.name, []).append(pump_event.time)
    return writer


def test_preprocess_data_for_currency_aggregates_ticks_into_trades() -> None:
    writer: PumpsFeatureWriter = _build_writer(pump_events=[])
    t1: datetime = datetime(2021, 1, 1, 8, 0, 0)
    t2: datetime = datetime(2021, 1, 1, 8, 1, 0)

    df_ticks: pl.DataFrame = pl.DataFrame(
        {
            TRADE_TIME: [t1, t1, t2, t2],
            PRICE: [10.0, 12.0, 20.0, 18.0],
            QUANTITY: [2.0, 1.0, 1.0, 3.0],
            IS_BUYER_MAKER: [False, True, True, True],
        }
    )

    df_trades: pl.DataFrame = writer.preprocess_data_for_currency(df=df_ticks)
    assert df_trades.shape[0] == 2

    row0 = df_trades.row(0, named=True)
    row1 = df_trades.row(1, named=True)

    assert np.isclose(row0["price_first"], 10.0)
    assert np.isclose(row0["price_last"], 12.0)
    assert np.isclose(row0["quote_abs"], 32.0)
    assert np.isclose(row0["quote_sign"], 8.0)
    assert np.isclose(row0["quantity_sign"], 1.0)
    assert bool(row0["is_long"]) is True
    assert np.isclose(row0["quote_slippage_abs"], 2.0)
    assert np.isclose(row0["quote_slippage_sign"], 2.0)

    assert np.isclose(row1["price_first"], 20.0)
    assert np.isclose(row1["price_last"], 18.0)
    assert np.isclose(row1["quote_abs"], 74.0)
    assert np.isclose(row1["quote_sign"], -74.0)
    assert np.isclose(row1["quantity_sign"], -4.0)
    assert bool(row1["is_long"]) is False
    assert np.isclose(row1["quote_slippage_abs"], 6.0)
    assert np.isclose(row1["quote_slippage_sign"], -6.0)
    assert np.isclose(row1["price_last_prev"], 12.0)
    assert row1["trade_time_prev"] == t1


def test_compute_features_matches_feature_definitions(monkeypatch) -> None:
    monkeypatch.setattr(feature_writer_module, "REGRESSOR_OFFSETS", [NamedTimeDelta.ONE_HOUR])
    monkeypatch.setattr(feature_writer_module, "DECAY_OFFSETS", [NamedTimeDelta.ONE_MINUTE])

    currency_pair: CurrencyPair = CurrencyPair.from_string("AAA-BTC")
    other_pair: CurrencyPair = CurrencyPair.from_string("BBB-BTC")

    # Pump time chosen so that ``rb = T - 15min`` lands on an hour boundary
    # (``T = 10:15``, ``rb = 10:00``). Bar labels aligned to natural hours
    # therefore neatly cover the 1H feature window ``[rb - 1h, rb) = [9:00, 10:00)``.
    pump_event: PumpEvent = PumpEvent(
        currency_pair=currency_pair,
        time=datetime(2021, 1, 1, 10, 15, 0),
        exchange=Exchange.BINANCE_SPOT,
    )
    writer: PumpsFeatureWriter = _build_writer(
        pump_events=[
            PumpEvent(
                currency_pair=currency_pair,
                time=datetime(2020, 1, 1, 0, 0, 0),
                exchange=Exchange.BINANCE_SPOT,
            ),
            PumpEvent(
                currency_pair=other_pair,
                time=datetime(2020, 3, 1, 0, 0, 0),
                exchange=Exchange.BINANCE_SPOT,
            ),
            PumpEvent(
                currency_pair=currency_pair,
                time=datetime(2020, 6, 1, 0, 0, 0),
                exchange=Exchange.BINANCE_SPOT,
            ),
            PumpEvent(
                currency_pair=currency_pair,
                time=datetime(2021, 2, 1, 0, 0, 0),
                exchange=Exchange.BINANCE_SPOT,
            ),
            pump_event,
        ]
    )

    df: pl.DataFrame = pl.DataFrame(
        {
            TRADE_TIME: [
                datetime(2021, 1, 1, 7, 10, 0),
                datetime(2021, 1, 1, 7, 40, 0),
                datetime(2021, 1, 1, 8, 30, 0),
                datetime(2021, 1, 1, 9, 0, 0),  # included lower boundary of 1H window (lb=9:00)
                datetime(2021, 1, 1, 9, 45, 0),
                datetime(2021, 1, 1, 10, 15, 10),  # first trade used for the 1MIN target
                datetime(2021, 1, 1, 10, 15, 40),
            ],
            "price_first": [100.0, 101.0, 200.0, 102.0, 103.0, 110.0, 115.0],
            "price_last": [101.0, 102.0, 400.0, 103.0, 108.0, 115.0, 120.0],
            "price_last_prev": [99.0, 100.0, 1.0, 102.0, 103.0, 109.0, 114.0],
            "quote_abs": [10.0, 20.0, 1000.0, 30.0, 50.0, 60.0, 70.0],
            "quote_sign": [5.0, -5.0, 1000.0, 10.0, 20.0, 30.0, 35.0],
            "quote_slippage_abs": [1.0, 2.0, 100.0, 3.0, 5.0, 6.0, 7.0],
            "quote_slippage_sign": [1.0, -2.0, 100.0, 3.0, 2.0, 3.0, 4.0],
            "is_long": [True, False, True, True, True, True, True],
        }
    )

    features = writer.compute_features(df=df, currency_pair=currency_pair, pump_event=pump_event)
    window: NamedTimeDelta = NamedTimeDelta.ONE_HOUR
    rb: datetime = pump_event.time - DECISION_LAG
    lb: datetime = rb - window.get_td()

    # 1H feature window (left-closed) should catch the two trades at 9:00 and 9:45.
    df_window: pl.DataFrame = df.filter(pl.col(TRADE_TIME).is_between(lb, rb, closed="left"))
    assert df_window.shape[0] == 2

    expected_return = ((108.0 / 102.0) - 1.0) * 1e4
    expected_share_long = 1.0
    expected_powerlaw = 1.0 + 2.0 / np.log(50.0 / 30.0)
    expected_slippage_imbalance = (3.0 + 2.0) / (3.0 + 5.0)
    expected_flow_imbalance = (10.0 + 20.0) / (30.0 + 50.0)
    expected_num_trades = 2
    expected_target_return = ((120.0 / 109.0) - 1.0) * 1e4

    # Rebuild the normaliser exactly the way the writer does: truncate trades
    # to strictly-pre-rb, aggregate into hourly bars anchored to rb via
    # ``group_by_dynamic(offset=<rb-mod-1h>)``. When ``rb`` lands on a calendar
    # hour boundary the anchor offset is zero and rb-anchored bars coincide
    # with the natural-hour bars.
    df_pre_rb = df.filter(pl.col(TRADE_TIME) < rb)
    offset_us: int = (rb.minute * 60 + rb.second) * 1_000_000 + rb.microsecond
    hourly = (
        df_pre_rb.group_by_dynamic(
            index_column=TRADE_TIME,
            period=timedelta(hours=1),
            every=timedelta(hours=1),
            offset=f"{offset_us}us",
        )
        .agg(
            asset_return_pips=(pl.col("price_last").last() / pl.col("price_first").first() - 1.0) * 1e4,
            quote_abs=pl.col("quote_abs").sum(),
        )
        .sort(TRADE_TIME)
    )
    hourly_full = hourly.filter(pl.col(TRADE_TIME) + pl.duration(hours=1) <= rb)
    # rb=10:00 sits on the hour boundary so bars [7:00,8:00), [8:00,9:00),
    # [9:00,10:00) are all full pre-rb hours and kept by the normaliser.
    assert hourly_full.shape[0] == 3

    asset_return_mean = float(np.mean(hourly_full["asset_return_pips"].to_numpy()))
    asset_return_std = float(np.std(hourly_full["asset_return_pips"].to_numpy(), ddof=1))
    quote_abs_mean = float(np.mean(hourly_full["quote_abs"].to_numpy()))
    quote_abs_std = float(np.std(hourly_full["quote_abs"].to_numpy(), ddof=1))

    expected_asset_return_zscore = (588.2352941176471 - asset_return_mean) / asset_return_std
    expected_quote_abs_zscore = (80.0 - quote_abs_mean) / quote_abs_std

    assert np.isclose(features[FeatureType.ASSET_RETURN.col_name(window)], expected_return)
    assert np.isclose(
        features[FeatureType.ASSET_RETURN_ZSCORE.col_name(window)],
        expected_asset_return_zscore,
    )
    assert np.isclose(
        features[FeatureType.QUOTE_ABS_ZSCORE.col_name(window)],
        expected_quote_abs_zscore,
    )
    assert np.isclose(features[FeatureType.SHARE_OF_LONG_TRADES.col_name(window)], expected_share_long)
    assert np.isclose(features[FeatureType.POWERLAW_ALPHA.col_name(window)], expected_powerlaw)
    assert np.isclose(
        features[FeatureType.SLIPPAGE_IMBALANCE.col_name(window)],
        expected_slippage_imbalance,
    )
    assert np.isclose(features[FeatureType.FLOW_IMBALANCE.col_name(window)], expected_flow_imbalance)
    assert np.isclose(features[FeatureType.NUM_TRADES.col_name(window)], expected_num_trades)
    assert features[FeatureType.NUM_PREV_PUMP.lower()] == 2
    assert np.isclose(features["target_return@1MIN"], expected_target_return)


def _calm_trade(ts: datetime) -> dict:
    return {
        TRADE_TIME: ts,
        "price_first": 100.0,
        "price_last": 100.0,
        "price_last_prev": 100.0,
        "quote_abs": 1000.0,
        "quote_sign": 500.0,
        "quote_slippage_abs": 1.0,
        "quote_slippage_sign": 1.0,
        "is_long": True,
    }


def _explosive_trade(ts: datetime, *, price: float, prev_price: float, quote_abs: float) -> dict:
    return {
        TRADE_TIME: ts,
        "price_first": price,
        "price_last": price,
        "price_last_prev": prev_price,
        "quote_abs": quote_abs,
        "quote_sign": quote_abs,
        "quote_slippage_abs": quote_abs / 100.0,
        "quote_slippage_sign": quote_abs / 100.0,
        "is_long": True,
    }


def _build_pre_pump_calm_trades(pump_time: datetime) -> pl.DataFrame:
    """Three calm pre-``rb`` hours of trades at price 100 and volume 1000.

    Each hour bucket contains four evenly spaced trades. Trades are anchored on
    natural hour boundaries relative to the pump minute so the resulting hourly
    aggregates cover the full pre-``rb`` window regardless of the exact minute
    of the pump.
    """
    rows: list[dict] = []
    hour_floor: datetime = pump_time.replace(minute=0, second=0, microsecond=0)
    for offset_hours in (-3, -2, -1):
        base: datetime = hour_floor + timedelta(hours=offset_hours)
        for minute in (0, 15, 30, 45):
            rows.append(_calm_trade(base + timedelta(minutes=minute)))
    return pl.DataFrame(rows)


def _build_extreme_post_pump_trades(pump_time: datetime) -> pl.DataFrame:
    """Massive-price, massive-volume trades placed strictly after the pump.

    Any leak of these bars into the z-score normalisers would blow up
    ``asset_return_std`` and ``quote_abs_std`` by several orders of magnitude.
    """
    return pl.DataFrame(
        [
            _explosive_trade(pump_time + timedelta(minutes=5), price=500.0, prev_price=100.0, quote_abs=1e7),
            _explosive_trade(pump_time + timedelta(minutes=30), price=800.0, prev_price=500.0, quote_abs=5e7),
        ]
    )


def _assert_features_equal(a: dict, b: dict, keys: list[str], where: str) -> None:
    """Compare feature dicts entry-by-entry, treating ``None`` as equal to ``None``.

    Under the new mid-hour ``rb`` geometry (T=15min lag) the 1H z-score can be
    ``None`` when no hourly bar fits the ``[lb, rb)`` window as a full pre-rb
    hour. That is the correct behaviour for both the clean and the leaky
    fixtures, and ``np.isclose(None, None)`` raises, so we short-circuit here.
    """
    for key in keys:
        va, vb = a[key], b[key]
        if va is None and vb is None:
            continue
        if va is None or vb is None:
            raise AssertionError(f"{where}: {key} differs (clean={va!r} leaky={vb!r})")
        assert np.isclose(va, vb, equal_nan=True), f"{where}: {key} differs (clean={va!r} leaky={vb!r})"


_ALL_REGRESSOR_OFFSETS: list[NamedTimeDelta] = [
    NamedTimeDelta.FIVE_MINUTES,
    NamedTimeDelta.FIFTEEN_MINUTES,
    NamedTimeDelta.ONE_HOUR,
    NamedTimeDelta.TWO_HOURS,
    NamedTimeDelta.FOUR_HOURS,
]


def _build_rich_pre_rb_trades(pump_time: datetime) -> pl.DataFrame:
    """Six pre-``rb`` hours of varied trades used by the rb-anchored leak tests.

    Trades are placed on a 5-minute cadence relative to ``rb`` so that every
    rb-anchored hourly bucket collects at least one tick. Prices and quote
    volumes drift monotonically across bars so the resulting hourly aggregates
    have non-zero variance (bar-to-bar differences); this is what lets the
    z-score denominator be finite and non-zero on the synthetic fixture.
    """
    rb: datetime = pump_time - DECISION_LAG
    rows: list[dict] = []
    for k in range(1, 6 * 12 + 1):  # 6h at 5-minute spacing = 72 trades
        ts: datetime = rb - timedelta(minutes=5 * k)
        bar_idx: int = (k - 1) // 12  # 0 = newest hourly bar, 5 = oldest
        intra: int = (k - 1) % 12  # position within the bar
        rows.append(
            {
                TRADE_TIME: ts,
                "price_first": 100.0 + bar_idx + 0.01 * intra,
                "price_last": 100.0 + bar_idx + 0.01 * intra + 0.005,
                "price_last_prev": 100.0 + bar_idx + 0.01 * intra - 0.005,
                # Monotone bar-by-bar volume: bar k has total ~ 12*(500 + 100*k).
                "quote_abs": 500.0 + 100.0 * bar_idx + intra,
                "quote_sign": 250.0 + 50.0 * bar_idx + intra * 0.5,
                "quote_slippage_abs": 5.0 + bar_idx + intra * 0.1,
                "quote_slippage_sign": 5.0 + bar_idx + intra * 0.1,
                "is_long": intra % 3 != 0,
            }
        )
    return pl.DataFrame(rows).sort(TRADE_TIME)


def _build_post_rb_extreme_trades(pump_time: datetime) -> pl.DataFrame:
    """Extreme trades inside ``[rb, T]`` AND after ``T``, all of which must have
    zero effect on any feature under the rb-anchored geometry."""
    rb: datetime = pump_time - DECISION_LAG
    return pl.DataFrame(
        [
            _explosive_trade(rb, price=1e6, prev_price=100.0, quote_abs=1e9),  # exactly at rb
            _explosive_trade(rb + timedelta(seconds=1), price=2e6, prev_price=1e6, quote_abs=2e9),  # in [rb, T)
            _explosive_trade(pump_time, price=3e6, prev_price=2e6, quote_abs=3e9),  # pump minute
            _explosive_trade(pump_time + timedelta(minutes=30), price=5e6, prev_price=3e6, quote_abs=5e9),
            _explosive_trade(pump_time + timedelta(hours=2), price=1e7, prev_price=5e6, quote_abs=1e10),
        ]
    )


def _calendar_anchored_compute_features(writer: PumpsFeatureWriter, df: pl.DataFrame, pump_event: PumpEvent) -> dict:
    """Deliberately mis-anchored reference implementation used to prove the leak
    tests catch mis-anchoring. This mimics the previous "calendar-hour bars +
    filter(label < rb) + drop partial" pipeline, which admits the calendar bar
    covering ``[floor_hour(rb), floor_hour(rb) + 1h)`` when ``rb`` sits mid-hour
    — a bar whose data spans past ``rb`` into ``[rb, T]``.
    """
    from features.feature_exprs import (
        compute_asset_return_zscore,
        compute_quote_abs_zscore,
        compute_return,
        compute_share_of_long_trades,
        compute_powerlaw_alpha,
        compute_slippage_imbalance,
        compute_flow_imbalance,
        compute_num_trades,
    )

    features: dict = {}
    rb: datetime = pump_event.time - DECISION_LAG
    df_hourly = df.group_by_dynamic(
        index_column=TRADE_TIME,
        every=timedelta(hours=1),
        period=timedelta(hours=1),
    ).agg(
        asset_return_pips=(pl.col("price_last").last() / pl.col("price_first").first() - 1) * 1e4,
        quote_abs=pl.col("quote_abs").sum(),
    )
    df_hourly_pre_rb = df_hourly.filter(pl.col(TRADE_TIME) < rb)
    asset_return_mean = df_hourly_pre_rb.select(pl.col("asset_return_pips").mean()).item()
    asset_return_std = df_hourly_pre_rb.select(pl.col("asset_return_pips").std()).item()
    quote_abs_mean = df_hourly_pre_rb.select(pl.col("quote_abs").mean()).item()
    quote_abs_std = df_hourly_pre_rb.select(pl.col("quote_abs").std()).item()

    for offset in feature_writer_module.REGRESSOR_OFFSETS:
        lb = rb - offset.get_td()
        df_filtered = df.filter(pl.col(TRADE_TIME).is_between(lb, rb))
        df_hourly_filtered = df_hourly_pre_rb.filter(pl.col(TRADE_TIME).is_between(lb, rb))
        w = df_filtered.select(
            compute_return().alias("asset_return"),
            compute_share_of_long_trades().alias("share_of_long_trades"),
            compute_powerlaw_alpha().alias("powerlaw_alpha"),
            compute_slippage_imbalance().alias("slippage_imbalance"),
            compute_flow_imbalance().alias("flow_imbalance"),
            compute_num_trades().alias("num_trades"),
        ).to_dicts()[0]
        h = df_hourly_filtered.select(
            compute_asset_return_zscore(
                asset_return_mean=asset_return_mean,
                asset_return_std=asset_return_std,
            ).alias("asset_return_zscore"),
            compute_quote_abs_zscore(quote_abs_mean=quote_abs_mean, quote_abs_std=quote_abs_std).alias(
                "quote_abs_zscore"
            ),
        ).to_dicts()[0]
        features[FeatureType.ASSET_RETURN.col_name(offset)] = w["asset_return"]
        features[FeatureType.SHARE_OF_LONG_TRADES.col_name(offset)] = w["share_of_long_trades"]
        features[FeatureType.POWERLAW_ALPHA.col_name(offset)] = w["powerlaw_alpha"]
        features[FeatureType.SLIPPAGE_IMBALANCE.col_name(offset)] = w["slippage_imbalance"]
        features[FeatureType.FLOW_IMBALANCE.col_name(offset)] = w["flow_imbalance"]
        features[FeatureType.NUM_TRADES.col_name(offset)] = w["num_trades"]
        features[FeatureType.ASSET_RETURN_ZSCORE.col_name(offset)] = h["asset_return_zscore"]
        features[FeatureType.QUOTE_ABS_ZSCORE.col_name(offset)] = h["quote_abs_zscore"]
    return features


def test_newest_rb_anchored_bar_ends_exactly_at_rb() -> None:
    """The newest hourly bar produced by the rb-anchored geometry must end
    exactly at ``rb`` for arbitrary sub-second ``rb`` values (this is what
    guarantees zero overlap with ``[rb, T]``).
    """
    for pump_time in (
        datetime(2021, 6, 1, 12, 0, 0),  # rb on the hour
        datetime(2021, 6, 1, 12, 15, 0),  # rb on quarter-hour
        datetime(2021, 6, 1, 12, 20, 0),  # rb mid-hour
        datetime(2021, 6, 1, 12, 20, 37, 123456),  # rb with sub-second
    ):
        rb: datetime = pump_time - DECISION_LAG
        df = _build_rich_pre_rb_trades(pump_time=pump_time)
        offset_us: int = (rb.minute * 60 + rb.second) * 1_000_000 + rb.microsecond
        hourly = (
            df.filter(pl.col(TRADE_TIME) < rb)
            .group_by_dynamic(
                index_column=TRADE_TIME,
                every=timedelta(hours=1),
                period=timedelta(hours=1),
                offset=f"{offset_us}us",
            )
            .agg(quote_abs=pl.col("quote_abs").sum())
            .sort(TRADE_TIME)
        )
        assert hourly.shape[0] >= 1, f"no bars for pump_time={pump_time}"
        newest_left = hourly.tail(1)[TRADE_TIME].item()
        newest_right = newest_left + timedelta(hours=1)
        assert newest_right == rb, f"pump_time={pump_time}: newest bar ends at {newest_right}, not rb={rb}"


def test_no_feature_can_see_post_rb_trades(monkeypatch) -> None:
    """No feature — z-score OR in-window — must react to trades placed in
    ``[rb, T]`` or after ``T``. Under the rb-anchored geometry the newest bar
    ends exactly at ``rb``, so this must hold across arbitrary pump minutes.
    """
    monkeypatch.setattr(feature_writer_module, "REGRESSOR_OFFSETS", _ALL_REGRESSOR_OFFSETS)
    monkeypatch.setattr(feature_writer_module, "DECAY_OFFSETS", [NamedTimeDelta.ONE_MINUTE])

    currency_pair: CurrencyPair = CurrencyPair.from_string("AAA-BTC")
    # A menu of pump minutes: on hour, quarter hour, awkward mid-hour, sub-second.
    for pump_time in (
        datetime(2021, 6, 1, 12, 0, 0),
        datetime(2021, 6, 1, 12, 15, 0),
        datetime(2021, 6, 1, 12, 20, 0),
        datetime(2021, 6, 1, 12, 20, 37, 123456),
    ):
        pump_event: PumpEvent = PumpEvent(currency_pair=currency_pair, time=pump_time, exchange=Exchange.BINANCE_SPOT)
        writer: PumpsFeatureWriter = _build_writer(pump_events=[pump_event])

        df_pre: pl.DataFrame = _build_rich_pre_rb_trades(pump_time=pump_time)
        df_post: pl.DataFrame = _build_post_rb_extreme_trades(pump_time=pump_time)
        df_leaky: pl.DataFrame = pl.concat([df_pre, df_post]).sort(TRADE_TIME)

        features_clean = writer.compute_features(df=df_pre, currency_pair=currency_pair, pump_event=pump_event)
        features_leaky = writer.compute_features(df=df_leaky, currency_pair=currency_pair, pump_event=pump_event)

        all_keys = [
            feature.col_name(offset)
            for feature in (
                FeatureType.ASSET_RETURN,
                FeatureType.SHARE_OF_LONG_TRADES,
                FeatureType.POWERLAW_ALPHA,
                FeatureType.SLIPPAGE_IMBALANCE,
                FeatureType.FLOW_IMBALANCE,
                FeatureType.NUM_TRADES,
                FeatureType.ASSET_RETURN_ZSCORE,
                FeatureType.QUOTE_ABS_ZSCORE,
            )
            for offset in _ALL_REGRESSOR_OFFSETS
        ]
        _assert_features_equal(features_clean, features_leaky, all_keys, where=f"post-rb leak (pump={pump_time})")


def test_leak_test_fails_against_calendar_anchored_reference(monkeypatch) -> None:
    """Verify the fixture is strong enough that a deliberately mis-anchored
    implementation (calendar hour bars + naive ``label < rb``) produces
    different values on the same synthetic data. If this ever stops failing,
    the fixture cannot detect a regression.

    Uses ``T = 12:20`` so ``rb = 12:05`` lies mid-hour: under the calendar
    anchor the bar ``[12:00, 13:00)`` is admitted by ``label < rb`` (12:00 <
    12:05) and its data extends past ``rb`` into ``[rb, T]``, so its z-score
    changes when the extreme post-rb trades are added. Under the rb-anchored
    geometry that bar has left edge ``11:05`` and right edge ``12:05 = rb``,
    so the same extreme trades cannot enter it.
    """
    monkeypatch.setattr(feature_writer_module, "REGRESSOR_OFFSETS", [NamedTimeDelta.ONE_HOUR])
    monkeypatch.setattr(feature_writer_module, "DECAY_OFFSETS", [NamedTimeDelta.ONE_MINUTE])

    currency_pair: CurrencyPair = CurrencyPair.from_string("AAA-BTC")
    pump_time: datetime = datetime(2021, 6, 1, 12, 20, 0)
    pump_event: PumpEvent = PumpEvent(currency_pair=currency_pair, time=pump_time, exchange=Exchange.BINANCE_SPOT)
    writer: PumpsFeatureWriter = _build_writer(pump_events=[pump_event])

    df_pre: pl.DataFrame = _build_rich_pre_rb_trades(pump_time=pump_time)
    df_post: pl.DataFrame = _build_post_rb_extreme_trades(pump_time=pump_time)
    df_leaky: pl.DataFrame = pl.concat([df_pre, df_post]).sort(TRADE_TIME)

    # Real (rb-anchored) writer: features are unchanged.
    features_clean = writer.compute_features(df=df_pre, currency_pair=currency_pair, pump_event=pump_event)
    features_leaky = writer.compute_features(df=df_leaky, currency_pair=currency_pair, pump_event=pump_event)
    window: NamedTimeDelta = NamedTimeDelta.ONE_HOUR
    zscore_key: str = FeatureType.ASSET_RETURN_ZSCORE.col_name(window)
    _assert_features_equal(features_clean, features_leaky, [zscore_key], where="rb-anchored (must match)")

    # Reference (calendar-anchored + naive filter): features SHOULD differ; if
    # they don't, the fixture is not tight enough to catch a mis-anchored
    # implementation and the leak test would silently pass.
    ref_clean = _calendar_anchored_compute_features(writer, df_pre, pump_event=pump_event)
    ref_leaky = _calendar_anchored_compute_features(writer, df_leaky, pump_event=pump_event)
    zc = ref_clean.get(zscore_key)
    zl = ref_leaky.get(zscore_key)
    assert (
        zc is not None and zl is not None
    ), f"reference impl returned None for {zscore_key} — fixture cannot distinguish anchoring"
    assert not np.isclose(zc, zl, equal_nan=True), (
        f"reference (calendar-anchored) impl produced identical values for {zscore_key}; "
        f"the fixture no longer catches mis-anchoring (clean={zc}, leaky={zl})"
    )


def test_short_offset_zscores_non_nan_on_synthetic_data(monkeypatch) -> None:
    """Short-offset z-scores must be finite and retain window-specific values.

    A previous implementation silently reused the same hourly numerator for
    ``5MIN``, ``15MIN`` and ``1H``, making all three columns identical.
    """
    monkeypatch.setattr(
        feature_writer_module,
        "REGRESSOR_OFFSETS",
        [
            NamedTimeDelta.FIVE_MINUTES,
            NamedTimeDelta.FIFTEEN_MINUTES,
            NamedTimeDelta.ONE_HOUR,
        ],
    )
    monkeypatch.setattr(feature_writer_module, "DECAY_OFFSETS", [NamedTimeDelta.ONE_MINUTE])

    currency_pair: CurrencyPair = CurrencyPair.from_string("AAA-BTC")
    pump_event: PumpEvent = PumpEvent(
        currency_pair=currency_pair,
        time=datetime(2021, 6, 1, 12, 20, 0),  # awkward mid-hour rb
        exchange=Exchange.BINANCE_SPOT,
    )
    writer: PumpsFeatureWriter = _build_writer(pump_events=[pump_event])

    df: pl.DataFrame = _build_rich_pre_rb_trades(pump_time=pump_event.time)
    features = writer.compute_features(df=df, currency_pair=currency_pair, pump_event=pump_event)

    for offset in (NamedTimeDelta.FIVE_MINUTES, NamedTimeDelta.FIFTEEN_MINUTES, NamedTimeDelta.ONE_HOUR):
        for feature in (FeatureType.ASSET_RETURN_ZSCORE, FeatureType.QUOTE_ABS_ZSCORE):
            key: str = feature.col_name(offset)
            value = features.get(key)
            assert value is not None and not (isinstance(value, float) and np.isnan(value)), (
                f"{key} is {value!r} — under rb-anchored geometry short-offset z-scores must be non-NaN "
                f"when the newest hourly bar has trades"
            )

    for feature in (FeatureType.ASSET_RETURN_ZSCORE, FeatureType.QUOTE_ABS_ZSCORE):
        values = [
            features[feature.col_name(offset)]
            for offset in (NamedTimeDelta.FIVE_MINUTES, NamedTimeDelta.FIFTEEN_MINUTES, NamedTimeDelta.ONE_HOUR)
        ]
        assert len({round(float(value), 12) for value in values}) == 3, f"{feature.name} reused a numerator: {values}"


def test_universe_eligibility_is_exactly_past_only() -> None:
    rb = datetime(2021, 6, 1, 12, 0)
    writer: PumpsFeatureWriter = _build_writer(pump_events=[])

    future_only = pl.DataFrame({TRADE_TIME: [rb, rb + timedelta(seconds=1)]})
    stale_only = pl.DataFrame({TRADE_TIME: [rb - timedelta(days=1)]})
    eligible = pl.DataFrame({TRADE_TIME: [rb - timedelta(microseconds=1)]})

    assert not writer.has_pre_decision_activity(df_ticks=future_only, rb=rb)
    assert writer.has_pre_decision_activity(df_ticks=stale_only, rb=rb)
    assert writer.has_pre_decision_activity(df_ticks=eligible, rb=rb)


def test_in_window_features_do_not_see_decision_lag_gap(monkeypatch) -> None:
    """The forbidden interval ``[T - DECISION_LAG, T]`` must never leak into any
    in-window feature (return, flow imbalance, slippage, powerlaw alpha, share
    of long trades, num_trades). Placing extreme trades inside that gap must
    not change the in-window feature values.
    """
    monkeypatch.setattr(feature_writer_module, "REGRESSOR_OFFSETS", [NamedTimeDelta.ONE_HOUR])
    monkeypatch.setattr(feature_writer_module, "DECAY_OFFSETS", [NamedTimeDelta.ONE_MINUTE])

    currency_pair: CurrencyPair = CurrencyPair.from_string("AAA-BTC")
    # ``rb = T - DECISION_LAG = 12:00`` lies on an hour boundary so the 1H
    # feature window ``[rb - 1h, rb) = [11:00, 12:00)`` is well-populated. The
    # decision-lag gap ``[12:00, 12:15]`` is where the extreme trades live.
    pump_event: PumpEvent = PumpEvent(
        currency_pair=currency_pair,
        time=datetime(2021, 6, 1, 12, 15, 0),
        exchange=Exchange.BINANCE_SPOT,
    )
    writer: PumpsFeatureWriter = _build_writer(pump_events=[pump_event])

    df_pre: pl.DataFrame = _build_rich_pre_rb_trades(pump_time=pump_event.time)
    gap_trades: pl.DataFrame = pl.DataFrame(
        [
            # Boundary of the forbidden zone (``rb = T - 15min``). The in-window
            # filter uses ``is_between(closed="left")`` which already excludes
            # exactly ``rb``; belt-and-braces to prove it.
            _explosive_trade(pump_event.time - DECISION_LAG, price=1e6, prev_price=100.0, quote_abs=1e9),
            # Strictly inside ``(T - 15min, T)``.
            _explosive_trade(pump_event.time - timedelta(minutes=5), price=2e6, prev_price=1e6, quote_abs=2e9),
            # The pump minute itself.
            _explosive_trade(pump_event.time, price=3e6, prev_price=2e6, quote_abs=3e9),
        ]
    )
    df_with_gap: pl.DataFrame = pl.concat([df_pre, gap_trades]).sort(TRADE_TIME)

    features_clean = writer.compute_features(df=df_pre, currency_pair=currency_pair, pump_event=pump_event)
    features_leaky = writer.compute_features(df=df_with_gap, currency_pair=currency_pair, pump_event=pump_event)

    window: NamedTimeDelta = NamedTimeDelta.ONE_HOUR
    in_window_keys = [
        feature.col_name(window)
        for feature in (
            FeatureType.ASSET_RETURN,
            FeatureType.SHARE_OF_LONG_TRADES,
            FeatureType.POWERLAW_ALPHA,
            FeatureType.SLIPPAGE_IMBALANCE,
            FeatureType.FLOW_IMBALANCE,
            FeatureType.NUM_TRADES,
            FeatureType.ASSET_RETURN_ZSCORE,
            FeatureType.QUOTE_ABS_ZSCORE,
        )
    ]
    _assert_features_equal(features_clean, features_leaky, in_window_keys, where="decision-lag gap leak")
