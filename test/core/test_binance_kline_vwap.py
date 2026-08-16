from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.utils.IndicativePriceProvider import IndicativePriceProvider
from core.columns import BINANCE_KLINES_COLS


def _utc_ms(ts: str) -> int:
    return int(pd.Timestamp(ts, tz="UTC").timestamp() * 1000)


def _write_1m_kline_zip(
    root_dir: Path,
    symbol: str,
    day: str,
    rows: list[list[float | int | str]],
) -> None:
    path = root_dir / symbol / f"klines@1m@{day}.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=BINANCE_KLINES_COLS).to_csv(
        path,
        compression="zip",
        index=False,
        header=False,
    )


def test_quote_to_usdt_uses_direct_symbol_vwap(tmp_path: Path) -> None:
    _write_1m_kline_zip(
        root_dir=tmp_path,
        symbol="BTC-USDT",
        day="2025-01-01",
        rows=[
            [
                _utc_ms("2025-01-01 12:00:00"),
                10000.0,
                10010.0,
                9990.0,
                10005.0,
                2.0,
                0,
                20_000.0,
                1,
                1.0,
                10_000.0,
                0,
            ],
            [
                _utc_ms("2025-01-01 12:01:00"),
                10500.0,
                10510.0,
                10490.0,
                10505.0,
                4.0,
                0,
                42_000.0,
                1,
                2.0,
                21_000.0,
                0,
            ],
        ],
    )

    converter = IndicativePriceProvider(raw_klines_dir=tmp_path)
    assert np.isclose(
        converter.get_quote_to_usdt_indicative_price("BTC", datetime(2025, 1, 1, 12, 1, 30)),
        10_000.0,
    )
    assert np.isclose(
        converter.get_quote_to_usdt_indicative_price("BTC", datetime(2025, 1, 1, 12, 2, 30)),
        10_500.0,
    )
    assert np.isclose(
        converter.get_quote_to_usdt_indicative_price("USDT", datetime(2025, 1, 1, 12, 2, 30)),
        1.0,
    )


def test_quote_to_usdt_supports_inverse_symbol(tmp_path: Path) -> None:
    _write_1m_kline_zip(
        root_dir=tmp_path,
        symbol="USDT-BTC",
        day="2025-01-01",
        rows=[
            [
                _utc_ms("2025-01-01 12:00:00"),
                0.00005,
                0.000051,
                0.000049,
                0.00005,
                20_000.0,
                0,
                1.0,
                1,
                10_000.0,
                0.5,
                0,
            ],
        ],
    )

    converter = IndicativePriceProvider(raw_klines_dir=tmp_path)
    assert np.isclose(
        converter.get_quote_to_usdt_indicative_price("BTC", datetime(2025, 1, 1, 12, 1, 0)),
        20_000.0,
    )


def test_indicative_price_never_uses_an_incomplete_or_future_bar(tmp_path: Path) -> None:
    _write_1m_kline_zip(
        root_dir=tmp_path,
        symbol="BTC-USDT",
        day="2025-01-01",
        rows=[
            [
                _utc_ms("2025-01-01 12:00:00"),
                10_000.0,
                10_010.0,
                9_990.0,
                10_005.0,
                2.0,
                0,
                20_000.0,
                1,
                1.0,
                10_000.0,
                0,
            ]
        ],
    )

    converter = IndicativePriceProvider(raw_klines_dir=tmp_path)
    assert converter.get_indicative_price("BTC-USDT", datetime(2025, 1, 1, 12, 0, 59)) is None
    assert np.isclose(converter.get_indicative_price("BTC-USDT", datetime(2025, 1, 1, 12, 1)), 10_000.0)


def test_indicative_price_can_use_previous_days_last_completed_bar(tmp_path: Path) -> None:
    _write_1m_kline_zip(
        root_dir=tmp_path,
        symbol="BTC-USDT",
        day="2024-12-31",
        rows=[
            [
                _utc_ms("2024-12-31 23:59:00"),
                10_000.0,
                10_010.0,
                9_990.0,
                10_005.0,
                2.0,
                0,
                20_000.0,
                1,
                1.0,
                10_000.0,
                0,
            ]
        ],
    )

    converter = IndicativePriceProvider(raw_klines_dir=tmp_path)
    assert np.isclose(converter.get_indicative_price("BTC-USDT", datetime(2025, 1, 1, 0, 0, 30)), 10_000.0)
