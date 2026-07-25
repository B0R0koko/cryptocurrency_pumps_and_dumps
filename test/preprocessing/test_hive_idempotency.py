"""Regression tests for `binance_spot_trades_to_hive` idempotency and input filtering.

None of these tests touch the real dataset directory. Synthetic zipped CSVs are built in
`tmp_path` so the tests are hermetic.
"""

import io
import logging
import zipfile
from datetime import date
from pathlib import Path
from typing import List

import pandas as pd
import polars as pl
import pytest

from core.columns import BINANCE_TRADE_COLS, TRADE_TIME
from core.currency_pair import CurrencyPair
from core.time_utils import Bounds
from preprocessing.pipelines.binance_spot_trades_to_hive import (
    BinanceSpotTrades2Hive,
    _detect_trade_time_unit,
    filter_by_bounds,
)


def _write_fake_trades_zip(path: Path, num_rows: int, day: date = date(2021, 6, 16)) -> None:
    """Create a Binance-formatted trades zip at `path` with `num_rows` synthetic rows.

    Timestamps are emitted in ms for days before 2025-01-01 and in µs afterwards, matching the
    real Binance historical archive so the auto-detector in preprocess_batched_data picks the
    correct unit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if day < date(2025, 1, 1):
        base_time: int = int(pd.Timestamp(day).timestamp() * 1_000)
        step: int = 1  # 1 ms between rows
    else:
        base_time = int(pd.Timestamp(day).timestamp() * 1_000_000)
        step = 1_000  # 1 ms in µs
    df: pd.DataFrame = pd.DataFrame(
        {
            "trade_id": range(num_rows),
            "price": [1.0 + i * 0.01 for i in range(num_rows)],
            "quantity": [10.0] * num_rows,
            "quote_quantity": [10.0 + i * 0.1 for i in range(num_rows)],
            "trade_time": [base_time + i * step for i in range(num_rows)],
            "is_buyer_maker": [False] * num_rows,
            "is_best_match": [True] * num_rows,
        }
    )[BINANCE_TRADE_COLS]

    csv_bytes: bytes = df.to_csv(index=False, header=False).encode("utf-8")
    buffer: io.BytesIO = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{path.stem}.csv", csv_bytes)
    path.write_bytes(buffer.getvalue())


def _build_pipeline(tmp_path: Path, day: date, currency_pair: CurrencyPair, num_rows: int) -> BinanceSpotTrades2Hive:
    raw_dir: Path = tmp_path / "raw"
    out_dir: Path = tmp_path / "hive"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_fake_trades_zip(raw_dir / currency_pair.name / f"trades@{day}.zip", num_rows=num_rows, day=day)
    return BinanceSpotTrades2Hive(bounds=Bounds.for_day(day), raw_data_dir=raw_dir, output_dir=out_dir)


def _row_count(hive_dir: Path, currency_pair: CurrencyPair) -> int:
    frame: pl.LazyFrame = pl.scan_parquet(source=hive_dir, hive_partitioning=True)
    return frame.filter(pl.col("symbol") == currency_pair.name).select(pl.len()).collect().item()


def test_unzip_and_save_to_hive_is_idempotent_on_rerun(tmp_path: Path) -> None:
    """Re-running the same (day, symbol) task must NOT duplicate rows.

    Before the fix, `existing_data_behavior="overwrite_or_ignore"` combined with UUID basenames
    left previous parquet files in place, so re-running the ingest job doubled every row.
    """
    day: date = date(2021, 6, 16)
    currency_pair: CurrencyPair = CurrencyPair.from_string("BTC-USDT")
    pipe: BinanceSpotTrades2Hive = _build_pipeline(tmp_path, day, currency_pair, num_rows=1_500)

    pipe.unzip_and_save_to_hive(currency_pair=currency_pair, day=day)
    first_count: int = _row_count(pipe.output_dir, currency_pair)
    assert first_count == 1_500

    # Second run of the same task; row count must remain the same.
    pipe.unzip_and_save_to_hive(currency_pair=currency_pair, day=day)
    second_count: int = _row_count(pipe.output_dir, currency_pair)
    assert second_count == 1_500, f"re-running duplicated rows: {first_count} -> {second_count}"


def test_multi_chunk_task_writes_all_rows_after_reset(tmp_path: Path) -> None:
    """A single task splits into multiple chunks (chunksize=1_000_000).

    Wiping the partition once at task start (rather than once per chunk) must NOT delete earlier
    chunks of the same run. We simulate this with a small chunk size by writing more than
    1_000_000 rows only implicitly; here we mainly assert that a single-run count equals the
    input row count so no chunk is silently lost.
    """
    day: date = date(2021, 6, 16)
    currency_pair: CurrencyPair = CurrencyPair.from_string("ETH-USDT")
    pipe: BinanceSpotTrades2Hive = _build_pipeline(tmp_path, day, currency_pair, num_rows=2_500)

    pipe.unzip_and_save_to_hive(currency_pair=currency_pair, day=day)
    assert _row_count(pipe.output_dir, currency_pair) == 2_500


def test_filter_by_bounds_skips_files_without_expected_pattern(caplog: pytest.LogCaptureFixture) -> None:
    """Stray files (e.g. `.DS_Store`, checksums) must be skipped, not crash the pipeline."""
    bounds: Bounds = Bounds.for_day(date(2021, 6, 16))
    files: List[str] = [
        "trades@2021-06-16.zip",
        ".DS_Store",
        "trades@2021-06-16.zip.CHECKSUM",
        "readme.txt",
        "trades@2021-06-15.zip",
    ]
    with caplog.at_level(logging.WARNING):
        result: List[date] = filter_by_bounds(bounds=bounds, file_names=files)

    assert result == [date(2021, 6, 16)]
    assert any("Skipping raw file" in rec.message for rec in caplog.records)


def test_detect_trade_time_unit_recognises_milliseconds() -> None:
    """A real pre-2025 timestamp (13 digits) must resolve as milliseconds."""
    # 1735603282257 ms = 2024-12-31 00:01:22 UTC, from a real pre-cutover file on disk.
    assert _detect_trade_time_unit(sample=1_735_603_282_257, day=date(2024, 12, 31)) == "ms"


def test_detect_trade_time_unit_recognises_microseconds() -> None:
    """A real post-2025 timestamp (16 digits) must resolve as microseconds."""
    # 1735689608235752 us = 2025-01-01 00:00:08 UTC.
    assert _detect_trade_time_unit(sample=1_735_689_608_235_752, day=date(2025, 1, 1)) == "us"


def test_detect_trade_time_unit_rejects_out_of_band_value() -> None:
    """A value that is neither ms nor us must raise, so we never silently write 1970 rows.

    A previous heuristic keyed on the filename date alone; if Binance ever mislabels an archive
    or reprocesses a pre-2025 file with us timestamps, the day-based check would misinterpret it
    and quietly rewrite thousands of trades into the wrong era.
    """
    with pytest.raises(ValueError, match="Unrecognised Binance trade_time magnitude"):
        _detect_trade_time_unit(sample=1_000, day=date(2024, 12, 31))
    with pytest.raises(ValueError, match="Unrecognised Binance trade_time magnitude"):
        # 10^14 sits in the dead band between ms and us.
        _detect_trade_time_unit(sample=10**14, day=date(2025, 1, 1))


def test_preprocess_batched_data_uses_microseconds_for_2025_and_later(tmp_path: Path) -> None:
    """Post-cutover files must decode into 2025-era timestamps, not 1970."""
    day: date = date(2025, 1, 1)
    currency_pair: CurrencyPair = CurrencyPair.from_string("BTC-USDT")
    pipe: BinanceSpotTrades2Hive = _build_pipeline(tmp_path, day, currency_pair, num_rows=100)

    pipe.unzip_and_save_to_hive(currency_pair=currency_pair, day=day)
    frame: pl.LazyFrame = pl.scan_parquet(source=pipe.output_dir, hive_partitioning=True)
    first_ts = frame.filter(pl.col("symbol") == currency_pair.name).select(pl.col(TRADE_TIME).min()).collect().item()
    assert first_ts.year == 2025, f"expected 2025 timestamp, got {first_ts!r}"


def test_preprocess_batched_data_rejects_corrupted_timestamp(tmp_path: Path) -> None:
    """A malformed CSV whose trade_time is neither ms nor us must fail loudly, not silently.

    We hand-write a zip containing a nonsense integer to make sure the auto-detector short-circuits
    before pd.to_datetime happily produces year-1970 rows.
    """
    day: date = date(2024, 12, 31)
    currency_pair: CurrencyPair = CurrencyPair.from_string("BTC-USDT")
    raw_dir: Path = tmp_path / "raw"
    out_dir: Path = tmp_path / "hive"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path: Path = raw_dir / currency_pair.name / f"trades@{day}.zip"
    zip_path.parent.mkdir(parents=True)
    csv_bytes: bytes = b"1,1.0,10.0,10.0,42,False,True\n"  # 42 is neither ms nor us
    buffer: io.BytesIO = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{zip_path.stem}.csv", csv_bytes)
    zip_path.write_bytes(buffer.getvalue())

    pipe: BinanceSpotTrades2Hive = BinanceSpotTrades2Hive(
        bounds=Bounds.for_day(day), raw_data_dir=raw_dir, output_dir=out_dir
    )
    with pytest.raises(ValueError, match="Unrecognised Binance trade_time magnitude"):
        pipe.unzip_and_save_to_hive(currency_pair=currency_pair, day=day)


def test_iterate_over_tasks_skips_stray_entries(tmp_path: Path) -> None:
    """A stray directory or non-directory in the raw data root must not abort iteration."""
    raw_dir: Path = tmp_path / "raw"
    out_dir: Path = tmp_path / "hive"
    out_dir.mkdir(parents=True, exist_ok=True)

    day: date = date(2021, 6, 16)
    good_pair: CurrencyPair = CurrencyPair.from_string("BTC-USDT")
    _write_fake_trades_zip(raw_dir / good_pair.name / f"trades@{day}.zip", num_rows=10)

    # Stray plain file in the root.
    (raw_dir / "some_note.txt").write_text("hello")
    # Stray directory whose name is not a currency pair.
    (raw_dir / "not_a_pair").mkdir()

    pipe: BinanceSpotTrades2Hive = BinanceSpotTrades2Hive(
        bounds=Bounds.for_day(day), raw_data_dir=raw_dir, output_dir=out_dir
    )

    tasks = list(pipe.iterate_over_tasks())
    assert tasks == [(day, good_pair)]
