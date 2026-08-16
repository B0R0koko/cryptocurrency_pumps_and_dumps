"""Regression tests for `core.currency_pair.get_cross_section_currencies`.

The previous implementation crashed with `IndexError` when a stray non-`symbol=` file lived under
a date partition (e.g. pyarrow's `_common_metadata`, `.DS_Store`, or a checksum file). We now
skip such entries with a warning and continue collecting the well-formed pairs.
"""

import logging
from datetime import date
from pathlib import Path
from typing import List, Set

import pytest

from core.currency_pair import CurrencyPair, get_cross_section_currencies
from core.time_utils import Bounds


def _make_partition(hive_dir: Path, day: date, symbols: List[str], extras: List[str]) -> None:
    part: Path = hive_dir / f"date={day}"
    part.mkdir(parents=True, exist_ok=True)
    for sym in symbols:
        (part / f"symbol={sym}").mkdir()
    for extra in extras:
        (part / extra).touch()


def test_get_cross_section_currencies_skips_non_symbol_entries(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """pyarrow-style stray files must be logged and skipped, not raised as IndexError."""
    day: date = date(2024, 5, 15)
    _make_partition(
        hive_dir=tmp_path,
        day=day,
        symbols=["BTC-USDT", "ETH-USDT"],
        extras=["_common_metadata", ".DS_Store", "readme.txt"],
    )

    bounds: Bounds = Bounds.for_days(date(2024, 5, 1), date(2024, 6, 1))

    with caplog.at_level(logging.WARNING):
        pairs: List[CurrencyPair] = get_cross_section_currencies(hive_dir=tmp_path, bounds=bounds)

    names: Set[str] = {p.name for p in pairs}
    assert names == {"BTC-USDT", "ETH-USDT"}
    warnings = [rec for rec in caplog.records if "Skipping non-symbol entry" in rec.message]
    assert len(warnings) >= 1


def test_get_cross_section_currencies_skips_partition_without_date_prefix(tmp_path: Path) -> None:
    """A stray directory whose name does not carry a `YYYY-MM-DD` token must be ignored entirely."""
    _make_partition(tmp_path, date(2024, 5, 15), symbols=["BTC-USDT"], extras=[])
    # Non-partition sibling.
    (tmp_path / "not_a_partition").mkdir()

    pairs = get_cross_section_currencies(
        hive_dir=tmp_path,
        bounds=Bounds.for_days(date(2024, 5, 1), date(2024, 6, 1)),
    )
    assert {p.name for p in pairs} == {"BTC-USDT"}


def test_get_cross_section_currencies_only_reads_partitions_inside_bounds(tmp_path: Path) -> None:
    """Symbols living under a date partition outside the requested bounds must be excluded."""
    _make_partition(tmp_path, date(2024, 5, 15), symbols=["BTC-USDT"], extras=[])
    _make_partition(tmp_path, date(2024, 7, 15), symbols=["ETH-USDT"], extras=[])

    pairs = get_cross_section_currencies(
        hive_dir=tmp_path,
        bounds=Bounds.for_days(date(2024, 5, 1), date(2024, 6, 1)),
    )
    assert {p.name for p in pairs} == {"BTC-USDT"}


def test_get_cross_section_currencies_returns_deterministic_symbol_order(tmp_path: Path) -> None:
    _make_partition(
        tmp_path,
        date(2024, 5, 15),
        symbols=["ZZZ-BTC", "AAA-BTC", "MMM-BTC"],
        extras=[],
    )
    pairs = get_cross_section_currencies(
        hive_dir=tmp_path,
        bounds=Bounds.for_days(date(2024, 5, 1), date(2024, 6, 1)),
    )
    assert [pair.name for pair in pairs] == ["AAA-BTC", "MMM-BTC", "ZZZ-BTC"]


def test_get_cross_section_currencies_skips_unparseable_pair_names(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """`symbol=<garbage>` must not crash; a clear warning must be emitted."""
    day: date = date(2024, 5, 15)
    part: Path = tmp_path / f"date={day}"
    part.mkdir(parents=True)
    (part / "symbol=BTC-USDT").mkdir()
    (part / "symbol=malformed_no_hyphen").mkdir()

    with caplog.at_level(logging.WARNING):
        pairs = get_cross_section_currencies(
            hive_dir=tmp_path,
            bounds=Bounds.for_days(date(2024, 5, 1), date(2024, 6, 1)),
        )

    assert {p.name for p in pairs} == {"BTC-USDT"}
    assert any("unparseable pair" in rec.message for rec in caplog.records)


def test_currency_pair_from_string_roundtrips_via_str() -> None:
    """CurrencyPair.from_string(str(pair)) must return an equal pair, including for `-` in quote."""
    pair: CurrencyPair = CurrencyPair.from_string("ADA-USDT")
    assert str(pair) == "ADA-USDT"
    assert pair.name == "ADA-USDT"
    assert pair.binance_name == "ADAUSDT"
    assert CurrencyPair.from_string(str(pair)) == pair


def test_currency_pair_hash_matches_equal_pairs() -> None:
    """Two pairs with the same name must share a hash so they collapse in sets/dicts."""
    a: CurrencyPair = CurrencyPair.from_string("BTC-USDT")
    b: CurrencyPair = CurrencyPair.from_string("BTC-USDT")
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
