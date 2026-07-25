"""Regression tests for `BinanceParser` download-integrity and href-filter guards.

These tests do not require the network; scrapy `Response` objects are constructed directly with
synthetic bodies, and the spider's file I/O is redirected under `tmp_path`.
"""

import logging
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import List

import pytest
from scrapy.http import Request, TextResponse

from core.currency_pair import CurrencyPair
from core.time_utils import Bounds
from market_data.parsers.binance.BinanceParser import (
    BinanceBaseParser,
    filter_hrefs_by_bounds,
)


class _StubBinanceParser(BinanceBaseParser):
    """Concrete subclass so the abstract `BinanceBaseParser` can be instantiated in tests."""

    name = "stub_binance_parser"

    def get_prefix(self, currency_pair: CurrencyPair) -> str:
        return f"data/spot/daily/trades/{currency_pair.binance_name}/"


def _make_parser(tmp_path: Path) -> _StubBinanceParser:
    return _StubBinanceParser(
        bounds=Bounds.for_day(date(2021, 6, 16)),
        currency_pairs=[CurrencyPair.from_string("BTC-USDT")],
        output_dir=tmp_path,
    )


def _make_zip_response(url: str, currency_pair: CurrencyPair, day: date, body: bytes) -> TextResponse:
    request: Request = Request(url=url, meta={"currency_pair": currency_pair, "day": day})
    return TextResponse(url=url, request=request, body=body, headers={})


def _make_real_zip_bytes() -> bytes:
    buffer: BytesIO = BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as zf:
        zf.writestr("file.csv", "a,b,c\n1,2,3\n")
    return buffer.getvalue()


def test_filter_hrefs_by_bounds_skips_href_with_no_date(caplog: pytest.LogCaptureFixture) -> None:
    bounds: Bounds = Bounds.for_day(date(2021, 6, 16))
    hrefs: List[str] = [
        "data/spot/daily/trades/BTCUSDT/BTCUSDT-trades-2021-06-16.zip",
        "data/spot/daily/trades/BTCUSDT/index.html",  # no date
        "data/spot/daily/trades/BTCUSDT/BTCUSDT-trades-2021-06-15.zip",
    ]

    with caplog.at_level(logging.WARNING):
        filtered, dates = filter_hrefs_by_bounds(hrefs=hrefs, bounds=bounds)

    assert filtered == ["data/spot/daily/trades/BTCUSDT/BTCUSDT-trades-2021-06-16.zip"]
    assert dates == [date(2021, 6, 16)]
    assert any("no parseable date" in rec.message for rec in caplog.records)


def test_parse_zip_file_writes_when_body_is_real_zip(tmp_path: Path) -> None:
    parser: _StubBinanceParser = _make_parser(tmp_path)
    currency_pair: CurrencyPair = CurrencyPair.from_string("BTC-USDT")
    day: date = date(2021, 6, 16)
    body: bytes = _make_real_zip_bytes()
    response: TextResponse = _make_zip_response("http://example/BTCUSDT.zip", currency_pair, day, body)

    parser._parse_zip_file(response)

    written: Path = parser.output_zip_path(currency_pair=currency_pair, day=day)
    assert written.exists()
    assert written.read_bytes() == body
    assert parser.failed_downloads == []


def test_parse_zip_file_rejects_empty_body(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    parser: _StubBinanceParser = _make_parser(tmp_path)
    currency_pair: CurrencyPair = CurrencyPair.from_string("BTC-USDT")
    day: date = date(2021, 6, 16)
    response: TextResponse = _make_zip_response("http://example/BTCUSDT.zip", currency_pair, day, body=b"")

    with caplog.at_level(logging.WARNING):
        parser._parse_zip_file(response)

    written: Path = parser.output_zip_path(currency_pair=currency_pair, day=day)
    assert not written.exists()
    assert parser.failed_downloads == [(currency_pair, day)]
    assert any("Rejecting non-zip response" in rec.message for rec in caplog.records)


def test_parse_zip_file_rejects_non_zip_body(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    parser: _StubBinanceParser = _make_parser(tmp_path)
    currency_pair: CurrencyPair = CurrencyPair.from_string("BTC-USDT")
    day: date = date(2021, 6, 16)
    # Simulates the AWS S3 XML error page that occasionally comes back instead of a zip.
    body: bytes = b'<?xml version="1.0"?><Error><Code>NoSuchKey</Code></Error>'
    response: TextResponse = _make_zip_response("http://example/BTCUSDT.zip", currency_pair, day, body)

    with caplog.at_level(logging.WARNING):
        parser._parse_zip_file(response)

    written: Path = parser.output_zip_path(currency_pair=currency_pair, day=day)
    assert not written.exists()
    assert parser.failed_downloads == [(currency_pair, day)]
