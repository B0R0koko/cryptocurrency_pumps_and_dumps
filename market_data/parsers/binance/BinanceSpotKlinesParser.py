import argparse
from datetime import date
from pathlib import Path
from typing import List

from scrapy.crawler import CrawlerProcess

from core.currency_pair import CurrencyPair
from core.paths import BINANCE_SPOT_RAW_KLINES
from core.time_utils import Bounds
from market_data.parsers.binance.BinanceParser import BinanceBaseParser
from market_data.parsers.settings import SETTINGS

_SUPPORTED_INTERVALS: set[str] = {
    "1s",
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
}


class BinanceSpotKlinesParser(BinanceBaseParser):
    name = "binance_spot_klines_parser"

    def __init__(
        self,
        bounds: Bounds,
        currency_pairs: List[CurrencyPair],
        output_dir: Path,
        interval: str,
    ):
        super().__init__(bounds=bounds, currency_pairs=currency_pairs, output_dir=output_dir)
        if interval not in _SUPPORTED_INTERVALS:
            raise ValueError(f"Unsupported Binance kline interval: {interval}")
        self.interval: str = interval

    def get_prefix(self, currency_pair: CurrencyPair) -> str:
        return f"data/spot/daily/klines/{currency_pair.binance_name}/{self.interval}/"

    def output_zip_path(self, currency_pair: CurrencyPair, day: date) -> Path:
        return self.output_dir / currency_pair.name / f"klines@{self.interval}@{str(day)}.zip"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Binance spot daily klines from data.binance.vision.")
    parser.add_argument("--start-date", required=True, help="Inclusive start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", required=True, help="Exclusive end date in YYYY-MM-DD format.")
    parser.add_argument("--interval", required=True, help="Binance kline interval, e.g. 1m, 1h, 1d.")
    parser.add_argument(
        "--quote-asset",
        default="BTC",
        help="Download only spot pairs quoted in this asset.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BINANCE_SPOT_RAW_KLINES),
        help="Directory where downloaded zip files will be stored.",
    )
    return parser.parse_args()


def run_main() -> None:
    bounds: Bounds = Bounds.for_days(date(2018, 1, 1), date(2026, 3, 8))
    process: CrawlerProcess = CrawlerProcess(settings=SETTINGS)

    process.crawl(
        BinanceSpotKlinesParser,
        bounds=bounds,
        currency_pairs=[CurrencyPair.from_string("BTC-USDT")],
        output_dir=BINANCE_SPOT_RAW_KLINES,
        interval="1m",
    )
    process.start()


if __name__ == "__main__":
    run_main()
