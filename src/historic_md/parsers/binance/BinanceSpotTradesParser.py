import logging
from datetime import date
from pathlib import Path
from typing import List

from scrapy.crawler import CrawlerProcess

from core.currency_pair import CurrencyPair, collect_all_spot_currency_pairs
from core.paths import BINANCE_SPOT_RAW_TRADES
from core.time_utils import Bounds
from core.utils import configure_logging
from historic_md.parsers.binance.BinanceParser import BinanceBaseParser
from historic_md.parsers.settings import SETTINGS


class BinanceSpotTradesParser(BinanceBaseParser):
    name = "binance_spot_trades_parser"

    def __init__(self, bounds: Bounds, currency_pairs: List[CurrencyPair], output_dir: Path):
        super().__init__(
            bounds=bounds,
            currency_pairs=currency_pairs,
            output_dir=output_dir
        )

    def get_prefix(self, currency_pair: CurrencyPair) -> str:
        return f"data/spot/daily/trades/{currency_pair.binance_name}/"


def run_main():
    configure_logging()
    bounds: Bounds = Bounds.for_days(
        date(2018, 1, 1),
        date(2019, 1, 1)
    )
    process: CrawlerProcess = CrawlerProcess(settings=SETTINGS)

    usdt_currencies: List[CurrencyPair] = [
        currency_pair for currency_pair in collect_all_spot_currency_pairs() if currency_pair.term == "BTC"
    ]

    logging.info("Collecting data for %s currencies", len(usdt_currencies))

    process.crawl(
        BinanceSpotTradesParser,
        bounds=bounds,
        currency_pairs=usdt_currencies,
        output_dir=BINANCE_SPOT_RAW_TRADES,
    )
    process.start()


if __name__ == "__main__":
    run_main()
