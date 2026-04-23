import os
import re
import threading
from abc import ABC, abstractmethod
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlencode

import scrapy
from scrapy import signals
from scrapy.http import Request, Response
from tqdm import tqdm

from core.currency_pair import CurrencyPair
from core.time_utils import Bounds

BINANCE_S3: str = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BINANCE_DATAVISION: str = "https://data.binance.vision"


def filter_hrefs_by_bounds(hrefs: List[str], bounds: Bounds) -> Tuple[List[str], List[date]]:
    """Takes bounds as input and returns a list of hrefs that matches passed in Bounds"""

    filtered_hrefs: List[str] = []
    href_dates: List[date] = []

    pattern_str: str = r"\d{4}-\d{2}-\d{2}"

    for href in hrefs:
        # Find date in href string and parse it to datetime
        href_date_string: str = re.search(pattern=pattern_str, string=href)[0]
        href_date: date = datetime.strptime(href_date_string, "%Y-%m-%d").date()
        if bounds.contain_days(day=href_date):
            filtered_hrefs.append(href)
            href_dates.append(href_date)

    return filtered_hrefs, href_dates


def get_zip_file_url(href: str) -> str:
    """Returns a formatted url string which leads to a zip file with trades data"""
    return f"{BINANCE_DATAVISION}/{href}"


class BinanceBaseParser(ABC, scrapy.Spider):
    def __init__(
        self,
        bounds: Bounds,
        currency_pairs: List[CurrencyPair],
        output_dir: Path,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bounds: Bounds = bounds
        self.currency_pairs: List[CurrencyPair] = currency_pairs
        self.output_dir: Path = output_dir
        self._scheduled_hrefs: Set[str] = set()
        self._pbar: Optional[tqdm] = None
        self._shutdown_timer_started: bool = False

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider._on_spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(spider._on_spider_closed, signal=signals.spider_closed)
        return spider

    def _on_spider_opened(self, **_kwargs) -> None:
        self._pbar = tqdm(total=0, desc=self.name, unit="file", dynamic_ncols=True)

    def _on_spider_closed(self, **_kwargs) -> None:
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
        if self._shutdown_timer_started:
            return
        self._shutdown_timer_started = True
        # Safety net: scrapy's reactor sometimes won't shut down after the
        # spider finishes because the HTTP connection pool is still holding
        # keep-alive connections. Force-exit after a short grace period so
        # the CLI actually returns.
        timer = threading.Timer(3.0, lambda: os._exit(0))
        timer.daemon = True
        timer.start()

    @abstractmethod
    def get_prefix(self, currency_pair: CurrencyPair) -> str:
        """Return prefix"""

    def output_zip_path(self, currency_pair: CurrencyPair, day: date) -> Path:
        """Where the downloaded zip for a given currency pair / day is stored."""
        return self.output_dir / currency_pair.name / f"trades@{str(day)}.zip"

    def _get_currency_url(self, currency_pair: CurrencyPair, marker: Optional[str] = None) -> str:
        params: Dict[str, str] = {
            "delimiter": "/",
            "prefix": self.get_prefix(currency_pair=currency_pair),
        }

        if marker is not None:
            params["marker"] = marker

        datavision_url: str = f"{BINANCE_S3}?{urlencode(params)}"
        return datavision_url

    def start_requests(self):
        """This method is run first when the Spider starts"""
        for currency_pair in self.currency_pairs:
            yield Request(
                url=self._get_currency_url(currency_pair=currency_pair),
                callback=self._parse_currency_pair,  # type: ignore
                meta={
                    "currency_pair": currency_pair,
                    "href_container": [],
                },  # mutable object
            )

    def _parse_currency_pair(self, response: Response):
        """Parse hrefs with zip files from currency_pair page"""

        currency_pair: Optional[CurrencyPair] = response.meta.get("currency_pair")
        href_container: List[str] = response.meta.get("href_container")

        assert currency_pair, "Currency pair must be supplied in scrapy.http.Response.meta"
        assert href_container is not None, "Href container must be supplied in scrapy.http.Response.meta"

        hrefs: List[str] = re.findall(pattern=r"<Key>(.*?)</Key>", string=response.text)
        hrefs: List[str] = [href for href in hrefs if "CHECKSUM" not in href]
        href_container.extend(hrefs)

        # if len is 500, then we need to send another request with marker param which is the last entry in hrefs
        if len(hrefs) == 500:
            yield scrapy.Request(
                url=self._get_currency_url(currency_pair=currency_pair, marker=hrefs[-1]),
                callback=self._parse_currency_pair,  # call itself one more time
                meta={"currency_pair": currency_pair, "href_container": href_container},
            )
        # Once we have collected all hrefs into response.meta.href_container we loop over it and send requests that
        # collect zip files
        # Filter hrefs by dates that we want to collect data for
        filtered_hrefs, href_dates = filter_hrefs_by_bounds(hrefs=href_container, bounds=self.bounds)

        for href, day in zip(filtered_hrefs, href_dates):
            # Pagination pages re-enter this callback with the cumulative href_container,
            # so guard against scheduling (and counting) the same href twice.
            if href in self._scheduled_hrefs:
                continue
            self._scheduled_hrefs.add(href)
            if self._pbar is not None:
                self._pbar.total += 1
                self._pbar.refresh()
            yield scrapy.Request(
                url=get_zip_file_url(href=href),
                callback=self._parse_zip_file,
                meta={"currency_pair": currency_pair, "day": day},
            )

    def _parse_zip_file(self, response: Response) -> None:
        day: date = response.meta.get("day")
        currency_pair: CurrencyPair = response.meta.get("currency_pair")
        path: Path = self.output_zip_path(currency_pair=currency_pair, day=day)
        os.makedirs(path.parent, exist_ok=True)

        with open(path, "wb") as file:
            file.write(response.body)

        if self._pbar is not None:
            self._pbar.update(1)
