import os
import re
from abc import ABC, abstractmethod
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from urllib.parse import urlencode

import scrapy
from scrapy.http import Request, Response

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
    def __init__(self, bounds: Bounds, currency_pairs: List[CurrencyPair], output_dir: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bounds: Bounds = bounds
        self.currency_pairs: List[CurrencyPair] = currency_pairs
        self.output_dir: Path = output_dir

    @abstractmethod
    def get_prefix(self, currency_pair: CurrencyPair) -> str:
        """Return prefix"""

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
                callback=self._parse_currency_pair,  # type:ignore
                meta={"currency_pair": currency_pair, "href_container": []},  # mutable object
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
            yield scrapy.Request(
                url=get_zip_file_url(href=href),
                callback=self._parse_zip_file,
                meta={"currency_pair": currency_pair, "day": day},
            )

    def _parse_zip_file(self, response: Response) -> None:
        day: date = response.meta.get("day")
        currency_pair: CurrencyPair = response.meta.get("currency_pair")
        path: Path = self.output_dir / currency_pair.name / f"trades@{str(day)}.zip"
        os.makedirs(path.parent, exist_ok=True)

        with open(path, "wb") as file:
            file.write(response.body)
