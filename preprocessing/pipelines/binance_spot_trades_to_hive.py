import logging
import os
import re
import shutil
from datetime import date, datetime
from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import List, Any, Generator, Literal, Optional, cast

import pandas as pd
from tqdm import tqdm

from core.columns import TRADE_TIME, BINANCE_TRADE_COLS, PRICE, QUANTITY, IS_BUYER_MAKER
from core.currency_pair import CurrencyPair
from core.paths import BINANCE_SPOT_RAW_TRADES, BINANCE_SPOT_HIVE_TRADES
from core.time_utils import Bounds

_USE_COLS: List[str] = [PRICE, QUANTITY, TRADE_TIME, IS_BUYER_MAKER]
_ARCHIVE_NAME_RE: re.Pattern[str] = re.compile(r"@(\d{4}-\d{2}-\d{2})\.zip$")

# Timestamps produced by Binance's historical spot trades archive are either milliseconds (13
# digits for dates in this decade) or microseconds (16 digits). Values in these bands map to
# realistic wall-clock times after the epoch; anything else indicates a corrupted or unknown feed.
_MS_MIN: int = 10**12
_MS_MAX: int = 10**13
_US_MIN: int = 10**15
_US_MAX: int = 10**16

_log = logging.getLogger(__name__)


TradeTimeUnit = Literal["ms", "us"]


def _detect_trade_time_unit(sample: int, day: date) -> TradeTimeUnit:
    """Infer whether a trade-time integer is milliseconds or microseconds.

    Binance switched the historical spot trades feed from ms to us at 2025-01-01 UTC. Rather than
    trust the filename date alone, we validate the numeric magnitude of the first row so a
    mislabelled or straddling archive fails loudly instead of silently producing 1970-era rows.
    """
    if _MS_MIN <= sample < _MS_MAX:
        return "ms"
    if _US_MIN <= sample < _US_MAX:
        return "us"
    raise ValueError(
        f"Unrecognised Binance trade_time magnitude {sample} for day {day}; "
        f"expected milliseconds ({_MS_MIN}..{_MS_MAX}) or microseconds ({_US_MIN}..{_US_MAX})."
    )


def filter_by_bounds(bounds: Bounds, file_names: List[str]) -> List[date]:
    """Returns a list of dates parsed from archive filenames that fall within bounds.

    Filenames that do not match the expected `trades@YYYY-MM-DD.zip` pattern are skipped with a
    warning rather than raising, so stray files in the raw directory do not abort ingestion.
    """
    valid_dates: List[date] = []

    for file in file_names:
        match: Optional[re.Match[str]] = _ARCHIVE_NAME_RE.search(file)
        if match is None:
            _log.warning("Skipping raw file with unexpected name: %s", file)
            continue
        day: date = datetime.strptime(match.group(1), "%Y-%m-%d").date()

        if bounds.contain_days(day=day):
            valid_dates.append(day)

    return valid_dates


class BinanceSpotTrades2Hive:

    def __init__(
        self,
        bounds: Bounds,
        raw_data_dir: Path,
        output_dir: Path,
    ):
        self.bounds: Bounds = bounds
        self.raw_data_dir: Path = raw_data_dir
        self.output_dir: Path = output_dir

    @staticmethod
    def preprocess_batched_data(df_batch: pd.DataFrame, currency_pair: CurrencyPair, day: date) -> pd.DataFrame:
        """Attach new columns and convert dtypes here before saving to hive structure.

        Binance switched the historical spot trades feed from ms to microseconds at 2025-01-01.
        We auto-detect the unit from the first row's magnitude so a mislabelled archive fails
        loudly rather than silently producing rows near the Unix epoch.
        """
        if df_batch.empty:
            df_batch[TRADE_TIME] = pd.to_datetime(df_batch[TRADE_TIME], unit="ms")
        else:
            sample: int = int(cast(int, df_batch[TRADE_TIME].iat[0]))
            unit: TradeTimeUnit = _detect_trade_time_unit(sample=sample, day=day)
            df_batch[TRADE_TIME] = pd.to_datetime(df_batch[TRADE_TIME], unit=unit)
        # Create a date column from TRADE_TIME
        df_batch["date"] = day
        # Create symbol column
        df_batch["symbol"] = currency_pair.name

        return df_batch

    def save_batched_data_to_hive(self, df_batch: pd.DataFrame) -> None:
        df_batch.to_parquet(
            self.output_dir,
            engine="pyarrow",
            compression="gzip",
            partition_cols=["date", "symbol"],
            existing_data_behavior="overwrite_or_ignore",
        )

    def _partition_dir(self, currency_pair: CurrencyPair, day: date) -> Path:
        """Return the HIVE partition directory for a single (day, symbol) task."""
        return self.output_dir / f"date={day}" / f"symbol={currency_pair.name}"

    def _reset_partition(self, currency_pair: CurrencyPair, day: date) -> None:
        """Delete the target partition directory before writing a task's chunks.

        With `existing_data_behavior="overwrite_or_ignore"` and UUID basenames, re-running the
        same task appends new parquet files alongside the previous run's data, silently doubling
        rows. Because each `unzip_and_save_to_hive` task owns a single (day, symbol) partition
        (multiprocessing workers handle disjoint pairs), it is safe to wipe the partition once at
        task start, before any chunk of the current run has been written.
        """
        partition_dir: Path = self._partition_dir(currency_pair=currency_pair, day=day)
        if partition_dir.exists():
            shutil.rmtree(partition_dir)

    def unzip_and_save_to_hive(self, currency_pair: CurrencyPair, day: date) -> None:
        self._reset_partition(currency_pair=currency_pair, day=day)

        csv_reader = pd.read_csv(
            self.raw_data_dir / currency_pair.name / f"trades@{str(day)}.zip",
            chunksize=1_000_000,
            header=None,
            names=BINANCE_TRADE_COLS,
            usecols=_USE_COLS,
        )

        for df_batch in csv_reader:
            df_batch = self.preprocess_batched_data(df_batch=df_batch, currency_pair=currency_pair, day=day)
            self.save_batched_data_to_hive(df_batch=df_batch)

    def iterate_over_tasks(self) -> Generator[tuple[date, CurrencyPair], Any, None]:
        for symbol in os.listdir(self.raw_data_dir):
            symbol_dir: Path = self.raw_data_dir / symbol
            if not symbol_dir.is_dir():
                _log.warning("Skipping non-directory entry in raw data dir: %s", symbol)
                continue
            try:
                currency_pair: CurrencyPair = CurrencyPair.from_string(symbol=symbol)
            except ValueError:
                _log.warning("Skipping directory that does not look like a CurrencyPair: %s", symbol)
                continue
            filtered_dates: List[date] = filter_by_bounds(bounds=self.bounds, file_names=os.listdir(symbol_dir))

            for day in filtered_dates:
                yield day, currency_pair

    def run_multiprocessing(self, processes: int = 10) -> None:
        with Pool(processes=processes) as pool:
            promises: List[AsyncResult] = []

            for day, currency_pair in self.iterate_over_tasks():
                promise: AsyncResult = pool.apply_async(
                    partial(
                        self.unzip_and_save_to_hive,
                        day=day,
                        currency_pair=currency_pair,
                    ),
                )
                promises.append(promise)

            for promise in tqdm(promises, desc="Saving zipped csv files to HiveDataset: "):
                promise.get()


def run_main():
    bounds: Bounds = Bounds.for_days(date(2018, 1, 1), date(2019, 1, 1))
    pipe = BinanceSpotTrades2Hive(
        bounds=bounds,
        raw_data_dir=BINANCE_SPOT_RAW_TRADES,
        output_dir=BINANCE_SPOT_HIVE_TRADES,
    )
    pipe.run_multiprocessing()


if __name__ == "__main__":
    run_main()
