import os
import re
from datetime import date, datetime
from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import List, Any, Generator

import pandas as pd
from tqdm import tqdm

from core.columns import TRADE_TIME, BINANCE_TRADE_COLS, PRICE, QUANTITY, IS_BUYER_MAKER
from core.currency_pair import CurrencyPair
from core.paths import BINANCE_SPOT_RAW_TRADES, BINANCE_SPOT_HIVE_TRADES
from core.time_utils import Bounds

_USE_COLS: List[str] = [PRICE, QUANTITY, TRADE_TIME, IS_BUYER_MAKER]


def filter_by_bounds(bounds: Bounds, file_names: List[str]) -> List[date]:
    """Returns a list of Tuple [(date(2025, 1, 1)]"""
    valid_dates: List[date] = []

    for file in file_names:
        date_part = re.search(r'@(\d{4}-\d{2}-\d{2})\.zip$', file)[1]
        day: date = datetime.strptime(date_part, "%Y-%m-%d").date()

        if bounds.contain_days(day=day):
            valid_dates.append(day)

    return valid_dates


class BinanceSpotTrades2Hive:

    def __init__(
            self, bounds: Bounds,
            raw_data_dir: Path,
            output_dir: Path,
    ):
        self.bounds: Bounds = bounds
        self.raw_data_dir: Path = raw_data_dir
        self.output_dir: Path = output_dir

    def preprocess_batched_data(self, df_batch: pd.DataFrame, currency_pair: CurrencyPair, day: date) -> pd.DataFrame:
        """Attach new columns and convert dtypes here before saving to hive structure"""
        # Since 2025-01-01 Binance Spot data is written not in "ms" but in "us" - microseconds
        unit: str = "ms" if day < date(2025, 1, 1) else "us"
        df_batch[TRADE_TIME] = pd.to_datetime(df_batch[TRADE_TIME], unit=unit)
        # Create date column from TRADE_TIME
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

    def unzip_and_save_to_hive(self, currency_pair: CurrencyPair, day: date) -> None:
        csv_reader = pd.read_csv(
            self.raw_data_dir / currency_pair.name / f"trades@{str(day)}.zip",
            chunksize=1_000_000,
            header=None,
            names=BINANCE_TRADE_COLS,
            usecols=_USE_COLS,
        )

        for batch_id, df_batch in enumerate(csv_reader):
            df_batch = self.preprocess_batched_data(df_batch=df_batch, currency_pair=currency_pair, day=day)
            self.save_batched_data_to_hive(df_batch=df_batch)

    def iterate_over_tasks(self) -> Generator[tuple[date, CurrencyPair], Any, None]:
        for symbol in os.listdir(self.raw_data_dir):
            filtered_dates: List[date] = filter_by_bounds(
                bounds=self.bounds, file_names=os.listdir(self.raw_data_dir / symbol)
            )
            currency_pair: CurrencyPair = CurrencyPair.from_string(symbol=symbol)

            for day in filtered_dates:
                yield day, currency_pair

    def run_multiprocessing(self, processes: int = 10) -> None:
        with Pool(processes=processes) as pool:
            promises: List[AsyncResult] = []

            for (day, currency_pair) in self.iterate_over_tasks():
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
    bounds: Bounds = Bounds.for_days(
        date(2019, 1, 1), date(2020, 1, 1)
    )
    pipe = BinanceSpotTrades2Hive(
        bounds=bounds,
        raw_data_dir=BINANCE_SPOT_RAW_TRADES,
        output_dir=BINANCE_SPOT_HIVE_TRADES
    )
    pipe.run_multiprocessing()


if __name__ == "__main__":
    run_main()
