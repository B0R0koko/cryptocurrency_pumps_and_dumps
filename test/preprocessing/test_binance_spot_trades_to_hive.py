import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl

from core.columns import SYMBOL
from core.currency_pair import CurrencyPair
from core.paths import BINANCE_SPOT_RAW_TRADES
from core.time_utils import Bounds
from core.utils import configure_logging
from preprocessing.pipelines.binance_spot_trades_to_hive import BinanceSpotTrades2Hive


def test_binance_spot_trades_to_hive() -> None:
    """
    This test makes sure that the way HiveDataset is created matches the result produced by simply unpacking a csv file
    and reading it with pandas, we simply compare shapes of two dataframes
    """
    configure_logging()
    day: date = date(2025, 5, 1)
    bounds: Bounds = Bounds.for_day(day)
    currency_pair: CurrencyPair = CurrencyPair.from_string("ADA-USDT")
    zip_file_name: str = "trades@2025-05-01.zip"

    with (tempfile.TemporaryDirectory() as temp_dir):
        # Create a hive structure in the test folder
        output_dir: Path = Path(temp_dir)
        pipe: BinanceSpotTrades2Hive = BinanceSpotTrades2Hive(
            bounds=bounds, output_dir=output_dir
        )

        pipe.unzip_and_save_to_hive(currency_pair=currency_pair, day=day)

        # Collect the number of rows lazily
        hive: pl.LazyFrame = pl.scan_parquet(source=output_dir, hive_partitioning=True)

        hive_size: int = (
            hive
            .filter(
                (pl.col(SYMBOL) == currency_pair.name)
            )
            .select(pl.len()).collect().item()
        )

        # unzip data using pandas by simply unpacking csv file and then reading csv with pandas
        zip_file_path: Path = BINANCE_SPOT_RAW_TRADES.joinpath(currency_pair.name).joinpath(zip_file_name)

        df_pandas: pd.DataFrame = pd.read_csv(zip_file_path, compression="zip", header=None)

        assert df_pandas.shape[0] == hive_size, "Polars HiveDataset and Pandas.DataFrame have different shapes"
