from pathlib import Path


def get_root_dir() -> Path:
    directory = Path().absolute()
    while not directory.joinpath("pyproject.toml").exists():
        directory = directory.parents[0]
    return directory


DATA_DIR: Path = Path("D:/data")
RAW_DATA_DIR: Path = DATA_DIR / "raw"
TRANSFORMED_DATA_DIR: Path = DATA_DIR / "transformed"

# Raw zip files
BINANCE_SPOT_RAW_TRADES: Path = RAW_DATA_DIR / "binance" / "spot" / "trades"
# HIVE locations
BINANCE_SPOT_HIVE_TRADES: Path = TRANSFORMED_DATA_DIR / "binance" / "spot" / "trades"
FEATURE_DIR: Path = DATA_DIR.joinpath("features")
