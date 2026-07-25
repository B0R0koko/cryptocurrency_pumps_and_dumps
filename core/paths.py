from pathlib import Path


def get_root_dir() -> Path:
    start: Path = Path().absolute()
    directory: Path = start
    while not directory.joinpath("pyproject.toml").exists():
        parent: Path = directory.parent
        if parent == directory:
            raise RuntimeError(f"pyproject.toml not found above {start}")
        directory = parent
    return directory


DATA_DIR: Path = Path("/data/pumps/data")
SQLITE_URL: str = "sqlite:////data/pumps/data/studies.db"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
TRANSFORMED_DATA_DIR: Path = DATA_DIR / "transformed"

# Raw zip files
BINANCE_SPOT_RAW_TRADES: Path = RAW_DATA_DIR / "binance" / "spot" / "trades"
BINANCE_SPOT_RAW_KLINES: Path = RAW_DATA_DIR / "binance" / "spot" / "klines"
# HIVE locations
BINANCE_SPOT_HIVE_TRADES: Path = TRANSFORMED_DATA_DIR / "binance" / "spot" / "trades"
FEATURE_DIR: Path = DATA_DIR.joinpath("features")
