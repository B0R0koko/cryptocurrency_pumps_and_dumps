import logging
import warnings


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    warnings.filterwarnings("ignore", category=FutureWarning)
