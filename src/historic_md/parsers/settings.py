# pylint: disable=line-too-long

GB = 8 * 1024 * 1024 * 1024

SETTINGS = dict(
    LOG_LEVEL="DEBUG",
    # Parser settings
    USER_AGENT="Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
    # Obey robots.txt rules
    ROBOTSTXT_OBEY=False,
    COOKIES_ENABLED=False,
    # Configure maximum concurrent requests performed by Scrapy (default: 16)
    CONCURRENT_REQUESTS=64,
    CONCURRENT_REQUESTS_PER_DOMAIN=64,
    REACTOR_THREADPOOL_MAXSIZE=64,
    DOWNLOAD_DELAY=0,
    # Change these to avoid timeouts to download and corresponding warnings
    DOWNLOAD_WARNSIZE=0.5 * GB,
    DOWNLOAD_MAXSIZE=15 * GB,
    DOWNLOAD_TIMEOUT=60 * 60 * 24,
)
