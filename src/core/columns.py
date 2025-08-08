from typing import List

TRADE_ID: str = "trade_id"
PRICE: str = "price"
QUANTITY: str = "quantity"
QUOTE_QUANTITY: str = "quote_quantity"
TRADE_TIME: str = "trade_time"
IS_BUYER_MAKER: str = "is_buyer_maker"
IS_BEST_MATCH: str = "is_best_match"

BINANCE_TRADE_COLS: List[str] = [
    TRADE_ID, PRICE, QUANTITY, QUOTE_QUANTITY, TRADE_TIME, IS_BUYER_MAKER, IS_BEST_MATCH
]

BINANCE_TRADE_USDM_COLS: List[str] = [
    TRADE_ID, PRICE, QUANTITY, QUOTE_QUANTITY, TRADE_TIME, IS_BUYER_MAKER
]

# Kline data example that comes from Binance API
# [
#   [
#     1499040000000,      // Kline open time
#     "0.01634790",       // Open price
#     "0.80000000",       // High price
#     "0.01575800",       // Low price
#     "0.01577100",       // Close price
#     "148976.11427815",  // Volume
#     1499644799999,      // Kline Close time
#     "2434.19055334",    // Quote asset volume
#     308,                // Number of trades
#     "1756.87402397",    // Taker buy base asset volume
#     "28.46694368",      // Taker buy quote asset volume
#     "0"                 // Unused field, ignore.
#   ]
# ]

OPEN_TIME: str = "open_time"
OPEN_PRICE: str = "open_price"
HIGH_PRICE: str = "high_price"
LOW_PRICE: str = "low_price"
CLOSE_PRICE: str = "close_price"
VOLUME: str = "volume"
CLOSE_TIME: str = "close_time"
QUOTE_ASSET_VOLUME: str = "quote_asset_volume"
NUM_TRADES: str = "num_trades"
TAKER_BUY_BASE_ASSET_VOLUME: str = "taker_buy_base_asset_volume"
TAKER_BUY_QUOTE_ASSET_VOLUME: str = "taker_buy_quote_asset_volume"
_UNUSED: str = "unused"

# BUY_VOL - (TOTAL_VOL - BUY_VOL)

BINANCE_KLINES_COLS: List[str] = [
    OPEN_TIME, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, CLOSE_PRICE, VOLUME, CLOSE_TIME, QUOTE_ASSET_VOLUME,
    NUM_TRADES, TAKER_BUY_BASE_ASSET_VOLUME, TAKER_BUY_QUOTE_ASSET_VOLUME, _UNUSED
]

SYMBOL: str = "symbol"
DATE: str = "date"
SAMPLED_TIME: str = "sampled_time"
