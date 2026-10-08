"""Binance Spot ingestion: message parser, REST client and raw-capture recorder.

All Binance-specific field names and endpoints are confined to this package.
"""

from stlens.ingestion.binance.parser import (
    EXCHANGE,
    BinanceAdapter,
    BinanceDepthMessage,
    BinanceParseError,
    ParsedMessage,
    TimeUnit,
    depth_messages_frame,
)

__all__ = [
    "EXCHANGE",
    "BinanceAdapter",
    "BinanceDepthMessage",
    "BinanceParseError",
    "ParsedMessage",
    "TimeUnit",
    "depth_messages_frame",
]
