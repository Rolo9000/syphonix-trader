"""Market regime detection (trending / ranging / high-volatility).

Classifies the current market state from recent price action and volatility so
the aggregator can weight strategies appropriately (e.g. favour breakout in
trending regimes, suppress it in choppy ranges).
"""

from __future__ import annotations

from enum import Enum

from core.mt5_client import MT5Client


class MarketRegime(str, Enum):
    """Coarse classification of the prevailing market state."""

    TRENDING = "trending"
    RANGING = "ranging"
    HIGH_VOL = "high_vol"


class RegimeDetector:
    """Infers the current :class:`MarketRegime` for an instrument."""

    def __init__(self, client: MT5Client, symbol: str, lookback: int = 100) -> None:
        """Bind to a broker client, instrument, and lookback window."""
        raise NotImplementedError

    def detect(self) -> MarketRegime:
        """Return the regime classification for the most recent bars."""
        raise NotImplementedError
