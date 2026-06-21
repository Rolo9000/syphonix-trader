"""Asian session range breakout strategy.

Builds the Asian-session high/low (see :func:`core.indicators.calculate_asian_range`) and
emits a breakout :class:`~core.models.TradeSignal` when price sweeps the range
and a market structure shift confirms the reversal.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from datetime import datetime
from typing import List

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:  # pragma: no cover - platform dependent
    mt5 = None
    MT5_AVAILABLE = False

from core.indicators import (
    calculate_asian_range,
    calculate_atr,
    detect_market_structure_shift,
)
from core.models import TradeSignal
from core.mt5_client import MT5Client
from core.risk_manager import RiskManager

logger = logging.getLogger(__name__)

try:
    import logfire
except ImportError:  # pragma: no cover
    logfire = None


def _span(name: str):
    if logfire is not None and hasattr(logfire, "span"):
        return logfire.span(name)
    return nullcontext()


class AsianBreakoutStrategy:
    """Range-breakout strategy keyed on the Asian trading session."""

    symbols: List[str] = ["USDJPY", "AUDUSD", "NZDUSD"]
    min_range_pips: float = 20.0
    risk_per_trade: float = 0.008
    atr_period: int = 14
    atr_stop_mult: float = 1.5

    def __init__(
        self,
        symbols: List[str] | None = None,
        min_range_pips: float | None = None,
        risk_per_trade: float | None = None,
        atr_period: int | None = None,
        atr_stop_mult: float | None = None,
    ) -> None:
        self.symbols = symbols if symbols is not None else list(self.symbols)
        if min_range_pips is not None:
            self.min_range_pips = min_range_pips
        if risk_per_trade is not None:
            self.risk_per_trade = risk_per_trade
        if atr_period is not None:
            self.atr_period = atr_period
        if atr_stop_mult is not None:
            self.atr_stop_mult = atr_stop_mult

    def _price_diff_to_pips(self, symbol: str, diff: float) -> float:
        symbol = symbol.upper()
        if symbol.endswith("JPY") or symbol.endswith("JPY"):
            return abs(diff) * 100.0
        if symbol.startswith("XAU") or "GOLD" in symbol:
            return abs(diff) * 100.0
        return abs(diff) * 10000.0

    def generate_signals(self, client: MT5Client, risk_manager: RiskManager) -> List[TradeSignal]:
        """Generate breakout signals for configured symbols."""
        signals: List[TradeSignal] = []
        with _span("AsianBreakoutStrategy.generate_signals"):
            for symbol in self.symbols:
                try:
                    session_range = calculate_asian_range(client, symbol)
                    if session_range is None:
                        logger.debug("No Asian range for %s", symbol)
                        continue

                    session_low, session_high = session_range
                    range_width = abs(session_high - session_low)
                    range_width_pips = self._price_diff_to_pips(symbol, range_width)
                    if range_width_pips < self.min_range_pips:
                        logger.debug("Asian range too small for %s: %s pips", symbol, range_width_pips)
                        continue

                    candles = client.get_candles(symbol, mt5.TIMEFRAME_M5, 50)
                    if candles.empty:
                        logger.debug("No M5 candles for %s", symbol)
                        continue

                    latest = candles.iloc[-1]
                    current_low = float(latest["low"])
                    current_high = float(latest["high"])
                    current_close = float(latest["close"])
                    current_open = float(latest["open"])

                    bullish_sweep = current_low < session_low
                    bearish_sweep = current_high > session_high
                    if not bullish_sweep and not bearish_sweep:
                        continue

                    market_structure = detect_market_structure_shift(candles)
                    if bullish_sweep and market_structure == "BULLISH_MSS":
                        entry = current_close
                        atr_value = calculate_atr(candles, self.atr_period)
                        stop_loss = current_low - atr_value * 0.5
                        take_profit = entry + range_width * 1.2
                        stop_loss_pips = self._price_diff_to_pips(symbol, abs(entry - stop_loss))
                        volume = risk_manager.calculate_position_size(symbol, stop_loss_pips, self.risk_per_trade)
                        if volume <= 0:
                            continue
                        signals.append(
                            TradeSignal(
                                symbol=symbol,
                                action="BUY",
                                entry_price=entry,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                volume=volume,
                                strategy_name="AsianBreakout",
                                confidence=0.6,
                                timestamp=datetime.utcnow(),
                            )
                        )
                    elif bearish_sweep and market_structure == "BEARISH_MSS":
                        entry = current_close
                        atr_value = calculate_atr(candles, self.atr_period)
                        stop_loss = current_high + atr_value * 0.5
                        take_profit = entry - range_width * 1.2
                        stop_loss_pips = self._price_diff_to_pips(symbol, abs(entry - stop_loss))
                        volume = risk_manager.calculate_position_size(symbol, stop_loss_pips, self.risk_per_trade)
                        if volume <= 0:
                            continue
                        signals.append(
                            TradeSignal(
                                symbol=symbol,
                                action="SELL",
                                entry_price=entry,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                volume=volume,
                                strategy_name="AsianBreakout",
                                confidence=0.6,
                                timestamp=datetime.utcnow(),
                            )
                        )
                except Exception:
                    logger.exception("Failed to generate signal for %s", symbol)
                    continue
        return signals
