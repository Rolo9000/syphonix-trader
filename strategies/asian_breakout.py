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
    calculate_trend_strength,
    is_rapid_decline,
    is_rapid_rally,
)
from core.models import TradeSignal
from core.mt5_client import MT5Client
from core.risk_manager import RiskManager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infra.state_store import StateStore

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

    symbols: List[str] = [
        "EURUSD",
        "GBPUSD",
        "USDCHF",
        "USDJPY",
        "USDCAD",
        "AUDUSD",
        "EURGBP",
        "EURCHF",
    ]
    min_range_pips: float = 20.0
    risk_per_trade: float = 0.010  # Increased from 0.005 (2x position sizing)
    atr_period: int = 14
    atr_stop_mult: float = 0.3

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

    def generate_signals(self, client: MT5Client, risk_manager: RiskManager, state_store: "StateStore | None" = None) -> List[TradeSignal]:
        """Generate breakout signals with trend confirmation.
        
        Hybrid approach:
        - Only trade breakouts in the direction of the underlying trend
        - Scale position size by trend strength (0.5x to 1.5x)
        - Sentiment boost: +30% volume when sentiment aligns
        - Boost confidence when trend confirms breakout
        """
        signals: List[TradeSignal] = []
        with _span("AsianBreakoutStrategy.generate_signals"):
            # Get existing positions to avoid stacking
            open_positions = {p.symbol for p in client.get_open_positions()}
            
            for symbol in self.symbols:
                try:
                    # Skip if we already have a position in this symbol
                    if symbol in open_positions:
                        logger.debug("Skipping %s - already have open position", symbol)
                        continue
                    
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

                    # Get M15 candles for trend detection (faster than H1)
                    try:
                        m15_candles = client.get_candles(symbol, mt5.TIMEFRAME_M15, 30)
                        trend_direction, trend_strength = calculate_trend_strength(m15_candles)
                        # Check rapid price movement
                        declining = is_rapid_decline(m15_candles, threshold=0.003, bars=4)
                        rallying = is_rapid_rally(m15_candles, threshold=0.003, bars=4)
                    except Exception:
                        trend_direction, trend_strength = "NEUTRAL", 0.0
                        declining, rallying = False, False

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
                    
                    # Get spread for filtering
                    bid, ask = client.get_current_price(symbol)
                    spread = float(ask - bid)
                    
                    # Only take bullish breakout if trend is UP or NEUTRAL (not against DOWN trend)
                    if bullish_sweep and market_structure == "BULLISH_MSS":
                        if trend_direction == "DOWN" and trend_strength >= 0.5:
                            logger.info("Skipping bullish breakout on %s - against DOWN trend (strength=%.2f)", symbol, trend_strength)
                            continue
                        # Don't buy into falling knife
                        if declining:
                            logger.info("Skipping bullish breakout on %s - rapid decline detected", symbol)
                            continue
                        
                        entry = current_close
                        atr_value = calculate_atr(candles, self.atr_period)
                        stop_loss = current_low - atr_value * 0.5
                        take_profit = entry + range_width * 1.2  # Let winners run (was 0.6x)
                        
                        # SANITY CHECK: TP must be above entry for BUY
                        if take_profit <= entry:
                            logger.error("BUG: BUY TP %.5f <= entry %.5f for %s, skipping", take_profit, entry, symbol)
                            continue
                        
                        # Spread filter: skip if spread > 30% of TP distance
                        tp_distance = abs(take_profit - entry)
                        if spread > tp_distance * 0.30:
                            logger.info("Skipping BUY %s - spread %.5f > 30%% of TP distance %.5f",
                                       symbol, spread, tp_distance)
                            continue
                        
                        stop_loss_pips = self._price_diff_to_pips(symbol, abs(entry - stop_loss))
                        
                        # Scale volume by trend confirmation (0.5x if neutral, 1.5x if trend confirms)
                        base_volume = risk_manager.calculate_position_size(symbol, stop_loss_pips, self.risk_per_trade)
                        if trend_direction == "UP":
                            volume = base_volume * (0.5 + trend_strength * 1.0)  # 0.5x to 1.5x
                            signal_confidence = 0.70 + (trend_strength * 0.25)
                        else:
                            volume = base_volume * 0.5  # Reduced size without trend confirmation
                            signal_confidence = 0.65
                        
                        # Sentiment boost: +30% when sentiment aligns with BUY
                        if state_store is not None:
                            try:
                                sentiment_result = state_store.get_sentiment(symbol)
                                if sentiment_result:
                                    sentiment = sentiment_result.sentiment.upper()
                                    if sentiment == "BULLISH":
                                        volume = volume * 1.30
                                        logger.info("%s BULLISH sentiment aligns with BUY - boosting volume 30%%", symbol)
                                    elif sentiment == "BEARISH":
                                        volume = volume * 0.70
                                        logger.info("%s BEARISH sentiment opposes BUY - reducing volume 30%%", symbol)
                            except Exception:
                                pass
                        
                        if volume <= 0:
                            continue
                            
                        logger.info("BUY signal: %s vol=%.2f (trend=%s, strength=%.2f, conf=%.2f)",
                                   symbol, volume, trend_direction, trend_strength, signal_confidence)
                        signals.append(
                            TradeSignal(
                                symbol=symbol,
                                action="BUY",
                                entry_price=entry,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                volume=volume,
                                strategy_name="AsianBreakout",
                                confidence=signal_confidence,
                                timestamp=datetime.utcnow(),
                            )
                        )
                    # Only take bearish breakout if trend is DOWN or NEUTRAL (not against UP trend)
                    elif bearish_sweep and market_structure == "BEARISH_MSS":
                        if trend_direction == "UP" and trend_strength >= 0.5:
                            logger.info("Skipping bearish breakout on %s - against UP trend (strength=%.2f)", symbol, trend_strength)
                            continue
                        # Don't sell into rallying market
                        if rallying:
                            logger.info("Skipping bearish breakout on %s - rapid rally detected", symbol)
                            continue
                        
                        entry = current_close
                        atr_value = calculate_atr(candles, self.atr_period)
                        stop_loss = current_high + atr_value * 0.5
                        take_profit = entry - range_width * 1.2  # Let winners run (was 0.6x)
                        
                        # SANITY CHECK: TP must be below entry for SELL
                        if take_profit >= entry:
                            logger.error("BUG: SELL TP %.5f >= entry %.5f for %s, skipping", take_profit, entry, symbol)
                            continue
                        
                        # Spread filter: skip if spread > 30% of TP distance
                        tp_distance = abs(take_profit - entry)
                        if spread > tp_distance * 0.30:
                            logger.info("Skipping SELL %s - spread %.5f > 30%% of TP distance %.5f",
                                       symbol, spread, tp_distance)
                            continue
                        
                        stop_loss_pips = self._price_diff_to_pips(symbol, abs(entry - stop_loss))
                        
                        # Scale volume by trend confirmation
                        base_volume = risk_manager.calculate_position_size(symbol, stop_loss_pips, self.risk_per_trade)
                        if trend_direction == "DOWN":
                            volume = base_volume * (0.5 + trend_strength * 1.0)  # 0.5x to 1.5x
                            signal_confidence = 0.70 + (trend_strength * 0.25)
                        else:
                            volume = base_volume * 0.5  # Reduced size without trend confirmation
                            signal_confidence = 0.65
                        
                        # Sentiment boost: +30% when sentiment aligns with SELL
                        if state_store is not None:
                            try:
                                sentiment_result = state_store.get_sentiment(symbol)
                                if sentiment_result:
                                    sentiment = sentiment_result.sentiment.upper()
                                    if sentiment == "BEARISH":
                                        volume = volume * 1.30
                                        logger.info("%s BEARISH sentiment aligns with SELL - boosting volume 30%%", symbol)
                                    elif sentiment == "BULLISH":
                                        volume = volume * 0.70
                                        logger.info("%s BULLISH sentiment opposes SELL - reducing volume 30%%", symbol)
                            except Exception:
                                pass
                        
                        if volume <= 0:
                            continue
                        
                        logger.info("SELL signal: %s vol=%.2f (trend=%s, strength=%.2f, conf=%.2f)",
                                   symbol, volume, trend_direction, trend_strength, signal_confidence)
                        signals.append(
                            TradeSignal(
                                symbol=symbol,
                                action="SELL",
                                entry_price=entry,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                volume=volume,
                                strategy_name="AsianBreakout",
                                confidence=signal_confidence,
                                timestamp=datetime.utcnow(),
                            )
                        )
                except Exception:
                    logger.exception("Failed to generate signal for %s", symbol)
                    continue
        return signals
