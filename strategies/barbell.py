"""Gold / BTC barbell portfolio rebalancing strategy.

Maintains a target allocation split across a defensive leg (gold) and a
high-volatility leg (BTC). Emits rebalancing :class:`~core.models.TradeSignal`
objects when realized weights drift beyond a tolerance band.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from datetime import datetime
from typing import Dict, List

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:  # pragma: no cover - platform dependent
    mt5 = None
    MT5_AVAILABLE = False
import pandas as pd

from core.indicators import calculate_atr, calculate_portfolio_weights, calculate_trend_strength, is_rapid_decline, is_rapid_rally
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


class BarbellStrategy:
    """Two-leg barbell allocation with drift-triggered rebalancing."""

    symbols: List[str] = [
        "XAUUSD",
        # "XAGUSD",  # Excluded - existing position we want to let ride
        "BTCUSD",
        "ETHUSD",
        "SOLUSD",
        "XRPUSD",
    ]
    rebalance_threshold: float = 0.10
    target_weights: Dict[str, float] = {
        "XAUUSD": 0.35,   # Boosted from 0.30
        # "XAGUSD": 0.10, # Excluded
        "BTCUSD": 0.30,   # Boosted from 0.25
        "ETHUSD": 0.15,
        "SOLUSD": 0.10,
        "XRPUSD": 0.10,
    }
    total_allocation_pct: float = 0.35  # Increased from 0.25 (40% more capital)

    def __init__(
        self,
        symbols: List[str] | None = None,
        target_weights: Dict[str, float] | None = None,
        rebalance_threshold: float | None = None,
        total_allocation_pct: float | None = None,
    ) -> None:
        if symbols is not None:
            self.symbols = symbols
        if target_weights is not None:
            self.target_weights = target_weights
        if rebalance_threshold is not None:
            self.rebalance_threshold = rebalance_threshold
        if total_allocation_pct is not None:
            self.total_allocation_pct = total_allocation_pct

    def _position_notional(self, symbol: str, positions: List[TradeSignal]) -> float:
        return 0.0

    def _current_notional(self, client: MT5Client) -> Dict[str, float]:
        exposures: Dict[str, float] = {symbol: 0.0 for symbol in self.symbols}
        for position in client.get_open_positions():
            if position.symbol not in exposures:
                continue
            price = float(position.current_price) if position.current_price else float(client.get_current_price(position.symbol)[0])
            exposures[position.symbol] += abs(float(position.volume) * price)
        return exposures

    def update_weights_mvo(self, client: MT5Client) -> Dict[str, float]:
        """Update target weights using mean-variance optimization."""
        with _span("BarbellStrategy.update_weights_mvo"):
            returns: Dict[str, pd.Series] = {}
            for symbol in self.symbols:
                try:
                    candles = client.get_candles(symbol, mt5.TIMEFRAME_D1, 30)
                    if candles.empty or "close" not in candles.columns:
                        logger.warning("Insufficient daily candles for %s", symbol)
                        continue
                    close = candles["close"].astype(float)
                    daily_returns = close.pct_change().dropna()
                    if daily_returns.empty:
                        logger.warning("Insufficient return data for %s", symbol)
                        continue
                    returns[symbol] = daily_returns
                except Exception:
                    logger.exception("Failed to collect returns for %s", symbol)
                    continue

            if not returns:
                logger.warning("No returns data available for MVO update")
                return self.target_weights

            weights = calculate_portfolio_weights(returns)
            self.target_weights = {symbol: weights.get(symbol, self.target_weights.get(symbol, 0.0)) for symbol in self.symbols}
            return self.target_weights

    def get_current_weights(self, client: MT5Client) -> Dict[str, float]:
        """Return the current realized portfolio weight distribution."""
        with _span("BarbellStrategy.get_current_weights"):
            exposures = self._current_notional(client)
            total = sum(exposures.values())
            if total <= 0:
                return {symbol: 0.0 for symbol in self.symbols}
            return {symbol: exposures[symbol] / total for symbol in self.symbols}

    def generate_rebalance_signals(self, client: MT5Client, risk_manager: RiskManager, state_store: "StateStore | None" = None) -> List[TradeSignal]:
        """Generate rebalancing trade signals with trend-weighted allocation.
        
        Hybrid approach inspired by competition leaders:
        - Base allocation: 35% of equity across all assets
        - Trend boost: Up to 2.5x weight for strongly trending assets
        - Sentiment boost: +30% volume when sentiment aligns with trend
        - Confidence-based sizing: 0.5x volume for neutral, 1.5x for strong trends
        - Conservative stops (0.8x ATR) and tight profits (0.7x ATR)
        - $50k position cap per trade
        """
        signals: List[TradeSignal] = []
        with _span("BarbellStrategy.generate_rebalance_signals"):
            safe, reason = risk_manager.is_safe_to_trade()
            if not safe:
                logger.warning("Skipping rebalance because trading is not safe: %s", reason)
                return signals

            state = client.get_account_info()
            base_allocation = float(state.equity) * float(self.total_allocation_pct)
            current_exposures = self._current_notional(client)
            current_weights = self.get_current_weights(client)
            
            # Track existing position directions to prevent stacking
            existing_positions = {}
            for pos in client.get_open_positions():
                if pos.symbol in self.symbols:
                    # Positive volume = BUY, Negative = SELL
                    direction = "BUY" if pos.volume > 0 else "SELL"
                    existing_positions[pos.symbol] = direction

            # Step 1: Calculate trend strength for all symbols and compute dynamic weights
            # Using M15 for faster trend response (was H1 which lagged badly)
            trend_data = {}
            total_trend_score = 0.0
            for symbol in self.symbols:
                try:
                    candles = client.get_candles(symbol, mt5.TIMEFRAME_M15, 30)
                    direction, strength = calculate_trend_strength(candles)
                    # Check for rapid price movement
                    declining = is_rapid_decline(candles, threshold=0.003, bars=4)
                    rallying = is_rapid_rally(candles, threshold=0.003, bars=4)
                    trend_data[symbol] = {
                        "direction": direction, 
                        "strength": strength,
                        "declining": declining,
                        "rallying": rallying
                    }
                    # Trending assets get boosted weight (1.0 to 2.5x)
                    if strength > 0.5:  # Stricter threshold (was 0.3)
                        trend_score = 1.0 + (strength * 1.5)  # Max 2.5x at full strength
                    else:
                        trend_score = 1.0
                    trend_data[symbol]["score"] = trend_score
                    total_trend_score += trend_score * float(self.target_weights.get(symbol, 0.0))
                except Exception:
                    trend_data[symbol] = {"direction": "NEUTRAL", "strength": 0.0, "score": 1.0, "declining": False, "rallying": False}
                    total_trend_score += float(self.target_weights.get(symbol, 0.0))

            # Step 2: Compute trend-adjusted allocation per symbol
            for symbol in self.symbols:
                try:
                    base_weight = float(self.target_weights.get(symbol, 0.0))
                    trend_score = trend_data[symbol]["score"]
                    trend_direction = trend_data[symbol]["direction"]
                    trend_strength = trend_data[symbol]["strength"]
                    
                    # Dynamic weight: boost trending assets proportionally
                    if total_trend_score > 0:
                        dynamic_weight = (base_weight * trend_score) / total_trend_score
                    else:
                        dynamic_weight = base_weight
                    
                    actual_weight = float(current_weights.get(symbol, 0.0))
                    drift = float(actual_weight - dynamic_weight)
                    
                    # Use tighter threshold for trending assets (want to be in trend quickly)
                    effective_threshold = float(self.rebalance_threshold) * (1.0 - trend_strength * 0.5)
                    if abs(drift) <= effective_threshold:
                        continue

                    bid, ask = client.get_current_price(symbol)
                    current_price = float(bid)
                    spread = float(ask - bid)
                    desired_notional = float(base_allocation) * dynamic_weight
                    current_notional = float(current_exposures.get(symbol, 0.0))
                    notional_diff = float(desired_notional - current_notional)
                    if abs(notional_diff) < float(current_price) * 0.0001:
                        continue

                    # Determine action - but only trade WITH the trend if trend is strong
                    raw_action = "BUY" if notional_diff > 0 else "SELL"
                    
                    # CRITICAL: Don't stack positions in the same direction
                    if symbol in existing_positions and existing_positions[symbol] == raw_action:
                        logger.debug("Skipping %s %s - already have %s position", raw_action, symbol, raw_action)
                        continue
                    
                    # RAPID MOVEMENT FILTER: Don't buy into falling knife, don't sell into rally
                    if raw_action == "BUY" and trend_data[symbol].get("declining", False):
                        logger.info("Skipping BUY %s - rapid decline detected (price dropping fast)", symbol)
                        continue
                    if raw_action == "SELL" and trend_data[symbol].get("rallying", False):
                        logger.info("Skipping SELL %s - rapid rally detected (price rising fast)", symbol)
                        continue
                    
                    # Skip trades against trends (stricter threshold: 0.5 instead of 0.3)
                    if trend_strength >= 0.5:
                        if (trend_direction == "UP" and raw_action == "SELL") or \
                           (trend_direction == "DOWN" and raw_action == "BUY"):
                            logger.info("Skipping %s %s - against trend (%s, strength=%.2f)",
                                       raw_action, symbol, trend_direction, trend_strength)
                            continue
                    
                    action = raw_action
                    entry_price = float(current_price)
                    
                    try:
                        candles = client.get_candles(symbol, mt5.TIMEFRAME_M15, 30)
                        atr = calculate_atr(candles)
                        if atr is None or atr <= 0.0:
                            raise ValueError("ATR calculation returned invalid value")
                    except Exception:
                        logger.warning("ATR calculation failed for %s, using fallback volatility", symbol)
                        atr = float(entry_price) * 0.01

                    # Wider profits to capture larger moves: 1.5x ATR TP, 1.0x ATR SL (1.5:1 R:R)
                    if action == "BUY":
                        stop_loss = float(entry_price - atr * 1.0)
                        take_profit = float(entry_price + atr * 1.5)
                    else:
                        stop_loss = float(entry_price + atr * 1.0)
                        take_profit = float(entry_price - atr * 1.5)
                    
                    # SANITY CHECK: TP must be profitable
                    if action == "BUY" and take_profit <= entry_price:
                        logger.error("BUG: BUY TP %.5f <= entry %.5f for %s, skipping", take_profit, entry_price, symbol)
                        continue
                    if action == "SELL" and take_profit >= entry_price:
                        logger.error("BUG: SELL TP %.5f >= entry %.5f for %s, skipping", take_profit, entry_price, symbol)
                        continue

                    # Spread filter: skip if spread > 30% of TP distance (kills profitability at high frequency)
                    tp_distance = abs(take_profit - entry_price)
                    if spread > tp_distance * 0.30:
                        logger.info("Skipping %s %s - spread %.5f > 30%% of TP distance %.5f",
                                   action, symbol, spread, tp_distance)
                        continue

                    stop_loss_pips = float(abs(entry_price - stop_loss) * (100.0 if symbol.endswith("JPY") else 10000.0))
                    base_volume = float(risk_manager.calculate_position_size(symbol, stop_loss_pips, risk_manager.risk_per_trade))
                    
                    # Confidence-based sizing: scale volume by trend strength
                    # - Neutral trend (0.0): 0.5x volume (conservative)
                    # - Strong trend (1.0): 1.5x volume (leader-style conviction)
                    confidence_multiplier = 0.5 + (trend_strength * 1.0)  # Range: 0.5 to 1.5
                    volume = base_volume * confidence_multiplier
                    
                    # Sentiment boost: +30% when sentiment aligns with trend/action
                    sentiment_boost = 1.0
                    if state_store is not None:
                        try:
                            sentiment_result = state_store.get_sentiment(symbol)
                            if sentiment_result:
                                sentiment = sentiment_result.sentiment.upper()
                                # Boost if sentiment aligns with action
                                if (sentiment == "BULLISH" and action == "BUY") or \
                                   (sentiment == "BEARISH" and action == "SELL"):
                                    sentiment_boost = 1.30  # +30% volume
                                    logger.info("%s sentiment %s aligns with %s - boosting volume 30%%",
                                               symbol, sentiment, action)
                                # Reduce if sentiment opposes action
                                elif (sentiment == "BULLISH" and action == "SELL") or \
                                     (sentiment == "BEARISH" and action == "BUY"):
                                    sentiment_boost = 0.70  # -30% volume
                                    logger.info("%s sentiment %s opposes %s - reducing volume 30%%",
                                               symbol, sentiment, action)
                        except Exception:
                            pass  # Sentiment unavailable, use default
                    volume = volume * sentiment_boost
                    
                    if volume <= 0.0:
                        continue

                    # Cap position at $100k notional maximum (increased for bigger winners)
                    try:
                        tick = mt5.symbol_info_tick(symbol)
                        if tick:
                            price = (float(tick.bid) + float(tick.ask)) / 2.0
                            symbol_info = mt5.symbol_info(symbol)
                            if symbol_info:
                                contract_size = float(symbol_info.trade_contract_size)
                                position_notional = volume * price * contract_size
                                if position_notional > 100000.0:
                                    volume = 100000.0 / (price * contract_size)
                                    volume = round(round(volume / float(symbol_info.volume_step)) * float(symbol_info.volume_step), 8)
                                    volume = max(float(symbol_info.volume_min), volume)
                    except Exception:
                        logger.warning("Position cap check failed for %s", symbol)

                    if volume <= 0.0:
                        continue

                    # Confidence: base 0.65 + trend_strength * 0.30 (range 0.65 to 0.95)
                    signal_confidence = 0.65 + (trend_strength * 0.30)
                    
                    logger.info("Signal: %s %s vol=%.2f (trend=%s, strength=%.2f, conf=%.2f, dynamic_wt=%.2f)",
                               action, symbol, volume, trend_direction, trend_strength, signal_confidence, dynamic_weight)

                    signals.append(
                        TradeSignal(
                            symbol=symbol,
                            action=action,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            volume=volume,
                            strategy_name="BarbellRebalance",
                            confidence=signal_confidence,
                            timestamp=datetime.utcnow(),
                        )
                    )
                except Exception:
                    logger.exception("Failed to generate rebalance signal for %s", symbol)
                    continue
        return signals
