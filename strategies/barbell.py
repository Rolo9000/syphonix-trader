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

from core.indicators import calculate_atr, calculate_portfolio_weights
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


class BarbellStrategy:
    """Two-leg barbell allocation with drift-triggered rebalancing."""

    symbols: List[str] = [
        "XAUUSD",
        "XAGUSD",
        "BTCUSD",
        "ETHUSD",
        "SOLUSD",
        "XRPUSD",
    ]
    rebalance_threshold: float = 0.05
    target_weights: Dict[str, float] = {
        "XAUUSD": 0.30,
        "XAGUSD": 0.10,
        "BTCUSD": 0.25,
        "ETHUSD": 0.15,
        "SOLUSD": 0.10,
        "XRPUSD": 0.10,
    }
    total_allocation_pct: float = 0.70

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

    def generate_rebalance_signals(self, client: MT5Client, risk_manager: RiskManager) -> List[TradeSignal]:
        """Generate rebalancing trade signals when weights drift outside tolerance."""
        signals: List[TradeSignal] = []
        with _span("BarbellStrategy.generate_rebalance_signals"):
            safe, reason = risk_manager.is_safe_to_trade()
            if not safe:
                logger.warning("Skipping rebalance because trading is not safe: %s", reason)
                return signals

            state = client.get_account_info()
            allocation_amount = float(state.equity) * float(self.total_allocation_pct)
            current_exposures = self._current_notional(client)
            current_weights = self.get_current_weights(client)

            for symbol in self.symbols:
                try:
                    target_weight = float(self.target_weights.get(symbol, 0.0))
                    actual_weight = float(current_weights.get(symbol, 0.0))
                    drift = float(actual_weight - target_weight)
                    if abs(drift) <= float(self.rebalance_threshold):
                        continue

                    current_price = float(client.get_current_price(symbol)[0])
                    desired_notional = float(allocation_amount) * target_weight
                    current_notional = float(current_exposures.get(symbol, 0.0))
                    notional_diff = float(desired_notional - current_notional)
                    if abs(notional_diff) < float(current_price) * 0.0001:
                        continue

                    action = "BUY" if notional_diff > 0 else "SELL"
                    entry_price = float(current_price)
                    try:
                        candles = client.get_candles(symbol, mt5.TIMEFRAME_H1, 20)
                        atr = calculate_atr(candles)
                        if atr is None or atr <= 0.0:
                            raise ValueError("ATR calculation returned invalid value")
                    except Exception:
                        logger.warning("ATR calculation failed for %s, using fallback volatility", symbol)
                        atr = float(entry_price) * 0.01

                    if action == "BUY":
                        stop_loss = float(entry_price - atr * 1.5)
                        take_profit = float(entry_price + atr * 2.0)
                    else:
                        stop_loss = float(entry_price + atr * 1.5)
                        take_profit = float(entry_price - atr * 2.0)

                    stop_loss_pips = float(abs(entry_price - stop_loss) * (100.0 if symbol.endswith("JPY") else 10000.0))
                    volume = float(risk_manager.calculate_position_size(symbol, stop_loss_pips, 0.05))
                    if volume <= 0.0:
                        continue

                    signals.append(
                        TradeSignal(
                            symbol=symbol,
                            action=action,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            volume=volume,
                            strategy_name="BarbellRebalance",
                            confidence=0.75,
                            timestamp=datetime.utcnow(),
                        )
                    )
                except Exception:
                    logger.exception("Failed to generate rebalance signal for %s", symbol)
                    continue
        return signals
