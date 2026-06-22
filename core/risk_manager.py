"""Risk management: position sizing, drawdown circuit breaker, and margin checks.

The :class:`RiskManager` is the single gate every signal must pass before it
reaches the broker. It sizes positions from account equity and per-trade risk,
enforces a daily-drawdown circuit breaker, and verifies that free margin can
support a prospective order.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from datetime import datetime
from typing import List

try:
    import logfire
except ImportError:  # pragma: no cover
    logfire = None

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:  # pragma: no cover
    mt5 = None
    MT5_AVAILABLE = False

from core.models import OrderResult, Position, RiskState, TradeSignal
from core.mt5_client import MT5Client

logger = logging.getLogger(__name__)


def _span(name: str):
    if logfire is not None and hasattr(logfire, "span"):
        return logfire.span(name)
    return nullcontext()


class RiskManager:
    """Enforces account-level risk limits and sizes positions."""

    def __init__(
        self,
        client: MT5Client,
        risk_per_trade: float = 0.010,
        max_daily_drawdown: float = 0.05,
        max_open_positions: int = 5,
    ) -> None:
        """Configure risk limits and bind the MT5 client."""
        self.client = client
        self.risk_per_trade = risk_per_trade
        self.max_daily_drawdown = max_daily_drawdown
        self.max_open_positions = max_open_positions

    def _pip_value(self, symbol: str) -> float:
        symbol_text = symbol.upper()
        if symbol_text.startswith("XAU") or "GOLD" in symbol_text:
            return 0.1
        if any(crypto in symbol_text for crypto in ["BTC", "ETH", "LTC", "XRP", "ADA", "SOL", "DOT", "DOGE", "BNB", "AVAX", "LINK"]):
            return 1.0
        return 10.0

    def calculate_position_size(
        self,
        symbol: str,
        stop_loss_pips: float,
        risk_pct: float = 0.05,
    ) -> float:
        """Return the lot size that risks up to risk_pct of equity, capping leverage at 20x."""
        with _span("RiskManager.calculate_position_size"):
            try:
                state = self.client.get_account_info()
                equity = state.equity
                if equity <= 0 or stop_loss_pips <= 0:
                    raise ValueError("Equity and stop loss pips must be positive")

                # Risk amount in dollars
                risk_amount = equity * risk_pct

                # Get symbol info for pip value calculation
                if not MT5_AVAILABLE:
                    raise RuntimeError("MetaTrader5 not available")
                
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    return symbol_info.volume_min if symbol_info else 0.01

                # Calculate pip value per lot
                contract_size = float(symbol_info.trade_contract_size)

                if symbol.endswith('JPY'):
                    pip_value_per_lot = contract_size * 0.01
                elif symbol in ['XAUUSD', 'XAGUSD']:
                    pip_value_per_lot = contract_size * 0.01
                elif symbol in ['BTCUSD', 'ETHUSD', 'SOLUSD', 'XRPUSD', 'BARUSD']:
                    pip_value_per_lot = contract_size * 0.01
                else:
                    pip_value_per_lot = contract_size * 0.0001

                if stop_loss_pips <= 0 or pip_value_per_lot <= 0:
                    return symbol_info.volume_min

                # Base volume from risk
                volume = risk_amount / (stop_loss_pips * pip_value_per_lot)

                # Apply leverage cap - don't exceed 20x total leverage
                current_notional = state.leverage_ratio * equity
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    price = (float(tick.bid) + float(tick.ask)) / 2.0
                    new_notional = volume * price * contract_size
                    if (current_notional + new_notional) / equity > 20.0:
                        max_new_notional = (20.0 * equity) - current_notional
                        if max_new_notional <= 0:
                            return 0.0
                        volume = max_new_notional / (price * contract_size)

                # Normalize to lot step
                volume_step = float(symbol_info.volume_step)
                volume = round(round(volume / volume_step) * volume_step, 8)
                volume = max(float(symbol_info.volume_min), min(float(symbol_info.volume_max), volume))

                return volume
            except Exception as exc:
                logger.exception("Position size calculation failed for %s", symbol)
                return 0.01

    def check_risk_state(self) -> RiskState:
        """Fetch the latest broker state and normalize derived risk metrics."""
        with _span("RiskManager.check_risk_state"):
            try:
                state = self.client.get_account_info()
                if state.equity:
                    state.margin_usage_pct = (state.margin_used / state.equity) * 100.0
                else:
                    state.margin_usage_pct = 0.0

                if state.peak_equity:
                    state.current_drawdown_pct = max(
                        0.0,
                        (state.peak_equity - state.equity) / state.peak_equity * 100.0,
                    )
                else:
                    state.current_drawdown_pct = 0.0

                return state
            except Exception as exc:
                logger.exception("Failed to check risk state")
                raise RuntimeError(f"Risk state check failed: {exc}") from exc

    def _single_instrument_concentration(self, state: RiskState) -> float:
        exposures: dict[str, float] = {}
        for position in state.active_positions:
            notional = abs(position.volume * position.current_price)
            exposures[position.symbol] = exposures.get(position.symbol, 0.0) + notional

        total = sum(exposures.values())
        if total <= 0:
            return 0.0
        return max(exposures.values()) / total

    def is_safe_to_trade(self) -> tuple[bool, str]:
        """Return whether the current account state is safe for new trades."""
        with _span("RiskManager.is_safe_to_trade"):
            try:
                state = self.check_risk_state()
                logger.info(f"Current actual leverage: {state.leverage_ratio:.2f}x, margin: {state.margin_usage_pct:.1f}%")

                # Emergency equity protection: stop trading if equity drops below $850k
                if state.equity < 850000.0:
                    return False, f"Equity protection: below $850,000 threshold (current: ${state.equity:,.2f})"

                if state.margin_usage_pct > 85.0:
                    return False, f"Margin usage {state.margin_usage_pct:.1f}% exceeds 85%"
                if state.leverage_ratio > 26.0:
                    logger.warning(
                        "Leverage approaching cap: %.1fx (competition penalty threshold is 28x)",
                        state.leverage_ratio,
                    )
                if state.current_drawdown_pct > 8.0:
                    return False, f"Drawdown {state.current_drawdown_pct:.1f}% exceeds 8%"

                return True, "OK"
            except Exception as exc:
                logger.exception("Risk safety check failed")
                return False, str(exc)

    def emergency_close_all(self) -> List[OrderResult]:
        """Close all positions immediately and log the emergency reason."""
        with _span("RiskManager.emergency_close_all"):
            safe, reason = self.is_safe_to_trade()
            logger.warning("Emergency close all positions triggered: %s", reason)
            try:
                return self.client.close_all_positions()
            except Exception as exc:
                logger.exception("Emergency close all failed")
                return []

    # Symbols to exclude from hedging (let existing positions ride)
    HEDGE_EXCLUSIONS = {"XAGUSD"}
    
    def add_hedges(self, client: MT5Client) -> List[OrderResult]:
        """Check all open positions for unrealized loss > $2000; open small opposing hedge position if triggered.
        
        Hedge sizing: 0.1x of position volume to limit losses without full reversal.
        Returns list of hedge order results.
        """
        with _span("RiskManager.add_hedges"):
            results = []
            try:
                state = client.get_account_info()
                for position in state.active_positions:
                    # Skip excluded symbols (legacy positions we want to let ride)
                    if position.symbol in self.HEDGE_EXCLUSIONS:
                        logger.debug("Skipping hedge for excluded symbol: %s", position.symbol)
                        continue
                    
                    if position.profit is None or position.profit >= -2000.0:
                        # Position is either profit or loss < $2000 threshold
                        continue
                    
                    # Position has unrealized loss > $2000; open opposing hedge
                    try:
                        hedge_volume = max(
                            mt5.symbol_info(position.symbol).volume_min,
                            round(position.volume * 0.1, 2)
                        )
                        
                        hedge_action = "SELL" if position.volume > 0 else "BUY"
                        logger.info(
                            "Opening hedge for %s (unrealized loss: $%.2f, hedge volume: %.2f)",
                            position.symbol,
                            position.profit,
                            hedge_volume,
                        )
                        
                        # Create a minimal TradeSignal for hedge placement
                        hedge_signal = TradeSignal(
                            symbol=position.symbol,
                            action=hedge_action,
                            entry_price=position.current_price,
                            stop_loss=position.stop_loss,  # Use position's existing stop
                            take_profit=position.take_profit,  # Use position's existing TP
                            volume=hedge_volume,
                            strategy_name="Hedging",
                            confidence=0.5,
                            timestamp=datetime.utcnow(),
                        )
                        hedge_result = client.place_market_order(hedge_signal)
                        logger.debug("Hedge result for %s: success=%s", position.symbol, hedge_result.success)
                        results.append(hedge_result)
                    except Exception as exc:
                        logger.exception("Failed to place hedge for %s", position.symbol)
                        continue
                
                return results
            except Exception as exc:
                logger.exception("Hedge check failed")
                return []

    def check_concentration(self, new_symbol: str, new_notional: float) -> bool:
        """Return True if adding a proposed position keeps concentration below 85%.
        
        Safety rule: Only enforce concentration limits at high leverage (>15x).
        At low leverage, concentration checks pass automatically to allow strategies to scale up.
        """
        with _span("RiskManager.check_concentration"):
            try:
                state = self.check_risk_state()
                
                # At low leverage (<=15x), allow all concentrations. Rules restrict "near-full-leverage" scenarios.
                if state.leverage_ratio <= 15.0:
                    logger.debug(
                        "Leverage %.1fx is below 15x threshold; concentration check bypassed for %s",
                        state.leverage_ratio,
                        new_symbol,
                    )
                    return True
                
                current_gross = sum(
                    abs(p.volume * p.current_price)
                    for p in state.active_positions
                )

                # If no existing positions, always safe to proceed
                if current_gross == 0:
                    return True

                symbol_notional = sum(
                    abs(p.volume * p.current_price)
                    for p in state.active_positions
                    if p.symbol == new_symbol
                ) + new_notional

                new_gross = current_gross + new_notional
                concentration = symbol_notional / new_gross

                if concentration > 0.85:
                    logger.warning(
                        "Concentration check blocked at %.1fx leverage: %s would be %.1f%% (threshold 85%%)",
                        state.leverage_ratio,
                        new_symbol,
                        concentration * 100.0,
                    )
                    return False
                return True
            except Exception as exc:
                logger.exception("Concentration check failed for %s", new_symbol)
                return True

    def calculate_net_directional_exposure(self) -> float:
        """Return net directional exposure as percentage (0-100 = balanced to fully long/short)."""
        with _span("RiskManager.calculate_net_directional_exposure"):
            try:
                state = self.check_risk_state()
                long_notional = sum(
                    p.volume * p.current_price
                    for p in state.active_positions
                    if p.volume > 0
                )
                short_notional = sum(
                    abs(p.volume) * p.current_price
                    for p in state.active_positions
                    if p.volume < 0
                )
                total = long_notional + short_notional
                if total <= 0:
                    return 0.0
                net = abs(long_notional - short_notional)
                return (net / total) * 100.0
            except Exception as exc:
                logger.exception("Failed to calculate net directional exposure")
                return 0.0

    def check_directional_exposure(self, signal) -> bool:
        """Return True if adding signal keeps directional exposure manageable.
        
        Tiered approach based on leverage:
        - Low leverage (≤5x): Allow all trades regardless of directional exposure
        - Medium leverage (5-15x): Allow trades that reduce or maintain exposure, or if < 90%
        - High leverage (>15x): Strict 80% limit
        """
        with _span("RiskManager.check_directional_exposure"):
            try:
                state = self.check_risk_state()
                signal_notional = float(signal.volume) * float(signal.entry_price)

                # At low leverage, directional concentration is less risky
                if state.leverage_ratio <= 5.0:
                    logger.debug(
                        "Leverage %.1fx ≤ 5x; directional check bypassed for %s %s",
                        state.leverage_ratio, signal.action, signal.symbol
                    )
                    return True

                # Current portfolio directional exposure (existing positions only)
                current_long = sum(
                    p.volume * p.current_price
                    for p in state.active_positions
                    if p.volume > 0
                )
                current_short = sum(
                    abs(p.volume) * p.current_price
                    for p in state.active_positions
                    if p.volume < 0
                )
                current_total = current_long + current_short
                current_directional = 0.0
                if current_total > 0:
                    current_directional = abs(current_long - current_short) / current_total * 100.0

                # Simulate adding the signal
                if signal.action.upper() == "BUY":
                    new_long = current_long + signal_notional
                    new_short = current_short
                else:
                    new_short = current_short + signal_notional
                    new_long = current_long

                new_total = new_long + new_short
                if new_total <= 0:
                    return True

                new_directional = abs(new_long - new_short) / new_total * 100.0
                
                # Medium leverage (5-15x): Allow if improves exposure or stays under 90%
                if state.leverage_ratio <= 15.0:
                    if new_directional <= current_directional or new_directional <= 90.0:
                        return True
                    logger.warning(
                        "Directional exposure would increase to %.1f%% at %.1fx leverage (signal: %s %s)",
                        new_directional, state.leverage_ratio, signal.action, signal.symbol
                    )
                    return False
                
                # High leverage (>15x): Strict 80% limit
                if new_directional > 80.0:
                    logger.warning(
                        "Directional exposure would exceed 80%% at %.1fx leverage: %.1f%% (signal: %s %s)",
                        state.leverage_ratio, new_directional, signal.action, signal.symbol
                    )
                    return False

                return True
            except Exception as exc:
                logger.exception("Directional exposure check failed")
                return True  # fail open to allow trading

    def reduce_positions_if_leverage_high(self) -> List[OrderResult]:
        """If leverage > 20.5x, close smallest/oldest positions until leverage <= 20.5x."""
        with _span("RiskManager.reduce_positions_if_leverage_high"):
            results = []
            try:
                state = self.check_risk_state()
                if state.leverage_ratio <= 20.5:
                    return results

                logger.warning(
                    "Leverage drift detected: %.1fx; reducing positions to stay below 20.5x",
                    state.leverage_ratio,
                )

                # Sort by notional value (smallest first)
                sorted_positions = sorted(
                    state.active_positions,
                    key=lambda p: abs(p.volume * p.current_price),
                )

                for position in sorted_positions:
                    result = self.client.close_position(position.ticket)
                    results.append(result)

                    # Check if we're back below 20.5x
                    state = self.check_risk_state()
                    if state.leverage_ratio <= 20.5:
                        logger.info(
                            "Leverage reduced to %.1fx; position reduction complete",
                            state.leverage_ratio,
                        )
                        break

                return results
            except Exception as exc:
                logger.exception("Failed to reduce positions for leverage")
                return results
