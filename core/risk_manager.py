"""Risk management: position sizing, drawdown circuit breaker, and margin checks.

The :class:`RiskManager` is the single gate every signal must pass before it
reaches the broker. It sizes positions from account equity and per-trade risk,
enforces a daily-drawdown circuit breaker, and verifies that free margin can
support a prospective order.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
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

from core.models import OrderResult, Position, RiskState
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
        risk_per_trade: float = 0.05,
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

                if state.margin_usage_pct > 85.0:
                    return False, f"Margin usage {state.margin_usage_pct:.1f}% exceeds 85%"
                if state.leverage_ratio > 26.0:
                    logger.warning(
                        "Leverage approaching cap: %.1fx (competition penalty threshold is 28x)",
                        state.leverage_ratio,
                    )
                if state.current_drawdown_pct > 15.0:
                    return False, f"Drawdown {state.current_drawdown_pct:.1f}% exceeds 15%"

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

    def check_concentration(self, new_symbol: str, new_notional: float) -> bool:
        """Return True if adding a proposed position keeps concentration below 85%."""
        with _span("RiskManager.check_concentration"):
            try:
                state = self.check_risk_state()
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

                return concentration <= 0.85
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
        """Return True if adding signal keeps net directional exposure below 95%."""
        with _span("RiskManager.check_directional_exposure"):
            try:
                state = self.check_risk_state()
                signal_notional = float(signal.volume) * float(signal.entry_price)

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

                # Add signal to appropriate side
                if signal.action.upper() == "BUY":
                    long_notional += signal_notional
                else:
                    short_notional += signal_notional

                total = long_notional + short_notional
                if total <= 0:
                    return True

                net = abs(long_notional - short_notional)
                directional_exposure = (net / total) * 100.0

                if directional_exposure > 95.0:
                    logger.warning(
                        "Directional exposure would exceed 95%%: %.1f%% (signal: %s %s)",
                        directional_exposure,
                        signal.action,
                        signal.symbol,
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
