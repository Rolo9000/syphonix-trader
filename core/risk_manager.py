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
        risk_per_trade: float = 0.02,
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
        risk_pct: float = 0.01,
    ) -> float:
        """Return the lot size that risks no more than ``risk_pct`` of equity."""
        with _span("RiskManager.calculate_position_size"):
            try:
                state = self.client.get_account_info()
                equity = state.equity
                if equity <= 0 or stop_loss_pips <= 0:
                    raise ValueError("Equity and stop loss pips must be positive")

                risk_amount = equity * risk_pct
                pip_value = self._pip_value(symbol)
                volume = risk_amount / (stop_loss_pips * pip_value)
                return max(volume, 0.0)
            except Exception as exc:
                logger.exception("Failed to calculate position size for %s", symbol)
                return 0.0

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
                if len(state.active_positions) > 0 and state.leverage_ratio > 29.5:
                    return False, f"Leverage {state.leverage_ratio:.1f}x exceeds 29.5x"
                if state.current_drawdown_pct > 15.0:
                    return False, f"Drawdown {state.current_drawdown_pct:.1f}% exceeds 15%"
                if self._single_instrument_concentration(state) > 0.85:
                    return False, "Single instrument concentration exceeds 85%"

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
