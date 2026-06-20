"""Core trading primitives: MT5 connectivity, risk control, indicators, and data models."""

from __future__ import annotations

from core.indicators import (
    calculate_asian_range,
    calculate_atr,
    calculate_portfolio_weights,
    detect_market_structure_shift,
    is_news_blackout,
)
from core.models import OrderResult, Position, RiskState, TradeSignal
from core.mt5_client import MT5Client
from core.risk_manager import RiskManager

__all__ = [
    "MT5Client",
    "RiskManager",
    "calculate_atr",
    "calculate_asian_range",
    "detect_market_structure_shift",
    "calculate_portfolio_weights",
    "is_news_blackout",
    "TradeSignal",
    "Position",
    "RiskState",
    "OrderResult",
]
