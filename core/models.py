"""Pydantic data models shared across the trading system.

These are the canonical wire/state types passed between strategies, the risk
manager, the MT5 client, and the state store. Validation and serialization
behaviour lives here so the rest of the system can assume well-formed data.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TradeSignal(BaseModel):
    symbol: str
    action: Literal['BUY', 'SELL', 'HOLD']
    entry_price: float
    stop_loss: float
    take_profit: float
    volume: float = Field(gt=0)
    strategy_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime

    model_config = {
        "frozen": False,
    }


class Position(BaseModel):
    ticket: int
    symbol: str
    order_type: str
    volume: float = Field(gt=0)
    open_price: float
    current_price: float
    profit: float
    stop_loss: float
    take_profit: float

    model_config = {
        "frozen": False,
    }


class RiskState(BaseModel):
    equity: float
    balance: float
    margin_used: float
    margin_free: float
    margin_usage_pct: float = Field(ge=0.0, le=100.0)
    current_drawdown_pct: float
    peak_equity: float
    leverage_ratio: float = Field(le=30.0)
    active_positions: list[Position]

    model_config = {
        "frozen": False,
    }


class OrderResult(BaseModel):
    success: bool
    ticket: int | None = None
    error_code: int | None = None
    error_msg: str | None = None
    symbol: str
    volume: float
    price: float

    model_config = {
        "frozen": False,
    }


class SentimentResult(BaseModel):
    symbol: str
    sentiment: Literal['BULLISH', 'BEARISH', 'NEUTRAL']
    confidence: float
    reasoning: str
    timestamp: datetime

    model_config = {
        "frozen": False,
    }


class MarketRegime(BaseModel):
    regime: Literal['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'HIGH_VOL']
    volatility_percentile: float
    trend_strength: float

    model_config = {
        "frozen": False,
    }
