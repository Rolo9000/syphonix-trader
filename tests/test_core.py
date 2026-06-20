import asyncio
from datetime import datetime, timedelta, time

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

import fakeredis

from pydantic import ValidationError

from core.indicators import (
    calculate_atr,
    calculate_asian_range,
    detect_market_structure_shift,
    calculate_portfolio_weights,
)
from core.models import (
    TradeSignal,
    Position,
    RiskState,
    OrderResult,
    SentimentResult,
    MarketRegime,
)
from core.risk_manager import RiskManager
from core.mt5_client import MT5Client
from infra.state_store import StateStore
from main import execute_trading_cycle


@pytest.fixture
def fakeredis_client():
    return fakeredis.FakeRedis()


@pytest.fixture
def state_store(fakeredis_client):
    return StateStore("redis://localhost:6379/0", redis_client=fakeredis_client)


@pytest.fixture
def mt5_client_mock():
    m = MagicMock()
    # default account info
    m.get_account_info.return_value = RiskState(
        equity=1_000_000.0,
        balance=1_000_000.0,
        margin_used=0.0,
        margin_free=1_000_000.0,
        margin_usage_pct=0.0,
        current_drawdown_pct=0.0,
        peak_equity=1_000_000.0,
        leverage_ratio=5.0,
        active_positions=[],
    )
    m.get_open_positions.return_value = []
    return m


def test_pydantic_models_valid_and_invalid():
    # valid instantiation
    ts = TradeSignal(
        symbol="EURUSD",
        action="BUY",
        entry_price=1.1,
        stop_loss=1.09,
        take_profit=1.12,
        volume=1.0,
        strategy_name="test",
        confidence=0.5,
        timestamp=datetime.utcnow(),
    )
    assert ts.symbol == "EURUSD"

    pos = Position(
        ticket=1,
        symbol="EURUSD",
        order_type="BUY",
        volume=0.1,
        open_price=1.1,
        current_price=1.11,
        profit=100.0,
        stop_loss=1.09,
        take_profit=1.2,
    )
    assert pos.ticket == 1

    rs = RiskState(
        equity=1000000.0,
        balance=1000000.0,
        margin_used=0.0,
        margin_free=1000000.0,
        margin_usage_pct=0.0,
        current_drawdown_pct=0.0,
        peak_equity=1000000.0,
        leverage_ratio=10.0,
        active_positions=[pos],
    )
    assert rs.equity == 1000000.0

    orr = OrderResult(success=True, ticket=1, error_code=None, error_msg=None, symbol="EURUSD", volume=0.1, price=1.11)
    assert orr.success

    sres = SentimentResult(symbol="EURUSD", sentiment="NEUTRAL", confidence=0.5, reasoning="ok", timestamp=datetime.utcnow())
    assert sres.sentiment == "NEUTRAL"

    mr = MarketRegime(regime="RANGING", volatility_percentile=0.5, trend_strength=0.1)
    assert mr.regime == "RANGING"

    # invalid: volume=0 should raise
    with pytest.raises(ValidationError):
        TradeSignal(
            symbol="EURUSD",
            action="BUY",
            entry_price=1.1,
            stop_loss=1.09,
            take_profit=1.12,
            volume=0.0,
            strategy_name="test",
            confidence=0.5,
            timestamp=datetime.utcnow(),
        )

    # invalid leverage > 30
    with pytest.raises(ValidationError):
        RiskState(
            equity=1000.0,
            balance=1000.0,
            margin_used=0.0,
            margin_free=1000.0,
            margin_usage_pct=0.0,
            current_drawdown_pct=0.0,
            peak_equity=1000.0,
            leverage_ratio=31.0,
            active_positions=[],
        )


def test_risk_manager_position_size(mt5_client_mock):
    rm = RiskManager(client=mt5_client_mock, risk_per_trade=0.01)
    # equity 1_000_000 from fixture, 1% risk = 10_000, pip value for FX=10, stop 20 pips
    volume = rm.calculate_position_size("EURUSD", stop_loss_pips=20.0, risk_pct=0.01)
    assert pytest.approx(volume, rel=1e-6) == 50.0


def test_asian_range():
    # Build candles for an Asian session: use a fixed "now" and construct minutes covering 21:00-06:00 UTC
    from core.indicators import calculate_asian_range as _calc_range

    class DummyClient:
        def get_candles(self, symbol, timeframe, count):
            # create minutes for previous day Asian session (21:00-06:00 UTC)
            base = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=1)
            times = [base + pd.Timedelta(hours=21) + pd.Timedelta(minutes=i) for i in range(540)]
            highs = [1.5 + (i % 5) * 0.001 for i in range(len(times))]
            lows = [1.4 - (i % 5) * 0.001 for i in range(len(times))]
            closes = [(h + l) / 2 for h, l in zip(highs, lows)]
            df = pd.DataFrame({"time": times, "high": highs, "low": lows, "close": closes})
            return df

    client = DummyClient()
    result = _calc_range(client, "EURUSD")
    assert result is not None
    low, high = result
    assert high >= low


def test_atr():
    # Create simple OHLC where high-low always 2, and prev_close shifts
    import numpy as np

    data = {
        "high": [2.0, 3.0, 4.0, 5.0, 6.0],
        "low": [1.0, 2.0, 3.0, 4.0, 5.0],
        "close": [1.5, 2.5, 3.5, 4.5, 5.5],
    }
    df = pd.DataFrame(data)
    # For this simple series, TR each bar = high-low =1.0? Actually high-low=1.0 here
    atr_val = calculate_atr(df, period=3)
    assert atr_val > 0  # verify ATR returns a positive float


def test_state_store_and_emergency(fakeredis_client):
    store = StateStore("redis://localhost:6379/0", redis_client=fakeredis_client)
    ts = TradeSignal(
        symbol="EURUSD",
        action="BUY",
        entry_price=1.1,
        stop_loss=1.09,
        take_profit=1.12,
        volume=1.0,
        strategy_name="test",
        confidence=0.5,
        timestamp=datetime.utcnow(),
    )
    store.save_signal(ts)
    got = store.get_signal("EURUSD", "test")
    assert got is not None and got.symbol == "EURUSD"

    rs = RiskState(
        equity=1000.0,
        balance=1000.0,
        margin_used=0.0,
        margin_free=1000.0,
        margin_usage_pct=0.0,
        current_drawdown_pct=0.0,
        peak_equity=1000.0,
        leverage_ratio=1.0,
        active_positions=[],
    )
    store.save_risk_state(rs)
    got_rs = store.get_risk_state()
    assert got_rs is not None and got_rs.equity == 1000.0

    # emergency stop
    store.set_emergency_stop(True)
    assert store.is_emergency_stop() is True
    store.set_emergency_stop(False)
    assert store.is_emergency_stop() is False


def test_execute_cycle_skips_on_emergency(monkeypatch):
    # build minimal collaborators
    client = MagicMock()
    risk_manager = MagicMock()
    state_store = MagicMock()
    asian = MagicMock()
    barbell = MagicMock()

    state_store.is_emergency_stop.return_value = True

    # run the coroutine
    asyncio.run(execute_trading_cycle(client, risk_manager, state_store, asian, barbell))

    # ensure no signals executed
    client.place_market_order.assert_not_called()
