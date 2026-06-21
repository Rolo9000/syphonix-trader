"""Technical indicators and risk utilities.

Pure functions over OHLC bar sequences and market state. Most functions are
pure and deterministic, with external API access only in ``is_news_blackout``.
"""

from __future__ import annotations

import json
import logging
import math
from contextlib import nullcontext
from datetime import datetime, timedelta, time
from typing import Literal

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:  # pragma: no cover - platform dependent
    mt5 = None
    MT5_AVAILABLE = False
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from core.mt5_client import MT5Client

logger = logging.getLogger(__name__)

try:
    import logfire
except ImportError:  # pragma: no cover
    logfire = None


def _span(name: str):
    if logfire is not None and hasattr(logfire, "span"):
        return logfire.span(name)
    return nullcontext()


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Return the most recent ATR value from a high/low/close DataFrame."""
    with _span("calculate_atr"):
        if not all(col in df.columns for col in ("high", "low", "close")):
            raise ValueError("DataFrame must contain high, low, and close columns")

        if len(df) < period + 1:
            raise ValueError("Not enough data to calculate ATR")

        bars = df.copy()
        bars = bars.loc[:, ["high", "low", "close"]].astype(float)
        bars["prev_close"] = bars["close"].shift(1)
        bars["tr"] = bars["high"].combine(bars["prev_close"], max) - bars["low"].combine(bars["prev_close"], min)
        atr_series = bars["tr"].rolling(period).mean()
        atr_value = float(atr_series.iloc[-1])
        if math.isnan(atr_value):
            raise ValueError("ATR calculation failed; check the input series")
        return atr_value


def calculate_asian_range(candles_or_client, symbol: str | None = None) -> tuple[float, float] | None:
    """Return the low/high of the most recent Asian session in UTC.

    Accept either a DataFrame (returned by `client.get_candles`) or an MT5Client
    and symbol pair. If a client is provided, the function will call
    `client.get_candles(symbol, mt5.TIMEFRAME_M1, lookback_minutes)` internally.
    """
    with _span("calculate_asian_range"):
        # Resolve input to a DataFrame
        if hasattr(candles_or_client, "get_candles") and symbol is not None:
            lookback_minutes = 24 * 60
            try:
                df = candles_or_client.get_candles(symbol, mt5.TIMEFRAME_M1, lookback_minutes)
            except Exception as exc:
                logger.exception("Failed to fetch candles from client: %s", exc)
                return None
        elif isinstance(candles_or_client, pd.DataFrame):
            df = candles_or_client
        else:
            logger.warning("calculate_asian_range received unsupported inputs: %s %s", type(candles_or_client), symbol)
            return None

        # Log head for debugging
        try:
            logger.debug("calculate_asian_range received df with columns %s and head:\n%s", df.columns.tolist(), df.head().to_dict())
        except Exception:
            logger.debug("calculate_asian_range received df (unable to display head)")

        if df is None or df.empty or "time" not in df.columns:
            return None

        candles = df.copy()
        # Normalize times to UTC timestamps
        candles["time"] = pd.to_datetime(candles["time"], utc=True)

        # Filter for Asian session hours: 21:00 - 06:00 UTC
        session_df = candles[(candles["time"].dt.hour >= 21) | (candles["time"].dt.hour < 6)]
        if session_df.empty:
            return None

        return float(session_df["low"].min()), float(session_df["high"].max())


def detect_market_structure_shift(df: pd.DataFrame) -> Literal["BULLISH_MSS", "BEARISH_MSS", "NONE"]:
    """Detect simple bullish/bearish market structure shifts from the last 20 candles."""
    with _span("detect_market_structure_shift"):
        if len(df) < 20:
            return "NONE"

        window = df.tail(20).reset_index(drop=True)
        if not all(col in window.columns for col in ("open", "high", "low", "close")):
            return "NONE"

        highs = window["high"]
        lows = window["low"]
        closes = window["close"]
        opens = window["open"]

        swing_low = float(lows[:-3].min())
        swing_high = float(highs[:-3].max())

        broke_below = any(lows[: -1] < swing_low)
        broke_above = any(highs[: -1] > swing_high)

        last_open = float(opens.iloc[-1])
        last_close = float(closes.iloc[-1])
        last_high = float(highs.iloc[-1])
        last_low = float(lows.iloc[-1])
        last_body = abs(last_close - last_open)
        last_range = last_high - last_low
        strong_bullish = last_close > last_open and last_body >= 0.5 * last_range
        strong_bearish = last_close < last_open and last_body >= 0.5 * last_range

        if broke_below and last_close > swing_low and strong_bullish:
            return "BULLISH_MSS"
        if broke_above and last_close < swing_high and strong_bearish:
            return "BEARISH_MSS"
        return "NONE"


def calculate_portfolio_weights(returns_dict: dict[str, pd.Series]) -> dict[str, float]:
    """Optimize portfolio weights for maximum Sharpe ratio subject to box constraints."""
    with _span("calculate_portfolio_weights"):
        if not returns_dict:
            return {}

        returns = pd.DataFrame(returns_dict).dropna(how="all")
        if returns.empty:
            raise ValueError("Returns dictionary must contain at least one non-empty series")

        returns = returns.dropna(axis=1, how="all")
        symbols = list(returns.columns)
        mean_returns = returns.mean()
        cov = returns.cov()
        rf = 0.052 / 252

        def sharpe_penalty(weights: np.ndarray) -> float:
            portfolio_return = float(np.dot(weights, mean_returns - rf))
            portfolio_vol = float(np.sqrt(weights @ cov.values @ weights))
            if portfolio_vol <= 0:
                return 1e6
            return -portfolio_return / portfolio_vol

        n = len(symbols)
        bounds = [(0.1, 0.9)] * n
        constraints = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
        guess = np.full(n, 1.0 / n)

        result = minimize(sharpe_penalty, guess, bounds=bounds, constraints=constraints, method="SLSQP")
        if not result.success:
            logger.warning("Portfolio optimization failed: %s", result.message)
            result = minimize(sharpe_penalty, guess, bounds=bounds, constraints=constraints, method="SLSQP", options={"maxiter": 200})

        weights = result.x if result.success else guess
        weights = np.clip(weights, 0.1, 0.9)
        weights /= float(np.sum(weights))
        return {symbol: float(weight) for symbol, weight in zip(symbols, weights)}


_NEWS_BLACKOUT_CACHE: dict[str, object] = {}


def is_news_blackout(minutes_before: int = 30, minutes_after: int = 15) -> bool:
    """Return True if a high-impact economic event is within the blackout window."""
    with _span("is_news_blackout"):
        now = datetime.utcnow()
        cache_key = f"{minutes_before}:{minutes_after}"
        cached = _NEWS_BLACKOUT_CACHE.get(cache_key)
        if cached is not None:
            cached_time = cached["timestamp"]
            if now - cached_time < timedelta(minutes=10):
                return bool(cached["value"])

        urls = [
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.json",
        ]
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.forexfactory.com/",
        }
        blackout = False
        for url in urls:
            try:
                response = requests.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "Accept": "application/json, text/plain, */*",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Accept-Encoding": "gzip, deflate, br",
                        "Connection": "keep-alive",
                        "Referer": "https://www.forexfactory.com/",
                        "Sec-Fetch-Dest": "empty",
                        "Sec-Fetch-Mode": "cors",
                        "Sec-Fetch-Site": "cross-site",
                    },
                    timeout=5,
                    verify=False,
                )
                payload = response.json()
                events = payload.get("events") if isinstance(payload, dict) else payload
                if not events:
                    continue

                for event in events:
                    impact = str(event.get("impact", ""))
                    if impact != "High":
                        continue

                    event_time = event.get("utc") or event.get("date") or event.get("time")
                    if event_time is None:
                        continue

                    event_dt = pd.to_datetime(event_time, utc=True)
                    if event_dt.tzinfo is None:
                        event_dt = event_dt.tz_localize("UTC")

                    if now - timedelta(minutes=minutes_after) <= event_dt.to_pydatetime() <= now + timedelta(minutes=minutes_before):
                        blackout = True
                        break
                if blackout:
                    break
            except Exception:
                if logfire is not None and hasattr(logfire, "warn"):
                    logfire.warn(f"News blackout URL failed: {url}")
                else:
                    logger.warning("News blackout URL failed: %s", url)
                continue

        if not blackout:
            logger.debug("No high-impact events in blackout window; allowing trading")

        _NEWS_BLACKOUT_CACHE[cache_key] = {
            "timestamp": now,
            "value": blackout,
        }
        return blackout
