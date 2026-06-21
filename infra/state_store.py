"""Redis-backed persistence for trading signals, risk state, and sentiment."""

from __future__ import annotations

import json
import logging

import redis

from core.models import RiskState, SentimentResult, TradeSignal

logger = logging.getLogger(__name__)


class StateStore:
    """Redis-backed persistence for signals and risk state."""

    def __init__(self, redis_url: str, redis_client=None) -> None:
        self.redis_url = redis_url
        if redis_client is not None:
            self._redis = redis_client
        else:
            try:
                if redis_url.startswith("rediss://"):
                    self._redis = redis.Redis.from_url(
                        redis_url,
                        decode_responses=True,
                        ssl_cert_reqs=None,
                    )
                else:
                    self._redis = redis.Redis.from_url(
                        redis_url,
                        decode_responses=True,
                    )
            except Exception as exc:
                logger.warning("Redis initialization failed: %s", exc)
                self._redis = None

    def _safe_execute(self, fn, *args, default=None, **kwargs):
        if self._redis is None:
            logger.warning("Redis unavailable; skipping operation")
            return default
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.warning("Redis error: %s", exc)
            return default

    def save_signal(self, signal: TradeSignal) -> None:
        key = f"signal:{signal.symbol}:{signal.strategy_name}"
        payload = signal.model_dump_json()
        self._safe_execute(self._redis.set, key, payload)

    def get_signal(self, symbol: str, strategy: str) -> TradeSignal | None:
        key = f"signal:{symbol}:{strategy}"
        payload = self._safe_execute(self._redis.get, key)
        if not payload:
            return None
        try:
            return TradeSignal.model_validate_json(payload)
        except Exception as exc:
            logger.warning("Failed to parse stored signal %s: %s", key, exc)
            return None

    def save_risk_state(self, state: RiskState) -> None:
        payload = state.model_dump_json()
        # store with TTL via `ex` kwarg for redis-py compatibility
        self._safe_execute(self._redis.set, "risk:current", payload, ex=60)

    def get_risk_state(self) -> RiskState | None:
        payload = self._safe_execute(self._redis.get, "risk:current")
        if not payload:
            return None
        try:
            return RiskState.model_validate_json(payload)
        except Exception as exc:
            logger.warning("Failed to parse stored risk state: %s", exc)
            return None

    def save_sentiment(self, result: SentimentResult) -> None:
        key = f"sentiment:{result.symbol}"
        payload = result.model_dump_json()
        self._safe_execute(self._redis.set, key, payload, ex=3600)

    def get_sentiment(self, symbol: str) -> SentimentResult | None:
        key = f"sentiment:{symbol}"
        payload = self._safe_execute(self._redis.get, key)
        if not payload:
            return None
        try:
            return SentimentResult.model_validate_json(payload)
        except Exception as exc:
            logger.warning("Failed to parse stored sentiment %s: %s", key, exc)
            return None

    def set_emergency_stop(self, value: bool) -> None:
        if self._redis is None:
            logger.warning("Redis unavailable; skipping emergency stop update")
            return
        try:
            self._redis.set("system:emergency_stop", "1" if value else "0")
        except Exception as exc:
            logger.warning("Redis error: %s", exc)

    def is_emergency_stop(self) -> bool:
        if self._redis is None:
            logger.warning("Redis unavailable; assuming emergency stop is off")
            return False
        try:
            val = self._redis.get("system:emergency_stop")
            return val in ("1", b"1")
        except Exception as exc:
            logger.warning("Redis error: %s", exc)
            return False

    def save_peak_equity(self, equity: float) -> None:
        payload = json.dumps(float(equity))
        self._safe_execute(self._redis.set, "peak_equity", payload)

    def get_peak_equity(self) -> float:
        payload = self._safe_execute(self._redis.get, "peak_equity")
        if not payload:
            return 0.0
        try:
            return float(json.loads(payload))
        except Exception as exc:
            logger.warning("Failed to parse peak equity: %s", exc)
            return 0.0
