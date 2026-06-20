"""MetaTrader 5 connection management and order execution.

Wraps the ``MetaTrader5`` terminal API behind a typed, testable surface. The
client owns the terminal session lifecycle (initialize / login / shutdown),
exposes market data reads, and translates :class:`~core.models.TradeSignal`
intents into broker orders, returning :class:`~core.models.OrderResult`.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional, Tuple

import MetaTrader5 as mt5
import pandas as pd

from core.models import OrderResult, Position, RiskState, TradeSignal

logger = logging.getLogger(__name__)


class MT5Client:
    """Typed wrapper around the MetaTrader 5 terminal API."""

    def __init__(
        self,
        login: int,
        password: str,
        server: str,
        path: str | None = None,
    ) -> None:
        """Store credentials; the terminal session is opened in :meth:`connect`."""
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self._connected = False

    def __enter__(self) -> "MT5Client":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool:
        self.disconnect()
        return False

    def connect(self) -> bool:
        """Initialize the terminal and authenticate. Returns ``True`` on success."""
        try:
            initialized = mt5.initialize(self.path) if self.path else mt5.initialize()
            if not initialized:
                error = mt5.last_error()
                raise RuntimeError(f"MT5 initialization failed: {error}")

            login_result = mt5.login(self.login, password=self.password, server=self.server)
            if not login_result:
                error = mt5.last_error()
                raise RuntimeError(f"MT5 login failed: {error}")

            self._connected = True
            logger.info("MT5 connected: login=%s server=%s", self.login, self.server)
            return True
        except Exception as exc:
            self.disconnect()
            logger.exception("MT5 connection error")
            raise RuntimeError(f"MT5 connection error: {exc}") from exc

    def disconnect(self) -> None:
        """Shut down the terminal session."""
        try:
            if mt5.shutdown():
                logger.info("MT5 shutdown complete")
        except Exception:
            logger.exception("MT5 disconnect failed")
        finally:
            self._connected = False

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("MT5Client is not connected")

    def get_account_info(self) -> RiskState:
        """Return account and risk snapshot from the broker."""
        try:
            self._ensure_connected()
            account_info = mt5.account_info()
            if account_info is None:
                raise RuntimeError("Failed to retrieve MT5 account info")

            equity = float(account_info.equity)
            balance = float(account_info.balance)
            margin_used = float(getattr(account_info, "margin", 0.0))
            margin_free = float(getattr(account_info, "margin_free", 0.0))
            leverage_ratio = float(getattr(account_info, "leverage", 0.0))
            peak_equity = float(getattr(account_info, "equity", equity))

            margin_usage_pct = (margin_used / equity * 100.0) if equity else 0.0
            current_drawdown_pct = float(getattr(account_info, "drawdown", 0.0))

            return RiskState(
                equity=equity,
                balance=balance,
                margin_used=margin_used,
                margin_free=margin_free,
                margin_usage_pct=margin_usage_pct,
                current_drawdown_pct=current_drawdown_pct,
                peak_equity=peak_equity,
                leverage_ratio=leverage_ratio,
                active_positions=self.get_open_positions(),
            )
        except Exception as exc:
            logger.exception("Failed to fetch MT5 account info")
            raise RuntimeError(f"Failed to fetch account info: {exc}") from exc

    def get_current_price(self, symbol: str) -> Tuple[float, float]:
        """Return the latest bid and ask for the requested symbol."""
        try:
            self._ensure_connected()
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                raise RuntimeError(f"No tick data available for symbol: {symbol}")
            return float(tick.bid), float(tick.ask)
        except Exception as exc:
            logger.exception("Failed to fetch current price for %s", symbol)
            raise RuntimeError(f"Failed to fetch current price for {symbol}: {exc}") from exc

    def get_candles(self, symbol: str, timeframe: int, count: int) -> pd.DataFrame:
        """Return recent OHLC bars with time, open, high, low, close, and volume."""
        try:
            self._ensure_connected()
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                raise RuntimeError(f"No candle data available for {symbol}")

            df = pd.DataFrame(rates)
            if "time" in df:
                df["time"] = pd.to_datetime(df["time"], unit="s")
            if "tick_volume" in df:
                df = df.rename(columns={"tick_volume": "volume"})
            elif "real_volume" in df:
                df = df.rename(columns={"real_volume": "volume"})

            return df[["time", "open", "high", "low", "close", "volume"]]
        except Exception as exc:
            logger.exception("Failed to fetch candles for %s", symbol)
            empty = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
            return empty

    def place_market_order(self, signal: TradeSignal) -> OrderResult:
        """Submit a market order for the provided trade signal."""
        try:
            self._ensure_connected()
            bid, ask = self.get_current_price(signal.symbol)
            order_type = mt5.ORDER_TYPE_BUY if signal.action == "BUY" else mt5.ORDER_TYPE_SELL
            price = ask if order_type == mt5.ORDER_TYPE_BUY else bid

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": signal.symbol,
                "volume": float(signal.volume),
                "type": order_type,
                "price": price,
                "sl": float(signal.stop_loss),
                "tp": float(signal.take_profit),
                "deviation": 20,
                "type_filling": mt5.ORDER_FILLING_FOK,
                "type_time": mt5.ORDER_TIME_GTC,
            }

            result = mt5.order_send(request)
            if result is None:
                raise RuntimeError("MT5 order_send returned no result")

            success = getattr(result, "retcode", 0) == mt5.TRADE_RETCODE_DONE
            return OrderResult(
                success=success,
                ticket=int(getattr(result, "order", 0) or getattr(result, "deal", 0) or 0) or None,
                error_code=int(getattr(result, "retcode", 0)) if getattr(result, "retcode", None) is not None else None,
                error_msg=str(getattr(result, "comment", "")),
                symbol=signal.symbol,
                volume=float(getattr(result, "volume", signal.volume)),
                price=float(getattr(result, "price", price)),
            )
        except Exception as exc:
            logger.exception("Market order failed for %s", signal.symbol)
            return OrderResult(
                success=False,
                ticket=None,
                error_code=None,
                error_msg=str(exc),
                symbol=signal.symbol,
                volume=float(signal.volume),
                price=0.0,
            )

    def close_position(self, ticket: int) -> OrderResult:
        """Close the position identified by the given ticket."""
        try:
            self._ensure_connected()
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                raise RuntimeError(f"Position {ticket} not found")

            position = positions[0]
            symbol = position.symbol
            volume = float(position.volume)
            order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            bid, ask = self.get_current_price(symbol)
            price = bid if order_type == mt5.ORDER_TYPE_SELL else ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "position": int(ticket),
                "price": price,
                "deviation": 20,
                "type_filling": mt5.ORDER_FILLING_FOK,
                "type_time": mt5.ORDER_TIME_GTC,
            }

            result = mt5.order_send(request)
            if result is None:
                raise RuntimeError("MT5 close order returned no result")

            success = getattr(result, "retcode", 0) == mt5.TRADE_RETCODE_DONE
            return OrderResult(
                success=success,
                ticket=int(getattr(result, "order", 0) or getattr(result, "deal", 0) or 0) or None,
                error_code=int(getattr(result, "retcode", 0)) if getattr(result, "retcode", None) is not None else None,
                error_msg=str(getattr(result, "comment", "")),
                symbol=symbol,
                volume=volume,
                price=float(getattr(result, "price", price)),
            )
        except Exception as exc:
            logger.exception("Close position failed for ticket %s", ticket)
            return OrderResult(
                success=False,
                ticket=None,
                error_code=None,
                error_msg=str(exc),
                symbol="",
                volume=0.0,
                price=0.0,
            )

    def close_all_positions(self) -> List[OrderResult]:
        """Close all open positions and return the list of results."""
        results: List[OrderResult] = []
        for position in self.get_open_positions():
            results.append(self.close_position(position.ticket))
        return results

    def get_open_positions(self) -> List[Position]:
        """Return active open positions from the broker."""
        try:
            self._ensure_connected()
            raw_positions = mt5.positions_get()
            if raw_positions is None:
                return []

            positions: List[Position] = []
            for raw in raw_positions:
                positions.append(
                    Position(
                        ticket=int(raw.ticket),
                        symbol=str(raw.symbol),
                        order_type="BUY" if raw.type == mt5.POSITION_TYPE_BUY else "SELL",
                        volume=float(raw.volume),
                        open_price=float(raw.price_open),
                        current_price=float(getattr(raw, "price_current", raw.price_open)),
                        profit=float(getattr(raw, "profit", 0.0)),
                        stop_loss=float(getattr(raw, "sl", 0.0)),
                        take_profit=float(getattr(raw, "tp", 0.0)),
                    )
                )
            return positions
        except Exception as exc:
            logger.exception("Failed to load open positions")
            return []

    def get_price(self, symbol: str) -> float:
        """Backward-compatible wrapper for get_current_price."""
        bid, ask = self.get_current_price(symbol)
        return (bid + ask) / 2.0

    def get_ohlc(self, symbol: str, timeframe: int, count: int) -> list[dict[str, float]]:
        """Backward-compatible wrapper for get_candles."""
        df = self.get_candles(symbol, timeframe, count)
        return df.to_dict(orient="records")

    def get_positions(self, symbol: str | None = None) -> list[Position]:
        """Backward-compatible wrapper for get_open_positions."""
        positions = self.get_open_positions()
        if symbol is None:
            return positions
        return [position for position in positions if position.symbol == symbol]

    def execute(self, signal: TradeSignal) -> OrderResult:
        """Backward-compatible wrapper for place_market_order."""
        return self.place_market_order(signal)
