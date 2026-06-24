"""Entry point and APScheduler job definitions for the Syphonix trader.

Wires together the broker client, strategies, agents, risk manager, and state
store, then registers the scheduled jobs (signal generation, execution, risk
monitoring) on a background scheduler and runs until interrupted.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

from agents.sentiment_agent import SentimentAgent
from core import MT5Client
from core.indicators import is_news_blackout, calculate_atr
from core.risk_manager import RiskManager
from infra.state_store import StateStore
from strategies.asian_breakout import AsianBreakoutStrategy
from strategies.barbell import BarbellStrategy

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

logger = logging.getLogger(__name__)


def _span(name: str):
    try:
        import logfire

        if hasattr(logfire, "span"):
            return logfire.span(name)
    except ImportError:
        pass
    return asyncio.nullcontext()


async def trailing_stop_manager(client: MT5Client) -> None:
    """Trail stop losses on profitable positions to lock in gains.
    
    NUCLEAR Strategy - Lock in profits FAST:
    - Activation: 0.15% profit (activate early!)
    - Trail distance: 0.1% behind current price (tight trail!)
    - This catches those $2000 profits before they reverse
    """
    with _span("main.trailing_stop_manager"):
        try:
            positions = client.get_open_positions()
            if not positions:
                return
            
            for pos in positions:
                try:
                    current_price = pos.current_price
                    entry_price = pos.open_price
                    current_sl = pos.stop_loss
                    
                    if current_price <= 0 or entry_price <= 0:
                        continue
                    
                    # Calculate profit percentage
                    if pos.order_type == "BUY":
                        profit_pct = (current_price - entry_price) / entry_price
                    else:  # SELL
                        profit_pct = (entry_price - current_price) / entry_price
                    
                    # NUCLEAR: Activate trailing EARLY at 0.15% profit
                    if profit_pct < 0.0015:
                        continue
                    
                    # NUCLEAR: Trail TIGHT - only 0.1% behind current price
                    trail_distance = current_price * 0.001
                    
                    if pos.order_type == "BUY":
                        new_sl = current_price - trail_distance
                        # Only move SL up, never down
                        if new_sl > current_sl:
                            logger.info(
                                "Trailing BUY %s #%d: SL %.5f -> %.5f (profit %.2f%%)",
                                pos.symbol, pos.ticket, current_sl, new_sl, profit_pct * 100
                            )
                            client.modify_position(pos.ticket, stop_loss=new_sl)
                    else:  # SELL
                        new_sl = current_price + trail_distance
                        # Only move SL down, never up
                        if current_sl == 0 or new_sl < current_sl:
                            logger.info(
                                "Trailing SELL %s #%d: SL %.5f -> %.5f (profit %.2f%%)",
                                pos.symbol, pos.ticket, current_sl, new_sl, profit_pct * 100
                            )
                            client.modify_position(pos.ticket, stop_loss=new_sl)
                            
                except Exception as exc:
                    logger.warning("Failed to trail position %d: %s", pos.ticket, exc)
                    
        except Exception:
            logger.exception("Trailing stop manager failed")


# Global flag to track if we've hit the victory condition
_VICTORY_ACHIEVED = False

async def execute_trading_cycle(
    client: MT5Client,
    risk_manager: RiskManager,
    state_store: StateStore,
    asian_breakout: AsianBreakoutStrategy,
    barbell: BarbellStrategy,
) -> None:
    global _VICTORY_ACHIEVED
    with _span("main.execute_trading_cycle"):

        if state_store.is_emergency_stop():
            logger.warning("Emergency stop is active; skipping trading cycle")
            return

        # 🏆 VICTORY CONDITION: If equity >= $1M, we WIN - close everything and stop
        if not _VICTORY_ACHIEVED:
            try:
                risk_state = risk_manager.check_risk_state()
                if risk_state.equity >= 1000000.0:
                    logger.critical("🏆🏆🏆 VICTORY! Equity reached $%.2f - CLOSING ALL POSITIONS 🏆🏆🏆", risk_state.equity)
                    # Close all positions to lock in the win
                    close_results = client.close_all_positions()
                    for result in close_results:
                        if result.success:
                            logger.info("Closed position: ticket=%s", result.ticket)
                        else:
                            logger.warning("Failed to close position: %s", result.error_message)
                    # Set emergency stop to prevent any more trading
                    state_store.set_emergency_stop(True)
                    _VICTORY_ACHIEVED = True
                    logger.critical("🎉 ALL TRADING STOPPED - WE WON! Final equity: $%.2f 🎉", risk_state.equity)
                    return
            except Exception as exc:
                logger.warning("Victory check failed: %s", exc)

        safe, reason = risk_manager.is_safe_to_trade()
        if not safe:
            logger.warning("Trading is not safe: %s", reason)
            return

        # Check positions for unrealized losses and add hedges if needed
        hedge_results = risk_manager.add_hedges(client)
        if hedge_results:
            logger.info("Opened %d hedge positions", sum(1 for r in hedge_results if r.success))

        if is_news_blackout():
            logger.warning("News blackout active; skipping trading cycle")
            return

        signals: list = []
        try:
            signals.extend(asian_breakout.generate_signals(client, risk_manager, state_store))
        except Exception:
            logger.exception("Asian breakout signal generation failed")

        try:
            signals.extend(barbell.generate_rebalance_signals(client, risk_manager, state_store))
        except Exception:
            logger.exception("Barbell rebalance signal generation failed")

        for signal in signals:
            try:
                if float(signal.confidence) <= 0.6:
                    continue

                notional = float(signal.volume) * float(signal.entry_price)
                if not risk_manager.check_concentration(signal.symbol, notional):
                    logger.warning("Concentration check failed for %s", signal.symbol)
                    continue

                if not risk_manager.check_directional_exposure(signal):
                    logger.warning("Directional exposure check failed for %s", signal.symbol)
                    continue

                result = client.place_market_order(signal)
                try:
                    logfire.info(
                        f"Order attempt: {signal.action} {signal.symbol} vol={signal.volume:.4f} confidence={signal.confidence:.2f}"
                    )
                except Exception:
                    logger.debug("logfire info unavailable for order attempt logging")

                if result.success:
                    try:
                        logfire.info(f"ORDER PLACED: {signal.symbol} ticket={result.ticket}")
                    except Exception:
                        logger.debug("logfire info unavailable for placed order logging")
                else:
                    try:
                        logfire.error(
                            f"ORDER FAILED: {signal.symbol} error={result.error_code} msg={result.error_msg}"
                        )
                    except Exception:
                        logger.debug("logfire error unavailable for failed order logging")

                logger.info(
                    "Executed signal %s %s %s lots: success=%s code=%s comment=%s",
                    signal.strategy_name,
                    signal.symbol,
                    signal.volume,
                    result.success,
                    result.error_code,
                    result.error_msg,
                )
            except Exception:
                logger.exception("Failed to execute signal for %s", signal.symbol)

        try:
            state = risk_manager.check_risk_state()
            state_store.save_risk_state(state)
        except Exception:
            logger.exception("Failed to save risk state after trading cycle")


async def sentiment_refresh(
    state_store: StateStore,
    sentiment_agent: SentimentAgent,
) -> None:
    with _span("main.sentiment_refresh"):
        try:
            results = await sentiment_agent.run_sentiment_cycle(["XAUUSD", "BTCUSD", "USDJPY"])
            for symbol, result in results.items():
                state_store.save_sentiment(result)
            logger.info(
                "Sentiment refresh complete: %s",
                ", ".join(f"{symbol}:{result.sentiment}" for symbol, result in results.items()),
            )
        except Exception:
            logger.exception("Sentiment refresh failed")


async def mvo_rebalance(barbell: BarbellStrategy, client: MT5Client) -> None:
    with _span("main.mvo_rebalance"):
        try:
            weights = barbell.update_weights_mvo(client)
            logger.info("MVO rebalance updated target weights: %s", weights)
        except Exception:
            logger.exception("MVO rebalance failed")


async def risk_monitor(
    risk_manager: RiskManager,
    state_store: StateStore,
) -> None:
    with _span("main.risk_monitor"):
        try:
            state = risk_manager.check_risk_state()
            if state.current_drawdown_pct > 15.0:
                logger.warning("Drawdown exceeded 15%%: %s%%; triggering emergency close", state.current_drawdown_pct)
                risk_manager.emergency_close_all()
                state_store.set_emergency_stop(True)
            state_store.save_risk_state(state)
        except Exception:
            logger.exception("Risk monitor failed")


def build_scheduler(
    client: MT5Client,
    risk_manager: RiskManager,
    state_store: StateStore,
    asian_breakout: AsianBreakoutStrategy,
    barbell: BarbellStrategy,
    sentiment_agent: SentimentAgent,
) -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone="UTC")

    loop = asyncio.get_event_loop()

    scheduler.add_job(
        lambda: loop.call_soon_threadsafe(asyncio.create_task, execute_trading_cycle(client, risk_manager, state_store, asian_breakout, barbell)),
        "interval",
        minutes=2,
        id="execute_trading_cycle",
        replace_existing=True,
    )
    scheduler.add_job(
        lambda: loop.call_soon_threadsafe(asyncio.create_task, sentiment_refresh(state_store, sentiment_agent)),
        "interval",
        minutes=60,
        id="sentiment_refresh",
        replace_existing=True,
    )
    scheduler.add_job(
        lambda: loop.call_soon_threadsafe(asyncio.create_task, mvo_rebalance(barbell, client)),
        "cron",
        hour=8,
        minute=0,
        id="mvo_rebalance",
        replace_existing=True,
    )
    scheduler.add_job(
        lambda: loop.call_soon_threadsafe(asyncio.create_task, risk_monitor(risk_manager, state_store)),
        "interval",
        minutes=1,
        id="risk_monitor",
        replace_existing=True,
    )
    
    # NUCLEAR: Trail stops every 10 seconds to catch profits before reversal
    scheduler.add_job(
        lambda: loop.call_soon_threadsafe(asyncio.create_task, trailing_stop_manager(client)),
        "interval",
        seconds=10,
        id="trailing_stop_manager",
        replace_existing=True,
    )

    return scheduler


async def main() -> None:
    load_dotenv()

    logfire_token = os.getenv("LOGFIRE_TOKEN")
    try:
        import logfire

        logfire.configure(token=logfire_token, service_name="syphonix-trader")
    except Exception:
        logger.exception("Failed to configure Logfire")

    mt5_client = MT5Client(
        login=int(os.getenv("MT5_LOGIN", "0")),
        password=os.getenv("MT5_PASSWORD", ""),
        server=os.getenv("MT5_SERVER", ""),
        path=os.getenv("MT5_PATH", None),
    )
    state_store = StateStore(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    risk_manager = RiskManager(mt5_client)
    asian_breakout = AsianBreakoutStrategy()
    barbell = BarbellStrategy()
    sentiment_agent = SentimentAgent()

    try:
        mt5_client.connect()

        # Startup verification
        logfire.info("Running startup checks...")

        # 1. Check MT5 connection
        try:
            account = mt5_client.get_account_info()
            logfire.info(f"MT5 connected: equity=${account.equity:,.2f}")
        except Exception as e:
            logfire.error(f"MT5 connection failed: {e}")
            account = None

        # 2. Check Redis
        try:
            if account is not None:
                state_store.save_peak_equity(account.equity)
            else:
                state_store.save_peak_equity(0.0)
            logfire.info("Redis connected OK")
        except Exception as e:
            logfire.error(f"Redis failed: {e}")

        # 3. Check news blackout
        try:
            blackout = is_news_blackout()
            logfire.info(f"News blackout check: {blackout}")
        except Exception as e:
            logfire.error(f"News blackout check failed: {e}")

        # 4. Run one immediate trading cycle
        logfire.info("Running immediate trading cycle...")
        await execute_trading_cycle(mt5_client, risk_manager, state_store, asian_breakout, barbell)

        logfire.info("Startup checks complete - scheduler starting")

        scheduler = build_scheduler(mt5_client, risk_manager, state_store, asian_breakout, barbell, sentiment_agent)
        scheduler.start()
        logger.info("Scheduler started")

        try:
            while True:
                await asyncio.sleep(3600)
        except KeyboardInterrupt:
            logger.info("Scheduler shutdown requested by KeyboardInterrupt")
        finally:
            scheduler.shutdown(wait=False)
    finally:
        mt5_client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
