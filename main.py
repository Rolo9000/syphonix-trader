"""Entry point and APScheduler job definitions for the Syphonix trader.

Wires together the broker client, strategies, agents, risk manager, and state
store, then registers the scheduled jobs (signal generation, execution, risk
monitoring) on a background scheduler and runs until interrupted.
"""

from __future__ import annotations

import asyncio
import logging
import os

from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

from agents.sentiment_agent import SentimentAgent
from core import MT5Client
from core.indicators import is_news_blackout
from core.risk_manager import RiskManager
from infra.state_store import StateStore
from strategies.asian_breakout import AsianBreakoutStrategy
from strategies.barbell import BarbellStrategy

logger = logging.getLogger(__name__)


def _span(name: str):
    try:
        import logfire

        if hasattr(logfire, "span"):
            return logfire.span(name)
    except ImportError:
        pass
    return asyncio.nullcontext()


async def execute_trading_cycle(
    client: MT5Client,
    risk_manager: RiskManager,
    state_store: StateStore,
    asian_breakout: AsianBreakoutStrategy,
    barbell: BarbellStrategy,
) -> None:
    with _span("main.execute_trading_cycle"):
        if state_store.is_emergency_stop():
            logger.warning("Emergency stop is active; skipping trading cycle")
            return

        safe, reason = risk_manager.is_safe_to_trade()
        if not safe:
            logger.warning("Trading is not safe: %s", reason)
            return

        if is_news_blackout():
            logger.warning("News blackout active; skipping trading cycle")
            return

        signals: list = []
        try:
            signals.extend(asian_breakout.generate_signals(client, risk_manager))
        except Exception:
            logger.exception("Asian breakout signal generation failed")

        try:
            signals.extend(barbell.generate_rebalance_signals(client, risk_manager))
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

                result = client.place_market_order(signal)
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
        minutes=5,
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
