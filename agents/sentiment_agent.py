"""News sentiment analysis agent built on Pydantic AI + Claude.

Wraps a Pydantic AI ``Agent`` that reads headlines / articles for an instrument
and returns a structured :class:`~core.models.SentimentResult`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from core.models import SentimentResult
from pydantic_ai import Agent

logger = logging.getLogger(__name__)

#: Pydantic AI model identifier for the Anthropic Claude Sonnet 4.6 model.
SENTIMENT_MODEL = "anthropic:claude-sonnet-4-6"
NEWSAPI_DEMO_KEY = "demo"

try:
    import logfire
except ImportError:  # pragma: no cover
    logfire = None


def _span(name: str):
    if logfire is not None and hasattr(logfire, "span"):
        return logfire.span(name)
    return asyncio.nullcontext()


if logfire is not None and hasattr(logfire, "instrument_pydantic_ai"):
    logfire.instrument_pydantic_ai()


_agent: Agent[None, SentimentResult] | None = None


def get_agent() -> Agent[None, SentimentResult]:
    """Return a cached Pydantic AI agent, creating it lazily on first use."""
    global _agent
    if _agent is None:
        _agent = Agent(
            SENTIMENT_MODEL,
            result_type=SentimentResult,
            system_prompt=(
                "You are a financial sentiment analyst for an algorithmic trading system. "
                "Analyse the provided news headlines and return a structured sentiment "
                "assessment. Be concise and precise. Focus on market-moving implications."
            ),
        )
    return _agent


class SentimentAgent:
    """Async sentiment analysis agent backed by Pydantic AI."""

    async def analyse_sentiment(self, symbol: str, headlines: list[str]) -> SentimentResult:
        """Format headlines into a prompt, run the sentiment agent, and return the result."""
        with _span("sentiment_agent.analyse_sentiment"):
            if not headlines:
                return SentimentResult(
                    symbol=symbol,
                    sentiment="NEUTRAL",
                    confidence=0.0,
                    reasoning="No headlines were available for analysis.",
                    timestamp=datetime.utcnow(),
                )

            prompt = (
                f"Symbol: {symbol}\n"
                "Headlines:\n"
                + "\n".join(f"- {headline}" for headline in headlines[:10])
                + "\n\n"
                "Provide a concise sentiment assessment in terms of bullish, bearish, or neutral bias. "
                "Return only a structured response consistent with the expected output schema."
            )

            try:
                agent = get_agent()
                result = await agent(prompt)
                if isinstance(result, SentimentResult):
                    return result
                return SentimentResult(
                    symbol=symbol,
                    sentiment="NEUTRAL",
                    confidence=0.0,
                    reasoning="Agent returned an unexpected response format.",
                    timestamp=datetime.utcnow(),
                )
            except Exception as exc:
                logger.exception("Sentiment analysis failed for %s", symbol)
                return SentimentResult(
                    symbol=symbol,
                    sentiment="NEUTRAL",
                    confidence=0.0,
                    reasoning=f"Sentiment analysis failed: {exc}",
                    timestamp=datetime.utcnow(),
                )

    async def fetch_headlines(self, symbol: str) -> list[str]:
        """Fetch the latest headlines for the provided symbol.

        Uses a free NewsAPI demo endpoint and returns the top 10 headline titles.
        """
        with _span("sentiment_agent.fetch_headlines"):
            terms = {
                "XAUUSD": "gold price",
                "BTCUSD": "bitcoin",
                "USDJPY": "USD JPY",
            }
            query = terms.get(symbol, symbol)
            encoded = quote_plus(query)
            url = (
                f"https://newsapi.org/v2/everything?q={encoded}&pageSize=10"
                f"&sortBy=publishedAt&apiKey={NEWSAPI_DEMO_KEY}"
            )

            def _fetch() -> list[str]:
                try:
                    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urlopen(request, timeout=10) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                        articles = payload.get("articles", [])
                        headlines = []
                        for article in articles[:10]:
                            title = article.get("title")
                            if title:
                                headlines.append(title)
                        return headlines
                except (HTTPError, URLError, json.JSONDecodeError, OSError) as exc:
                    logger.warning("Headline fetch failed for %s: %s", symbol, exc)
                    return []

            return await asyncio.to_thread(_fetch)

    async def run_sentiment_cycle(self, symbols: list[str]) -> dict[str, SentimentResult]:
        """Fetch headlines and run sentiment analysis in parallel for each symbol."""
        with _span("sentiment_agent.run_sentiment_cycle"):
            async def _process(symbol: str) -> tuple[str, SentimentResult]:
                headlines = await self.fetch_headlines(symbol)
                result = await self.analyse_sentiment(symbol, headlines)
                return symbol, result

            tasks = [asyncio.create_task(_process(symbol)) for symbol in symbols]
            results = await asyncio.gather(*tasks)
            return {symbol: sentiment for symbol, sentiment in results}
