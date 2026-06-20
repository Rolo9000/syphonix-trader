"""Combine signals from multiple strategies into a single decision.

The aggregator collects raw :class:`~core.models.TradeSignal` objects from each
strategy, optionally weights them by sentiment and market regime, resolves
conflicts (e.g. opposing sides on the same symbol), and produces the final set
of signals handed to the risk manager.
"""

from __future__ import annotations

from core.models import TradeSignal


class SignalAggregator:
    """Merges and de-conflicts signals from all registered strategies."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        """Configure per-source weights used when combining signals."""
        raise NotImplementedError

    def aggregate(
        self,
        signals: list[TradeSignal],
        sentiment: float | None = None,
        regime: str | None = None,
    ) -> list[TradeSignal]:
        """Combine and de-conflict ``signals`` into a final decision set."""
        raise NotImplementedError
