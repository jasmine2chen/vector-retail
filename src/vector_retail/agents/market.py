"""
agents/market.py
Market Intelligence Agent — Layer 3.

Fetches 1-month returns, volume, 52-week range for each holding.
Provides balanced, factual market context — no price predictions.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
import yfinance as yf

from ..core.models import AgentResult, GraphState, UserProfile
from .base import BaseFinanceAgent

log = structlog.get_logger("agent.market")


class MarketIntelAgent(BaseFinanceAgent):
    AGENT_ID = "market_intel"

    def run(self, state: GraphState) -> AgentResult:
        t0 = time.time()
        reasoning: list[str] = []
        profile = UserProfile(**state.user_profile)

        symbols = [h["symbol"] for h in state.holdings[:5]]  # Cap at 5 for latency
        reasoning.append(f"Analysing market signals for: {', '.join(symbols)}")

        market_context: dict[str, Any] = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1mo")
                if not hist.empty and len(hist) > 1:
                    one_month_return = hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1
                    market_context[symbol] = {
                        "1m_return_pct": round(float(one_month_return) * 100, 2),
                        "avg_daily_volume": int(hist["Volume"].mean()),
                        "period_high": round(float(hist["High"].max()), 2),
                        "period_low": round(float(hist["Low"].min()), 2),
                    }
                    reasoning.append(f"{symbol}: 1M return {one_month_return:.1%}")
            except Exception as exc:
                log.warning("market_data_failed", symbol=symbol, error=str(exc))
                market_context[symbol] = {"error": "data unavailable"}

        market_summary = "\n".join(
            f"  {sym}: 1M {d.get('1m_return_pct', 'N/A')}%"
            for sym, d in market_context.items()
            if "error" not in d
        )

        llm_commentary = self._call_llm(
            system_prompt=(
                "You are a market analyst providing factual, balanced market context "
                "for a retail investor. Never predict future prices. "
                "Flag any data limitations clearly. Focus on observable trends and risk factors. "
                f"User risk tolerance: {profile.risk_tolerance.value}."
            ),
            user_content=(
                f"Query: {state.user_query}\n\n"
                f"Recent 1-month market data:\n{market_summary}\n\n"
                "Provide 3-4 sentences of balanced market context. "
                "Note if any data was unavailable."
            ),
        )

        reasoning.append("Market intel LLM analysis complete")
        self.audit.record("agent", self.AGENT_ID, "completed", {"symbols": len(market_context)})

        # ── Dynamic confidence ─────────────────────────────────────────────
        conf = self._confidence()
        failed_symbols = [s for s, d in market_context.items() if "error" in d]
        for sym in failed_symbols:
            conf.penalize("insufficient_data", f"Market data unavailable for {sym}", sym)
        if self._is_llm_error(llm_commentary):
            conf.penalize("llm_failure", "Market commentary unavailable")

        return AgentResult(
            agent_id=self.AGENT_ID,
            confidence=conf.score(),
            reasoning_chain=reasoning,
            findings={"market_context": market_context, "llm_commentary": llm_commentary},
            data_sources=["yfinance_history_1mo"],
            latency_ms=round((time.time() - t0) * 1000),
        )
