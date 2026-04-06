"""
agents/risk.py
Risk Assessment Agent — Layer 3.

Computes:
  - 95% 1-day Value at Risk (historical simulation)
  - Maximum drawdown per holding (3-month window)
  - Plain-language explanation for retail investors

VaR methodology: equal-weighted historical simulation over 3-month window.
Production upgrade: replace with parametric or Monte Carlo VaR with
                    correlation matrix for more accurate portfolio-level VaR.
"""

from __future__ import annotations

import json
import time

import numpy as np
import structlog
import yfinance as yf

from ..core.models import AgentResult, GraphState, PortfolioHolding, UserProfile
from .base import BaseFinanceAgent

log = structlog.get_logger("agent.risk")


class RiskAssessmentAgent(BaseFinanceAgent):
    AGENT_ID = "risk_assessment"

    def run(self, state: GraphState) -> AgentResult:
        t0 = time.time()
        reasoning: list[str] = []
        profile = UserProfile(**state.user_profile)
        holdings = [PortfolioHolding(**h) for h in state.holdings]
        quotes = state.quotes

        # ── Historical returns ─────────────────────────────────────────────
        returns_data: dict[str, np.ndarray] = {}
        for h in holdings[:5]:
            try:
                ticker = yf.Ticker(h.symbol)
                hist = ticker.history(period="3mo")
                if len(hist) > 10:
                    daily_returns = hist["Close"].pct_change().dropna().values
                    returns_data[h.symbol] = daily_returns
            except Exception as exc:
                log.warning("history_fetch_failed", symbol=h.symbol, error=str(exc))

        # ── Portfolio value ────────────────────────────────────────────────
        total_val = sum(
            h.quantity * quotes.get(h.symbol, {}).get("verified_price", h.cost_basis_per_share)
            for h in holdings
        )

        # ── VaR (95%, 1-day, historical simulation) ────────────────────────
        portfolio_var_95_usd: float | None = None
        if returns_data and total_val > 0:
            all_series = [r for r in returns_data.values() if len(r) > 0]
            if all_series:
                min_len = min(len(r) for r in all_series)
                portfolio_returns = np.mean([r[-min_len:] for r in all_series], axis=0)
                var_95_pct = float(np.percentile(portfolio_returns, 5))
                portfolio_var_95_usd = round(var_95_pct * total_val, 2)
                reasoning.append(
                    f"95% 1-day VaR: {var_95_pct:.2%} = ${abs(portfolio_var_95_usd):,.0f}"
                )

        # ── Max drawdown per holding ───────────────────────────────────────
        max_drawdowns: dict[str, float] = {}
        for sym, returns in returns_data.items():
            cumulative = np.cumprod(1 + returns)
            rolling_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - rolling_max) / rolling_max
            max_drawdowns[sym] = round(float(np.min(drawdowns)) * 100, 2)
            reasoning.append(f"{sym} max drawdown (3mo): {max_drawdowns[sym]:.1f}%")

        # ── LLM plain-language explanation ────────────────────────────────
        var_display = f"${abs(portfolio_var_95_usd):,.0f}" if portfolio_var_95_usd else "N/A"
        llm_explanation = self._call_llm(
            system_prompt=(
                "You are a risk analyst at a regulated retail brokerage. "
                "Explain risk metrics in plain language suitable for a retail investor. "
                "Never use jargon without defining it. "
                "Always recommend seeking professional advice for complex risk decisions. "
                f"Client risk profile: {profile.risk_tolerance.value}."
            ),
            user_content=(
                f"Query: {state.user_query}\n\n"
                f"Portfolio 95% 1-day VaR (approximate maximum daily loss at 95% confidence): "
                f"{var_display}\n"
                f"3-month maximum drawdowns by holding: {json.dumps(max_drawdowns, indent=2)}\n\n"
                "Explain these in 3-4 plain-language sentences for a retail investor. "
                "Define VaR simply. Contextualise the drawdowns."
            ),
        )

        self.audit.record(
            "agent",
            self.AGENT_ID,
            "completed",
            {"var_95_usd": portfolio_var_95_usd, "holdings_analysed": len(returns_data)},
        )

        # ── Dynamic confidence ─────────────────────────────────────────────
        conf = self._confidence()
        total_holdings = len(holdings[:5])
        missing_history = total_holdings - len(returns_data)
        for _ in range(missing_history):
            conf.penalize(
                "insufficient_data",
                f"Missing 3M history for {missing_history} holding(s)",
                missing_history,
            )
        if portfolio_var_95_usd is None:
            conf.penalize("var_computation_failed", "Could not compute portfolio VaR")
        if self._is_llm_error(llm_explanation):
            conf.penalize("llm_failure", "Risk explanation unavailable")

        return AgentResult(
            agent_id=self.AGENT_ID,
            confidence=conf.score(),
            reasoning_chain=reasoning,
            findings={
                "var_95_usd": portfolio_var_95_usd,
                "max_drawdowns_pct": max_drawdowns,
                "total_portfolio_value": round(total_val, 2),
                "llm_risk_explanation": llm_explanation,
            },
            data_sources=["yfinance_history_3mo"],
            latency_ms=round((time.time() - t0) * 1000),
        )
