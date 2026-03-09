"""
agents/portfolio.py
Portfolio Analysis Agent — Layer 3.

Responsibilities:
  - Compute current market values and unrealised P&L per position
  - Calculate sector and asset-class exposure
  - Run concentration checks via PolicyEngine
  - Produce an LLM-enhanced portfolio health assessment
"""
from __future__ import annotations

import time
from typing import Dict, List

import structlog

from ..core.models import AgentResult, GraphState, PortfolioHolding, UserProfile
from ..core.policy import PolicyEngine
from .base import BaseFinanceAgent

log = structlog.get_logger("agent.portfolio")


class PortfolioAnalysisAgent(BaseFinanceAgent):
    AGENT_ID = "portfolio_analysis"

    def run(self, state: GraphState) -> AgentResult:
        t0 = time.time()
        reasoning: List[str] = []

        holdings = [PortfolioHolding(**h) for h in state.holdings]
        quotes = state.quotes
        profile = UserProfile(**state.user_profile)
        policy = PolicyEngine(profile, self.audit.record)

        # ── Compute metrics ────────────────────────────────────────────────
        total_value = 0.0
        position_values: Dict[str, float] = {}
        unrealised_pnl: Dict[str, float] = {}
        sector_exposure: Dict[str, float] = {}

        for h in holdings:
            q = quotes.get(h.symbol, {})
            mkt_price = q.get("verified_price", h.cost_basis_per_share)
            mkt_val = h.quantity * mkt_price
            position_values[h.symbol] = mkt_val
            unrealised_pnl[h.symbol] = round(mkt_val - h.cost_basis_total, 2)
            sector_exposure[h.sector] = sector_exposure.get(h.sector, 0) + mkt_val
            total_value += mkt_val

        reasoning.append(f"Total portfolio market value: ${total_value:,.2f}")

        # ── Concentration checks ───────────────────────────────────────────
        concentration_flags: List[str] = []
        for sym, val in position_values.items():
            ok, reason = policy.check_position_concentration(sym, val, total_value)
            if not ok:
                concentration_flags.append(reason)

        # ── Sector exposure checks ─────────────────────────────────────────
        for sector, val in sector_exposure.items():
            ok, reason = policy.check_sector_exposure(sector, val, total_value)
            if not ok:
                concentration_flags.append(reason)

        reasoning.append(f"Policy flags: {len(concentration_flags)}")

        # ── LLM assessment ─────────────────────────────────────────────────
        holdings_summary = "\n".join(
            f"  {h.symbol}: {h.quantity} shares, "
            f"market value ${position_values.get(h.symbol, 0):,.0f}, "
            f"P&L ${unrealised_pnl.get(h.symbol, 0):+,.0f}"
            for h in holdings
        )
        sector_pcts = ", ".join(
            f"{s}: {v / total_value:.1%}" for s, v in sector_exposure.items()
        ) if total_value > 0 else "N/A"

        llm_assessment = self._call_llm(
            system_prompt=(
                "You are a portfolio analyst for a regulated retail brokerage. "
                "Provide a concise, factual 3-5 sentence portfolio health assessment. "
                "Never make specific buy/sell recommendations without suitability caveats. "
                "Always note that past performance does not guarantee future results. "
                f"User risk profile: {profile.risk_tolerance.value}. "
                f"Account type: {profile.account_type.value}."
            ),
            user_content=(
                f"Query: {state.user_query}\n\n"
                f"Portfolio holdings:\n{holdings_summary}\n\n"
                f"Sector exposure: {sector_pcts}\n"
                f"Total market value: ${total_value:,.2f}\n"
                f"Concentration/sector issues: "
                f"{'; '.join(concentration_flags) if concentration_flags else 'None detected'}\n\n"
                "Provide a 3-5 sentence portfolio health assessment."
            ),
        )

        reasoning.append("LLM portfolio assessment complete")

        # ── Dynamic confidence ─────────────────────────────────────────────
        conf = self._confidence()
        for h in holdings:
            q = quotes.get(h.symbol, {})
            if "verified_price" not in q:
                conf.penalize("missing_quote", f"No verified quote for {h.symbol}", h.symbol)
            elif q.get("is_stale"):
                conf.penalize("stale_quote", f"Stale quote for {h.symbol}", h.symbol)
        if concentration_flags:
            conf.penalize("policy_violation", f"{len(concentration_flags)} policy flag(s)", len(concentration_flags))
        if self._is_llm_error(llm_assessment):
            conf.penalize("llm_failure", "LLM assessment unavailable")

        self.audit.record(
            "agent", self.AGENT_ID, "completed",
            {"total_value": round(total_value, 2), "flags": len(concentration_flags)},
        )

        return AgentResult(
            agent_id=self.AGENT_ID,
            confidence=conf.score(),
            reasoning_chain=reasoning,
            findings={
                "total_value_usd": round(total_value, 2),
                "position_values": {k: round(v, 2) for k, v in position_values.items()},
                "unrealised_pnl": unrealised_pnl,
                "sector_exposure_pct": {
                    s: round(v / total_value, 4) for s, v in sector_exposure.items()
                } if total_value > 0 else {},
                "llm_assessment": llm_assessment,
            },
            recommendations=concentration_flags,
            data_sources=["yfinance", "alpha_vantage_stub"],
            policy_flags=concentration_flags,
            latency_ms=round((time.time() - t0) * 1000),
        )
