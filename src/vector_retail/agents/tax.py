"""
agents/tax.py
Tax Optimisation Agent — Layer 3.

Identifies:
  - Tax-loss harvesting candidates (unrealised loss > $500)
  - Short vs long-term holding classification (365-day rule)
  - Wash-sale risk warnings (positions purchased < 30 days ago)
  - IRA/Roth IRA account type notes

IMPORTANT: This agent provides general tax observations ONLY.
           It does NOT provide tax advice. All output directs users
           to a qualified CPA or tax attorney.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import structlog

from ..core.models import AgentResult, GraphState, PortfolioHolding, UserProfile
from ..core.policy import PolicyEngine
from .base import BaseFinanceAgent

log = structlog.get_logger("agent.tax")


class TaxOptimizationAgent(BaseFinanceAgent):
    AGENT_ID = "tax_optimization"

    def run(self, state: GraphState) -> AgentResult:
        t0 = time.time()
        reasoning: List[str] = []
        profile = UserProfile(**state.user_profile)
        holdings = [PortfolioHolding(**h) for h in state.holdings]
        quotes = state.quotes
        policy = PolicyEngine(profile, self.audit.record)

        today = datetime.now(timezone.utc).date()
        loss_harvest_candidates: List[Dict[str, Any]] = []
        wash_sale_warnings: List[str] = []

        for h in holdings:
            q = quotes.get(h.symbol, {})
            current_price = q.get("verified_price", h.cost_basis_per_share)
            mkt_val = h.quantity * current_price
            pnl = mkt_val - h.cost_basis_total

            try:
                purchase_date = datetime.fromisoformat(h.purchase_date).date()
            except ValueError:
                reasoning.append(f"{h.symbol}: invalid purchase_date, skipping")
                continue

            holding_days = (today - purchase_date).days

            # Tax-loss harvesting candidate
            if pnl < -500:
                is_long_term = holding_days >= 365
                loss_harvest_candidates.append({
                    "symbol": h.symbol,
                    "unrealised_loss_usd": round(pnl, 2),
                    "holding_days": holding_days,
                    "is_long_term": is_long_term,
                    "tax_treatment": "long-term capital loss" if is_long_term else "short-term capital loss",
                    "wash_sale_caution": "Verify 30-day rule before selling and repurchasing",
                })
                reasoning.append(
                    f"{h.symbol}: harvest candidate, loss ${pnl:,.0f}, "
                    f"{'long' if is_long_term else 'short'}-term ({holding_days}d)"
                )

            # Wash-sale warning: purchased within last 30 days
            if holding_days < 30:
                wash_sale_warnings.append(
                    f"{h.symbol}: purchased {holding_days} days ago — "
                    f"wash-sale rule (IRC §1091) may apply if sold at a loss"
                )

        # IRA/Roth note
        ira_note = policy.check_ira_tax_applicability()
        if ira_note:
            reasoning.append("IRA account detected — tax treatment differs")

        llm_commentary = self._call_llm(
            system_prompt=(
                "You are a tax-aware investment analyst. Provide general tax observations only — "
                "NOT specific tax advice. Always direct clients to a qualified CPA or tax attorney. "
                "Mention wash-sale rules and relevant account type implications. "
                f"Account type: {profile.account_type.value}. "
                f"Jurisdiction: {profile.jurisdiction.value}."
            ),
            user_content=(
                f"Query: {state.user_query}\n\n"
                f"Tax-loss harvesting candidates:\n{json.dumps(loss_harvest_candidates, indent=2)}\n"
                f"Wash-sale warnings: {wash_sale_warnings}\n"
                f"Account note: {ira_note}\n\n"
                "Summarise in 3-4 sentences. "
                "Remind client this is general information only and to consult a tax professional."
            ),
        )

        self.audit.record(
            "agent", self.AGENT_ID, "completed",
            {"candidates": len(loss_harvest_candidates), "wash_sale_warnings": len(wash_sale_warnings)},
        )

        # ── Dynamic confidence ─────────────────────────────────────────────
        conf = self._confidence()
        for h in holdings:
            try:
                datetime.fromisoformat(h.purchase_date)
            except ValueError:
                conf.penalize("date_parse_failure", f"Invalid purchase_date for {h.symbol}", h.symbol)
            q = quotes.get(h.symbol, {})
            if "verified_price" not in q:
                conf.penalize("stale_quote", f"No live quote for {h.symbol}, using cost basis", h.symbol)
        if self._is_llm_error(llm_commentary):
            conf.penalize("llm_failure", "Tax commentary unavailable")

        return AgentResult(
            agent_id=self.AGENT_ID,
            confidence=conf.score(),
            reasoning_chain=reasoning,
            findings={
                "loss_harvest_candidates": loss_harvest_candidates,
                "wash_sale_warnings": wash_sale_warnings,
                "ira_note": ira_note,
                "llm_tax_commentary": llm_commentary,
            },
            recommendations=[
                f"Review {c['symbol']} for {c['tax_treatment']}: ${abs(c['unrealised_loss_usd']):,.0f} loss"
                for c in loss_harvest_candidates
            ],
            data_sources=["portfolio_holdings", "yfinance"],
            latency_ms=round((time.time() - t0) * 1000),
        )
