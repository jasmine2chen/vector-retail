"""
agents/rebalance.py
Rebalancing Agent — Layer 3.

Computes drift from target allocation by asset class.
Flags trades that exceed the HITL threshold for human review.

Target allocations are risk-profile-driven and defined here as
class-level data (production: move to config/policy_rules.json).
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import structlog

from ..core.models import AgentResult, GraphState, PortfolioHolding, UserProfile
from ..core.enums import RiskTolerance
from ..core.policy import POLICY_RULES, PolicyEngine
from .base import BaseFinanceAgent

log = structlog.get_logger("agent.rebalance")


class RebalanceAgent(BaseFinanceAgent):
    AGENT_ID = "rebalance"

    # Target allocations loaded from config/policy_rules.json (single source of truth).
    # Fallback values used only if config key is missing.
    _FALLBACK_ALLOCATIONS: Dict[str, Dict[str, float]] = {
        RiskTolerance.CONSERVATIVE: {"equity": 0.40, "fixed_income": 0.50, "cash": 0.10},
        RiskTolerance.MODERATE:     {"equity": 0.60, "fixed_income": 0.30, "cash": 0.10},
        RiskTolerance.AGGRESSIVE:   {"equity": 0.80, "fixed_income": 0.15, "cash": 0.05},
    }

    @classmethod
    def _get_target_allocations(cls) -> Dict[str, Dict[str, float]]:
        """Load target allocations from policy config, falling back to defaults."""
        config_allocs = POLICY_RULES.get("target_allocations", {})
        if config_allocs:
            # Config uses string keys; map to enum values for consistency
            return {
                RiskTolerance(k): v
                for k, v in config_allocs.items()
                if k in [e.value for e in RiskTolerance]
            }
        return cls._FALLBACK_ALLOCATIONS

    def run(self, state: GraphState) -> AgentResult:
        t0 = time.time()
        reasoning: List[str] = []
        profile = UserProfile(**state.user_profile)
        holdings = [PortfolioHolding(**h) for h in state.holdings]
        quotes = state.quotes
        policy = PolicyEngine(profile, self.audit.record)

        # ── Current allocation ─────────────────────────────────────────────
        current_alloc: Dict[str, float] = {}
        total_val = 0.0
        for h in holdings:
            q = quotes.get(h.symbol, {})
            price = q.get("verified_price", h.cost_basis_per_share)
            val = h.quantity * price
            total_val += val
            current_alloc[h.asset_class] = current_alloc.get(h.asset_class, 0) + val

        current_pct = (
            {k: v / total_val for k, v in current_alloc.items()} if total_val > 0 else {}
        )

        target_allocations = self._get_target_allocations()
        target = target_allocations.get(
            profile.risk_tolerance, target_allocations.get(RiskTolerance.MODERATE, {})
        )

        # ── Drift calculation ──────────────────────────────────────────────
        rebalance_actions: List[Dict[str, Any]] = []
        policy_flags: List[str] = []

        for asset_class, target_pct in target.items():
            current = current_pct.get(asset_class, 0.0)
            drift = current - target_pct
            drift_usd = drift * total_val

            reasoning.append(
                f"{asset_class}: current {current:.1%}, target {target_pct:.1%}, "
                f"drift {drift:+.1%} (${drift_usd:+,.0f})"
            )

            if abs(drift) > 0.05:  # Only flag > 5% drift
                action = "REDUCE" if drift > 0 else "INCREASE"
                hitl_needed = policy.check_trade_hitl_required(abs(drift_usd))

                rebalance_actions.append({
                    "asset_class": asset_class,
                    "action": action,
                    "current_pct": round(current, 4),
                    "target_pct": round(target_pct, 4),
                    "drift_pct": round(drift * 100, 1),
                    "drift_usd": round(drift_usd, 2),
                    "requires_hitl": hitl_needed,
                })

                if hitl_needed:
                    policy_flags.append(
                        f"HITL_REQUIRED: {action} {asset_class} by ${abs(drift_usd):,.0f} "
                        f"exceeds ${POLICY_RULES['trade_value_hitl_threshold_usd']:,.0f} threshold"
                    )

        hitl_required = any(a["requires_hitl"] for a in rebalance_actions)

        llm_commentary = self._call_llm(
            system_prompt=(
                "You are a portfolio rebalancing specialist. "
                "Provide general rebalancing observations only — not specific trade orders. "
                "All specific trades must be reviewed by a licensed advisor before execution. "
                f"Target allocation for {profile.risk_tolerance.value} profile: {target}."
            ),
            user_content=(
                f"Query: {state.user_query}\n\n"
                f"Current allocations: {json.dumps(current_pct, indent=2)}\n"
                f"Target allocations: {json.dumps(target, indent=2)}\n"
                f"Rebalancing actions (>5% drift): {json.dumps(rebalance_actions, indent=2)}\n\n"
                "Summarise in 3-4 sentences. "
                "Note any large trades that require advisor review."
            ),
        )

        self.audit.record(
            "agent", self.AGENT_ID, "completed",
            {"actions": len(rebalance_actions), "hitl_required": hitl_required},
        )

        # ── Dynamic confidence ─────────────────────────────────────────────
        conf = self._confidence()
        for h in holdings:
            q = quotes.get(h.symbol, {})
            if "verified_price" not in q:
                conf.penalize("stale_quote", f"Using cost basis for {h.symbol}", h.symbol)
        if hitl_required:
            conf.penalize("hitl_trade_flagged", "Large trade(s) require human review")
        for asset_class in target:
            if asset_class not in current_pct:
                conf.penalize("missing_asset_class", f"Portfolio missing target class '{asset_class}'", asset_class)
        if self._is_llm_error(llm_commentary):
            conf.penalize("llm_failure", "Rebalance commentary unavailable")

        return AgentResult(
            agent_id=self.AGENT_ID,
            confidence=conf.score(),
            reasoning_chain=reasoning,
            findings={
                "current_allocation_pct": current_pct,
                "target_allocation_pct": target,
                "rebalance_actions": rebalance_actions,
                "total_portfolio_value": round(total_val, 2),
                "llm_rebalance_commentary": llm_commentary,
            },
            recommendations=[
                f"{a['action']} {a['asset_class']} by ${abs(a['drift_usd']):,.0f}"
                for a in rebalance_actions
            ],
            data_sources=["portfolio_holdings", "yfinance"],
            policy_flags=policy_flags,
            requires_hitl=hitl_required,
            hitl_reason="Rebalance trade exceeds HITL threshold" if hitl_required else None,
            latency_ms=round((time.time() - t0) * 1000),
        )
