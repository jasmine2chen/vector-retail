"""
orchestrator.py
LangGraph Master Orchestrator — Layer 8.

Builds and runs the full agent graph:

  data_fetch
      ├─► portfolio_analysis ─┐
      ├─► market_intel        ├─► meta_critic ─► hitl_gate ─► synthesis ─► END
      ├─► risk_assessment     ┘
      ├─► tax_optimization    ┘
      └─► rebalance           ┘

Parallel agents run concurrently via LangGraph's fan-out/fan-in.
All agent outputs converge at meta_critic before any response is delivered.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List

import structlog
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, StateGraph

from .agents.market import MarketIntelAgent
from .agents.meta_critic import MetaCriticAgent
from .agents.portfolio import PortfolioAnalysisAgent
from .agents.rebalance import RebalanceAgent
from .agents.risk import RiskAssessmentAgent
from .agents.synthesizer import ResponseSynthesizer
from .agents.tax import TaxOptimizationAgent
from .core.audit import AuditTrail
from .core.enums import DeploymentSlot
from .core.models import AgentResult, GraphState, PortfolioHolding, UserProfile
from .core.policy import POLICY_RULES, POLICY_VERSION, PolicyEngine
from .data.oracle import DataOracle
from .evaluation.hitl import HITLGate
from .evaluation.shadow_eval import ShadowEvaluator
from .security.rbac import SecurityLayer

log = structlog.get_logger("orchestrator")


def _get_llm() -> ChatAnthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. "
            "Add it to your .env file or export it as an environment variable."
        )
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=api_key,
        temperature=0.1,    # Low temperature for financial content — consistency over creativity
        max_tokens=1500,
        timeout=30,
    )


class VectorRetailAgent:
    """
    Production retail finance AI agent.

    Usage:
        agent = VectorRetailAgent(deployment_slot=DeploymentSlot.BLUE)
        result = agent.run(
            user_query="How is my portfolio performing?",
            user_profile=profile,
            holdings=holdings,
            auth_token="...",
            role="retail_client",
        )
    """

    def __init__(self, deployment_slot: DeploymentSlot = DeploymentSlot.BLUE):
        self.deployment_slot = deployment_slot
        self.llm = _get_llm()
        self._log = log.bind(slot=deployment_slot.value)

    # ── Graph construction ─────────────────────────────────────────────────

    def _build_graph(self, security: SecurityLayer, audit: AuditTrail) -> Any:
        """Construct the LangGraph state machine for this session."""

        # Instantiate all agents (shared LLM and audit trail)
        portfolio_agent = PortfolioAnalysisAgent(self.llm, audit)
        market_agent    = MarketIntelAgent(self.llm, audit)
        risk_agent      = RiskAssessmentAgent(self.llm, audit)
        tax_agent       = TaxOptimizationAgent(self.llm, audit)
        rebalance_agent = RebalanceAgent(self.llm, audit)
        meta_critic     = MetaCriticAgent(self.llm, audit)

        # ── Node functions ─────────────────────────────────────────────────

        def data_fetch_node(state: dict) -> dict:
            gs = GraphState(**state)
            oracle = DataOracle(audit_fn=audit.record)
            holdings = [PortfolioHolding(**h) for h in gs.holdings]
            quotes = oracle.get_portfolio_quotes(holdings)
            gs.quotes = {sym: q.model_dump() for sym, q in quotes.items()}
            return gs.model_dump()

        def portfolio_node(state: dict) -> dict:
            gs = GraphState(**state)
            result = portfolio_agent.run(gs)
            gs.agent_results["portfolio_analysis"] = result.model_dump()
            gs.policy_flags.extend(result.policy_flags)
            return gs.model_dump()

        def market_node(state: dict) -> dict:
            gs = GraphState(**state)
            result = market_agent.run(gs)
            gs.agent_results["market_intel"] = result.model_dump()
            return gs.model_dump()

        def risk_node(state: dict) -> dict:
            gs = GraphState(**state)
            result = risk_agent.run(gs)
            gs.agent_results["risk_assessment"] = result.model_dump()
            return gs.model_dump()

        def tax_node(state: dict) -> dict:
            gs = GraphState(**state)
            result = tax_agent.run(gs)
            gs.agent_results["tax_optimization"] = result.model_dump()
            gs.policy_flags.extend(result.policy_flags)
            return gs.model_dump()

        def rebalance_node(state: dict) -> dict:
            gs = GraphState(**state)
            result = rebalance_agent.run(gs)
            gs.agent_results["rebalance"] = result.model_dump()
            gs.policy_flags.extend(result.policy_flags)
            return gs.model_dump()

        def meta_critic_node(state: dict) -> dict:
            gs = GraphState(**state)
            result = meta_critic.run(gs)
            gs.meta_audit_result = result.model_dump()
            return gs.model_dump()

        def hitl_gate_node(state: dict) -> dict:
            gs = GraphState(**state)
            meta = AgentResult(**gs.meta_audit_result)
            gate = HITLGate(audit_fn=audit.record)
            requires_hitl, priority = gate.evaluate(gs, meta)
            if requires_hitl:
                ticket = gate.escalate(gs, meta, priority)
                gs.hitl_queue.append(ticket)
            return gs.model_dump()

        def synthesis_node(state: dict) -> dict:
            gs = GraphState(**state)
            profile = UserProfile(**gs.user_profile)
            synthesizer = ResponseSynthesizer(self.llm, audit_fn=audit.record)
            meta = AgentResult(**gs.meta_audit_result)
            agent_results = {k: AgentResult(**v) for k, v in gs.agent_results.items()}
            hitl_ticket = gs.hitl_queue[0] if gs.hitl_queue else None
            gs.final_response = synthesizer.synthesize(
                gs, agent_results, meta, hitl_ticket, profile
            )
            return gs.model_dump()

        # ── Wire the graph ─────────────────────────────────────────────────
        graph = StateGraph(dict)

        for name, fn in [
            ("data_fetch",          data_fetch_node),
            ("portfolio_analysis",  portfolio_node),
            ("market_intel",        market_node),
            ("risk_assessment",     risk_node),
            ("tax_optimization",    tax_node),
            ("rebalance",           rebalance_node),
            ("meta_critic",         meta_critic_node),
            ("hitl_gate",           hitl_gate_node),
            ("synthesis",           synthesis_node),
        ]:
            graph.add_node(name, fn)

        # data_fetch fans out to all 5 parallel agents
        graph.set_entry_point("data_fetch")
        for agent in ["portfolio_analysis", "market_intel", "risk_assessment",
                      "tax_optimization", "rebalance"]:
            graph.add_edge("data_fetch", agent)
            graph.add_edge(agent, "meta_critic")  # All converge at meta_critic

        graph.add_edge("meta_critic", "hitl_gate")
        graph.add_edge("hitl_gate",   "synthesis")
        graph.add_edge("synthesis",   END)

        return graph.compile()

    # ── Public entry point ─────────────────────────────────────────────────

    def run(
        self,
        user_query: str,
        user_profile: UserProfile,
        holdings: List[PortfolioHolding],
        auth_token: str = "demo_token_placeholder",
        role: str = "retail_client",
    ) -> Dict[str, Any]:
        """
        Execute a full advisory session.

        Returns a structured dict with:
          response             — final client-facing text
          policy_flags         — any compliance flags raised
          hitl_escalated       — whether human review was triggered
          agent_confidences    — per-agent confidence scores
          meta_confidence      — overall meta-critic confidence
          audit_chain_integrity — True if audit chain is intact
          shadow_eval_score    — quality score if this session was sampled
          total_latency_ms     — end-to-end wall time
        """
        session_id = str(uuid.uuid4())
        t_start = time.time()
        self._log.info("session_start", session_id=session_id)

        # ── Layer 0: Security ──────────────────────────────────────────────
        security = SecurityLayer(session_id)
        audit = AuditTrail(session_id, user_profile.user_id)

        try:
            security.validate_jwt_stub(auth_token)
        except PermissionError as exc:
            audit.record("auth", "validate_token", "failed", {"error": str(exc)})
            return {"error": "Authentication failed", "session_id": session_id}

        if not security.validate_permission(role, "request_advice"):
            audit.record("rbac", "request_advice", "denied", {"role": role})
            return {"error": "Insufficient permissions", "session_id": session_id}

        audit.record("auth", "session_opened", "success", {"role": role})

        # ── Shadow eval sampling decision ──────────────────────────────────
        shadow_eval = ShadowEvaluator(audit_fn=audit.record)
        is_shadow = shadow_eval.should_shadow()

        # ── Initial graph state ────────────────────────────────────────────
        from .security import pii
        initial_state = GraphState(
            session_id=session_id,
            user_query=pii.redact(user_query, session_id=session_id),
            user_profile=user_profile.model_dump(),
            holdings=[h.model_dump() for h in holdings],
        ).model_dump()

        # ── Execute graph ──────────────────────────────────────────────────
        app = self._build_graph(security, audit)
        try:
            final_state_dict = app.invoke(initial_state)
            final_state = GraphState(**final_state_dict)
        except Exception as exc:
            self._log.error("graph_execution_failed", error=str(exc))
            audit.record("orchestrator", "graph_run", "failed", {"error": str(exc)})
            return {
                "error": "Analysis pipeline error. Please retry or contact support.",
                "session_id": session_id,
            }

        total_latency = round((time.time() - t_start) * 1000)
        audit.record("orchestrator", "graph_run", "success", {"latency_ms": total_latency})

        # ── Shadow eval scoring ────────────────────────────────────────────
        shadow_score = None
        if is_shadow and final_state.final_response:
            eval_result = shadow_eval.evaluate(
                session_id,
                final_state.final_response,
                {k: AgentResult(**v) for k, v in final_state.agent_results.items()},
                self.deployment_slot,
            )
            shadow_score = eval_result.overall_score

        # ── Audit chain integrity ──────────────────────────────────────────
        chain_ok = audit.verify_chain_integrity()
        if not chain_ok:
            self._log.error("audit_chain_compromised", session_id=session_id)

        self._log.info(
            "session_complete",
            session_id=session_id,
            latency_ms=total_latency,
            chain_ok=chain_ok,
            shadow_score=shadow_score,
            hitl=bool(final_state.hitl_queue),
        )

        return {
            "session_id":           session_id,
            "response":             final_state.final_response,
            "policy_flags":         final_state.policy_flags,
            "hitl_escalated":       bool(final_state.hitl_queue),
            "hitl_tickets":         final_state.hitl_queue,
            "agent_confidences":    {
                k: v.get("confidence") for k, v in final_state.agent_results.items()
            },
            "meta_confidence":      (
                final_state.meta_audit_result.get("confidence")
                if final_state.meta_audit_result else None
            ),
            "data_sources":         list({
                src
                for v in final_state.agent_results.values()
                for src in v.get("data_sources", [])
            }),
            "total_latency_ms":     total_latency,
            "audit_chain_integrity": chain_ok,
            "audit_trail_length":   len(audit),
            "shadow_eval_score":    shadow_score,
            "deployment_slot":      self.deployment_slot.value,
            "policy_version":       POLICY_VERSION,
        }
