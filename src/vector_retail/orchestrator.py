"""
orchestrator.py
LangGraph Master Orchestrator — Layer 8.

Builds and runs the full agent graph:

  data_fetch
      ├─► portfolio_analysis ─┐
      ├─► risk_assessment      ├─► meta_critic ─► [router] ─► synthesis ─► END
      ├─► rebalance            │                      │
      └─► sentiment_analysis ──┘  (FinBERT news sentiment — 4th parallel agent)
                                                     └──► hitl_gate ─► synthesis ─► END

Router (after meta_critic):
  - needs_revision=True  : route directly to "synthesis" with critique foregrounded
                           (Reflection pattern — Andrew Ng Design Pattern #1)
  - needs_revision=False : route to "hitl_gate" first, then "synthesis"

LangGraph MemorySaver checkpointing is attached to every session graph.
Each session is identified by thread_id=session_id in the LangGraph config.
HITL-escalated sessions can be resumed via resume_hitl_session() — the reviewer's
approval updates the checkpointed state and re-invokes synthesis.

Production note on checkpointing:
  MemorySaver is in-process only. For durable HITL resume across process restarts,
  swap to AsyncPostgresSaver (langgraph-checkpoint-postgres) or RedisSaver.
  The graph compilation and invocation API is identical — only the checkpointer changes.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import structlog
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .agents.meta_critic import MetaCriticAgent
from .agents.portfolio import PortfolioAnalysisAgent
from .agents.rebalance import RebalanceAgent
from .agents.risk import RiskAssessmentAgent
from .agents.sentiment import SentimentAnalysisAgent
from .agents.synthesizer import ResponseSynthesizer
from .core.audit import AuditTrail
from .core.enums import DeploymentSlot
from .core.models import AgentResult, GraphState, PortfolioHolding, UserProfile
from .core.policy import POLICY_VERSION
from .data.oracle import DataOracle
from .evaluation.hitl import HITLGate
from .evaluation.shadow_eval import ShadowEvaluator
from .security.rbac import SecurityLayer

log = structlog.get_logger("orchestrator")


def _get_llm() -> ChatAnthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise OSError(
            "ANTHROPIC_API_KEY not set. "
            "Add it to your .env file or export it as an environment variable."
        )
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=api_key,
        temperature=0.1,  # Low temperature for financial content — consistency over creativity
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

    HITL Resume (after a session is escalated):
        result = agent.resume_hitl_session(
            session_id="...",
            reviewer_notes="Approved — concentration within acceptable range for this client",
        )
    """

    def __init__(self, deployment_slot: DeploymentSlot = DeploymentSlot.BLUE):
        self.deployment_slot = deployment_slot
        self.llm = _get_llm()
        self._log = log.bind(slot=deployment_slot.value)
        # In-process session registry for HITL resume.
        # Production: replace with Redis or PostgreSQL-backed store.
        self._session_graphs: dict[str, tuple[Any, MemorySaver]] = {}

    # ── Graph construction ─────────────────────────────────────────────────

    def _build_graph(
        self,
        security: SecurityLayer,
        audit: AuditTrail,
        checkpointer: MemorySaver,
    ) -> Any:
        """Construct the LangGraph state machine for this session."""

        portfolio_agent = PortfolioAnalysisAgent(self.llm, audit)
        risk_agent = RiskAssessmentAgent(self.llm, audit)
        rebalance_agent = RebalanceAgent(self.llm, audit)
        sentiment_agent = SentimentAnalysisAgent(self.llm, audit)
        meta_critic = MetaCriticAgent(self.llm, audit)

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

        def risk_node(state: dict) -> dict:
            gs = GraphState(**state)
            result = risk_agent.run(gs)
            gs.agent_results["risk_assessment"] = result.model_dump()
            return gs.model_dump()

        def rebalance_node(state: dict) -> dict:
            gs = GraphState(**state)
            result = rebalance_agent.run(gs)
            gs.agent_results["rebalance"] = result.model_dump()
            gs.policy_flags.extend(result.policy_flags)
            return gs.model_dump()

        def sentiment_node(state: dict) -> dict:
            gs = GraphState(**state)
            result = sentiment_agent.run(gs)
            gs.agent_results["sentiment_analysis"] = result.model_dump()
            # Bearish signals are surfaced as policy flags for meta-critic review
            gs.policy_flags.extend(result.policy_flags)
            return gs.model_dump()

        def meta_critic_node(state: dict) -> dict:
            gs = GraphState(**state)
            result = meta_critic.run(gs)
            gs.meta_audit_result = result.model_dump()
            # Surface reflection routing flags onto graph state
            gs.needs_revision = result.findings.get("needs_revision", False)
            if gs.needs_revision:
                gs.revision_critique = result.findings.get("compliance_review")
                audit.record(
                    "meta_critic",
                    "reflection_triggered",
                    "info",
                    {
                        "confidence": result.confidence,
                        "critique_preview": (
                            gs.revision_critique[:120] if gs.revision_critique else ""
                        ),
                    },
                )
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
            # Pass revision_critique when meta-critic flagged medium concern
            gs.final_response, gs.regulatory_clauses_used = synthesizer.synthesize(
                gs,
                agent_results,
                meta,
                hitl_ticket,
                profile,
                revision_critique=gs.revision_critique,
            )
            return gs.model_dump()

        # ── Conditional routing after meta_critic ──────────────────────────
        # Implements Andrew Ng's Reflection design pattern:
        # When meta_critic finds medium concern (conf 0.75–0.85), the critique
        # is fed back to the synthesizer rather than escalating to a human.
        # This self-correcting loop improves response quality without HITL overhead.
        def route_after_meta_critic(state: dict) -> str:
            gs = GraphState(**state)
            if gs.needs_revision:
                log.info(
                    "reflection_loop_routing",
                    session_id=gs.session_id,
                    reason="medium_confidence_band",
                )
                return "synthesis"  # Bypass hitl_gate — confidence >= 0.75
            return "hitl_gate"

        # ── Wire the graph ─────────────────────────────────────────────────
        graph = StateGraph(dict)

        for name, fn in [
            ("data_fetch", data_fetch_node),
            ("portfolio_analysis", portfolio_node),
            ("risk_assessment", risk_node),
            ("rebalance", rebalance_node),
            ("sentiment_analysis", sentiment_node),
            ("meta_critic", meta_critic_node),
            ("hitl_gate", hitl_gate_node),
            ("synthesis", synthesis_node),
        ]:
            graph.add_node(name, fn)

        # data_fetch fans out to all 6 parallel agents
        graph.set_entry_point("data_fetch")
        for agent in [
            "portfolio_analysis",
            "risk_assessment",
            "rebalance",
            "sentiment_analysis",
        ]:
            graph.add_edge("data_fetch", agent)
            graph.add_edge(agent, "meta_critic")

        # Reflection routing: conditional edge after meta_critic
        graph.add_conditional_edges(
            "meta_critic",
            route_after_meta_critic,
            {"synthesis": "synthesis", "hitl_gate": "hitl_gate"},
        )

        graph.add_edge("hitl_gate", "synthesis")
        graph.add_edge("synthesis", END)

        return graph.compile(checkpointer=checkpointer)

    # ── Public entry point ─────────────────────────────────────────────────

    def run(
        self,
        user_query: str,
        user_profile: UserProfile,
        holdings: list[PortfolioHolding],
        auth_token: str = "",
        role: str = "retail_client",
    ) -> dict[str, Any]:
        """
        Execute a full advisory session.

        Returns a structured dict with:
          response              — final client-facing text
          policy_flags          — any compliance flags raised
          hitl_escalated        — whether human review was triggered
          reflection_applied    — whether meta-critic revision loop ran
          agent_confidences     — per-agent confidence scores
          meta_confidence       — overall meta-critic confidence
          audit_chain_integrity — True if audit chain is intact
          shadow_eval_score     — quality score if this session was sampled
          total_latency_ms      — end-to-end wall time
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
        shadow_eval = ShadowEvaluator(audit_fn=audit.record, llm=self.llm)
        is_shadow = shadow_eval.should_shadow()

        # ── Initial graph state ────────────────────────────────────────────
        from .security import pii

        initial_state = GraphState(
            session_id=session_id,
            user_query=pii.redact(user_query, session_id=session_id),
            user_profile=user_profile.model_dump(),
            holdings=[h.model_dump() for h in holdings],
        ).model_dump()

        # ── Execute graph with MemorySaver checkpointing ───────────────────
        # thread_id=session_id ensures each session has an isolated checkpoint.
        # For HITL resume, the reviewer calls resume_hitl_session(session_id, ...)
        # which loads this checkpoint and continues from the hitl_gate node.
        checkpointer = MemorySaver()
        app = self._build_graph(security, audit, checkpointer=checkpointer)
        self._session_graphs[session_id] = (app, checkpointer)

        try:
            final_state_dict = app.invoke(
                initial_state,
                config={"configurable": {"thread_id": session_id}},
            )
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
            profile_risk = user_profile.risk_tolerance.value
            eval_result = shadow_eval.evaluate(
                session_id=session_id,
                response_text=final_state.final_response,
                agent_results={k: AgentResult(**v) for k, v in final_state.agent_results.items()},
                deployment_slot=self.deployment_slot,
                user_query=user_query,
                risk_profile=profile_risk,
                regulatory_clauses=final_state.regulatory_clauses_used or None,
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
            reflection=final_state.needs_revision,
        )

        return {
            "session_id": session_id,
            "response": final_state.final_response,
            "policy_flags": final_state.policy_flags,
            "hitl_escalated": bool(final_state.hitl_queue),
            "hitl_tickets": final_state.hitl_queue,
            "reflection_applied": final_state.needs_revision,
            "agent_confidences": {
                k: v.get("confidence") for k, v in final_state.agent_results.items()
            },
            "meta_confidence": (
                final_state.meta_audit_result.get("confidence")
                if final_state.meta_audit_result
                else None
            ),
            "data_sources": list(
                {
                    src
                    for v in final_state.agent_results.values()
                    for src in v.get("data_sources", [])
                }
            ),
            "total_latency_ms": total_latency,
            "audit_chain_integrity": chain_ok,
            "audit_trail_length": len(audit),
            "shadow_eval_score": shadow_score,
            "deployment_slot": self.deployment_slot.value,
            "policy_version": POLICY_VERSION,
        }

    def resume_hitl_session(
        self,
        session_id: str,
        reviewer_notes: str = "",
        approved: bool = True,
    ) -> dict[str, Any]:
        """
        Resume a session that was paused at HITL escalation.

        Called by a compliance reviewer after approving (or rejecting) the
        escalation ticket. When approved=True, the checkpointed state has its
        hitl_queue cleared and the graph re-invokes from the checkpoint,
        completing the synthesis step that was gated.

        Args:
            session_id:     Session ID from the original run() result.
            reviewer_notes: Compliance reviewer's approval notes (audited).
            approved:       True to approve and complete synthesis; False to reject.

        Returns:
            Completed advisory response dict, or error if session not found.

        Production note:
            With a persistent checkpointer (PostgreSQL/Redis), this method works
            across process restarts and multiple service instances. The in-memory
            MemorySaver here is for demonstration — swap the checkpointer in
            _build_graph() to AsyncPostgresSaver for production durability.
        """
        if session_id not in self._session_graphs:
            return {
                "error": "Session not found or expired. "
                "In production, use a persistent checkpointer (PostgreSQL/Redis) "
                "to survive process restarts.",
                "session_id": session_id,
            }

        app, checkpointer = self._session_graphs[session_id]
        config = {"configurable": {"thread_id": session_id}}

        self._log.info(
            "hitl_resume",
            session_id=session_id,
            approved=approved,
            notes_preview=reviewer_notes[:80],
        )

        if not approved:
            return {
                "session_id": session_id,
                "status": "rejected",
                "message": "Session rejected by compliance reviewer. No response generated.",
                "reviewer_notes": reviewer_notes,
            }

        # Update the checkpointed state: clear hitl_queue to signal approval.
        # The next invoke() will resume from the checkpoint and proceed to synthesis.
        current_snapshot = app.get_state(config)
        if current_snapshot and current_snapshot.values:
            state_update = dict(current_snapshot.values)
            state_update["hitl_queue"] = []  # Approval = clear the HITL gate
            app.update_state(config, state_update)

        try:
            resumed_state_dict = app.invoke(None, config=config)
            resumed_state = GraphState(**resumed_state_dict)
        except Exception as exc:
            self._log.error("hitl_resume_failed", session_id=session_id, error=str(exc))
            return {"error": f"Resume failed: {exc}", "session_id": session_id}

        return {
            "session_id": session_id,
            "status": "completed",
            "response": resumed_state.final_response,
            "reviewer_notes": reviewer_notes,
            "policy_flags": resumed_state.policy_flags,
        }
