"""
agents/meta_critic.py
Self-Reflective Meta-Critic — Layer 4.

Audits ALL parallel agent outputs BEFORE any response is delivered.
This is the key differentiator from a naive agent pipeline.

Checks performed:
  1. Confidence audit — flags agents with confidence < 0.70
  2. Policy flag sweep — aggregates all policy flags from all agents
  3. Hallucination signal detection — patterns suggesting ungrounded specifics
  4. Data verification check — are all quotes dual-source verified?
  5. HITL escalation check — any agent flagged requires_hitl?
  6. Self-reflection LLM pass — compliance officer reviews all outputs

If overall_confidence < min_confidence_for_auto_response (0.75 default),
the session is routed to the HITL gate regardless of other flags.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List

import structlog

from ..core.models import AgentResult, GraphState
from ..core.policy import POLICY_RULES
from .base import BaseFinanceAgent

log = structlog.get_logger("agent.meta_critic")


class MetaCriticAgent(BaseFinanceAgent):
    AGENT_ID = "meta_critic"

    def run(self, state: GraphState) -> AgentResult:
        t0 = time.time()
        reasoning: List[str] = []
        results = state.agent_results
        flags: List[str] = []
        overall_confidence = 1.0

        # ── 1. Confidence audit ────────────────────────────────────────────
        confidences = {k: v.get("confidence", 0.0) for k, v in results.items()}
        low_conf = [k for k, c in confidences.items() if c < 0.70]
        if low_conf:
            flags.append(f"LOW_CONFIDENCE: agents {', '.join(low_conf)}")
            overall_confidence *= 0.85
        reasoning.append(f"Confidence scores: {confidences}")

        # ── 2. Policy flag sweep ───────────────────────────────────────────
        all_policy_flags: List[str] = []
        for agent_id, result in results.items():
            agent_flags = result.get("policy_flags", [])
            all_policy_flags.extend(agent_flags)

        if all_policy_flags:
            flags.extend(all_policy_flags)
            reasoning.append(f"Policy flags from agents: {len(all_policy_flags)}")

        # ── 3. Hallucination signal detection ─────────────────────────────
        # Heuristic: LLM outputs with > 5 specific dollar amounts without
        # clear data grounding may indicate hallucination.
        hallucination_signals: List[str] = []
        for agent_id, result in results.items():
            for key, val in result.get("findings", {}).items():
                if isinstance(val, str) and len(val) > 50:
                    dollar_matches = re.findall(r"\$[\d,]+(?:\.\d{2})?", val)
                    if len(dollar_matches) > 5:
                        hallucination_signals.append(
                            f"{agent_id}.{key}: {len(dollar_matches)} specific dollar figures — verify grounding"
                        )

        if hallucination_signals:
            flags.append(f"HALLUCINATION_RISK: {'; '.join(hallucination_signals)}")
            overall_confidence *= 0.90
        reasoning.append(f"Hallucination signals detected: {len(hallucination_signals)}")

        # ── 4. Data verification check ────────────────────────────────────
        unverified = any(
            not state.quotes.get(sym, {}).get("is_verified", False)
            for sym in state.quotes
        )
        if unverified:
            flags.append("DATA_QUALITY: One or more prices not dual-source verified")
            overall_confidence *= 0.92
        reasoning.append(f"Quote verification: {'partial' if unverified else 'all verified'}")

        # ── 5. HITL escalation check ───────────────────────────────────────
        hitl_from_agents = any(r.get("requires_hitl", False) for r in results.values())
        if hitl_from_agents:
            flags.append("HITL_ESCALATION: Agent-level HITL flag detected")
            overall_confidence *= 0.95

        # ── 6. Self-reflection LLM pass ────────────────────────────────────
        agent_summaries = "\n".join(
            f"  [{k}] confidence={v.get('confidence', 0):.0%}, "
            f"flags={v.get('policy_flags', [])}, "
            f"top_rec={v.get('recommendations', ['none'])[:1]}"
            for k, v in results.items()
        )

        llm_critique = self._call_llm(
            system_prompt=(
                "You are a senior compliance officer reviewing AI-generated financial analysis. "
                "Your job: identify inconsistencies, unsupported claims, or regulatory concerns. "
                "Be direct and concise. If outputs appear consistent and policy-compliant, confirm this. "
                "Flag anything that could expose the firm to regulatory risk."
            ),
            user_content=(
                f"User query: {state.user_query}\n\n"
                f"Agent output summary:\n{agent_summaries}\n\n"
                f"Detected flags: {flags}\n\n"
                "In 3-4 sentences: Are these outputs internally consistent? "
                "Any compliance concerns? Should any content be withheld pending human review?"
            ),
        )

        reasoning.append("Meta-critic self-reflection LLM pass complete")
        overall_confidence = max(0.0, min(1.0, overall_confidence))
        min_conf = POLICY_RULES["min_confidence_for_auto_response"]
        requires_hitl = hitl_from_agents or overall_confidence < min_conf

        self.audit.record(
            "meta_critic", "plan_audit", "completed",
            {
                "overall_confidence": round(overall_confidence, 4),
                "flags_count": len(flags),
                "hallucination_signals": len(hallucination_signals),
                "requires_hitl": requires_hitl,
            },
        )

        return AgentResult(
            agent_id=self.AGENT_ID,
            confidence=overall_confidence,
            reasoning_chain=reasoning,
            findings={
                "compliance_review": llm_critique,
                "detected_flags": flags,
                "hallucination_signals": hallucination_signals,
                "agent_confidences": confidences,
                "data_fully_verified": not unverified,
            },
            policy_flags=flags,
            requires_hitl=requires_hitl,
            hitl_reason=(
                f"Overall confidence {overall_confidence:.0%} below {min_conf:.0%} threshold"
                if overall_confidence < min_conf
                else ("Agent-level HITL flag" if hitl_from_agents else None)
            ),
            latency_ms=round((time.time() - t0) * 1000),
        )
