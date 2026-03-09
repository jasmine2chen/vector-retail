"""
agents/base.py
Base class for all parallel finance agents.

Every specialist agent inherits from BaseFinanceAgent and must implement run().
The base class provides:
  - LLM call with automatic PII redaction
  - Structured latency tracking
  - Error handling that never crashes the graph
  - Auditable confidence scoring via ConfidenceCalculator
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.models import AgentResult, GraphState
from ..core.policy import POLICY_RULES
from ..security import pii

if TYPE_CHECKING:
    from ..core.audit import AuditTrail

log = structlog.get_logger("base_agent")

# ── Confidence penalties from policy config ──────────────────────────────
_CONFIDENCE_PENALTIES: Dict[str, float] = POLICY_RULES.get("confidence_penalties", {})


class ConfidenceCalculator:
    """
    Auditable confidence scoring for production financial systems.

    Each specialist agent creates one per run(). Penalties are applied via
    penalize(), and the final score is computed + written to the audit trail
    via score().  This gives compliance a full chain of custody for every
    confidence decision.

    Penalties are multiplicative: base × p1 × p2 × ... × pN
    Penalty weights are loaded from config/policy_rules.json so compliance
    can tune sensitivity without a code deploy.
    """

    def __init__(self, agent_id: str, audit_fn, base: float = 0.95):
        self._agent_id = agent_id
        self._audit = audit_fn
        self._base = base
        self._penalties: List[Dict[str, Any]] = []

    def penalize(self, signal: str, reason: str, observed: Any = None) -> None:
        """
        Apply a configurable multiplicative penalty.

        Args:
            signal:   Key into confidence_penalties config (e.g. 'stale_quote').
            reason:   Human-readable explanation for the audit trail.
            observed: Optional observed value that triggered the penalty.
        """
        factor = _CONFIDENCE_PENALTIES.get(signal, 0.90)  # conservative default
        self._penalties.append({
            "signal": signal,
            "observed_value": observed,
            "penalty_factor": factor,
            "reason": reason,
        })

    def score(self) -> float:
        """
        Compute final confidence and audit the full penalty chain.

        Returns:
            Clamped [0.0, 1.0] confidence score.
        """
        result = self._base
        for p in self._penalties:
            result *= p["penalty_factor"]
        result = round(max(0.0, min(1.0, result)), 4)

        self._audit(
            "confidence", f"{self._agent_id}_score", "computed",
            {
                "final_score": result,
                "base": self._base,
                "penalty_count": len(self._penalties),
                "penalties": self._penalties,
            },
        )
        return result


class BaseFinanceAgent:
    """
    Template for all specialist agents in the parallel mesh.

    Subclasses must define:
        AGENT_ID   : str  — unique identifier used in audit trail and graph state
        run()      : GraphState -> AgentResult
    """

    AGENT_ID: str = "base"
    AGENT_VERSION: str = "2.0.0"

    def __init__(self, llm: ChatAnthropic, audit: "AuditTrail"):
        self.llm = llm
        self.audit = audit
        self._log = log.bind(agent=self.AGENT_ID)

    def _call_llm(self, system_prompt: str, user_content: str) -> str:
        """
        Invoke the LLM with PII-redacted content.
        Returns the response text, or an error string on failure
        (errors never propagate out — callers handle gracefully).
        """
        safe_content = pii.redact(user_content, session_id=self.audit.session_id)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=safe_content),
        ]
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as exc:
            self._log.error("llm_call_failed", error=str(exc))
            self.audit.record("agent", f"{self.AGENT_ID}_llm_call", "failed", {"error": str(exc)})
            return f"[{self.AGENT_ID}] Analysis temporarily unavailable: {exc}"

    @staticmethod
    def _is_llm_error(response: str) -> bool:
        """Check whether an LLM response is actually an error fallback."""
        return response.startswith("[") and "temporarily unavailable" in response

    def _confidence(self) -> ConfidenceCalculator:
        """Create a new auditable confidence calculator for this agent run."""
        return ConfidenceCalculator(self.AGENT_ID, self.audit.record)

    def run(self, state: GraphState) -> AgentResult:
        """
        Execute this agent's analysis.
        Must be overridden by every subclass.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement run()")

