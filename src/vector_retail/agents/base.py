"""
agents/base.py
Base class for all parallel finance agents.

Every specialist agent inherits from BaseFinanceAgent and must implement run().
The base class provides:
  - LLM call with automatic PII redaction
  - Prompt injection defense (OWASP LLM Top 10 #1)
  - LangFuse observability: per-call traces with latency, model, confidence
  - Structured latency tracking
  - Error handling that never crashes the graph
  - Auditable confidence scoring via ConfidenceCalculator
  - Versioned prompts via PromptRegistry
"""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.models import AgentResult, GraphState
from ..core.policy import POLICY_RULES
from ..core.prompts import get_prompt_version
from ..security import pii
from ..security import prompt_guard

if TYPE_CHECKING:
    from ..core.audit import AuditTrail

log = structlog.get_logger("base_agent")

# ── Confidence penalties from policy config ──────────────────────────────
_CONFIDENCE_PENALTIES: dict[str, float] = POLICY_RULES.get("confidence_penalties", {})

# ── LangFuse client (no-op singleton if not configured) ──────────────────
# Enable by setting LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY env vars.
# All LLM calls are automatically traced with agent_id, model, latency,
# input/output tokens, and confidence score as a scored evaluation.
_langfuse: Any = None
try:
    from langfuse import Langfuse  # type: ignore[import-untyped]

    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        _langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        log.info("langfuse_enabled", host=os.getenv("LANGFUSE_HOST", "cloud"))
    else:
        log.debug("langfuse_disabled", reason="LANGFUSE_PUBLIC_KEY not set")
except ImportError:
    log.debug("langfuse_not_installed", hint="pip install langfuse to enable tracing")


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
        self._penalties: list[dict[str, Any]] = []

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

    def score(self, langfuse_trace: Any = None) -> float:
        """
        Compute final confidence, audit the full penalty chain, and optionally
        emit a LangFuse score so confidence is visible in the observability dashboard.

        Args:
            langfuse_trace: Optional LangFuse trace object to attach the score to.

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

        # Emit confidence as a LangFuse score for dashboard visibility
        if langfuse_trace and _langfuse:
            try:
                langfuse_trace.score(
                    name="agent_confidence",
                    value=result,
                    comment=f"{len(self._penalties)} penalties applied",
                )
            except Exception:
                pass  # Observability failure must never affect the main flow

        return result


class BaseFinanceAgent:
    """
    Template for all specialist agents in the parallel mesh.

    Subclasses must define:
        AGENT_ID   : str  — unique identifier used in audit trail and graph state
        run()      : GraphState -> AgentResult

    Subclasses may override PROMPT_FALLBACK to provide an inline default prompt
    used when the agent's config/prompts/<AGENT_ID>.yaml file is absent.
    """

    AGENT_ID: str = "base"
    AGENT_VERSION: str = "2.0.0"
    PROMPT_FALLBACK: str = ""  # Inline default; overridden per agent

    def __init__(self, llm: ChatAnthropic, audit: AuditTrail):
        self.llm = llm
        self.audit = audit
        self._log = log.bind(agent=self.AGENT_ID)
        self._prompt_version = get_prompt_version(self.AGENT_ID)

    def _call_llm(
        self,
        system_prompt: str,
        user_content: str,
        langfuse_trace: Any = None,
    ) -> str:
        """
        Invoke the LLM with:
          1. Prompt injection scan (OWASP LLM Top 10 #1)
          2. PII redaction (OWASP LLM Top 10 LLM06)
          3. LangFuse generation trace (latency, model, input/output)
          4. Graceful error handling (never propagates exceptions)

        Args:
            system_prompt:   System-level instructions for the LLM.
            user_content:    User/data content to analyse (will be PII-redacted).
            langfuse_trace:  Optional LangFuse trace to nest this generation under.

        Returns:
            LLM response text, or a safe error string on failure.
        """
        # ── 1. Prompt injection scan ───────────────────────────────────────
        injection_result = prompt_guard.scan(user_content, session_id=self.audit.session_id)
        if not injection_result.is_safe:
            self.audit.record(
                "security",
                f"{self.AGENT_ID}_injection_scan",
                "flagged",
                {
                    "categories": injection_result.categories,
                    "severities": injection_result.severities,
                },
            )
            self._log.warning(
                "prompt_injection_detected",
                categories=injection_result.categories,
                critical=injection_result.critical,
            )
            # For critical injections, return a safe refusal rather than proceeding.
            # For warnings, proceed with a sanitised version but flag it.
            if injection_result.critical:
                return (
                    f"[{self.AGENT_ID}] Input flagged for security review. "
                    "Analysis cannot be completed for this query."
                )

        # ── 2. PII redaction ───────────────────────────────────────────────
        safe_content = pii.redact(user_content, session_id=self.audit.session_id)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=safe_content),
        ]

        # ── 3. LangFuse generation trace ───────────────────────────────────
        generation = None
        if _langfuse:
            try:
                trace = langfuse_trace or _langfuse.trace(
                    name=f"agent.{self.AGENT_ID}",
                    session_id=self.audit.session_id,
                    metadata={"agent_version": self.AGENT_VERSION},
                )
                generation = trace.generation(
                    name=f"{self.AGENT_ID}.llm_call",
                    model=getattr(self.llm, "model", "claude-sonnet-4-20250514"),
                    model_parameters={"temperature": getattr(self.llm, "temperature", 0.1)},
                    input=[
                        {"role": "system", "content": system_prompt[:500]},
                        {"role": "user", "content": safe_content[:500]},
                    ],
                    metadata={
                        "prompt_version": self._prompt_version,
                        "agent_id": self.AGENT_ID,
                    },
                )
            except Exception:
                generation = None  # Tracing failure must never block the pipeline

        # ── 4. LLM invocation ──────────────────────────────────────────────
        t0 = time.time()
        try:
            response = self.llm.invoke(messages)
            latency_ms = round((time.time() - t0) * 1000)

            if generation:
                try:
                    generation.end(
                        output=response.content[:1000],
                        usage={
                            "input": getattr(response, "usage_metadata", {}).get(
                                "input_tokens", 0
                            ),
                            "output": getattr(response, "usage_metadata", {}).get(
                                "output_tokens", 0
                            ),
                        },
                        metadata={"latency_ms": latency_ms},
                    )
                except Exception:
                    pass

            return response.content

        except Exception as exc:
            latency_ms = round((time.time() - t0) * 1000)
            if generation:
                try:
                    generation.end(level="ERROR", status_message=str(exc))
                except Exception:
                    pass
            self._log.error("llm_call_failed", error=str(exc), latency_ms=latency_ms)
            self.audit.record(
                "agent", f"{self.AGENT_ID}_llm_call", "failed", {"error": str(exc)}
            )
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
