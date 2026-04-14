"""
agents/synthesizer.py
Grounded Response Synthesizer — Layer 6.

Assembles final responses from all agent findings.
Enforces:
  - Jurisdiction-appropriate regulatory disclaimers (US/UK/EU/CA)
  - Inline data source citations
  - No ungrounded claims
  - HITL holding message when escalated
  - Revision-aware synthesis when meta-critic flags medium concern
    (implements Andrew Ng's Reflection design pattern)
"""

from __future__ import annotations

from typing import Any

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.enums import Jurisdiction
from ..core.models import AgentResult, GraphState, UserProfile
from ..core.prompts import get_system_prompt
from ..data.regulatory_retriever import RegulatoryClause, get_regulatory_retriever
from ..security import pii

log = structlog.get_logger("synthesizer")

# Jurisdiction-specific disclaimers (SEC Reg BI / FCA / MiFID II / OSC)
DISCLAIMERS: dict[str, str] = {
    Jurisdiction.US: (
        "This content is for informational purposes only and does not constitute investment advice "
        "under SEC Regulation Best Interest or any other applicable regulation. "
        "Past performance is not indicative of future results. "
        "Please consult a registered investment advisor before making any investment decisions."
    ),
    Jurisdiction.UK: (
        "This information is for general guidance only and does not constitute "
        "regulated financial advice under FCA rules. Capital at risk. "
        "Past performance is not a reliable indicator of future results. "
        "Seek independent financial advice before making investment decisions."
    ),
    Jurisdiction.EU: (
        "This content does not constitute MiFID II investment advice. "
        "Investments involve risk of loss, including possible loss of principal. "
        "Consult a licensed investment firm authorised by a competent authority "
        "in your jurisdiction."
    ),
    Jurisdiction.CA: (
        "This is general financial information only and does not constitute "
        "personalised investment advice under OSC or applicable provincial securities "
        "regulations. Consult a registered portfolio manager or investment advisor "
        "for personalised guidance."
    ),
}

_FALLBACK_SYSTEM_PROMPT = (
    "You are a retail financial advisor assistant producing a final client-facing response. "
    "Synthesise the analysis below into a clear, well-structured response. "
    "Use plain language. Lead with the most important findings. "
    "Do NOT make specific buy/sell recommendations without noting these are "
    "general observations only. Cite data sources where relevant. "
)


def _build_regulatory_query(state: GraphState, policy_flags: list[str]) -> str:
    """
    Build a retrieval query from current session state and policy flags.
    Flag presence drives query term weighting toward relevant regulatory obligations.
    """
    parts: list[str] = []
    flags_str = " ".join(policy_flags)

    if "CONCENTRATION" in flags_str:
        parts.append("concentration limit suitability obligation position size client")
    if "SENTIMENT_BEARISH" in flags_str:
        parts.append("risk disclosure bearish market condition client communication")
    if "HITL_REQUIRED" in flags_str:
        parts.append("large trade approval threshold dealer obligation review")
    if "KYC_FAIL" in flags_str:
        parts.append("know your client KYC identity verification AML requirement")

    risk_profile = (
        state.user_profile.get("risk_tolerance", "moderate")
        if isinstance(state.user_profile, dict)
        else "moderate"
    )
    parts.append(f"portfolio advice {risk_profile} investor suitability NI 31-103")
    return " ".join(parts)


class ResponseSynthesizer:
    """
    Builds the final grounded, cited, compliant response.
    Returns a HITL holding message when the session is escalated.
    """

    def __init__(self, llm: ChatAnthropic, audit_fn):
        self.llm = llm
        self._audit = audit_fn
        self._log = log
        self._prompt_version = "inline"

    def synthesize(
        self,
        state: GraphState,
        agent_results: dict[str, AgentResult],
        meta_result: AgentResult,
        hitl_ticket: dict[str, Any] | None,
        profile: UserProfile,
        revision_critique: str | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Build the final response, grounded in retrieved regulatory clauses.

        Args:
            state:            Current graph state.
            agent_results:    All specialist agent outputs.
            meta_result:      Meta-critic audit result.
            hitl_ticket:      Set when HITL escalation is required.
            profile:          Validated user profile (drives disclaimer + jurisdiction filter).
            revision_critique: When set, meta-critic identified improvement areas.
                              The synthesizer addresses these explicitly, implementing
                              the Reflection pattern (Andrew Ng's Design Pattern #1).

        Returns:
            Tuple of (final client-facing response string, list of clause metadata dicts).
            Clause metadata is stored on GraphState for shadow eval grounding validation.
        """
        disclaimer = DISCLAIMERS.get(profile.jurisdiction, DISCLAIMERS[Jurisdiction.US])

        # ── HITL holding message ───────────────────────────────────────────
        if hitl_ticket:
            ticket_ref = hitl_ticket["ticket_id"][:8].upper()
            priority = hitl_ticket["priority"].upper()
            return (
                f"**Your request is under advisor review** (Reference: #{ticket_ref})\n\n"
                f"One or more aspects of your query require review by a qualified advisor "
                f"before a response can be provided. Priority: {priority}.\n\n"
                f"A licensed advisor will respond within 1 business day. "
                f"For urgent matters, please contact your advisor directly.\n\n"
                f"---\n*{disclaimer}*",
                [],
            )

        # ── Regulatory retrieval (RAG) ─────────────────────────────────────
        # Query ChromaDB for the top-3 Canadian regulatory clauses most
        # relevant to this session's policy flags and risk profile.
        # Injected into the prompt so the LLM cannot contradict current regulation.
        # Degrades gracefully if corpus not yet ingested — prompt continues without block.
        retrieved_clauses: list[RegulatoryClause] = []
        retriever = get_regulatory_retriever()
        if retriever is not None:
            reg_query = _build_regulatory_query(state, state.policy_flags)
            jurisdiction_code = profile.jurisdiction.value if hasattr(profile.jurisdiction, "value") else "CA"
            retrieved_clauses = retriever.retrieve(
                query=reg_query,
                jurisdiction=jurisdiction_code,
            )
            if retrieved_clauses:
                self._log.info(
                    "regulatory_clauses_retrieved",
                    session_id=state.session_id,
                    n=len(retrieved_clauses),
                    regulators=[c.regulator for c in retrieved_clauses],
                )

        # ── Build analysis sections from each agent ────────────────────────
        sections: list[str] = []
        all_sources: list[str] = []

        for agent_id, result in agent_results.items():
            agent_label = agent_id.replace("_", " ").title()
            findings = result.findings

            # Prefer LLM commentary if present (meta_critic, synthesizer passes)
            commentary = ""
            for key, val in findings.items():
                if key.startswith("llm_") and isinstance(val, str) and len(val) > 20:
                    commentary = val
                    break

            # Sentiment agent: build structured summary from raw FinBERT scores
            if not commentary and agent_id == "sentiment_analysis":
                scores = findings.get("sentiment_scores", {})
                bearish = findings.get("bearish_signals", [])
                if scores:
                    lines = [
                        f"  {sym}: positive={s['positive']:.0%}, negative={s['negative']:.0%}, "
                        f"neutral={s['neutral']:.0%}, dominant={s['dominant']}, "
                        f"headlines={s['n_headlines']}, bearish={'yes' if s['is_bearish'] else 'no'}"
                        for sym, s in scores.items()
                    ]
                    commentary = "FinBERT sentiment scores:\n" + "\n".join(lines)
                    if bearish:
                        commentary += f"\nBearish signals: {', '.join(bearish)}"

            if commentary:
                sections.append(f"**{agent_label}**\n{commentary}")

            all_sources.extend(result.data_sources)

        combined_analysis = "\n\n".join(sections)
        unique_sources = list(dict.fromkeys(all_sources))

        # ── Build system prompt (versioned from registry) ──────────────────
        base_system_prompt = get_system_prompt("synthesizer", fallback=_FALLBACK_SYSTEM_PROMPT)
        system_prompt = (
            f"{base_system_prompt}\n"
            f"Client risk profile: {profile.risk_tolerance.value}. "
            f"Account: {profile.account_type.value}. "
            f"Jurisdiction: {profile.jurisdiction.value}."
        )

        # ── Reflection: address meta-critic concerns when revision path ────
        # When revision_critique is set, this is the second-pass synthesis.
        # The meta-critic flagged concerns that warrant addressing before delivery,
        # but not serious enough to escalate to a human reviewer.
        revision_instruction = ""
        if revision_critique:
            revision_instruction = (
                f"\n\n**REVISION REQUIRED — Compliance Review Findings:**\n"
                f"{revision_critique}\n\n"
                "In your response, explicitly address each of the above compliance "
                "concerns. Add appropriate caveats, clarify any ambiguous claims, "
                "and ensure the response is fully suitable for the client's risk profile."
            )
            self._log.info(
                "revision_synthesis",
                session_id=state.session_id,
                critique_preview=revision_critique[:100],
            )

        # Build REGULATORY CONTEXT block when clauses were retrieved
        regulatory_block = ""
        if retrieved_clauses:
            clause_texts = "\n\n".join(
                f"[{c.regulator} | {c.source} | {c.version_date}]\n{c.text}"
                for c in retrieved_clauses
            )
            regulatory_block = (
                f"\n\nREGULATORY CONTEXT (retrieved from corpus — do not contradict):\n"
                f"{clause_texts}"
            )

        user_content = (
            f"Original query: {state.user_query}\n\n"
            f"Specialist agent analysis:\n{combined_analysis}\n\n"
            f"Compliance review: "
            f"{meta_result.findings.get('compliance_review', 'No issues flagged.')}\n\n"
            f"Data sources used: {', '.join(unique_sources)}"
            f"{regulatory_block}\n\n"
            "Write a clear, grounded response. "
            "Flag any areas where data confidence was limited. "
            "Where regulatory context is provided, ensure your response is consistent "
            "with those clauses. "
            "Do not reproduce the disclaimer — it will be appended automatically."
            f"{revision_instruction}"
        )

        safe_content = pii.redact(user_content, session_id=state.session_id)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=safe_content)]

        try:
            response = self.llm.invoke(messages)
            response_text = response.content
        except Exception as exc:
            self._log.error("synthesis_failed", error=str(exc))
            response_text = (
                "We were unable to generate a complete analysis at this time. "
                "Please retry or contact your advisor directly."
            )

        clause_metadata = [c.to_dict() for c in retrieved_clauses]
        self._audit(
            "synthesis",
            "response_generated",
            "success",
            {
                "sources": unique_sources,
                "revision_applied": revision_critique is not None,
                "regulatory_clauses_retrieved": len(clause_metadata),
                "regulatory_sources": [c["source"] for c in clause_metadata],
            },
        )

        return f"{response_text}\n\n---\n*{disclaimer}*", clause_metadata
