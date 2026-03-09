"""
agents/synthesizer.py
Grounded Response Synthesizer — Layer 6.

Assembles final responses from all agent findings.
Enforces:
  - Jurisdiction-appropriate regulatory disclaimers (US/UK/EU/CA)
  - Inline data source citations
  - No ungrounded claims
  - HITL holding message when escalated
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.enums import Jurisdiction
from ..core.models import AgentResult, GraphState, UserProfile
from ..security import pii

log = structlog.get_logger("synthesizer")

# Jurisdiction-specific disclaimers (SEC Reg BI / FCA / MiFID II / OSC)
DISCLAIMERS: Dict[str, str] = {
    Jurisdiction.US: (
        "This content is for informational purposes only and does not constitute investment advice "
        "under SEC Regulation Best Interest or any other applicable regulation. "
        "Past performance is not indicative of future results. "
        "Please consult a registered investment advisor before making any investment decisions."
    ),
    Jurisdiction.UK: (
        "This information is for general guidance only and does not constitute regulated financial advice "
        "under FCA rules. Capital at risk. Past performance is not a reliable indicator of future results. "
        "Seek independent financial advice before making investment decisions."
    ),
    Jurisdiction.EU: (
        "This content does not constitute MiFID II investment advice. "
        "Investments involve risk of loss, including possible loss of principal. "
        "Consult a licensed investment firm authorised by a competent authority in your jurisdiction."
    ),
    Jurisdiction.CA: (
        "This is general financial information only and does not constitute personalised investment advice "
        "under OSC or applicable provincial securities regulations. "
        "Consult a registered portfolio manager or investment advisor for personalised guidance."
    ),
}


class ResponseSynthesizer:
    """
    Builds the final grounded, cited, compliant response.
    Returns a HITL holding message when the session is escalated.
    """

    def __init__(self, llm: ChatAnthropic, audit_fn):
        self.llm = llm
        self._audit = audit_fn
        self._log = log

    def synthesize(
        self,
        state: GraphState,
        agent_results: Dict[str, AgentResult],
        meta_result: AgentResult,
        hitl_ticket: Optional[Dict[str, Any]],
        profile: UserProfile,
    ) -> str:
        """
        Build the final response.

        If HITL escalated: return holding message with ticket reference.
        Otherwise: synthesise grounded response with citations and disclaimer.
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
                f"---\n*{disclaimer}*"
            )

        # ── Extract LLM commentary from each agent ─────────────────────────
        sections: List[str] = []
        all_sources: List[str] = []

        for agent_id, result in agent_results.items():
            agent_label = agent_id.replace("_", " ").title()
            findings = result.findings

            # Pull the primary LLM output from each agent
            commentary = ""
            for key, val in findings.items():
                if key.startswith("llm_") and isinstance(val, str) and len(val) > 20:
                    commentary = val
                    break

            if commentary:
                sections.append(f"**{agent_label}**\n{commentary}")

            all_sources.extend(result.data_sources)

        combined_analysis = "\n\n".join(sections)
        unique_sources = list(dict.fromkeys(all_sources))

        # ── Final LLM synthesis ────────────────────────────────────────────
        system_prompt = (
            "You are a retail financial advisor assistant producing a final client-facing response. "
            "Synthesise the analysis below into a clear, well-structured response. "
            "Use plain language. Lead with the most important findings. "
            "Do NOT make specific buy/sell recommendations without noting these are general observations only. "
            "Cite data sources where relevant. "
            f"Client risk profile: {profile.risk_tolerance.value}. "
            f"Account: {profile.account_type.value}. "
            f"Jurisdiction: {profile.jurisdiction.value}."
        )

        user_content = (
            f"Original query: {state.user_query}\n\n"
            f"Specialist agent analysis:\n{combined_analysis}\n\n"
            f"Compliance review: "
            f"{meta_result.findings.get('compliance_review', 'No issues flagged.')}\n\n"
            f"Data sources used: {', '.join(unique_sources)}\n\n"
            "Write a clear, grounded response. "
            "Flag any areas where data confidence was limited. "
            "Do not reproduce the disclaimer — it will be appended automatically."
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

        self._audit("synthesis", "response_generated", "success", {"sources": unique_sources})

        return f"{response_text}\n\n---\n*{disclaimer}*"
