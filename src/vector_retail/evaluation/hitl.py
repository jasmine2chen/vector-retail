"""
evaluation/hitl.py
Human-in-the-Loop Gate — Layer 5.

Evaluates whether a session requires human review before responding.
Creates and queues HITL tickets with priority scoring.

Production integration points:
  - Replace _queue with a real ticket system (ServiceNow, PagerDuty, Zendesk)
  - Add SLA timers and escalation logic
  - Expose ticket status endpoint for client polling
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import structlog

from ..core.enums import HITLPriority
from ..core.models import AgentResult, GraphState
from ..core.policy import POLICY_RULES

log = structlog.get_logger("hitl_gate")


class HITLGate:
    """
    Evaluates HITL necessity and creates escalation tickets.
    One instance per session — queue is session-scoped.
    """

    def __init__(self, audit_fn):
        self._audit = audit_fn
        self._queue: list[dict[str, Any]] = []
        self._log = log

    def evaluate(
        self, state: GraphState, meta_result: AgentResult
    ) -> tuple[bool, HITLPriority]:
        """
        Determine if HITL is required and assign a priority level.

        Priority logic:
          CRITICAL — HITL_REQUIRED flag + confidence < 0.70
          HIGH     — HITL_REQUIRED flag present
          MEDIUM   — confidence below auto-response threshold
          LOW      — DATA_QUALITY flag only
        """
        flags = " ".join(meta_result.policy_flags)
        confidence = meta_result.confidence
        min_conf = POLICY_RULES["min_confidence_for_auto_response"]

        if "HITL_REQUIRED" in flags and confidence < 0.70:
            return True, HITLPriority.CRITICAL
        if "HITL_REQUIRED" in flags:
            return True, HITLPriority.HIGH
        if confidence < min_conf:
            return True, HITLPriority.MEDIUM
        if "DATA_QUALITY" in flags:
            return True, HITLPriority.LOW

        return False, HITLPriority.LOW

    def escalate(
        self,
        state: GraphState,
        meta_result: AgentResult,
        priority: HITLPriority,
    ) -> dict[str, Any]:
        """
        Create and queue a HITL ticket.
        In production, this would POST to your ticketing system.
        """
        ticket: dict[str, Any] = {
            "ticket_id": str(uuid.uuid4()),
            "session_id": state.session_id,
            "priority": priority.value,
            "user_query": state.user_query,
            "flags": meta_result.policy_flags,
            "confidence": meta_result.confidence,
            "assigned_to": "compliance_queue",
            "status": "pending_review",
            "sla_hours": {"critical": 2, "high": 4, "medium": 8, "low": 24}[priority.value],
            "created_at": datetime.now(UTC).isoformat(),
        }

        self._queue.append(ticket)

        self._log.info(
            "hitl_ticket_created",
            ticket_id=ticket["ticket_id"][:8],
            priority=priority.value,
            sla_hours=ticket["sla_hours"],
        )
        self._audit(
            "hitl", "escalate", f"ticket_{ticket['ticket_id'][:8]}",
            {"priority": priority.value, "flags": meta_result.policy_flags},
        )
        return ticket

    @property
    def queue(self) -> list[dict[str, Any]]:
        return list(self._queue)
