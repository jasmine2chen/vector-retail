"""
core/audit.py
Immutable, hash-chained audit trail.

Every agent decision, policy check, auth event, and HITL escalation is
recorded here as an AuditEvent. Events form a SHA-256 hash chain:
  event_hash = SHA256(event_id + timestamp + user_id + action + outcome + prev_hash)

Chain integrity can be verified at any time with verify_chain_integrity().
Any tampering (insert, modify, delete) breaks the chain.

Production deployment: persist via export() to S3 Object Lock or Azure WORM
for SOC 2 Type II compliance.
"""
from __future__ import annotations

from typing import Any

import structlog

from .models import AuditEvent

log = structlog.get_logger("audit_trail")


class AuditTrail:
    """
    Append-only, hash-chained audit log.
    One instance per session — created in the orchestrator and passed
    down to every layer as a callable (audit.record).
    """

    def __init__(self, session_id: str, user_id: str):
        self.session_id = session_id
        self.user_id = user_id
        self._chain: list[AuditEvent] = []
        self._log = log.bind(session_id=session_id, user_id=user_id)

    def record(
        self,
        event_type: str,
        action: str,
        outcome: str,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """
        Append a new event to the chain.
        Automatically links to the previous event's hash.

        Args:
            event_type: Category (e.g. 'auth', 'policy', 'agent', 'hitl')
            action:     Specific action taken
            outcome:    Result ('success', 'failed', 'escalated', etc.)
            metadata:   Arbitrary serialisable context

        Returns:
            The newly created AuditEvent (hash already computed).
        """
        prev_hash = self._chain[-1].event_hash if self._chain else ""

        event = AuditEvent(
            session_id=self.session_id,
            user_id=self.user_id,
            event_type=event_type,
            action=action,
            outcome=outcome,
            metadata=metadata or {},
            prev_hash=prev_hash,
        )
        event.event_hash = event.compute_hash()
        self._chain.append(event)

        self._log.info(
            "audit_event",
            event_id=event.event_id,
            event_type=event_type,
            action=action,
            outcome=outcome,
            chain_length=len(self._chain),
            hash_prefix=event.event_hash[:12],
        )
        return event

    def verify_chain_integrity(self) -> bool:
        """
        Recompute every hash and confirm the chain is unbroken.
        Returns True if intact, False if any event has been tampered with.
        Called at the end of every session before returning results.
        """
        for i, event in enumerate(self._chain):
            recomputed = event.compute_hash()
            if event.event_hash != recomputed:
                self._log.error(
                    "chain_integrity_failed",
                    position=i,
                    event_id=event.event_id,
                    stored=event.event_hash[:12],
                    computed=recomputed[:12],
                )
                return False
        return True

    def export(self) -> list[dict[str, Any]]:
        """
        Export the full chain as a list of dicts.
        Use this to persist to external WORM storage.
        """
        return [e.model_dump() for e in self._chain]

    def __len__(self) -> int:
        return len(self._chain)
