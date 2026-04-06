"""
tests/unit/test_audit.py
Unit tests for the immutable hash-chained audit trail.
"""

from vector_retail.core.audit import AuditTrail


class TestAuditTrail:

    def _make_trail(self) -> AuditTrail:
        return AuditTrail(session_id="sess-test-001", user_id="user-test-001")

    def test_record_appends_event(self):
        trail = self._make_trail()
        trail.record("auth", "login", "success")
        assert len(trail) == 1

    def test_event_has_hash(self):
        trail = self._make_trail()
        event = trail.record("auth", "login", "success")
        assert len(event.event_hash) == 64  # SHA-256 hex digest

    def test_chain_links_correctly(self):
        trail = self._make_trail()
        e1 = trail.record("auth", "login", "success")
        e2 = trail.record("policy", "check", "passed")
        assert e2.prev_hash == e1.event_hash

    def test_first_event_has_empty_prev_hash(self):
        trail = self._make_trail()
        e1 = trail.record("auth", "login", "success")
        assert e1.prev_hash == ""

    def test_chain_integrity_valid(self):
        trail = self._make_trail()
        trail.record("auth", "login", "success")
        trail.record("agent", "run", "completed")
        trail.record("synthesis", "generate", "success")
        assert trail.verify_chain_integrity() is True

    def test_tamper_detection(self):
        trail = self._make_trail()
        trail.record("auth", "login", "success")
        trail.record("policy", "check", "passed")
        # Tamper with the first event
        trail._chain[0].event_hash = "tampered_" + "x" * 55
        assert trail.verify_chain_integrity() is False

    def test_export_returns_list_of_dicts(self):
        trail = self._make_trail()
        trail.record("auth", "login", "success")
        trail.record("agent", "run", "completed")
        exported = trail.export()
        assert isinstance(exported, list)
        assert len(exported) == 2
        assert all(isinstance(e, dict) for e in exported)
        assert all("event_hash" in e for e in exported)

    def test_metadata_stored(self):
        trail = self._make_trail()
        event = trail.record("policy", "check", "passed", {"value": 12345})
        assert event.metadata["value"] == 12345
