"""
tests/unit/test_pii.py
Unit tests for PII redaction layer.
"""
from vector_retail.security.pii import redact, has_pii


class TestPIIRedaction:

    def test_ssn_redacted(self):
        assert redact("My SSN is 123-45-6789") == "My SSN is [SSN-REDACTED]"

    def test_email_redacted(self):
        result = redact("Contact user@example.com for details")
        assert "[EMAIL-REDACTED]" in result
        assert "user@example.com" not in result

    def test_phone_redacted(self):
        result = redact("Call 555-867-5309 now")
        assert "[PHONE-REDACTED]" in result
        assert "555-867-5309" not in result

    def test_no_pii_unchanged(self):
        text = "No personal information here, just market analysis."
        assert redact(text) == text

    def test_multiple_pii_types(self):
        text = "SSN: 123-45-6789, email: a@b.com, phone: 555-123-4567"
        result = redact(text)
        assert "123-45-6789" not in result
        assert "a@b.com" not in result
        assert "555-123-4567" not in result

    def test_has_pii_true(self):
        assert has_pii("My email is test@example.com") is True

    def test_has_pii_false(self):
        assert has_pii("Normal market analysis text") is False

    def test_empty_string(self):
        assert redact("") == ""

    def test_session_id_param(self):
        # Should not raise with session_id provided
        result = redact("test@example.com", session_id="sess-123")
        assert "[EMAIL-REDACTED]" in result
