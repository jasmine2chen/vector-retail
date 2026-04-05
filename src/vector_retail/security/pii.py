"""
security/pii.py
PII Redaction Layer.

Intercepts ALL text before it is passed to an LLM call.
Patterns cover: SSN, TIN, email, US phone, major card formats, IBAN,
16-digit account numbers.

In production, augment with Microsoft Presidio for named-entity PII
(person names, addresses, DOBs) beyond regex coverage.

OWASP LLM Top 10 — LLM06: Sensitive Information Disclosure mitigation.
"""
from __future__ import annotations

import re

import structlog

log = structlog.get_logger("pii_redactor")

# ── Compiled patterns ──────────────────────────────────────────────────────────
# Order matters: more-specific patterns first to avoid partial matches.

_PATTERNS: list[tuple[re.Pattern, str]] = [
    # US Social Security Number
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN-REDACTED]"),
    # US Employer Identification Number (EIN: XX-XXXXXXX format)
    (re.compile(r"\b\d{2}-\d{7}\b"), "[TIN-REDACTED]"),
    # Email addresses
    (re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    ), "[EMAIL-REDACTED]"),
    # US phone numbers (multiple formats)
    (re.compile(
        r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ), "[PHONE-REDACTED]"),
    # Visa / Mastercard / Amex card numbers
    (re.compile(
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"
    ), "[CARD-REDACTED]"),
    # IBAN (international bank account number)
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b"), "[IBAN-REDACTED]"),
    # Generic 16-digit account numbers (with optional separators)
    (re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"), "[ACCT-REDACTED]"),
]


def redact(text: str, session_id: str = "") -> str:
    """
    Redact all recognised PII patterns from text.

    Args:
        text:       Raw user input or data string
        session_id: Optional session ID for audit logging

    Returns:
        Text with PII replaced by placeholder tokens.
    """
    _log = log.bind(session_id=session_id)
    redacted = text
    redactions: list[str] = []

    for pattern, replacement in _PATTERNS:
        new_text, count = re.subn(pattern, replacement, redacted)
        if count:
            redactions.append(f"{replacement}x{count}")
        redacted = new_text

    if redactions:
        _log.info("pii_redacted", tokens=redactions, char_delta=len(text) - len(redacted))

    return redacted


def has_pii(text: str) -> bool:
    """Returns True if text contains any detectable PII."""
    return any(pattern.search(text) for pattern, _ in _PATTERNS)
