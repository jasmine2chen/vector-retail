"""
security/prompt_guard.py
Prompt Injection Defense — OWASP LLM Top 10 #1 Mitigation.

Intercepts all user-supplied text BEFORE it reaches any LLM call.
Detects instruction-override, persona-hijack, prompt-extraction, and
template-injection attack patterns.

Architecture note: This is a heuristic (pattern-based) first-line defense.
For production systems handling adversarial users, augment with a lightweight
LLM classifier (e.g. a fine-tuned DistilBERT or a fast Haiku call) as a
second-pass guard — accuracy > pure regex, latency ~50ms.

References:
  OWASP LLM Top 10 2025 — LLM01: Prompt Injection
  https://owasp.org/www-project-top-10-for-large-language-model-applications/
"""
from __future__ import annotations

import re

import structlog

log = structlog.get_logger("prompt_guard")

# ── Injection pattern registry ────────────────────────────────────────────────
# Each entry: (compiled_regex, category_label, severity)
# Severity: "critical" = block-worthy, "warning" = flag and log

_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # Instruction override attacks
    (
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass|skip)\b.{0,40}"
            r"\b(previous|prior|all|above|earlier)\b.{0,40}"
            r"\b(instructions?|prompts?|context|rules?|constraints?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "instruction_override",
        "critical",
    ),
    # Direct system prompt extraction
    (
        re.compile(
            r"\b(repeat|print|output|reveal|show|display|return|tell me|what is)\b"
            r".{0,40}\b(system|initial|original|full|your)\b.{0,20}\bprompt\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "prompt_extraction",
        "critical",
    ),
    # Persona hijacking — "you are now X", "act as X"
    (
        re.compile(
            r"\b(you are now|pretend (you are|to be)|act as|roleplay as|"
            r"simulate being|switch to|become)\b",
            re.IGNORECASE,
        ),
        "persona_hijack",
        "critical",
    ),
    # Context wipe attempts
    (
        re.compile(
            r"\b(forget|clear|reset|wipe|delete|erase)\b.{0,30}"
            r"\b(everything|all|prior|previous|above|context|memory|instructions?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "context_wipe",
        "critical",
    ),
    # XML/tag injection (common in RAG and tool-use pipelines)
    (
        re.compile(
            r"<\s*(system|instruction|prompt|context|tool_call|function_call"
            r"|assistant|human|im_start|im_end)\s*[>/]",
            re.IGNORECASE,
        ),
        "xml_injection",
        "critical",
    ),
    # LLM template token injection (Llama/Mistral/ChatML special tokens)
    (
        re.compile(
            r"(\[INST\]|\[\/INST\]|<\|im_start\|>|<\|im_end\|>"
            r"|\[SYS\]|\[\/SYS\]|<<SYS>>|<</SYS>>)",
            re.IGNORECASE,
        ),
        "template_token_injection",
        "critical",
    ),
    # Known jailbreak phrases
    (
        re.compile(
            r"\b(jailbreak|DAN mode|developer mode|god mode|unrestricted mode"
            r"|no restrictions|no limits|no rules)\b",
            re.IGNORECASE,
        ),
        "jailbreak_attempt",
        "critical",
    ),
    # Indirect injection via data payloads (common in agentic RAG scenarios)
    (
        re.compile(
            r"(when you read this|if you are an? (ai|llm|assistant)|"
            r"note to (ai|llm|model|assistant))",
            re.IGNORECASE,
        ),
        "indirect_injection",
        "warning",
    ),
]


class InjectionScanResult:
    """
    Result of a prompt injection scan.

    Attributes:
        is_safe:    True if no injection patterns detected.
        categories: List of detected pattern categories.
        severities: Corresponding severity levels per detection.
        critical:   True if any critical-severity pattern was found.
    """

    __slots__ = ("is_safe", "categories", "severities", "critical")

    def __init__(
        self,
        categories: list[str],
        severities: list[str],
    ) -> None:
        self.categories = categories
        self.severities = severities
        self.is_safe = len(categories) == 0
        self.critical = "critical" in severities

    def as_policy_flag(self) -> str | None:
        """Return a policy flag string suitable for AgentResult.policy_flags."""
        if self.is_safe:
            return None
        cats = ", ".join(self.categories)
        level = "CRITICAL" if self.critical else "WARNING"
        return f"PROMPT_INJECTION_{level}: {cats}"


def scan(text: str, session_id: str = "") -> InjectionScanResult:
    """
    Scan text for prompt injection patterns.

    Args:
        text:       User-supplied input to scan.
        session_id: Session identifier for audit logging.

    Returns:
        InjectionScanResult — check `.is_safe` before proceeding.

    Usage in agents (base.py _call_llm):
        result = prompt_guard.scan(user_content, session_id=self.audit.session_id)
        if not result.is_safe:
            self.audit.record("security", "injection_scan", "flagged",
                              {"categories": result.categories})
            # Still process (don't silently drop) — log and continue.
            # For critical injections, callers may choose to block or flag.
    """
    _log = log.bind(session_id=session_id)
    detected_categories: list[str] = []
    detected_severities: list[str] = []

    for pattern, category, severity in _PATTERNS:
        if pattern.search(text):
            detected_categories.append(category)
            detected_severities.append(severity)

    result = InjectionScanResult(detected_categories, detected_severities)

    if not result.is_safe:
        _log.warning(
            "prompt_injection_detected",
            categories=result.categories,
            severities=result.severities,
            critical=result.critical,
            text_preview=text[:120].replace("\n", " "),
        )

    return result
