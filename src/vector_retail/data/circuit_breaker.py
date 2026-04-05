"""
data/circuit_breaker.py
Circuit Breaker pattern for external API calls.

States:
  CLOSED  — normal operation, requests pass through
  OPEN    — too many failures, requests blocked, cooldown timer running
  HALF-OPEN — cooldown expired, next request is a probe

Prevents cascading failures when market data APIs are degraded.
"""
from __future__ import annotations

import time

import structlog

log = structlog.get_logger("circuit_breaker")


class CircuitBreaker:
    """
    Simple three-state circuit breaker.

    Args:
        name:             Identifier for this breaker (used in logs)
        max_failures:     Failures before opening the circuit
        cooldown_seconds: Time before allowing a probe after opening
    """

    def __init__(self, name: str, max_failures: int = 3, cooldown_seconds: int = 60):
        self.name = name
        self.max_failures = max_failures
        self.cooldown_seconds = cooldown_seconds
        self._failures = 0
        self._opened_at: float | None = None
        self._log = log.bind(circuit=name)

    @property
    def is_open(self) -> bool:
        """True if circuit is open (requests should be blocked)."""
        if self._opened_at is None:
            return False
        elapsed = time.time() - self._opened_at
        if elapsed > self.cooldown_seconds:
            # Half-open: allow one probe through
            self._log.info("circuit_half_open", elapsed_seconds=round(elapsed))
            return False
        return True

    def record_success(self) -> None:
        """Called after a successful API call — resets failure count."""
        if self._failures > 0:
            self._log.info("circuit_reset", previous_failures=self._failures)
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        """Called after a failed API call — may open the circuit."""
        self._failures += 1
        self._log.warning("circuit_failure_recorded", failures=self._failures, max=self.max_failures)
        if self._failures >= self.max_failures:
            self._opened_at = time.time()
            self._log.error(
                "circuit_opened",
                failures=self._failures,
                cooldown_seconds=self.cooldown_seconds,
            )

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "CLOSED"
        if time.time() - self._opened_at > self.cooldown_seconds:
            return "HALF-OPEN"
        return "OPEN"
