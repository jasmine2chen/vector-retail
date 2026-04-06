"""
security/rbac.py
JWT Validation + Role-Based Access Control (RBAC).

Enforces least-privilege: every action is gated against a role's
permission list before execution.

Production integration points (marked PRODUCTION):
  - Replace validate_jwt_stub() with python-jose JWT decode
    using your IdP's public key
  - Fetch role assignments from your IAM system / database
  - Add token revocation check against a Redis deny-list

OWASP LLM Top 10 — LLM08: Excessive Agency mitigation.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import structlog

log = structlog.get_logger("rbac")

# ── Role permission registry ───────────────────────────────────────────────────
# Least-privilege: each role has the minimum set of permissions needed.
# '*' means all permissions (admin only).

ROLE_PERMISSIONS: dict[str, list[str]] = {
    "retail_client": [
        "read_portfolio",
        "read_market",
        "request_advice",
    ],
    "advisor": [
        "read_portfolio",
        "read_market",
        "request_advice",
        "approve_hitl",
        "view_client_profiles",
    ],
    "compliance": [
        "read_all",
        "read_audit",
        "approve_hitl",
        "export_audit_logs",
        "view_shadow_eval",
    ],
    "admin": ["*"],
}


class SecurityLayer:
    """
    Per-session security context.
    Wraps PII redaction, JWT validation, and RBAC checks.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._log = log.bind(session_id=session_id)

    def validate_jwt_stub(self, token: str) -> dict[str, Any]:
        """
        Validates an auth token and returns claims.

        STUB: accepts any non-empty token >= 10 chars and returns
              demo claims. Replace the body below with real JWT decode.

        PRODUCTION:
            from jose import jwt, JWTError
            PUBLIC_KEY = os.getenv("JWT_PUBLIC_KEY")
            try:
                claims = jwt.decode(token, PUBLIC_KEY, algorithms=["RS256"])
                # Also check claims["exp"] and revocation list
                return claims
            except JWTError as e:
                raise PermissionError(f"Invalid token: {e}")
        """
        if not token or len(token) < 10:
            raise PermissionError("Invalid or missing auth token")

        # Stub claims — replace with real decode in production
        return {
            "sub": "user_demo",
            "role": "retail_client",
            "iss": "vector-retail-auth",
            "iat": datetime.now(UTC).isoformat(),
            "exp": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        }

    def validate_permission(self, role: str, action: str) -> bool:
        """
        RBAC check: does this role have permission for this action?

        Args:
            role:   Role string (must be in ROLE_PERMISSIONS)
            action: Action string (must be in role's permission list)

        Returns:
            True if permitted, False otherwise (also logs the denial).
        """
        permissions = ROLE_PERMISSIONS.get(role, [])

        if "*" in permissions or action in permissions:
            return True

        self._log.warning("rbac_denied", role=role, action=action)
        return False
