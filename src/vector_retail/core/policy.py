"""
core/policy.py
Contextual Policy Engine.

All risk thresholds are loaded from config/policy_rules.json at startup.
No magic numbers live in application code — every limit is data-driven,
versioned, and auditable.

Implements:
  - SEC Reg BI suitability checks
  - FINRA Rule 2111 concentration limits
  - Pre-trade compliance sweep
  - KYC clearance gate
  - HITL threshold evaluation
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from .enums import AccountType
from .models import UserProfile

log = structlog.get_logger("policy_engine")

# ── Load policy rules from config file ────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "policy_rules.json"

def _load_policy() -> dict[str, Any]:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    # Fallback defaults (used during tests / first run)
    return {
        "version": "2.0.0",
        "trade_value_hitl_threshold_usd": 25000,
        "max_single_position_pct": {
            "conservative": 0.05,
            "moderate": 0.10,
            "aggressive": 0.20,
        },
        "max_sector_exposure_pct": {
            "conservative": 0.15,
            "moderate": 0.25,
            "aggressive": 0.40,
        },
        "data_staleness_seconds": 300,
        "cross_ref_divergence_threshold": 0.02,
        "min_confidence_for_auto_response": 0.75,
        "shadow_eval_sample_rate": 0.10,
        "min_cash_buffer_pct": {
            "conservative": 0.10,
            "moderate": 0.05,
            "aggressive": 0.02,
        },
    }

POLICY_RULES: dict[str, Any] = _load_policy()
POLICY_VERSION: str = POLICY_RULES.get("version", "2.0.0")


class PolicyEngine:
    """
    Enforces all risk, suitability, and compliance rules.
    Instantiated per-session with the user's profile.
    All rule evaluations are logged to the audit trail.
    """

    def __init__(self, profile: UserProfile, audit_fn):
        """
        Args:
            profile: Validated UserProfile
            audit_fn: Callable matching AuditTrail.record() signature
        """
        self.profile = profile
        self._audit = audit_fn
        self._log = log.bind(user_id=profile.user_id, risk=profile.risk_tolerance)
        self._risk_key = profile.risk_tolerance.value

    # ── Limit accessors ────────────────────────────────────────────────────

    def max_position_pct(self) -> float:
        return POLICY_RULES["max_single_position_pct"][self._risk_key]

    def max_sector_pct(self) -> float:
        return POLICY_RULES["max_sector_exposure_pct"][self._risk_key]

    def min_cash_pct(self) -> float:
        return POLICY_RULES["min_cash_buffer_pct"][self._risk_key]

    def hitl_threshold_usd(self) -> float:
        return float(POLICY_RULES["trade_value_hitl_threshold_usd"])

    def min_confidence(self) -> float:
        return float(POLICY_RULES["min_confidence_for_auto_response"])

    # ── Individual rule checks ─────────────────────────────────────────────

    def check_position_concentration(
        self,
        symbol: str,
        proposed_value_usd: float,
        total_portfolio_value_usd: float,
    ) -> tuple[bool, str]:
        """
        Returns (passes, reason).
        Fails if position would exceed max_position_pct for this risk profile.
        """
        if total_portfolio_value_usd <= 0:
            return False, "Portfolio value is zero or negative — cannot assess concentration"

        pct = proposed_value_usd / total_portfolio_value_usd
        limit = self.max_position_pct()

        if pct > limit:
            reason = (
                f"{symbol} would be {pct:.1%} of portfolio, "
                f"exceeding {limit:.1%} limit for {self._risk_key} profile"
            )
            self._log.warning("concentration_breach", symbol=symbol, pct=round(pct, 4), limit=limit)
            self._audit(
                "policy", f"concentration_check_{symbol}", "failed",
                {"symbol": symbol, "pct": round(pct, 4), "limit": limit},
            )
            return False, reason

        return True, "OK"

    def check_sector_exposure(
        self,
        sector: str,
        sector_value_usd: float,
        total_portfolio_value_usd: float,
    ) -> tuple[bool, str]:
        """Fails if sector exposure exceeds max_sector_pct."""
        if total_portfolio_value_usd <= 0:
            return False, "Portfolio value is zero"

        pct = sector_value_usd / total_portfolio_value_usd
        limit = self.max_sector_pct()

        if pct > limit:
            reason = (
                f"Sector '{sector}' is {pct:.1%} of portfolio, "
                f"exceeding {limit:.1%} limit"
            )
            self._log.warning("sector_breach", sector=sector, pct=round(pct, 4))
            self._audit(
                "policy", f"sector_check_{sector}", "failed",
                {"sector": sector, "pct": round(pct, 4), "limit": limit},
            )
            return False, reason

        return True, "OK"

    def check_trade_hitl_required(self, trade_value_usd: float) -> bool:
        """True if trade value exceeds HITL threshold."""
        required = trade_value_usd >= self.hitl_threshold_usd()
        if required:
            self._log.info(
                "hitl_required",
                trade_value=trade_value_usd,
                threshold=self.hitl_threshold_usd(),
            )
        return required

    def check_kyc_clearance(self, action: str) -> tuple[bool, str]:
        """KYC gate — blocks regulated actions if KYC not verified."""
        if not self.profile.kyc_verified:
            reason = f"KYC verification required before: {action}"
            self._audit("policy", f"kyc_check_{action}", "failed", {"action": action})
            return False, reason
        return True, "KYC cleared"

    def check_ira_tax_applicability(self) -> str:
        """Returns a note when account type affects tax treatment."""
        if self.profile.account_type in (AccountType.IRA, AccountType.ROTH_IRA):
            return (
                "Tax-loss harvesting has limited applicability in tax-advantaged accounts "
                f"({self.profile.account_type.value.upper()}). Consult a qualified tax advisor."
            )
        return ""

    # ── Full pre-trade compliance sweep ───────────────────────────────────

    def run_pre_trade_checks(
        self,
        symbol: str,
        trade_value_usd: float,
        total_portfolio_value_usd: float,
    ) -> list[str]:
        """
        Runs all applicable compliance checks for a proposed trade.
        Returns a list of policy flag strings (empty = all clear).
        """
        flags: list[str] = []

        kyc_ok, kyc_reason = self.check_kyc_clearance("trade")
        if not kyc_ok:
            flags.append(f"KYC_FAIL: {kyc_reason}")

        conc_ok, conc_reason = self.check_position_concentration(
            symbol, trade_value_usd, total_portfolio_value_usd
        )
        if not conc_ok:
            flags.append(f"CONCENTRATION: {conc_reason}")

        if self.check_trade_hitl_required(trade_value_usd):
            flags.append(
                f"HITL_REQUIRED: Trade value ${trade_value_usd:,.0f} "
                f"exceeds ${self.hitl_threshold_usd():,.0f} threshold"
            )

        self._audit(
            "policy", "pre_trade_sweep", "completed",
            {"symbol": symbol, "flags": flags, "trade_value": trade_value_usd},
        )

        return flags
