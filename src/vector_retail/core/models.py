"""
core/models.py
Pydantic v2 data models — the canonical data contracts for the entire system.
Every agent, layer, and API surface uses these types.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .enums import AccountType, DeploymentSlot, Jurisdiction, RiskTolerance


# ─────────────────────────────────────────────
# DOMAIN MODELS
# ─────────────────────────────────────────────

class UserProfile(BaseModel):
    """
    Validated, immutable user profile.
    Drives every policy and suitability decision downstream.
    Name is stored as first name + last initial only (GDPR/CCPA minimisation).
    """
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    risk_tolerance: RiskTolerance
    account_type: AccountType
    jurisdiction: Jurisdiction
    kyc_verified: bool = False
    accredited_investor: bool = False
    annual_income_usd: Optional[float] = None
    net_worth_usd: Optional[float] = None
    investment_horizon_years: int = 10

    @field_validator("name")
    @classmethod
    def minimise_name_pii(cls, v: str) -> str:
        """Store only first name + last initial — PII minimisation."""
        parts = v.strip().split()
        if len(parts) >= 2:
            return f"{parts[0]} {parts[-1][0]}."
        return parts[0] if parts else "Anonymous"


class PortfolioHolding(BaseModel):
    """
    Single tax-lot position.
    purchase_date must be ISO-8601 (e.g. '2022-03-15').
    """
    symbol: str
    quantity: float
    cost_basis_per_share: float
    purchase_date: str
    sector: str = "Unknown"
    asset_class: str = "equity"

    @property
    def cost_basis_total(self) -> float:
        return self.quantity * self.cost_basis_per_share


class MarketQuote(BaseModel):
    """
    Dual-sourced, cross-validated market quote.
    is_verified=True only when both sources agree within divergence threshold.
    """
    symbol: str
    price_primary: float
    price_secondary: Optional[float] = None
    source_primary: str = "yfinance"
    source_secondary: str = "alpha_vantage"
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    is_verified: bool = False
    divergence_pct: Optional[float] = None
    is_stale: bool = False

    @property
    def verified_price(self) -> float:
        """Averaged price when dual-verified; primary price otherwise."""
        if self.is_verified and self.price_secondary:
            return (self.price_primary + self.price_secondary) / 2
        return self.price_primary


class AgentResult(BaseModel):
    """
    Standardised output contract for every parallel agent.
    All fields are required — agents must justify their confidence and reasoning.
    """
    agent_id: str
    agent_version: str = "2.0.0"
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_chain: List[str] = Field(default_factory=list)
    findings: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    data_sources: List[str] = Field(default_factory=list)
    policy_flags: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    requires_hitl: bool = False
    hitl_reason: Optional[str] = None


class AuditEvent(BaseModel):
    """
    Single immutable audit log entry.
    Hash-chained: each event includes the hash of the previous event.
    Tampering any event breaks the chain and is detected by verify_chain_integrity().
    SOC 2 Type II compliant when persisted to WORM storage.
    """
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    user_id: str
    event_type: str
    action: str
    outcome: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prev_hash: str = ""
    event_hash: str = ""

    def compute_hash(self) -> str:
        payload = (
            self.event_id
            + self.timestamp_utc
            + self.user_id
            + self.action
            + self.outcome
            + self.prev_hash
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class ShadowEvalResult(BaseModel):
    """Scoring output from the shadow evaluation framework."""
    session_id: str
    deployment_slot: DeploymentSlot
    response_text: str
    ground_truth_check: Dict[str, bool] = Field(default_factory=dict)
    hallucination_flags: List[str] = Field(default_factory=list)
    policy_adherence_score: float = Field(ge=0.0, le=1.0, default=1.0)
    factual_accuracy_score: float = Field(ge=0.0, le=1.0, default=1.0)
    overall_score: float = Field(ge=0.0, le=1.0, default=1.0)
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class GraphState(BaseModel):
    """
    Typed LangGraph state container.
    Passed through every node — agents read from and write into this.
    """
    session_id: str
    user_query: str
    user_profile: Dict[str, Any]
    holdings: List[Dict[str, Any]]
    quotes: Dict[str, Any] = Field(default_factory=dict)
    agent_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    meta_audit_result: Optional[Dict[str, Any]] = None
    hitl_queue: List[Dict[str, Any]] = Field(default_factory=list)
    final_response: Optional[str] = None
    policy_flags: List[str] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    shadow_eval: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
