"""
tests/unit/test_policy.py
Unit tests for the Policy Engine.
"""
from vector_retail.core.enums import AccountType, Jurisdiction, RiskTolerance
from vector_retail.core.models import UserProfile
from vector_retail.core.policy import PolicyEngine


# Dummy audit function for testing
def _noop_audit(event_type, action, outcome, metadata=None):
    pass


def _make_profile(risk: RiskTolerance, kyc: bool = True) -> UserProfile:
    return UserProfile(
        name="Test User",
        risk_tolerance=risk,
        account_type=AccountType.INDIVIDUAL,
        jurisdiction=Jurisdiction.US,
        kyc_verified=kyc,
    )


class TestConcentrationChecks:

    def test_moderate_within_limit(self):
        policy = PolicyEngine(_make_profile(RiskTolerance.MODERATE), _noop_audit)
        ok, _ = policy.check_position_concentration("AAPL", 8000, 100000)
        assert ok is True  # 8% < 10% limit

    def test_moderate_exceeds_limit(self):
        policy = PolicyEngine(_make_profile(RiskTolerance.MODERATE), _noop_audit)
        ok, reason = policy.check_position_concentration("AAPL", 12000, 100000)
        assert ok is False
        assert "AAPL" in reason

    def test_conservative_within_limit(self):
        policy = PolicyEngine(_make_profile(RiskTolerance.CONSERVATIVE), _noop_audit)
        ok, _ = policy.check_position_concentration("VTI", 4000, 100000)
        assert ok is True  # 4% < 5% limit

    def test_conservative_exceeds_limit(self):
        policy = PolicyEngine(_make_profile(RiskTolerance.CONSERVATIVE), _noop_audit)
        ok, _ = policy.check_position_concentration("VTI", 6000, 100000)
        assert ok is False  # 6% > 5% limit

    def test_aggressive_higher_limit(self):
        policy = PolicyEngine(_make_profile(RiskTolerance.AGGRESSIVE), _noop_audit)
        ok, _ = policy.check_position_concentration("TSLA", 18000, 100000)
        assert ok is True  # 18% < 20% limit

    def test_zero_portfolio_value(self):
        policy = PolicyEngine(_make_profile(RiskTolerance.MODERATE), _noop_audit)
        ok, reason = policy.check_position_concentration("AAPL", 1000, 0)
        assert ok is False
        assert "zero" in reason.lower()


class TestHITLThreshold:

    def test_below_threshold_no_hitl(self):
        policy = PolicyEngine(_make_profile(RiskTolerance.MODERATE), _noop_audit)
        assert policy.check_trade_hitl_required(24999) is False

    def test_at_threshold_hitl(self):
        policy = PolicyEngine(_make_profile(RiskTolerance.MODERATE), _noop_audit)
        assert policy.check_trade_hitl_required(25000) is True

    def test_above_threshold_hitl(self):
        policy = PolicyEngine(_make_profile(RiskTolerance.MODERATE), _noop_audit)
        assert policy.check_trade_hitl_required(100000) is True


class TestKYCGate:

    def test_kyc_verified_passes(self):
        policy = PolicyEngine(_make_profile(RiskTolerance.MODERATE, kyc=True), _noop_audit)
        ok, _ = policy.check_kyc_clearance("trade")
        assert ok is True

    def test_kyc_not_verified_fails(self):
        policy = PolicyEngine(_make_profile(RiskTolerance.MODERATE, kyc=False), _noop_audit)
        ok, reason = policy.check_kyc_clearance("trade")
        assert ok is False
        assert "KYC" in reason


class TestPreTradeSweep:

    def test_clean_trade_no_flags(self):
        policy = PolicyEngine(_make_profile(RiskTolerance.MODERATE, kyc=True), _noop_audit)
        flags = policy.run_pre_trade_checks("AAPL", 5000, 100000)
        assert flags == []

    def test_kyc_fail_in_sweep(self):
        policy = PolicyEngine(_make_profile(RiskTolerance.MODERATE, kyc=False), _noop_audit)
        flags = policy.run_pre_trade_checks("AAPL", 5000, 100000)
        assert any("KYC_FAIL" in f for f in flags)

    def test_hitl_in_sweep(self):
        policy = PolicyEngine(_make_profile(RiskTolerance.MODERATE, kyc=True), _noop_audit)
        flags = policy.run_pre_trade_checks("AAPL", 30000, 100000)
        assert any("HITL_REQUIRED" in f for f in flags)
