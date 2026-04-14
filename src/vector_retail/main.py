"""
main.py
Entry point and demo runner.

Run:
    python -m vector_retail.main
    # or
    python src/vector_retail/main.py
"""

from __future__ import annotations

import os

from .core.enums import AccountType, DeploymentSlot, Jurisdiction, RiskTolerance
from .core.models import PortfolioHolding, UserProfile
from .orchestrator import VectorRetailAgent


def run_demo() -> dict:
    """
    Smoke test with a realistic retail investor scenario.
    Replace profile and holdings with real data in production.
    """
    print("\n" + "═" * 65)
    print("  VECTOR RETAIL — Production Finance AI Agent v2.0")
    print("═" * 65 + "\n")

    profile = UserProfile(
        name="Alexandra Pemberton",
        risk_tolerance=RiskTolerance.MODERATE,
        account_type=AccountType.INDIVIDUAL,
        jurisdiction=Jurisdiction.US,
        kyc_verified=True,
        investment_horizon_years=15,
    )

    holdings = [
        PortfolioHolding(
            symbol="AAPL",
            quantity=50,
            cost_basis_per_share=145.00,
            purchase_date="2022-03-15",
            sector="Technology",
            asset_class="equity",
        ),
        PortfolioHolding(
            symbol="MSFT",
            quantity=30,
            cost_basis_per_share=280.00,
            purchase_date="2021-11-01",
            sector="Technology",
            asset_class="equity",
        ),
        PortfolioHolding(
            symbol="VTI",
            quantity=100,
            cost_basis_per_share=195.00,
            purchase_date="2020-06-01",
            sector="Diversified",
            asset_class="equity",
        ),
        PortfolioHolding(
            symbol="BND",
            quantity=80,
            cost_basis_per_share=75.00,
            purchase_date="2020-06-01",
            sector="Fixed Income",
            asset_class="fixed_income",
        ),
    ]

    agent = VectorRetailAgent(deployment_slot=DeploymentSlot.BLUE)

    result = agent.run(
        user_query=(
            "How is my portfolio performing? "
            "Do I need to rebalance?"
        ),
        user_profile=profile,
        holdings=holdings,
        auth_token=os.getenv("AUTH_TOKEN", ""),
        role="retail_client",
    )

    # ── Output ─────────────────────────────────────────────────────────────
    print(f"Session ID       : {result.get('session_id', 'N/A')}")
    print(f"Deployment Slot  : {result.get('deployment_slot', 'N/A')}")
    print(f"Policy Version   : {result.get('policy_version', 'N/A')}")
    print(f"Total Latency    : {result.get('total_latency_ms', 0):.0f}ms")
    print(f"Audit Events     : {result.get('audit_trail_length', 0)}")
    chain_ok = result.get("audit_chain_integrity")
    print(f"Chain Integrity  : {'✓ VALID' if chain_ok else '✗ COMPROMISED'}")
    print(f"HITL Escalated   : {'YES ⚠' if result.get('hitl_escalated') else 'No'}")
    print(f"Shadow Eval      : {result.get('shadow_eval_score') or 'Not sampled this run'}")
    print()

    print("Agent Confidences:")
    for agent_id, conf in (result.get("agent_confidences") or {}).items():
        bar = "█" * int((conf or 0) * 10)
        print(f"  {agent_id:<24}: {bar:<12} {(conf or 0):.0%}")

    meta_conf = result.get("meta_confidence") or 0
    print(f"\nMeta Confidence  : {'█' * int(meta_conf * 10):<12} {meta_conf:.0%}")
    print()

    if result.get("policy_flags"):
        print("Policy Flags:")
        for flag in result["policy_flags"]:
            print(f"  ⚑  {flag}")
        print()

    print("─" * 65)
    print("RESPONSE:")
    print("─" * 65)
    print(result.get("response", result.get("error", "No response generated")))
    print("─" * 65 + "\n")

    return result


if __name__ == "__main__":
    run_demo()
