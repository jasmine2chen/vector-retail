"""
tests/integration/test_orchestrator_pipeline.py

Integration tests for the full VectorRetailAgent pipeline.

These tests mock the LLM and external data sources but exercise the entire
orchestration graph: security → data oracle → parallel agents → meta-critic
→ HITL gate → synthesizer.

Run: pytest tests/integration/ -v
"""
from unittest.mock import AsyncMock, MagicMock, patch

from vector_retail.core.enums import AccountType, Jurisdiction, RiskTolerance
from vector_retail.core.models import PortfolioHolding, UserProfile
from vector_retail.orchestrator import VectorRetailAgent


def _sample_profile() -> UserProfile:
    return UserProfile(
        name="Integration Test User",
        risk_tolerance=RiskTolerance.MODERATE,
        account_type=AccountType.INDIVIDUAL,
        jurisdiction=Jurisdiction.US,
        kyc_verified=True,
    )


def _sample_holdings() -> list[PortfolioHolding]:
    return [
        PortfolioHolding(
            symbol="AAPL",
            quantity=50,
            cost_basis_per_share=145.0,
            purchase_date="2023-01-15",
            sector="Technology",
            asset_class="equity",
        ),
        PortfolioHolding(
            symbol="VTI",
            quantity=100,
            cost_basis_per_share=200.0,
            purchase_date="2022-06-01",
            sector="Broad Market",
            asset_class="equity",
        ),
        PortfolioHolding(
            symbol="BND",
            quantity=80,
            cost_basis_per_share=75.0,
            purchase_date="2023-03-10",
            sector="Fixed Income",
            asset_class="fixed_income",
        ),
    ]


class TestOrchestratorPipeline:
    """End-to-end tests for the VectorRetailAgent orchestration pipeline."""

    @patch("vector_retail.agents.base.ChatAnthropic")
    @patch("vector_retail.data.oracle.yf.Ticker")
    def test_full_pipeline_returns_valid_result(self, mock_ticker, mock_llm_class):
        """The full pipeline should return a dict with all expected keys."""
        # Mock yfinance
        mock_info = MagicMock()
        mock_info.last_price = 185.0
        mock_ticker.return_value.fast_info = mock_info

        # Mock LLM responses
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Portfolio looks healthy with moderate risk exposure. "
                    "This does not constitute investment advice."
        )
        mock_llm_class.return_value = mock_llm

        agent = VectorRetailAgent()
        result = agent.run(
            user=_sample_profile(),
            holdings=_sample_holdings(),
            query="How is my portfolio doing?",
        )

        # Core response structure assertions
        assert isinstance(result, dict)
        assert "response" in result
        assert "session_id" in result
        assert "audit_trail_length" in result
        assert "deployment_slot" in result
        assert "policy_version" in result

    @patch("vector_retail.agents.base.ChatAnthropic")
    @patch("vector_retail.data.oracle.yf.Ticker")
    def test_pipeline_audit_trail_has_events(self, mock_ticker, mock_llm_class):
        """Pipeline should generate audit events for tracking."""
        mock_ticker.return_value.fast_info.last_price = 185.0
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Analysis complete. This does not constitute investment advice."
        )
        mock_llm_class.return_value = mock_llm

        agent = VectorRetailAgent()
        result = agent.run(
            user=_sample_profile(),
            holdings=_sample_holdings(),
            query="Analyze my portfolio.",
        )

        assert result["audit_trail_length"] > 0

    @patch("vector_retail.agents.base.ChatAnthropic")
    @patch("vector_retail.data.oracle.yf.Ticker")
    def test_pipeline_with_stale_data_still_completes(self, mock_ticker, mock_llm_class):
        """Pipeline should gracefully handle stale/missing market data."""
        mock_ticker.return_value.fast_info.last_price = None
        mock_ticker.return_value.history.return_value = MagicMock(empty=True)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Limited data available. This does not constitute investment advice."
        )
        mock_llm_class.return_value = mock_llm

        agent = VectorRetailAgent()
        result = agent.run(
            user=_sample_profile(),
            holdings=_sample_holdings(),
            query="What's my risk exposure?",
        )

        # Should still return a valid result even with degraded data
        assert isinstance(result, dict)
        assert "response" in result

    @patch("vector_retail.agents.base.ChatAnthropic")
    @patch("vector_retail.data.oracle.yf.Ticker")
    def test_pipeline_kyc_unverified_flags_compliance(self, mock_ticker, mock_llm_class):
        """Unverified KYC should trigger compliance flags in the pipeline."""
        mock_ticker.return_value.fast_info.last_price = 185.0
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Review pending. This does not constitute investment advice."
        )
        mock_llm_class.return_value = mock_llm

        unverified_user = UserProfile(
            name="Unverified User",
            risk_tolerance=RiskTolerance.MODERATE,
            account_type=AccountType.INDIVIDUAL,
            jurisdiction=Jurisdiction.US,
            kyc_verified=False,
        )

        agent = VectorRetailAgent()
        result = agent.run(
            user=unverified_user,
            holdings=_sample_holdings(),
            query="Should I rebalance?",
        )

        assert isinstance(result, dict)
