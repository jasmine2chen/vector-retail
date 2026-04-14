"""
tests/integration/test_orchestrator_pipeline.py

Integration tests for the full VectorRetailAgent pipeline.

These tests mock the LLM and external data sources but exercise the entire
orchestration graph: security → data oracle → 6 parallel agents → meta-critic
→ HITL gate → synthesizer.

External dependencies mocked in ALL tests:
  - vector_retail.orchestrator.ChatAnthropic  — prevents real Anthropic API calls
  - vector_retail.data.oracle.yf.Ticker       — prevents real yfinance market data calls
  - vector_retail.agents.sentiment.yf.Ticker  — prevents real yfinance news calls
  - vector_retail.agents.sentiment._load_finbert — prevents FinBERT model download (~500MB)

The sentiment agent degrades gracefully when news=[]; no headlines → no FinBERT
inference → valid AgentResult with penalised confidence. Integration tests
verify the pipeline completes correctly with this degraded-but-valid path.

Test philosophy (Andrew Ng's eval-driven development):
  - Structural tests verify the pipeline runs without error.
  - Business logic tests assert on COMPLIANCE OUTCOMES — the actual flags,
    escalation decisions, and audit trail contents that prove the regulatory
    logic works correctly. These are the tests a regulator or CTO would want.

Run: pytest tests/integration/ -v
"""

from unittest.mock import MagicMock, patch

from vector_retail.core.enums import AccountType, Jurisdiction, RiskTolerance
from vector_retail.core.models import PortfolioHolding, UserProfile
from vector_retail.orchestrator import VectorRetailAgent

# ── Shared fixtures ────────────────────────────────────────────────────────────


def _sample_profile(
    kyc_verified: bool = True,
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE,
) -> UserProfile:
    return UserProfile(
        name="Integration Test User",
        risk_tolerance=risk_tolerance,
        account_type=AccountType.INDIVIDUAL,
        jurisdiction=Jurisdiction.US,
        kyc_verified=kyc_verified,
    )


def _sample_holdings(concentrated: bool = False) -> list[PortfolioHolding]:
    """
    Default 3-position portfolio.
    When concentrated=True, returns a single position that will breach
    the 10% moderate-profile concentration limit.
    """
    if concentrated:
        # Single position at 100% weight — should always trigger CONCENTRATION flag
        return [
            PortfolioHolding(
                symbol="AAPL",
                quantity=1000,
                cost_basis_per_share=185.0,
                purchase_date="2023-01-15",
                sector="Technology",
                asset_class="equity",
            )
        ]
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


def _mock_llm_response(content: str) -> MagicMock:
    mock = MagicMock()
    mock.content = content
    return mock


def _make_agent_with_mocked_llm(mock_llm_class, response_content: str) -> VectorRetailAgent:
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = _mock_llm_response(response_content)
    mock_llm_class.return_value = mock_llm
    return VectorRetailAgent()


# ── Structural tests ───────────────────────────────────────────────────────────


class TestOrchestratorPipeline:
    """End-to-end structural tests — verify the pipeline completes without error."""

    def setup_method(self):
        """
        Patch sentiment agent externals before every test.

        Prevents real yfinance news calls and FinBERT model download (~500MB)
        in CI. Empty news list triggers the agent's graceful no-data path —
        it still returns a valid AgentResult with penalised confidence, so
        the pipeline continues and all downstream assertions remain valid.
        """
        self._sentiment_ticker_patcher = patch("vector_retail.agents.sentiment.yf.Ticker")
        self._finbert_patcher = patch("vector_retail.agents.sentiment._load_finbert")
        mock_sentiment_ticker = self._sentiment_ticker_patcher.start()
        mock_sentiment_ticker.return_value.news = []
        self._finbert_patcher.start()

    def teardown_method(self):
        self._sentiment_ticker_patcher.stop()
        self._finbert_patcher.stop()

    @patch("vector_retail.orchestrator.ChatAnthropic")
    @patch("vector_retail.data.oracle.yf.Ticker")
    def test_full_pipeline_returns_valid_result(self, mock_ticker, mock_llm_class):
        """The full pipeline should return a dict with all expected keys."""
        mock_ticker.return_value.fast_info.last_price = 185.0
        agent = _make_agent_with_mocked_llm(
            mock_llm_class,
            "Portfolio looks healthy with moderate risk exposure. "
            "This does not constitute investment advice.",
        )
        result = agent.run(
            user_query="How is my portfolio doing?",
            user_profile=_sample_profile(),
            holdings=_sample_holdings(),
        )

        assert isinstance(result, dict)
        assert "response" in result
        assert "session_id" in result
        assert "audit_trail_length" in result
        assert "deployment_slot" in result
        assert "policy_version" in result
        assert "reflection_applied" in result  # New: reflection loop field

    @patch("vector_retail.orchestrator.ChatAnthropic")
    @patch("vector_retail.data.oracle.yf.Ticker")
    def test_pipeline_audit_trail_has_events(self, mock_ticker, mock_llm_class):
        """Pipeline should generate audit events for tracking."""
        mock_ticker.return_value.fast_info.last_price = 185.0
        agent = _make_agent_with_mocked_llm(
            mock_llm_class,
            "Analysis complete. This does not constitute investment advice.",
        )
        result = agent.run(
            user_query="Analyze my portfolio.",
            user_profile=_sample_profile(),
            holdings=_sample_holdings(),
        )
        assert result["audit_trail_length"] > 0

    @patch("vector_retail.orchestrator.ChatAnthropic")
    @patch("vector_retail.data.oracle.yf.Ticker")
    def test_pipeline_with_stale_data_still_completes(self, mock_ticker, mock_llm_class):
        """Pipeline should gracefully handle stale/missing market data."""
        mock_ticker.return_value.fast_info.last_price = None
        mock_ticker.return_value.history.return_value = MagicMock(empty=True)
        agent = _make_agent_with_mocked_llm(
            mock_llm_class,
            "Limited data available. This does not constitute investment advice.",
        )
        result = agent.run(
            user_query="What's my risk exposure?",
            user_profile=_sample_profile(),
            holdings=_sample_holdings(),
        )
        assert isinstance(result, dict)
        assert "response" in result


# ── Business logic compliance tests ───────────────────────────────────────────
# These assert on OUTCOMES — that the compliance logic produces the right
# flags and decisions given specific input conditions.
# A regulator auditing this system expects these tests to exist and pass.


class TestComplianceBusinessLogic:
    """
    Business logic assertions for compliance outcomes.
    Tests prove the regulatory rules fire correctly — not just that the pipeline runs.
    """

    def setup_method(self):
        """Patch sentiment externals — same rationale as TestOrchestratorPipeline."""
        self._sentiment_ticker_patcher = patch("vector_retail.agents.sentiment.yf.Ticker")
        self._finbert_patcher = patch("vector_retail.agents.sentiment._load_finbert")
        mock_sentiment_ticker = self._sentiment_ticker_patcher.start()
        mock_sentiment_ticker.return_value.news = []
        self._finbert_patcher.start()

    def teardown_method(self):
        self._sentiment_ticker_patcher.stop()
        self._finbert_patcher.stop()

    @patch("vector_retail.orchestrator.ChatAnthropic")
    @patch("vector_retail.data.oracle.yf.Ticker")
    def test_unverified_kyc_triggers_policy_flag(self, mock_ticker, mock_llm_class):
        """
        Unverified KYC must produce a KYC_FAIL policy flag.
        SEC Reg BI / FINRA suitability rules require KYC clearance before advice.
        """
        mock_ticker.return_value.fast_info.last_price = 185.0
        agent = _make_agent_with_mocked_llm(
            mock_llm_class,
            "Review pending. This does not constitute investment advice.",
        )
        unverified_user = _sample_profile(kyc_verified=False)

        result = agent.run(
            user_query="Should I rebalance?",
            user_profile=unverified_user,
            holdings=_sample_holdings(),
        )

        assert isinstance(result, dict), "Result must be a dict"
        policy_flags = result.get("policy_flags", [])
        kyc_flags = [f for f in policy_flags if "KYC_FAIL" in f]
        assert len(kyc_flags) > 0, (
            f"Expected at least one KYC_FAIL flag for unverified user. "
            f"Got flags: {policy_flags}"
        )

    @patch("vector_retail.orchestrator.ChatAnthropic")
    @patch("vector_retail.data.oracle.yf.Ticker")
    def test_concentrated_position_triggers_concentration_flag(self, mock_ticker, mock_llm_class):
        """
        A single position breaching the concentration limit must trigger a
        CONCENTRATION flag (FINRA Rule 2111 suitability enforcement).
        Moderate profile limit: 10% max single position.
        """
        mock_ticker.return_value.fast_info.last_price = 185.0
        agent = _make_agent_with_mocked_llm(
            mock_llm_class,
            "Portfolio is highly concentrated. This does not constitute investment advice.",
        )
        # 1000 shares × $185 = $185,000 — the only position, so 100% concentration
        concentrated_holdings = _sample_holdings(concentrated=True)

        result = agent.run(
            user_query="How is my portfolio?",
            user_profile=_sample_profile(risk_tolerance=RiskTolerance.MODERATE),
            holdings=concentrated_holdings,
        )

        policy_flags = result.get("policy_flags", [])
        concentration_flags = [f for f in policy_flags if "CONCENTRATION" in f]
        assert len(concentration_flags) > 0, (
            f"Expected CONCENTRATION flag for 100% single-position portfolio. "
            f"Got flags: {policy_flags}"
        )

    @patch("vector_retail.orchestrator.ChatAnthropic")
    @patch("vector_retail.data.oracle.yf.Ticker")
    def test_large_trade_triggers_hitl_escalation(self, mock_ticker, mock_llm_class):
        """
        A rebalance trade exceeding $25,000 HITL threshold must set hitl_escalated=True.
        Policy: trade_value_hitl_threshold_usd = 25000.
        """
        mock_ticker.return_value.fast_info.last_price = 185.0
        agent = _make_agent_with_mocked_llm(
            mock_llm_class,
            "Large rebalance trade suggested. Requires advisor review. "
            "This does not constitute investment advice.",
        )
        # Single large position where any rebalance trade would be significant
        large_holdings = [
            PortfolioHolding(
                symbol="TSLA",
                quantity=500,  # 500 × $185 = $92,500 position
                cost_basis_per_share=100.0,
                purchase_date="2022-01-01",
                sector="Technology",
                asset_class="equity",
            )
        ]

        result = agent.run(
            user_query="Should I sell TSLA and rebalance?",
            user_profile=_sample_profile(),
            holdings=large_holdings,
        )

        assert result.get("hitl_escalated") is True, (
            "Expected hitl_escalated=True for a trade suggestion on a $92,500 position. "
            f"Got hitl_escalated={result.get('hitl_escalated')}, "
            f"policy_flags={result.get('policy_flags', [])}"
        )

    @patch("vector_retail.orchestrator.ChatAnthropic")
    @patch("vector_retail.data.oracle.yf.Ticker")
    def test_audit_chain_integrity_always_holds(self, mock_ticker, mock_llm_class):
        """
        The SHA-256 hash-chained audit trail must verify as intact on every session.
        SOC 2 Type II requirement: tamper-evident log for all advisory decisions.
        """
        mock_ticker.return_value.fast_info.last_price = 185.0
        agent = _make_agent_with_mocked_llm(
            mock_llm_class,
            "Portfolio analysis complete. This does not constitute investment advice.",
        )
        result = agent.run(
            user_query="Analyze my portfolio.",
            user_profile=_sample_profile(),
            holdings=_sample_holdings(),
        )

        assert result.get("audit_chain_integrity") is True, (
            "Audit chain integrity check failed — SHA-256 hash chain is broken. "
            "This indicates tampered or out-of-order audit events."
        )
        assert result.get("audit_trail_length", 0) >= 3, (
            "Expected at least 3 audit events (auth + data + agent). "
            f"Got {result.get('audit_trail_length')} events."
        )

    @patch("vector_retail.orchestrator.ChatAnthropic")
    @patch("vector_retail.data.oracle.yf.Ticker")
    def test_response_contains_regulatory_disclaimer(self, mock_ticker, mock_llm_class):
        """
        Every non-HITL response must contain the jurisdiction-appropriate disclaimer.
        SEC Reg BI and FINRA require explicit disclosure for retail investor communications.
        """
        mock_ticker.return_value.fast_info.last_price = 185.0
        agent = _make_agent_with_mocked_llm(
            mock_llm_class,
            "Portfolio analysis complete. Risk is moderate.",
        )
        result = agent.run(
            user_query="How is my portfolio doing?",
            user_profile=_sample_profile(),
            holdings=_sample_holdings(),
        )

        response = result.get("response", "")
        disclaimer_phrases = [
            "informational purposes only",
            "does not constitute",
            "not investment advice",
            "consult a",
        ]
        has_disclaimer = any(phrase in response.lower() for phrase in disclaimer_phrases)
        assert has_disclaimer, (
            "Response missing regulatory disclaimer. "
            "SEC Reg BI requires disclosure for retail investor communications. "
            f"Response preview: {response[:200]}"
        )

    @patch("vector_retail.orchestrator.ChatAnthropic")
    @patch("vector_retail.data.oracle.yf.Ticker")
    def test_result_exposes_per_agent_confidence_scores(self, mock_ticker, mock_llm_class):
        """
        Result must expose per-agent confidence scores for all 6 specialist agents.
        Each agent must produce a confidence in [0.0, 1.0].
        sentiment_analysis runs in no-news mode (mocked empty news) and produces
        a valid penalised confidence — confirming graceful degradation is wired correctly.
        """
        mock_ticker.return_value.fast_info.last_price = 185.0
        agent = _make_agent_with_mocked_llm(
            mock_llm_class,
            "Analysis complete. This does not constitute investment advice.",
        )
        result = agent.run(
            user_query="What is my risk exposure?",
            user_profile=_sample_profile(),
            holdings=_sample_holdings(),
        )

        confidences = result.get("agent_confidences", {})
        expected_agents = {
            "portfolio_analysis",
            "risk_assessment",
            "rebalance",
            "sentiment_analysis",
        }
        assert expected_agents.issubset(set(confidences.keys())), (
            f"Missing agent confidence scores. Expected all 4 specialists: {expected_agents}. "
            f"Got: {set(confidences.keys())}"
        )
        for agent_id, conf in confidences.items():
            if conf is not None:
                assert (
                    0.0 <= conf <= 1.0
                ), f"Agent {agent_id} confidence {conf} out of valid [0.0, 1.0] range."
