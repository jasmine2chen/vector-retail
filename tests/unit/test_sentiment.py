"""
tests/unit/test_sentiment.py

Unit tests for the SentimentAnalysisAgent (FinBERT-powered news sentiment).

Testing strategy:
  - FinBERT model is mocked in ALL tests to avoid network downloads and GPU
    dependency in CI. The mock returns deterministic scores that exercise
    every code path (bearish, neutral, model failure, no-news).
  - yfinance news fetch is mocked to return controlled headline sets.
  - All tests are CPU-only and run in < 1 second.

Run: pytest tests/unit/test_sentiment.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vector_retail.agents.sentiment import SentimentAnalysisAgent, SentimentScore
from vector_retail.core.enums import AccountType, Jurisdiction, RiskTolerance
from vector_retail.core.models import GraphState, UserProfile

# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_state(symbols: list[str] | None = None) -> GraphState:
    symbols = symbols or ["AAPL", "MSFT"]
    holdings = [
        {
            "symbol": sym,
            "quantity": 10,
            "cost_basis_per_share": 100.0,
            "purchase_date": "2023-01-01",
            "sector": "Technology",
            "asset_class": "equity",
        }
        for sym in symbols
    ]
    profile = UserProfile(
        name="Test User",
        risk_tolerance=RiskTolerance.MODERATE,
        account_type=AccountType.INDIVIDUAL,
        jurisdiction=Jurisdiction.US,
        kyc_verified=True,
    )
    return GraphState(
        session_id="test-session",
        user_query="How is market sentiment for my holdings?",
        user_profile=profile.model_dump(),
        holdings=holdings,
    )


def _make_agent(mock_llm) -> SentimentAnalysisAgent:
    """Return an agent with a mocked LLM and a stubbed audit trail."""
    audit = MagicMock()
    audit.session_id = "test-session"
    audit.record = MagicMock()
    return SentimentAnalysisAgent(llm=mock_llm, audit=audit)


def _finbert_output(positive: float, negative: float) -> list[list[dict]]:
    """
    Build a mock FinBERT pipeline output for a single headline.
    Scores must sum to 1.0 (probability distribution).
    """
    neutral = max(0.0, round(1.0 - positive - negative, 4))
    return [
        [
            {"label": "positive", "score": positive},
            {"label": "negative", "score": negative},
            {"label": "neutral", "score": neutral},
        ]
    ]


# ── SentimentScore unit tests ──────────────────────────────────────────────────


class TestSentimentScore:
    """Tests for the SentimentScore value object."""

    def test_dominant_label_positive(self):
        score = SentimentScore("AAPL", positive=0.70, negative=0.10, neutral=0.20, n_headlines=5)
        assert score.dominant == "positive"

    def test_dominant_label_negative(self):
        score = SentimentScore("TSLA", positive=0.10, negative=0.65, neutral=0.25, n_headlines=3)
        assert score.dominant == "negative"

    def test_dominant_label_neutral(self):
        score = SentimentScore("BND", positive=0.20, negative=0.15, neutral=0.65, n_headlines=4)
        assert score.dominant == "neutral"

    def test_bearish_flag_triggers_above_threshold(self):
        """is_bearish=True when negative > 0.40."""
        score = SentimentScore("X", positive=0.10, negative=0.45, neutral=0.45, n_headlines=2)
        assert score.is_bearish is True

    def test_bearish_flag_does_not_trigger_below_threshold(self):
        """is_bearish=False when negative <= 0.40."""
        score = SentimentScore("X", positive=0.30, negative=0.35, neutral=0.35, n_headlines=2)
        assert score.is_bearish is False

    def test_to_dict_roundtrip(self):
        score = SentimentScore("AAPL", 0.70, 0.10, 0.20, 5)
        d = score.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["positive"] == 0.70
        assert d["negative"] == 0.10
        assert d["neutral"] == 0.20
        assert d["n_headlines"] == 5
        assert d["dominant"] == "positive"
        assert d["is_bearish"] is False


# ── Agent behaviour tests ──────────────────────────────────────────────────────


class TestSentimentAgent:
    """Integration-style unit tests for SentimentAnalysisAgent.run()."""

    def setup_method(self):
        """Disable the RAG retriever for all tests — no ChromaDB in unit tests."""
        self._retriever_patcher = patch(
            "vector_retail.agents.sentiment.get_retriever", return_value=None
        )
        self._retriever_patcher.start()

    def teardown_method(self):
        self._retriever_patcher.stop()

    @patch("vector_retail.agents.sentiment._load_finbert")
    @patch("vector_retail.agents.sentiment.yf.Ticker")
    @patch("vector_retail.orchestrator.ChatAnthropic")
    def test_bullish_headlines_produce_positive_sentiment(
        self, mock_llm_class, mock_ticker, mock_load_finbert
    ):
        """Headlines scoring strongly positive should produce positive dominant signal."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Sentiment is broadly positive across holdings."
        )

        # Mock FinBERT: all headlines are bullish
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = _finbert_output(0.85, 0.05) * 4  # 4 headlines, all bullish
        mock_load_finbert.return_value = mock_pipeline

        # Mock yfinance news
        mock_ticker.return_value.news = [
            {"title": f"Strong earnings beat for AAPL Q{i}"} for i in range(4)
        ]

        agent = _make_agent(mock_llm)
        state = _make_state(["AAPL"])
        result = agent.run(state)

        assert result.agent_id == "sentiment_analysis"
        assert result.confidence > 0.0
        scores = result.findings["sentiment_scores"]
        assert "AAPL" in scores
        assert scores["AAPL"]["dominant"] == "positive"
        assert scores["AAPL"]["is_bearish"] is False
        assert len(result.findings["bearish_signals"]) == 0

    @patch("vector_retail.agents.sentiment._load_finbert")
    @patch("vector_retail.agents.sentiment.yf.Ticker")
    @patch("vector_retail.orchestrator.ChatAnthropic")
    def test_bearish_headlines_produce_policy_flag(
        self, mock_llm_class, mock_ticker, mock_load_finbert
    ):
        """
        Headlines scoring negative > 0.40 must:
          1. Set is_bearish=True in SentimentScore
          2. Add SENTIMENT_BEARISH policy flag to AgentResult
          3. Include symbol in bearish_signals list
        """
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Negative sentiment detected. This does not constitute investment advice."
        )

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = _finbert_output(0.05, 0.80) * 5  # 5 headlines, all bearish
        mock_load_finbert.return_value = mock_pipeline

        mock_ticker.return_value.news = [
            {"title": f"TSLA faces regulatory probe {i}"} for i in range(5)
        ]

        agent = _make_agent(mock_llm)
        state = _make_state(["TSLA"])
        result = agent.run(state)

        assert "TSLA" in result.findings["bearish_signals"]
        sentiment_flags = [f for f in result.policy_flags if "SENTIMENT_BEARISH" in f]
        assert len(sentiment_flags) == 1
        assert "TSLA" in sentiment_flags[0]

    @patch("vector_retail.agents.sentiment.yf.Ticker")
    @patch("vector_retail.orchestrator.ChatAnthropic")
    def test_no_news_returns_gracefully_with_penalised_confidence(
        self, mock_llm_class, mock_ticker
    ):
        """
        When no news is available, agent must:
          1. Return a valid AgentResult (never raise)
          2. Have confidence < 0.95 (base) due to penalty
          3. Have empty sentiment_scores
        """
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="No news available for any holding.")
        mock_ticker.return_value.news = []  # No news at all

        agent = _make_agent(mock_llm)
        state = _make_state(["AAPL"])
        result = agent.run(state)

        assert result.agent_id == "sentiment_analysis"
        assert result.confidence < 0.95, "Confidence must be penalised when no news"
        assert result.findings["sentiment_scores"] == {}
        assert result.findings["bearish_signals"] == []

    @patch("vector_retail.agents.sentiment._load_finbert")
    @patch("vector_retail.agents.sentiment.yf.Ticker")
    @patch("vector_retail.orchestrator.ChatAnthropic")
    def test_finbert_failure_degrades_gracefully(
        self, mock_llm_class, mock_ticker, mock_load_finbert
    ):
        """
        If FinBERT fails to load (e.g. download error), the agent must:
          1. Not raise an exception
          2. Return AgentResult with reduced confidence
          3. Include 'model: unavailable' in findings
        """
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Sentiment analysis unavailable due to model error."
        )

        # Simulate model load failure
        mock_load_finbert.side_effect = RuntimeError("Model download failed: connection timeout")

        mock_ticker.return_value.news = [{"title": "AAPL reports record revenue"}]

        agent = _make_agent(mock_llm)
        state = _make_state(["AAPL"])
        result = agent.run(state)

        assert result.agent_id == "sentiment_analysis"
        assert result.confidence < 0.95, "Confidence must be penalised on model failure"
        assert result.findings.get("model") == "unavailable"

    @patch("vector_retail.agents.sentiment._load_finbert")
    @patch("vector_retail.agents.sentiment.yf.Ticker")
    @patch("vector_retail.orchestrator.ChatAnthropic")
    def test_multi_symbol_batch_inference(self, mock_llm_class, mock_ticker, mock_load_finbert):
        """
        Multiple symbols should all appear in sentiment_scores.
        Verifies that batch inference correctly routes results back to symbols.
        """
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Mixed sentiment across holdings.")

        # 3 bullish headlines per symbol — 6 total in batch
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = _finbert_output(0.75, 0.10) * 6
        mock_load_finbert.return_value = mock_pipeline

        mock_ticker.return_value.news = [{"title": f"Headline {i}"} for i in range(3)]

        agent = _make_agent(mock_llm)
        state = _make_state(["AAPL", "MSFT"])
        result = agent.run(state)

        scores = result.findings["sentiment_scores"]
        assert "AAPL" in scores
        assert "MSFT" in scores
        assert scores["AAPL"]["n_headlines"] == 3
        assert scores["MSFT"]["n_headlines"] == 3

    @patch("vector_retail.agents.sentiment._load_finbert")
    @patch("vector_retail.agents.sentiment.yf.Ticker")
    @patch("vector_retail.orchestrator.ChatAnthropic")
    def test_confidence_penalised_for_thin_news(
        self, mock_llm_class, mock_ticker, mock_load_finbert
    ):
        """
        When only 1 headline is available (below MIN_HEADLINES_FOR_SIGNAL=2),
        confidence must be penalised.
        """
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Limited news available.")

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = _finbert_output(0.60, 0.20) * 1  # Only 1 headline
        mock_load_finbert.return_value = mock_pipeline

        mock_ticker.return_value.news = [{"title": "Single headline only"}]

        agent = _make_agent(mock_llm)
        state = _make_state(["AAPL"])

        # Capture confidence with thin news
        result_thin = agent.run(state)

        # Reset mock to give ample headlines
        mock_pipeline.return_value = _finbert_output(0.60, 0.20) * 5
        mock_ticker.return_value.news = [{"title": f"Headline {i}"} for i in range(5)]
        agent2 = _make_agent(mock_llm)
        result_full = agent2.run(state)

        assert (
            result_thin.confidence < result_full.confidence
        ), "Thin news should produce lower confidence than ample news"

    @patch("vector_retail.agents.sentiment._load_finbert")
    @patch("vector_retail.agents.sentiment.yf.Ticker")
    @patch("vector_retail.orchestrator.ChatAnthropic")
    def test_data_sources_always_populated(self, mock_llm_class, mock_ticker, mock_load_finbert):
        """AgentResult.data_sources must always reference both yfinance and FinBERT."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Neutral sentiment overall.")

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = _finbert_output(0.33, 0.33) * 3
        mock_load_finbert.return_value = mock_pipeline
        mock_ticker.return_value.news = [{"title": f"VTI headline {i}"} for i in range(3)]

        agent = _make_agent(mock_llm)
        result = agent.run(_make_state(["VTI"]))

        assert "yfinance_news" in result.data_sources
        assert any(
            "finbert" in src for src in result.data_sources
        ), "data_sources must include the FinBERT model identifier"

    @patch("vector_retail.agents.sentiment._load_finbert")
    @patch("vector_retail.agents.sentiment.yf.Ticker")
    @patch("vector_retail.orchestrator.ChatAnthropic")
    def test_news_fetch_failure_produces_no_crash(
        self, mock_llm_class, mock_ticker, mock_load_finbert
    ):
        """yfinance news fetch failure must be absorbed — no uncaught exception."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="No data available.")

        mock_ticker.side_effect = Exception("yfinance connection timeout")
        mock_load_finbert.return_value = MagicMock()

        agent = _make_agent(mock_llm)
        result = agent.run(_make_state(["AAPL"]))

        # Must return a valid AgentResult, not raise
        assert result.agent_id == "sentiment_analysis"
        assert isinstance(result.confidence, float)
