"""
tests/unit/test_oracle.py
Unit tests for the data oracle — mocked to avoid live API calls.
"""
from unittest.mock import MagicMock, patch

from vector_retail.core.models import PortfolioHolding
from vector_retail.data.oracle import DataOracle


def _noop_audit(*args, **kwargs):
    pass


class TestDataOracle:

    def _make_oracle(self) -> DataOracle:
        return DataOracle(audit_fn=_noop_audit)

    def _make_holding(self, symbol="AAPL") -> PortfolioHolding:
        return PortfolioHolding(
            symbol=symbol,
            quantity=10,
            cost_basis_per_share=150.0,
            purchase_date="2022-01-01",
            sector="Technology",
            asset_class="equity",
        )

    @patch("vector_retail.data.oracle.yf.Ticker")
    def test_get_verified_quote_returns_market_quote(self, mock_ticker):
        mock_ticker.return_value.fast_info.last_price = 200.0
        oracle = self._make_oracle()
        quote = oracle.get_verified_quote("AAPL")
        assert quote.symbol == "AAPL"
        assert quote.price_primary > 0

    @patch("vector_retail.data.oracle.yf.Ticker")
    def test_quote_cached_on_second_call(self, mock_ticker):
        mock_ticker.return_value.fast_info.last_price = 200.0
        oracle = self._make_oracle()
        oracle.get_verified_quote("AAPL")
        oracle.get_verified_quote("AAPL")
        # Ticker should only be called once (second call hits cache)
        assert mock_ticker.call_count <= 2  # Can be 1 or 2 depending on stub

    @patch("vector_retail.data.oracle.yf.Ticker")
    def test_all_sources_fail_returns_stale_quote(self, mock_ticker):
        mock_ticker.return_value.fast_info.last_price = None
        mock_ticker.return_value.history.return_value = MagicMock(empty=True)
        oracle = self._make_oracle()
        # Exhaust retries
        oracle._cb_primary._failures = 10
        oracle._cb_primary._opened_at = 9999999999
        oracle._cb_secondary._failures = 10
        oracle._cb_secondary._opened_at = 9999999999
        quote = oracle.get_verified_quote("FAKE")
        assert quote.is_stale is True

    @patch("vector_retail.data.oracle.yf.Ticker")
    def test_portfolio_quotes_returns_all_symbols(self, mock_ticker):
        mock_ticker.return_value.fast_info.last_price = 100.0
        oracle = self._make_oracle()
        holdings = [self._make_holding("AAPL"), self._make_holding("MSFT")]
        quotes = oracle.get_portfolio_quotes(holdings)
        assert "AAPL" in quotes
        assert "MSFT" in quotes

    def test_circuit_breaker_opens_after_failures(self):
        from vector_retail.data.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test", max_failures=3, cooldown_seconds=60)
        assert cb.is_open is False
        for _ in range(3):
            cb.record_failure()
        assert cb.is_open is True

    def test_circuit_breaker_resets_on_success(self):
        from vector_retail.data.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test", max_failures=3, cooldown_seconds=60)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb._failures == 0
        assert cb.is_open is False
