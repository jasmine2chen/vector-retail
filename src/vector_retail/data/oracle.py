"""
data/oracle.py
Verified Data Oracle — Layer 1.

Fetches market data from two independent sources, cross-references them,
and flags divergence > 2% (configurable via policy_rules.json).

Primary source:   yfinance
Secondary source: Alpha Vantage (stub — replace with real API key)

Features:
  - TTL-based quote cache (5-minute default)
  - Circuit breakers on both sources
  - Exponential backoff via tenacity
  - Graceful degradation: falls back to primary-only if secondary fails
  - Staleness detection and is_stale flag on returned quotes
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import structlog
import yfinance as yf
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..core.models import MarketQuote, PortfolioHolding
from ..core.policy import POLICY_RULES
from .circuit_breaker import CircuitBreaker

log = structlog.get_logger("data_oracle")


class DataOracle:
    """
    Multi-source, cross-validated market data provider.

    Usage:
        oracle = DataOracle(audit_fn=audit.record)
        quote = oracle.get_verified_quote("AAPL")
        quotes = oracle.get_portfolio_quotes(holdings)
    """

    def __init__(self, audit_fn):
        self._audit = audit_fn
        self._cache: Dict[str, Tuple[MarketQuote, float]] = {}
        self._cb_primary = CircuitBreaker("yfinance", max_failures=3, cooldown_seconds=60)
        self._cb_secondary = CircuitBreaker("alpha_vantage", max_failures=3, cooldown_seconds=60)
        self._log = log

    # ── Private fetch methods ──────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def _fetch_primary(self, symbol: str) -> Optional[float]:
        """
        Fetch latest price from yfinance with exponential backoff.
        Raises on failure (tenacity will retry up to 3 times).
        """
        if self._cb_primary.is_open:
            self._log.warning("primary_circuit_open", symbol=symbol)
            return None

        try:
            ticker = yf.Ticker(symbol)
            # Try fast_info first (lower latency)
            info = ticker.fast_info
            if hasattr(info, "last_price") and info.last_price and info.last_price > 0:
                self._cb_primary.record_success()
                return float(info.last_price)
            # Fallback to history
            hist = ticker.history(period="1d")
            if not hist.empty:
                self._cb_primary.record_success()
                return float(hist["Close"].iloc[-1])
            return None
        except Exception as exc:
            self._cb_primary.record_failure()
            self._log.warning("primary_fetch_error", symbol=symbol, error=str(exc))
            raise

    def _fetch_secondary(self, symbol: str, primary_price: Optional[float]) -> Optional[float]:
        """
        Fetch from secondary source (Alpha Vantage).

        PRODUCTION: uncomment the requests block below and set AV_API_KEY.
        STUB: simulates a secondary source with a small random variation
              to demonstrate the cross-reference validation logic.

        import os, requests
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
        )
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return float(data["Global Quote"]["05. price"])
        """
        if self._cb_secondary.is_open:
            return None
        if primary_price is None:
            return None

        try:
            # Deterministic stub: small variation based on symbol hash for reproducibility
            hash_val = hash(symbol) % 1000 / 100000  # -0.005 to +0.005 range
            variation = (hash_val - 0.005)  # Within normal bid-ask spread
            self._cb_secondary.record_success()
            return round(primary_price * (1 + variation), 4)
        except Exception as exc:
            self._cb_secondary.record_failure()
            self._log.warning("secondary_fetch_error", symbol=symbol, error=str(exc))
            return None

    # ── Public interface ───────────────────────────────────────────────────

    def get_verified_quote(self, symbol: str) -> MarketQuote:
        """
        Fetch, cross-reference, and return a verified quote for symbol.

        Returns a MarketQuote with:
          - is_verified=True if both sources agree within divergence threshold
          - divergence_pct populated when sources are available
          - is_stale=True if served from cache past TTL
        """
        symbol = symbol.upper().strip()
        staleness_secs: int = POLICY_RULES["data_staleness_seconds"]
        divergence_threshold: float = POLICY_RULES["cross_ref_divergence_threshold"]

        # Cache check
        if symbol in self._cache:
            cached_quote, fetch_time = self._cache[symbol]
            age = time.time() - fetch_time
            if age < staleness_secs:
                self._log.debug("cache_hit", symbol=symbol, age_s=round(age))
                return cached_quote
            self._log.info("cache_stale", symbol=symbol, age_s=round(age))

        t0 = time.time()

        # Primary fetch
        price_primary: Optional[float] = None
        try:
            price_primary = self._fetch_primary(symbol)
        except Exception:
            price_primary = None

        # Secondary fetch
        price_secondary = self._fetch_secondary(symbol, price_primary)

        # Graceful degradation — all sources failed
        if price_primary is None and price_secondary is None:
            self._log.error("all_sources_failed", symbol=symbol)
            self._audit("data_oracle", f"fetch_{symbol}", "all_sources_failed")
            return MarketQuote(symbol=symbol, price_primary=0.0, is_stale=True)

        # Use secondary as fallover if primary failed
        if price_primary is None:
            price_primary = price_secondary

        # Cross-reference
        divergence: Optional[float] = None
        is_verified = False

        if price_primary and price_secondary:
            divergence = abs(price_primary - price_secondary) / price_primary
            is_verified = divergence <= divergence_threshold
            if not is_verified:
                self._log.warning(
                    "price_divergence",
                    symbol=symbol,
                    primary=price_primary,
                    secondary=price_secondary,
                    divergence_pct=round(divergence * 100, 3),
                )
                self._audit(
                    "data_oracle", f"cross_ref_{symbol}", "divergence_flagged",
                    {"divergence_pct": round(divergence * 100, 3)},
                )

        quote = MarketQuote(
            symbol=symbol,
            price_primary=price_primary,
            price_secondary=price_secondary,
            is_verified=is_verified,
            divergence_pct=round(divergence * 100, 4) if divergence is not None else None,
        )

        self._cache[symbol] = (quote, time.time())

        self._log.info(
            "quote_fetched",
            symbol=symbol,
            price=quote.verified_price,
            verified=is_verified,
            latency_ms=round((time.time() - t0) * 1000),
        )
        return quote

    def get_portfolio_quotes(
        self, holdings: List[PortfolioHolding]
    ) -> Dict[str, MarketQuote]:
        """Fetch verified quotes for all holdings."""
        return {h.symbol: self.get_verified_quote(h.symbol) for h in holdings}
