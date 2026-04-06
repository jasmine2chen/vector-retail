"""
agents/sentiment.py
Sentiment Analysis Agent — Layer 3 (6th parallel specialist).

Uses FinBERT (ProsusAI/finbert) — a BERT model fine-tuned on financial text
(Malo et al., 2014; Araci, 2019) — to score recent news headlines for each
holding. Provides a data-driven, model-backed sentiment signal alongside the
rule-based analysis from the other five agents.

Architecture:
  - FinBERT is loaded lazily at class level (thread-safe singleton).
    The first call triggers a one-time ~500MB model download to
    ~/.cache/huggingface/. Subsequent calls use the local cache.
    For production Docker images, pre-bake with:
        RUN python -c "from transformers import pipeline; \
            pipeline('text-classification', model='ProsusAI/finbert')"

  - Headlines are fetched from yfinance.Ticker.news (typically 8–12 per symbol).
    Batch inference is used for efficiency (all headlines across all symbols
    in a single forward pass).

  - Sentiment is aggregated as a weighted mean, with recency weighting applied
    so the most recent headline contributes more than older ones.

  - Confidence degrades gracefully:
      * No news available  → penalty applied, neutral sentiment assumed
      * Model load failure → penalty applied, agent returns gracefully
      * LLM commentary failure → penalty applied, raw scores surfaced

Why FinBERT over general BERT:
  FinBERT is fine-tuned on ~10,000 sentences from financial news and earnings
  releases. On Financial PhraseBank (all-agree split), it achieves ~97% accuracy
  versus ~73% for general BERT and ~78% for a TF-IDF + LR baseline.
  See notebooks/model_evaluation.ipynb for the full baseline comparison.

References:
  ProsusAI/finbert: https://huggingface.co/ProsusAI/finbert
  Araci (2019): https://arxiv.org/abs/1908.10063
  Malo et al. (2014): Financial PhraseBank dataset
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import structlog
import yfinance as yf

from ..core.models import AgentResult, GraphState, PortfolioHolding
from ..core.prompts import get_system_prompt
from .base import BaseFinanceAgent

log = structlog.get_logger("agent.sentiment")

# ── FinBERT model singleton ────────────────────────────────────────────────────
# Shared across all agent instances to avoid redundant memory allocation.
# Thread-safe double-checked locking ensures only one load happens concurrently.
_finbert_pipeline: Any = None
_finbert_lock = threading.Lock()
_finbert_load_error: str | None = None  # Cached load error so we fail-fast on retry

_FINBERT_MODEL = "ProsusAI/finbert"
_MAX_HEADLINES_PER_SYMBOL = 8
_MAX_SYMBOLS_FOR_SENTIMENT = 10  # Cap to control latency
_MIN_HEADLINES_FOR_SIGNAL = 2  # Below this we flag low-data confidence

_FALLBACK_SYSTEM_PROMPT = (
    "You are a market sentiment analyst at a regulated retail brokerage. "
    "Interpret FinBERT sentiment scores from recent news headlines. "
    "Flag strongly negative signals as potential risk factors. "
    "Note that sentiment is a noisy, lagging signal — never predict prices from it. "
    "Be concise (3-5 sentences)."
)


def _load_finbert() -> Any:
    """
    Load the FinBERT text-classification pipeline (thread-safe, lazy).

    Returns the pipeline on success, raises RuntimeError on failure.
    Caches the load error so subsequent calls fail immediately rather than
    retrying an expensive download that already failed.
    """
    global _finbert_pipeline, _finbert_load_error

    if _finbert_pipeline is not None:
        return _finbert_pipeline

    if _finbert_load_error is not None:
        raise RuntimeError(f"FinBERT load previously failed: {_finbert_load_error}")

    with _finbert_lock:
        # Double-checked locking — re-test inside the lock
        if _finbert_pipeline is not None:
            return _finbert_pipeline
        if _finbert_load_error is not None:
            raise RuntimeError(f"FinBERT load previously failed: {_finbert_load_error}")

        try:
            from transformers import pipeline  # type: ignore[import-untyped]

            log.info("finbert_loading", model=_FINBERT_MODEL)
            t0 = time.time()
            _finbert_pipeline = pipeline(
                "text-classification",
                model=_FINBERT_MODEL,
                return_all_scores=True,  # Get all three class scores, not just argmax
                device=-1,  # CPU; set to 0 to use first GPU
                truncation=True,
                max_length=512,
            )
            log.info(
                "finbert_loaded",
                model=_FINBERT_MODEL,
                load_ms=round((time.time() - t0) * 1000),
            )
            return _finbert_pipeline

        except Exception as exc:
            _finbert_load_error = str(exc)
            log.error("finbert_load_failed", model=_FINBERT_MODEL, error=str(exc))
            raise RuntimeError(f"FinBERT failed to load: {exc}") from exc


class SentimentScore:
    """
    Structured sentiment result for a single symbol.

    Attributes:
        symbol:         Ticker symbol (e.g. "AAPL")
        positive:       Mean positive-class probability across headlines [0, 1]
        negative:       Mean negative-class probability across headlines [0, 1]
        neutral:        Mean neutral-class probability across headlines [0, 1]
        dominant:       Dominant sentiment label ("positive"|"negative"|"neutral")
        n_headlines:    Number of headlines scored
        is_bearish:     True when negative > 0.40 — flagged as a risk signal
    """

    __slots__ = (
        "symbol",
        "positive",
        "negative",
        "neutral",
        "dominant",
        "n_headlines",
        "is_bearish",
    )

    def __init__(
        self,
        symbol: str,
        positive: float,
        negative: float,
        neutral: float,
        n_headlines: int,
    ) -> None:
        self.symbol = symbol
        self.positive = round(positive, 4)
        self.negative = round(negative, 4)
        self.neutral = round(neutral, 4)
        self.n_headlines = n_headlines
        self.dominant = max(
            ("positive", positive),
            ("negative", negative),
            ("neutral", neutral),
            key=lambda x: x[1],
        )[0]
        self.is_bearish = negative > 0.40

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "positive": self.positive,
            "negative": self.negative,
            "neutral": self.neutral,
            "dominant": self.dominant,
            "n_headlines": self.n_headlines,
            "is_bearish": self.is_bearish,
        }


class SentimentAnalysisAgent(BaseFinanceAgent):
    """
    FinBERT-powered news sentiment agent.

    Fetches recent news headlines via yfinance, runs batch FinBERT inference,
    and produces per-symbol sentiment scores with recency weighting.
    An LLM pass contextualises the raw scores in plain language for the client.
    """

    AGENT_ID = "sentiment_analysis"
    PROMPT_FALLBACK = _FALLBACK_SYSTEM_PROMPT

    # ── News fetching ──────────────────────────────────────────────────────

    def _fetch_headlines(self, symbol: str) -> list[tuple[str, int]]:
        """
        Fetch recent news headlines for a symbol from yfinance.

        Returns:
            List of (headline_text, recency_rank) tuples.
            recency_rank=0 is most recent; used for decay weighting.
        """
        try:
            ticker = yf.Ticker(symbol)
            news_items = ticker.news or []
            headlines = []
            for rank, item in enumerate(news_items[:_MAX_HEADLINES_PER_SYMBOL]):
                title = item.get("title", "").strip()
                if title and len(title) > 5:
                    headlines.append((title, rank))
            return headlines
        except Exception as exc:
            self._log.warning("news_fetch_failed", symbol=symbol, error=str(exc))
            return []

    # ── FinBERT inference ──────────────────────────────────────────────────

    def _run_finbert_batch(
        self,
        headlines_by_symbol: dict[str, list[tuple[str, int]]],
    ) -> dict[str, SentimentScore]:
        """
        Run FinBERT inference in a single batch across all symbols' headlines.

        Batching is critical for throughput: processing 40 headlines as one
        batch is ~4× faster than 40 individual calls due to parallelised
        matrix multiplication.

        Applies exponential recency decay: headline at rank r contributes
        weight = exp(-0.15 * r), so the most recent headline has weight 1.0
        and the 8th headline has weight ~0.33.

        Args:
            headlines_by_symbol: {symbol: [(headline, recency_rank), ...]}

        Returns:
            {symbol: SentimentScore}
        """
        model = _load_finbert()

        # Flatten all headlines into a single list, tracking provenance
        flat_headlines: list[str] = []
        flat_meta: list[tuple[str, int]] = []  # (symbol, recency_rank)

        for symbol, items in headlines_by_symbol.items():
            for text, rank in items:
                flat_headlines.append(text)
                flat_meta.append((symbol, rank))

        if not flat_headlines:
            return {}

        # Single batched forward pass — O(n) not O(n²)
        try:
            batch_results = model(flat_headlines, batch_size=16)
        except Exception as exc:
            self._log.error("finbert_inference_failed", error=str(exc))
            raise

        # Aggregate results per symbol with recency decay
        symbol_accum: dict[str, dict[str, list[float]]] = {
            sym: {"positive": [], "negative": [], "neutral": [], "weights": []}
            for sym in headlines_by_symbol
        }

        for (symbol, rank), scores_list in zip(flat_meta, batch_results, strict=False):
            weight = float(np.exp(-0.15 * rank))  # Recency decay
            scores_map = {s["label"].lower(): s["score"] for s in scores_list}
            accum = symbol_accum[symbol]
            accum["positive"].append(scores_map.get("positive", 0.0))
            accum["negative"].append(scores_map.get("negative", 0.0))
            accum["neutral"].append(scores_map.get("neutral", 1.0))
            accum["weights"].append(weight)

        results: dict[str, SentimentScore] = {}
        for symbol, accum in symbol_accum.items():
            weights = np.array(accum["weights"])
            if weights.sum() == 0:
                continue
            results[symbol] = SentimentScore(
                symbol=symbol,
                positive=float(np.average(accum["positive"], weights=weights)),
                negative=float(np.average(accum["negative"], weights=weights)),
                neutral=float(np.average(accum["neutral"], weights=weights)),
                n_headlines=len(accum["weights"]),
            )

        return results

    # ── Main agent entrypoint ──────────────────────────────────────────────

    def run(self, state: GraphState) -> AgentResult:
        t0 = time.time()
        reasoning: list[str] = []
        conf = self._confidence()
        policy_flags: list[str] = []

        holdings = [PortfolioHolding(**h) for h in state.holdings]
        symbols = [h.symbol for h in holdings[:_MAX_SYMBOLS_FOR_SENTIMENT]]

        # ── Fetch headlines ────────────────────────────────────────────────
        headlines_by_symbol: dict[str, list[tuple[str, int]]] = {}
        total_headlines = 0
        for sym in symbols:
            items = self._fetch_headlines(sym)
            headlines_by_symbol[sym] = items
            total_headlines += len(items)
            if not items:
                conf.penalize(
                    "insufficient_data",
                    f"No news headlines found for {sym}",
                    observed=0,
                )

        reasoning.append(f"Fetched {total_headlines} headlines across {len(symbols)} symbols")

        symbols_with_news = [s for s, h in headlines_by_symbol.items() if h]
        if not symbols_with_news:
            # No news at all — return gracefully with low confidence
            conf.penalize("missing_quote", "No news available for any holding")
            self.audit.record(
                "agent",
                self.AGENT_ID,
                "completed_no_data",
                {"symbols": symbols, "total_headlines": 0},
            )
            return AgentResult(
                agent_id=self.AGENT_ID,
                prompt_version=self._prompt_version,
                confidence=conf.score(),
                reasoning_chain=reasoning,
                findings={
                    "sentiment_scores": {},
                    "bearish_signals": [],
                    "llm_sentiment_commentary": (
                        "No recent news headlines were available for any holding."
                    ),
                    "model": _FINBERT_MODEL,
                },
                data_sources=["yfinance_news"],
                latency_ms=round((time.time() - t0) * 1000),
            )

        # ── FinBERT inference ──────────────────────────────────────────────
        sentiment_scores: dict[str, SentimentScore] = {}
        model_used = _FINBERT_MODEL

        try:
            t_infer = time.time()
            sentiment_scores = self._run_finbert_batch(
                {s: headlines_by_symbol[s] for s in symbols_with_news}
            )
            infer_ms = round((time.time() - t_infer) * 1000)
            reasoning.append(
                f"FinBERT batch inference: {len(sentiment_scores)} symbols in {infer_ms}ms"
            )
            self._log.info(
                "finbert_inference_complete",
                symbols=len(sentiment_scores),
                headlines=total_headlines,
                latency_ms=infer_ms,
            )
        except RuntimeError as exc:
            # FinBERT unavailable (download failed, import error, etc.)
            # Degrade gracefully: skip sentiment, log warning, penalise confidence
            conf.penalize("llm_failure", f"FinBERT unavailable: {exc}")
            model_used = "unavailable"
            reasoning.append(f"FinBERT inference failed — model unavailable: {exc}")
            self._log.warning("finbert_skipped", error=str(exc))

        # ── Analyse results ────────────────────────────────────────────────
        bearish_signals: list[str] = []
        for sym, score in sentiment_scores.items():
            if score.n_headlines < _MIN_HEADLINES_FOR_SIGNAL:
                conf.penalize(
                    "insufficient_data",
                    f"{sym}: only {score.n_headlines} headline(s) — low signal reliability",
                    observed=score.n_headlines,
                )
            if score.is_bearish:
                bearish_signals.append(sym)
                policy_flags.append(
                    f"SENTIMENT_BEARISH: {sym} — negative sentiment {score.negative:.0%} "
                    f"across {score.n_headlines} headlines (threshold: 40%)"
                )
                reasoning.append(
                    f"{sym}: bearish signal — negative={score.negative:.0%}, "
                    f"positive={score.positive:.0%}"
                )
            else:
                reasoning.append(
                    f"{sym}: dominant={score.dominant} "
                    f"(pos={score.positive:.0%} neg={score.negative:.0%} "
                    f"neu={score.neutral:.0%}) from {score.n_headlines} headlines"
                )

        # ── LLM contextualisation ──────────────────────────────────────────
        scores_summary = (
            "\n".join(
                f"  {sym}: positive={s.positive:.0%}, negative={s.negative:.0%}, "
                f"neutral={s.neutral:.0%}, dominant={s.dominant}, "
                f"n_headlines={s.n_headlines}, bearish={s.is_bearish}"
                for sym, s in sentiment_scores.items()
            )
            or "  No scores available."
        )

        system_prompt = get_system_prompt(self.AGENT_ID, fallback=_FALLBACK_SYSTEM_PROMPT)

        llm_commentary = self._call_llm(
            system_prompt=system_prompt,
            user_content=(
                f"Client query: {state.user_query}\n\n"
                f"FinBERT sentiment scores (model: {model_used}):\n{scores_summary}\n\n"
                f"Bearish signals: {bearish_signals or 'None'}\n\n"
                "Interpret these scores for the client. Flag any bearish signals "
                "as risk factors. Remind that sentiment is noisy and should not "
                "drive investment decisions alone."
            ),
        )

        if self._is_llm_error(llm_commentary):
            conf.penalize("llm_failure", "Sentiment commentary LLM call failed")

        # ── Audit ──────────────────────────────────────────────────────────
        self.audit.record(
            "agent",
            self.AGENT_ID,
            "completed",
            {
                "symbols_analysed": len(sentiment_scores),
                "total_headlines": total_headlines,
                "bearish_signals": bearish_signals,
                "model": model_used,
            },
        )

        return AgentResult(
            agent_id=self.AGENT_ID,
            prompt_version=self._prompt_version,
            confidence=conf.score(),
            reasoning_chain=reasoning,
            findings={
                "sentiment_scores": {
                    sym: score.to_dict() for sym, score in sentiment_scores.items()
                },
                "bearish_signals": bearish_signals,
                "llm_sentiment_commentary": llm_commentary,
                "model": model_used,
                "total_headlines_analysed": total_headlines,
            },
            policy_flags=policy_flags,
            data_sources=["yfinance_news", f"finbert:{model_used}"],
            latency_ms=round((time.time() - t0) * 1000),
        )
