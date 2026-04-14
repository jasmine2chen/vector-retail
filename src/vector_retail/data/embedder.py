"""
data/embedder.py
Real-time news ingestion pipeline — fetch → embed → upsert into ChromaDB.

Architecture:
  NewsEmbedder — fetches recent news articles per ticker symbol from
                 yfinance.Ticker.news (no API key required), builds
                 enriched text documents, and upserts into a local
                 ChromaDB collection at .chroma_db/market_news/.

Why real-time news over SEC filings:
  The SentimentAnalysisAgent scores current market mood from recent headlines.
  SEC filings are 30-365 days old and backward-looking. News articles from
  the past 7 days capture analyst reactions, earnings surprises, regulatory
  actions, and macro events — the same signals that move prices today.
  RAG over recent news gives the LLM specific, current context to ground
  its commentary rather than relying on stale quarterly disclosures.

Data source:
  yfinance.Ticker.news — 8-12 most recent articles per symbol.
  Fields used: title, publisher, link, providerPublishTime, relatedTickers.
  No API key required; same dependency already used by DataOracle and
  SentimentAgent for headline fetching.

Staleness:
  Articles older than 7 days are flagged by the retriever. Run
  scripts/ingest_news.py daily (cron/scheduler) to keep the corpus fresh.

Usage (called by scripts/ingest_news.py):
    from vector_retail.data.embedder import NewsEmbedder
    emb = NewsEmbedder()
    emb.ingest(symbol="AAPL")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger("news.embedder")

_UPSERT_BATCH = 50


# ── Data class ────────────────────────────────────────────────────────────────


@dataclass
class NewsChunk:
    """A single news article document ready for embedding and upsert."""

    symbol: str
    source: str          # Publisher name
    published_date: str  # ISO-8601 date
    url: str             # Article link
    article_index: int   # Ordering within the batch (0 = most recent)
    text: str            # Enriched text: title + publisher + related tickers
    ingested_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    @property
    def doc_id(self) -> str:
        safe_date = self.published_date.replace("-", "")
        return f"{self.symbol}_{safe_date}_{self.article_index}"

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "source": self.source,
            "published_date": self.published_date,
            "url": self.url,
            "article_index": self.article_index,
            "ingested_at": self.ingested_at,
        }


# ── Orchestrator ──────────────────────────────────────────────────────────────


class NewsEmbedder:
    """
    Orchestrates the news ingestion pipeline:
      fetch articles → build enriched text → embed → upsert into ChromaDB.

    Idempotent: deletes existing articles for a symbol before re-ingesting,
    so daily re-runs are safe (no duplicate chunks).

    Instantiate once; reuse across multiple ingest() calls.
    """

    def __init__(self, db_path: Path | None = None, model_name: str | None = None) -> None:
        from .retriever import (
            NewsRetriever,
            _DEFAULT_DB_PATH,
            _DEFAULT_MODEL,
        )

        self._retriever = NewsRetriever(
            db_path=db_path or _DEFAULT_DB_PATH,
            model_name=model_name or _DEFAULT_MODEL,
        )
        self._log = log

    def ingest(self, symbol: str) -> int:
        """
        Fetch, embed, and upsert recent news articles for symbol.

        Returns the total number of articles upserted.
        Raises nothing — all errors are logged and skipped.
        """
        self._log.info("ingest_start", symbol=symbol)

        chunks = self._fetch_and_build_chunks(symbol)
        if not chunks:
            self._log.warning("no_articles_found", symbol=symbol)
            return 0

        self._delete_existing(symbol=symbol)
        upserted = self._upsert_chunks(chunks)

        self._log.info("ingest_complete", symbol=symbol, articles=upserted)
        return upserted

    # ── Private ───────────────────────────────────────────────────────────────

    def _fetch_and_build_chunks(self, symbol: str) -> list[NewsChunk]:
        """Fetch news from yfinance and build enriched NewsChunk documents."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            news_items = ticker.news or []
        except Exception as exc:
            self._log.warning("news_fetch_failed", symbol=symbol, error=str(exc))
            return []

        chunks: list[NewsChunk] = []
        for idx, item in enumerate(news_items):
            title = item.get("title", "").strip()
            if not title:
                continue

            publisher = item.get("publisher", "")
            url = item.get("link", "")
            pub_ts = item.get("providerPublishTime", 0)
            related = item.get("relatedTickers", [])

            # Convert Unix timestamp to ISO date
            try:
                pub_date = datetime.fromtimestamp(pub_ts, tz=UTC).date().isoformat()
            except Exception:
                pub_date = datetime.now(UTC).date().isoformat()

            # Enriched text: title + publisher + related tickers
            # More context per document → better semantic matches at retrieval time
            text_parts = [title]
            if publisher:
                text_parts.append(f"Source: {publisher}.")
            if related:
                text_parts.append(f"Related tickers: {', '.join(str(t) for t in related[:5])}.")

            chunks.append(
                NewsChunk(
                    symbol=symbol,
                    source=publisher,
                    published_date=pub_date,
                    url=url,
                    article_index=idx,
                    text=" ".join(text_parts),
                )
            )

        self._log.info("articles_fetched", symbol=symbol, count=len(chunks))
        return chunks

    def _delete_existing(self, symbol: str) -> None:
        """Remove all existing articles for symbol (idempotent re-ingest)."""
        try:
            col = self._retriever._collection
            existing = col.get(where={"symbol": {"$eq": symbol}}, include=[])
            ids = existing.get("ids", [])
            if ids:
                col.delete(ids=ids)
                self._log.info("existing_articles_deleted", symbol=symbol, count=len(ids))
        except Exception as exc:
            self._log.warning("delete_existing_failed", symbol=symbol, error=str(exc))

    def _upsert_chunks(self, chunks: list[NewsChunk]) -> int:
        """Upsert chunks in batches. Returns count upserted."""
        col = self._retriever._collection
        upserted = 0

        for i in range(0, len(chunks), _UPSERT_BATCH):
            batch = chunks[i : i + _UPSERT_BATCH]
            try:
                col.upsert(
                    ids=[c.doc_id for c in batch],
                    documents=[c.text for c in batch],
                    metadatas=[c.metadata for c in batch],
                )
                upserted += len(batch)
                self._log.debug("batch_upserted", batch_start=i, batch_size=len(batch))
            except Exception as exc:
                self._log.warning("batch_upsert_failed", batch_start=i, error=str(exc))

        return upserted
