"""
data/retriever.py
Market News Retrieval — RAG layer for the SentimentAnalysisAgent.

Retrieves semantically relevant recent news articles from a local ChromaDB
vector collection. Articles are ingested daily via scripts/ingest_news.py
(yfinance.Ticker.news → embed → upsert).

Why news instead of SEC filings:
  SEC filings are 30-365 days old. The sentiment agent scores current market
  mood from recent headlines — the RAG layer should provide the same signal
  at article depth, not backward-looking disclosures. A 7-day staleness
  threshold enforces recency: if the news corpus is older than 7 days,
  confidence is penalised.

Design principles:
  - Thread-safe singleton — one ChromaDB client per process (mirrors FinBERT pattern)
  - Graceful degradation — empty or unavailable store returns [] with no crash
  - TTL-aware — articles carry age_days; agent penalises confidence for stale data
  - Swappable backend — replace ChromaDB with Pinecone/pgvector by implementing
    the same retrieve() interface; no agent code changes required

Collection schema:
  documents  : enriched article text (title + publisher + related tickers)
  metadatas  : {symbol, source, published_date, url, article_index, ingested_at}
  ids        : "{symbol}_{published_date}_{article_index}"
  similarity : cosine (HNSW index)

Embedding model:
  BAAI/bge-small-en-v1.5 — 130MB, CPU-only, top MTEB retrieval leaderboard score
  for its size class. No API key required. Pre-downloaded to ~/.cache/huggingface/.

Production notes:
  - ChromaDB PersistentClient suits single-node deployment (same host as the app).
  - For multi-pod Kubernetes: swap to pgvector or Pinecone without changing agent code.

Ingest before use (run daily to keep corpus fresh):
    python scripts/ingest_news.py --symbols AAPL MSFT TSLA NVDA
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import structlog

log = structlog.get_logger("news.retriever")

# ── Configuration ─────────────────────────────────────────────────────────────
_DEFAULT_DB_PATH = Path(os.getenv("NEWS_DB_PATH", ".chroma_db/market_news"))
_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"   # 130MB; strong MTEB retrieval score
_DEFAULT_TOP_K = 4
_STALE_WARN_DAYS = 3      # Warn in logs but still use
_STALE_PENALISE_DAYS = 7  # Return staleness flag for confidence penalty


# ── Data contract ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RetrievedPassage:
    """A single retrieved article from the market news corpus."""

    symbol: str
    text: str
    source: str             # Publisher / news outlet
    published_date: str     # ISO-8601 date of publication
    url: str                # Article URL for provenance
    relevance_score: float  # cosine similarity [0, 1]; higher = more relevant
    age_days: int           # Calendar days since publication


# ── Thread-safe singleton ─────────────────────────────────────────────────────

_instance: "NewsRetriever | None" = None
_init_lock = threading.Lock()
_init_error: str | None = None  # Cached so we fail-fast on repeated init attempts


def get_retriever(
    db_path: Path = _DEFAULT_DB_PATH,
    model_name: str = _DEFAULT_MODEL,
) -> "NewsRetriever | None":
    """
    Return the process-level singleton NewsRetriever.

    Returns None — never raises — if ChromaDB or sentence-transformers are
    unavailable, or if the DB path does not exist yet (pre-ingestion).
    Callers must handle None and degrade gracefully.
    """
    global _instance, _init_error

    if _instance is not None:
        return _instance
    if _init_error is not None:
        log.debug("retriever_previously_failed", error=_init_error)
        return None

    with _init_lock:
        # Double-checked locking — re-test inside lock
        if _instance is not None:
            return _instance
        if _init_error is not None:
            return None
        try:
            _instance = NewsRetriever(db_path=db_path, model_name=model_name)
            log.info("retriever_ready", db_path=str(db_path), model=model_name)
            return _instance
        except Exception as exc:
            _init_error = str(exc)
            log.warning("retriever_init_failed", error=str(exc))
            return None


# ── Retriever ─────────────────────────────────────────────────────────────────

class NewsRetriever:
    """
    Semantic retriever over market news corpus stored in ChromaDB.

    Instantiate via get_retriever() to ensure singleton behaviour.
    Direct instantiation is fine for testing.
    """

    def __init__(
        self,
        db_path: Path = _DEFAULT_DB_PATH,
        model_name: str = _DEFAULT_MODEL,
        top_k: int = _DEFAULT_TOP_K,
    ) -> None:
        # Lazy import — allows module to be imported even if chromadb not installed.
        # SentimentAgent will degrade gracefully if retriever returns None.
        import chromadb
        from chromadb.config import Settings
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )

        self._top_k = top_k
        self._log = log

        db_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(
                anonymized_telemetry=False,  # No external telemetry — fintech requirement
                allow_reset=False,            # Prevent accidental collection wipe
            ),
        )
        self._ef = SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            device="cpu",
            normalize_embeddings=True,       # Required for cosine similarity to be correct
        )
        self._collection = self._client.get_or_create_collection(
            name="market_news",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._ef,
        )
        self._log.info(
            "collection_loaded",
            name="market_news",
            total_articles=self._collection.count(),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        symbol: str,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedPassage]:
        """
        Return top-k recent news articles for symbol semantically relevant to query.

        Returns [] if:
          - No articles exist for this symbol (not yet ingested)
          - ChromaDB query fails for any reason
        Never raises — caller degrades gracefully on empty result.
        """
        k = top_k or self._top_k
        total = self._collection.count()
        if total == 0:
            return []

        try:
            result = self._collection.query(
                query_texts=[query],
                n_results=min(k, total),
                where={"symbol": {"$eq": symbol}},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            self._log.warning("retrieval_failed", symbol=symbol, error=str(exc))
            return []

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        if not docs:
            return []

        today = datetime.now(UTC).date()
        passages: list[RetrievedPassage] = []
        for text, meta, dist in zip(docs, metas, dists):
            pub_date = meta.get("published_date", "")
            try:
                age = (today - datetime.fromisoformat(pub_date).date()).days
            except ValueError:
                age = 0

            # Distance is 1 - cosine_similarity for cosine space in ChromaDB
            relevance = round(max(0.0, 1.0 - float(dist)), 4)

            passages.append(
                RetrievedPassage(
                    symbol=symbol,
                    text=text,
                    source=meta.get("source", ""),
                    published_date=pub_date,
                    url=meta.get("url", ""),
                    relevance_score=relevance,
                    age_days=age,
                )
            )

        self._log.debug(
            "retrieved",
            symbol=symbol,
            n=len(passages),
            top_score=passages[0].relevance_score if passages else 0.0,
            top_age_days=passages[0].age_days if passages else None,
        )
        return passages

    def has_documents(self, symbol: str) -> bool:
        """True if at least one article exists for this symbol."""
        try:
            result = self._collection.get(
                where={"symbol": {"$eq": symbol}},
                limit=1,
                include=["metadatas"],
            )
            return len(result.get("ids", [])) > 0
        except Exception:
            return False

    def get_newest_age_days(self, symbol: str) -> int | None:
        """
        Days since most recent article for symbol.
        Returns None if no articles exist.
        Used by SentimentAgent to apply staleness confidence penalty.
        """
        try:
            result = self._collection.get(
                where={"symbol": {"$eq": symbol}},
                include=["metadatas"],
            )
            dates = [
                m.get("published_date", "")
                for m in result.get("metadatas", [])
                if m.get("published_date")
            ]
            if not dates:
                return None
            parsed = []
            for d in dates:
                try:
                    parsed.append(datetime.fromisoformat(d).date())
                except ValueError:
                    continue
            if not parsed:
                return None
            newest = max(parsed)
            return (datetime.now(UTC).date() - newest).days
        except Exception:
            return None

    def collection_count(self) -> int:
        """Total articles in the collection across all symbols."""
        try:
            return self._collection.count()
        except Exception:
            return 0
