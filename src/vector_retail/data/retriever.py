"""
data/retriever.py
Fundamentals Retrieval — RAG layer for the SentimentAnalysisAgent.

Retrieves semantically relevant passages from SEC filings
(10-K risk factors, 10-Q MD&A) stored in a local ChromaDB vector collection.

Design principles:
  - Thread-safe singleton — one ChromaDB client per process (mirrors FinBERT pattern)
  - Graceful degradation — empty or unavailable store returns [] with no crash
  - TTL-aware — passages carry age_days; agent penalises confidence for stale data
  - Audit-logged — every retrieval records source metadata in the findings dict
  - Swappable backend — replace ChromaDB with Pinecone/pgvector by implementing
    the same retrieve() interface; no agent code changes required

Collection schema:
  documents  : chunked filing text (400-word paragraphs with 50-word overlap)
  metadatas  : {symbol, filing_type, filed_date, section, chunk_index, ingested_at}
  ids        : "{symbol}_{filing_type}_{filed_date}_{chunk_index}"
  similarity : cosine (HNSW index)

Embedding model:
  BAAI/bge-small-en-v1.5 — 130MB, CPU-only, top MTEB retrieval leaderboard score
  for its size class. No API key required. Pre-downloaded to ~/.cache/huggingface/.

Production notes:
  - ChromaDB PersistentClient suits single-node deployment (same host as the app).
  - For multi-pod Kubernetes: swap to pgvector (reuses existing PostgreSQL session
    store, ACID guarantees) or Pinecone (managed, no ops overhead).
  - Collection schema is identical regardless of backend.

Run offline ingestion before first use:
    python scripts/ingest_fundamentals.py --symbols AAPL MSFT TSLA NVDA
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import structlog

log = structlog.get_logger("fundamentals.retriever")

# ── Configuration ─────────────────────────────────────────────────────────────
_DEFAULT_DB_PATH = Path(os.getenv("FUNDAMENTALS_DB_PATH", ".chroma_db/fundamentals"))
_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"   # 130MB; strong MTEB retrieval score
_DEFAULT_TOP_K = 4
_STALE_WARN_DAYS = 90     # Warn in logs but still use
_STALE_PENALISE_DAYS = 180  # Return staleness flag for confidence penalty


# ── Data contract ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RetrievedPassage:
    """A single retrieved chunk from the SEC filing corpus."""

    symbol: str
    text: str
    filing_type: str        # "10-K" | "10-Q"
    filed_date: str         # ISO-8601, e.g. "2024-11-01"
    section: str            # "risk_factors" | "mda"
    relevance_score: float  # cosine similarity [0, 1]; higher = more relevant
    age_days: int           # calendar days since filing date


# ── Thread-safe singleton ─────────────────────────────────────────────────────

_instance: "FundamentalsRetriever | None" = None
_init_lock = threading.Lock()
_init_error: str | None = None  # Cached so we fail-fast on repeated init attempts


def get_retriever(
    db_path: Path = _DEFAULT_DB_PATH,
    model_name: str = _DEFAULT_MODEL,
) -> "FundamentalsRetriever | None":
    """
    Return the process-level singleton FundamentalsRetriever.

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
            _instance = FundamentalsRetriever(db_path=db_path, model_name=model_name)
            log.info("retriever_ready", db_path=str(db_path), model=model_name)
            return _instance
        except Exception as exc:
            _init_error = str(exc)
            log.warning("retriever_init_failed", error=str(exc))
            return None


# ── Retriever ─────────────────────────────────────────────────────────────────

class FundamentalsRetriever:
    """
    Semantic retriever over SEC filing corpus stored in ChromaDB.

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
            name="sec_fundamentals",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._ef,
        )
        self._log.info(
            "collection_loaded",
            name="sec_fundamentals",
            total_chunks=self._collection.count(),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        symbol: str,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedPassage]:
        """
        Return top-k passages for symbol semantically relevant to query.

        Returns [] if:
          - No documents exist for this symbol (pre-ingestion or CIK not found)
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
            filed = meta.get("filed_date", "")
            try:
                age = (today - datetime.fromisoformat(filed).date()).days
            except ValueError:
                age = 0

            # Distance is 1 - cosine_similarity for cosine space in ChromaDB
            relevance = round(max(0.0, 1.0 - float(dist)), 4)

            passages.append(
                RetrievedPassage(
                    symbol=symbol,
                    text=text,
                    filing_type=meta.get("filing_type", ""),
                    filed_date=filed,
                    section=meta.get("section", ""),
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
        """True if at least one document exists for this symbol."""
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
        Days since most recent filing for symbol.
        Returns None if no documents exist.
        Used by SentimentAgent to apply staleness confidence penalty.
        """
        try:
            result = self._collection.get(
                where={"symbol": {"$eq": symbol}},
                include=["metadatas"],
            )
            dates = [
                m.get("filed_date", "")
                for m in result.get("metadatas", [])
                if m.get("filed_date")
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
        """Total chunks in the collection across all symbols."""
        try:
            return self._collection.count()
        except Exception:
            return 0
