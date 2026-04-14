"""
data/regulatory_retriever.py
Canadian Regulatory Corpus — RAG retriever for the Synthesizer.

Retrieves semantically relevant clauses from OSFI/FINTRAC/CSA regulatory
documents stored in ChromaDB. The Synthesizer queries this before its LLM
call, injecting top-3 clauses into the prompt so every client-facing response
is grounded in current Canadian regulation.

Corpus sources:
  - OSFI: B-20 (mortgage underwriting), E-21 (operational risk), advisories
  - FINTRAC: AML/ATF compliance guidance
  - CSA/IIROC: NI 31-103 suitability obligations, client disclosure requirements

Collection schema:
  documents  : chunked regulatory text (~400 words, 50-word overlap)
  metadatas  : {source, regulator, jurisdiction, version_date, chunk_index, ingested_at}
  ids        : "{regulator}_{source}_{chunk_index}"
  similarity : cosine (HNSW index)

Staleness:
  Regulatory documents are versioned. Chunks carry version_date; the agent
  applies a confidence penalty if the newest chunk for a given regulator is
  older than 12 months (OSFI/FINTRAC update cadence).

Ingest before use:
    python scripts/ingest_regulatory.py --source-dir data/regulatory/ \\
        --regulator OSFI --jurisdiction CA --version-date 2024-11-01
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import structlog

log = structlog.get_logger("regulatory.retriever")

# ── Configuration ─────────────────────────────────────────────────────────────
_DEFAULT_DB_PATH = Path(os.getenv("REGULATORY_DB_PATH", ".chroma_db/regulatory"))
_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_DEFAULT_TOP_K = 3
_STALE_MONTHS = 12  # Flag corpus if newest chunk older than 12 months


# ── Data contract ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RegulatoryClause:
    """A single retrieved clause from the regulatory corpus."""

    text: str
    source: str        # Document identifier e.g. "OSFI_B20_2024-11"
    regulator: str     # "OSFI" | "FINTRAC" | "CSA"
    jurisdiction: str  # "CA"
    version_date: str  # ISO-8601 date of the document version
    relevance_score: float
    age_days: int
    chunk_index: int

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "regulator": self.regulator,
            "jurisdiction": self.jurisdiction,
            "version_date": self.version_date,
            "relevance_score": self.relevance_score,
            "age_days": self.age_days,
            "chunk_index": self.chunk_index,
            "text_preview": self.text[:200],
        }


# ── Thread-safe singleton ─────────────────────────────────────────────────────

_instance: "RegulatoryRetriever | None" = None
_init_lock = threading.Lock()
_init_error: str | None = None


def get_regulatory_retriever(
    db_path: Path = _DEFAULT_DB_PATH,
    model_name: str = _DEFAULT_MODEL,
) -> "RegulatoryRetriever | None":
    """
    Return the process-level singleton RegulatoryRetriever.

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
        if _instance is not None:
            return _instance
        if _init_error is not None:
            return None
        try:
            _instance = RegulatoryRetriever(db_path=db_path, model_name=model_name)
            log.info("regulatory_retriever_ready", db_path=str(db_path), model=model_name)
            return _instance
        except Exception as exc:
            _init_error = str(exc)
            log.warning("regulatory_retriever_init_failed", error=str(exc))
            return None


# ── Retriever ─────────────────────────────────────────────────────────────────

class RegulatoryRetriever:
    """
    Semantic retriever over Canadian regulatory corpus stored in ChromaDB.

    Instantiate via get_regulatory_retriever() to ensure singleton behaviour.
    Direct instantiation is fine for testing.
    """

    def __init__(
        self,
        db_path: Path = _DEFAULT_DB_PATH,
        model_name: str = _DEFAULT_MODEL,
        top_k: int = _DEFAULT_TOP_K,
    ) -> None:
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
                anonymized_telemetry=False,
                allow_reset=False,
            ),
        )
        self._ef = SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            device="cpu",
            normalize_embeddings=True,
        )
        self._collection = self._client.get_or_create_collection(
            name="regulatory_corpus",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._ef,
        )
        self._log.info(
            "collection_loaded",
            name="regulatory_corpus",
            total_chunks=self._collection.count(),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        jurisdiction: str = "CA",
        top_k: int | None = None,
    ) -> list[RegulatoryClause]:
        """
        Return top-k regulatory clauses semantically relevant to query,
        filtered by jurisdiction.

        Returns [] if:
          - No documents ingested yet
          - ChromaDB query fails
        Never raises.
        """
        k = top_k or self._top_k
        total = self._collection.count()
        if total == 0:
            return []

        try:
            result = self._collection.query(
                query_texts=[query],
                n_results=min(k, total),
                where={"jurisdiction": {"$eq": jurisdiction}},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            self._log.warning("retrieval_failed", query=query[:80], error=str(exc))
            return []

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        if not docs:
            return []

        today = datetime.now(UTC).date()
        clauses: list[RegulatoryClause] = []
        for text, meta, dist in zip(docs, metas, dists):
            version_date = meta.get("version_date", "")
            try:
                age = (today - datetime.fromisoformat(version_date).date()).days
            except ValueError:
                age = 0

            relevance = round(max(0.0, 1.0 - float(dist)), 4)
            clauses.append(
                RegulatoryClause(
                    text=text,
                    source=meta.get("source", ""),
                    regulator=meta.get("regulator", ""),
                    jurisdiction=meta.get("jurisdiction", jurisdiction),
                    version_date=version_date,
                    relevance_score=relevance,
                    age_days=age,
                    chunk_index=meta.get("chunk_index", 0),
                )
            )

        self._log.debug(
            "retrieved",
            n=len(clauses),
            top_score=clauses[0].relevance_score if clauses else 0.0,
            top_regulator=clauses[0].regulator if clauses else None,
        )
        return clauses

    def collection_count(self) -> int:
        try:
            return self._collection.count()
        except Exception:
            return 0
