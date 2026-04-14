"""
data/regulatory_embedder.py
Canadian Regulatory Corpus ingestion pipeline — chunk → embed → upsert into ChromaDB.

Architecture:
  RegulatoryEmbedder — reads plain-text regulatory documents, splits them into
                       ~400-word overlapping chunks, and upserts into ChromaDB.
                       Idempotent: deletes existing chunks for a source before
                       re-ingesting, so updates are safe to re-run.

Supported document formats:
  Plain text (.txt) — paste the relevant sections from OSFI/FINTRAC/CSA PDFs.
  The embedder reads all .txt files in the source directory.

Chunking:
  400 words per chunk, 50-word overlap — same strategy as industry standard RAG.
  Chunks shorter than 30 words are discarded (likely headers or table noise).

Usage (called by scripts/ingest_regulatory.py):
    from vector_retail.data.regulatory_embedder import RegulatoryEmbedder
    emb = RegulatoryEmbedder()
    emb.ingest(
        source_path=Path("data/regulatory/osfi_b20.txt"),
        source="OSFI_B20",
        regulator="OSFI",
        jurisdiction="CA",
        version_date="2024-11-01",
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger("regulatory.embedder")

_CHUNK_WORDS = 400
_CHUNK_OVERLAP_WORDS = 50
_MIN_CHUNK_WORDS = 30
_UPSERT_BATCH = 50


# ── Data class ────────────────────────────────────────────────────────────────


@dataclass
class RegDocChunk:
    """A single text chunk from a regulatory document, ready for embedding."""

    source: str        # e.g. "OSFI_B20"
    regulator: str     # "OSFI" | "FINTRAC" | "CSA"
    jurisdiction: str  # "CA"
    version_date: str  # ISO-8601
    chunk_index: int
    text: str
    ingested_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    @property
    def doc_id(self) -> str:
        safe_source = re.sub(r"[^a-zA-Z0-9_]", "_", self.source)
        return f"{safe_source}_{self.chunk_index}"

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "regulator": self.regulator,
            "jurisdiction": self.jurisdiction,
            "version_date": self.version_date,
            "chunk_index": self.chunk_index,
            "ingested_at": self.ingested_at,
        }


# ── Embedder ──────────────────────────────────────────────────────────────────


class RegulatoryEmbedder:
    """
    Ingests regulatory text documents into ChromaDB.

    Instantiate once; reuse across multiple ingest() calls.
    """

    def __init__(self, db_path: Path | None = None, model_name: str | None = None) -> None:
        from .regulatory_retriever import (
            RegulatoryRetriever,
            _DEFAULT_DB_PATH,
            _DEFAULT_MODEL,
        )

        self._retriever = RegulatoryRetriever(
            db_path=db_path or _DEFAULT_DB_PATH,
            model_name=model_name or _DEFAULT_MODEL,
        )
        self._log = log

    def ingest(
        self,
        source_path: Path,
        source: str,
        regulator: str,
        jurisdiction: str = "CA",
        version_date: str | None = None,
    ) -> int:
        """
        Read source_path, chunk, embed, and upsert into ChromaDB.

        Args:
            source_path:  Path to a plain-text (.txt) regulatory document.
            source:       Unique identifier for this document (e.g. "OSFI_B20").
            regulator:    "OSFI" | "FINTRAC" | "CSA"
            jurisdiction: Metadata filter used at retrieval time. Default: "CA".
            version_date: ISO-8601 date of this document version. Defaults to today.

        Returns:
            Number of chunks upserted. 0 on empty file or error.
        """
        if version_date is None:
            version_date = datetime.now(UTC).date().isoformat()

        self._log.info(
            "ingest_start",
            source=source,
            regulator=regulator,
            path=str(source_path),
        )

        try:
            text = source_path.read_text(encoding="utf-8")
        except Exception as exc:
            self._log.error("file_read_failed", path=str(source_path), error=str(exc))
            return 0

        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            self._log.warning("empty_document", source=source)
            return 0

        chunks = self._chunk_text(
            text=text,
            source=source,
            regulator=regulator,
            jurisdiction=jurisdiction,
            version_date=version_date,
        )
        if not chunks:
            self._log.warning("no_chunks_produced", source=source)
            return 0

        self._delete_existing(source=source)
        upserted = self._upsert_chunks(chunks)

        self._log.info("ingest_complete", source=source, chunks=upserted)
        return upserted

    # ── Private ───────────────────────────────────────────────────────────────

    def _chunk_text(
        self,
        text: str,
        source: str,
        regulator: str,
        jurisdiction: str,
        version_date: str,
    ) -> list[RegDocChunk]:
        words = text.split()
        step = _CHUNK_WORDS - _CHUNK_OVERLAP_WORDS
        chunks: list[RegDocChunk] = []
        chunk_index = 0

        i = 0
        while i < len(words):
            chunk_words = words[i : i + _CHUNK_WORDS]
            if len(chunk_words) >= _MIN_CHUNK_WORDS:
                chunks.append(
                    RegDocChunk(
                        source=source,
                        regulator=regulator,
                        jurisdiction=jurisdiction,
                        version_date=version_date,
                        chunk_index=chunk_index,
                        text=" ".join(chunk_words),
                    )
                )
                chunk_index += 1
            i += step

        return chunks

    def _delete_existing(self, source: str) -> None:
        """Delete all existing chunks for this source (idempotent re-ingest)."""
        try:
            col = self._retriever._collection
            existing = col.get(where={"source": {"$eq": source}}, include=[])
            ids = existing.get("ids", [])
            if ids:
                col.delete(ids=ids)
                self._log.info("existing_chunks_deleted", source=source, count=len(ids))
        except Exception as exc:
            self._log.warning("delete_existing_failed", source=source, error=str(exc))

    def _upsert_chunks(self, chunks: list[RegDocChunk]) -> int:
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
