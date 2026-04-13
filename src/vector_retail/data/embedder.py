"""
data/embedder.py
SEC EDGAR ingestion pipeline — fetch → chunk → embed → upsert into ChromaDB.

Architecture:
  SECEdgarClient   — thin wrapper around the public SEC EDGAR REST API.
                     No API key required; User-Agent header is mandatory
                     under SEC fair-access policy.
  FilingChunker    — extracts risk_factors / mda sections from raw HTML and
                     splits them into 400-word overlapping paragraphs.
  FundamentalsEmbedder — orchestrates the fetch→chunk→embed→upsert pipeline.
                     Deletes existing chunks for a symbol+form_type before
                     re-ingestion (idempotent; safe to re-run).

Rate limits:
  SEC EDGAR enforces a maximum of 10 requests per second.
  The client sleeps 0.12 s between requests to stay safely under the limit.

Usage (called by scripts/ingest_fundamentals.py):
    from vector_retail.data.embedder import FundamentalsEmbedder
    emb = FundamentalsEmbedder()
    emb.ingest(symbol="AAPL", form_type="10-K", n_filings=2)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger("fundamentals.embedder")

# ── Constants ─────────────────────────────────────────────────────────────────

_EDGAR_BASE = "https://data.sec.gov"
_EDGAR_COMPANY_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%22{symbol}%22&dateRange=custom&startdt={start}&enddt={end}&forms={form_type}"  # noqa: E501
_USER_AGENT = "VectorRetail/2.0 (contact@vectorretail.example)"  # SEC requires identify

_CHUNK_WORDS = 400
_CHUNK_OVERLAP_WORDS = 50
_UPSERT_BATCH = 50           # ChromaDB recommended batch size
_RATE_LIMIT_SLEEP = 0.12     # seconds; keeps us at ~8 req/s (< 10 limit)

# Section header patterns (case-insensitive)
_SECTION_PATTERNS: dict[str, list[str]] = {
    "risk_factors": [
        r"item\s+1a[\.\s]+risk\s+factors",
        r"risk\s+factors",
    ],
    "mda": [
        r"item\s+7[\.\s]+management.{0,20}discussion",
        r"management.{0,20}discussion\s+and\s+analysis",
        r"item\s+2[\.\s]+management.{0,20}discussion",  # 10-Q uses Item 2
    ],
}

# Next-section boundary — stop extraction when we hit the next Item header
_NEXT_SECTION_PATTERN = re.compile(
    r"^\s*item\s+\d+[a-z]?[\.\s]",
    re.IGNORECASE | re.MULTILINE,
)


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class FilingChunk:
    """A single text chunk ready for embedding and upsert."""

    symbol: str
    filing_type: str   # "10-K" | "10-Q"
    filed_date: str    # ISO-8601
    section: str       # "risk_factors" | "mda"
    chunk_index: int
    text: str
    ingested_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    @property
    def doc_id(self) -> str:
        safe_date = self.filed_date.replace("-", "")
        return f"{self.symbol}_{self.filing_type.replace('-', '')}_{safe_date}_{self.chunk_index}"

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "filing_type": self.filing_type,
            "filed_date": self.filed_date,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "ingested_at": self.ingested_at,
        }


# ── SEC EDGAR client ──────────────────────────────────────────────────────────


class SECEdgarClient:
    """
    Minimal client for the SEC EDGAR public REST API.

    Endpoints used:
      /submissions/{cik}.json  — filing history for a company
      /Archives/edgar/...      — full-text filing documents

    Rate limit: 10 req/s (we sleep 0.12s between calls → ~8 req/s).
    """

    def __init__(self) -> None:
        # Lazy import: requests is a standard dependency but we avoid
        # importing at module level so the module loads without network I/O.
        import requests

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})

    def get_recent_filings(
        self,
        symbol: str,
        form_type: str,
        n: int = 2,
    ) -> list[dict[str, Any]]:
        """
        Return metadata for the N most recent filings of form_type for symbol.

        Each dict contains: {accession_number, filed_date, primary_document, cik}.
        Returns [] if CIK lookup or EDGAR query fails.
        """
        cik = self._lookup_cik(symbol)
        if cik is None:
            log.warning("cik_not_found", symbol=symbol)
            return []

        return self._fetch_filing_list(cik=cik, symbol=symbol, form_type=form_type, n=n)

    def fetch_filing_text(self, cik: str, accession_number: str, primary_doc: str) -> str:
        """
        Fetch the raw HTML/text of the primary document for a filing.
        Returns "" on any error.
        """
        # EDGAR path convention: accession number with dashes removed
        acc_clean = accession_number.replace("-", "")
        url = f"{_EDGAR_BASE}/Archives/edgar/full-index/{cik}/{acc_clean}/{primary_doc}"
        try:
            resp = self._get(url)
            return resp.text if resp else ""
        except Exception as exc:
            log.warning("filing_fetch_failed", url=url, error=str(exc))
            return ""

    # ── Private ───────────────────────────────────────────────────────────────

    def _lookup_cik(self, symbol: str) -> str | None:
        """Return zero-padded 10-digit CIK for a ticker symbol, or None."""
        url = f"{_EDGAR_BASE}/submissions/CIK{symbol}.json"
        # EDGAR ticker→CIK mapping
        ticker_url = f"{_EDGAR_BASE}/files/company_tickers.json"
        try:
            resp = self._get(ticker_url)
            if resp is None:
                return None
            data: dict[str, Any] = resp.json()
            for entry in data.values():
                if entry.get("ticker", "").upper() == symbol.upper():
                    cik_int = entry["cik_str"]
                    return str(cik_int).zfill(10)
        except Exception as exc:
            log.warning("cik_lookup_failed", symbol=symbol, error=str(exc))
        return None

    def _fetch_filing_list(
        self,
        cik: str,
        symbol: str,
        form_type: str,
        n: int,
    ) -> list[dict[str, Any]]:
        """Pull filing history from EDGAR submissions API and return the N most recent."""
        url = f"{_EDGAR_BASE}/submissions/CIK{cik}.json"
        try:
            resp = self._get(url)
            if resp is None:
                return []
            data = resp.json()
        except Exception as exc:
            log.warning("submission_fetch_failed", cik=cik, error=str(exc))
            return []

        recent: dict[str, Any] = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        filed_dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])

        results = []
        for form, acc, dated, doc in zip(forms, accessions, filed_dates, primary_docs):
            if form == form_type:
                results.append(
                    {
                        "cik": cik,
                        "accession_number": acc,
                        "filed_date": dated,
                        "primary_document": doc,
                        "symbol": symbol,
                    }
                )
            if len(results) >= n:
                break

        return results

    def _get(self, url: str) -> Any:
        """GET with rate-limit sleep; returns Response or None on error."""
        time.sleep(_RATE_LIMIT_SLEEP)
        resp = self._session.get(url, timeout=15)
        resp.raise_for_status()
        return resp


# ── Section extractor + chunker ───────────────────────────────────────────────


class FilingChunker:
    """
    Extracts risk_factors and mda sections from SEC filing HTML and
    splits them into overlapping word-level chunks.

    Chunking strategy:
      - 400 words per chunk with 50-word overlap (standard RAG sliding window).
      - Preserves sentence boundaries where possible (splits on space after
        reaching the word limit, rather than mid-word).
      - Small chunks (< 50 words) are discarded — likely table noise or headers.
    """

    def chunk_filing(
        self,
        symbol: str,
        filing_type: str,
        filed_date: str,
        raw_html: str,
    ) -> list[FilingChunk]:
        """Parse raw_html and return all text chunks for all sections."""
        text = self._html_to_text(raw_html)
        all_chunks: list[FilingChunk] = []
        chunk_index = 0

        for section_name, patterns in _SECTION_PATTERNS.items():
            section_text = self._extract_section(text, patterns)
            if not section_text:
                log.debug(
                    "section_not_found",
                    symbol=symbol,
                    filing=filing_type,
                    section=section_name,
                )
                continue

            for chunk_text in self._sliding_window(section_text):
                all_chunks.append(
                    FilingChunk(
                        symbol=symbol,
                        filing_type=filing_type,
                        filed_date=filed_date,
                        section=section_name,
                        chunk_index=chunk_index,
                        text=chunk_text,
                    )
                )
                chunk_index += 1

        log.info(
            "filing_chunked",
            symbol=symbol,
            filing=filing_type,
            filed_date=filed_date,
            total_chunks=len(all_chunks),
        )
        return all_chunks

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Strip HTML tags; collapse whitespace."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "lxml")
            # Remove script/style noise
            for tag in soup(["script", "style", "table"]):
                tag.decompose()
            text = soup.get_text(separator=" ")
        except Exception:
            # Fall back to regex stripping if bs4 unavailable
            text = re.sub(r"<[^>]+>", " ", html)

        # Collapse whitespace
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _extract_section(text: str, patterns: list[str]) -> str:
        """
        Find the first matching section header and return the text up to
        the next Item header (or EOF).
        """
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                start = m.end()
                # Find next Item header after section start
                boundary = _NEXT_SECTION_PATTERN.search(text, start)
                end = boundary.start() if boundary else len(text)
                return text[start:end].strip()
        return ""

    @staticmethod
    def _sliding_window(text: str) -> list[str]:
        """
        Split text into overlapping word chunks.
        Returns chunks with >= 50 words; discards shorter tail fragments.
        """
        words = text.split()
        chunks: list[str] = []
        step = _CHUNK_WORDS - _CHUNK_OVERLAP_WORDS

        i = 0
        while i < len(words):
            chunk_words = words[i : i + _CHUNK_WORDS]
            if len(chunk_words) >= 50:
                chunks.append(" ".join(chunk_words))
            i += step

        return chunks


# ── Orchestrator ──────────────────────────────────────────────────────────────


class FundamentalsEmbedder:
    """
    Orchestrates the full ingest pipeline:
      fetch filings → chunk text → embed → upsert into ChromaDB.

    Idempotent: deletes existing chunks for symbol+form_type before
    re-ingesting, so re-runs are safe (no duplicate chunks).

    Instantiate once; reuse across multiple ingest() calls.
    """

    def __init__(self, db_path: Path | None = None, model_name: str | None = None) -> None:
        from .retriever import (
            FundamentalsRetriever,
            _DEFAULT_DB_PATH,
            _DEFAULT_MODEL,
        )

        self._retriever = FundamentalsRetriever(
            db_path=db_path or _DEFAULT_DB_PATH,
            model_name=model_name or _DEFAULT_MODEL,
        )
        self._edgar = SECEdgarClient()
        self._chunker = FilingChunker()
        self._log = log

    def ingest(
        self,
        symbol: str,
        form_type: str = "10-K",
        n_filings: int = 2,
    ) -> int:
        """
        Fetch, chunk, embed, and upsert the N most recent filings for symbol.

        Returns the total number of chunks upserted.
        Raises nothing — all errors are logged and skipped.
        """
        self._log.info("ingest_start", symbol=symbol, form_type=form_type, n=n_filings)

        filings = self._edgar.get_recent_filings(
            symbol=symbol, form_type=form_type, n=n_filings
        )
        if not filings:
            self._log.warning("no_filings_found", symbol=symbol, form_type=form_type)
            return 0

        # Delete stale chunks for this symbol+form_type before re-ingesting
        self._delete_existing(symbol=symbol, form_type=form_type)

        total_upserted = 0
        for filing in filings:
            chunks = self._process_filing(filing)
            if chunks:
                upserted = self._upsert_chunks(chunks)
                total_upserted += upserted

        self._log.info(
            "ingest_complete",
            symbol=symbol,
            form_type=form_type,
            filings_processed=len(filings),
            total_chunks=total_upserted,
        )
        return total_upserted

    # ── Private ───────────────────────────────────────────────────────────────

    def _process_filing(self, filing: dict[str, Any]) -> list[FilingChunk]:
        """Fetch + chunk a single filing. Returns [] on error."""
        try:
            raw_html = self._edgar.fetch_filing_text(
                cik=filing["cik"],
                accession_number=filing["accession_number"],
                primary_doc=filing["primary_document"],
            )
            if not raw_html:
                return []

            return self._chunker.chunk_filing(
                symbol=filing["symbol"],
                filing_type=filing.get("form_type", "10-K"),
                filed_date=filing["filed_date"],
                raw_html=raw_html,
            )
        except Exception as exc:
            self._log.warning(
                "filing_processing_failed",
                accession=filing.get("accession_number"),
                error=str(exc),
            )
            return []

    def _delete_existing(self, symbol: str, form_type: str) -> None:
        """Remove all existing chunks for symbol+form_type (idempotent re-ingest)."""
        try:
            col = self._retriever._collection
            existing = col.get(
                where={
                    "$and": [
                        {"symbol": {"$eq": symbol}},
                        {"filing_type": {"$eq": form_type}},
                    ]
                },
                include=[],
            )
            ids = existing.get("ids", [])
            if ids:
                col.delete(ids=ids)
                self._log.info(
                    "existing_chunks_deleted",
                    symbol=symbol,
                    form_type=form_type,
                    count=len(ids),
                )
        except Exception as exc:
            self._log.warning("delete_existing_failed", symbol=symbol, error=str(exc))

    def _upsert_chunks(self, chunks: list[FilingChunk]) -> int:
        """Upsert chunks in batches of _UPSERT_BATCH. Returns count upserted."""
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
                self._log.debug(
                    "batch_upserted",
                    batch_start=i,
                    batch_size=len(batch),
                )
            except Exception as exc:
                self._log.warning("batch_upsert_failed", batch_start=i, error=str(exc))

        return upserted
