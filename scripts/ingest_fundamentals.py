#!/usr/bin/env python3
"""
scripts/ingest_fundamentals.py
Offline ingestion pipeline for SEC filing fundamentals.

Fetches 10-K / 10-Q filings from SEC EDGAR, chunks text into 400-word
overlapping paragraphs, embeds with BAAI/bge-small-en-v1.5, and upserts
into a local ChromaDB collection at .chroma_db/fundamentals/.

This script must be run before the SentimentAnalysisAgent can use
fundamentals-augmented context. It is idempotent — safe to re-run.

Usage:
    # Ingest 10-K filings for specific symbols
    python scripts/ingest_fundamentals.py --symbols AAPL MSFT TSLA NVDA

    # Ingest both 10-K and 10-Q for a symbol
    python scripts/ingest_fundamentals.py --symbols AAPL --form-type 10-K 10-Q

    # Ingest holdings from a JSON file ({"holdings": [{"symbol": "AAPL"}, ...]})
    python scripts/ingest_fundamentals.py --from-holdings holdings.json

    # Custom DB path and number of filings
    python scripts/ingest_fundamentals.py --symbols AAPL --db-path /data/chroma --n-filings 3

Environment variables:
    FUNDAMENTALS_DB_PATH  — override default DB path (.chroma_db/fundamentals)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure the package is importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger("ingest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest SEC filings into ChromaDB for SentimentAgent RAG"
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--symbols",
        nargs="+",
        metavar="SYMBOL",
        help="Ticker symbols to ingest (e.g. AAPL MSFT TSLA)",
    )
    source_group.add_argument(
        "--from-holdings",
        metavar="FILE",
        help="JSON file containing holdings list with 'symbol' field",
    )

    parser.add_argument(
        "--form-type",
        nargs="+",
        default=["10-K"],
        choices=["10-K", "10-Q"],
        metavar="FORM",
        help="Filing form type(s) to ingest (default: 10-K)",
    )
    parser.add_argument(
        "--n-filings",
        type=int,
        default=2,
        metavar="N",
        help="Number of most-recent filings per symbol per form type (default: 2)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        metavar="PATH",
        help="Override ChromaDB path (default: .chroma_db/fundamentals)",
    )

    return parser.parse_args()


def load_symbols_from_holdings(filepath: str) -> list[str]:
    """Load unique symbols from a holdings JSON file."""
    path = Path(filepath)
    if not path.exists():
        log.error("holdings_file_not_found", path=str(path))
        sys.exit(1)

    with path.open() as f:
        data = json.load(f)

    holdings = data.get("holdings", data) if isinstance(data, dict) else data
    symbols = list({h["symbol"] for h in holdings if "symbol" in h})
    symbols.sort()
    log.info("symbols_loaded_from_holdings", path=str(path), count=len(symbols))
    return symbols


def main() -> None:
    args = parse_args()

    # Resolve symbols
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = load_symbols_from_holdings(args.from_holdings)

    if not symbols:
        log.error("no_symbols_to_ingest")
        sys.exit(1)

    log.info(
        "ingest_job_start",
        symbols=symbols,
        form_types=args.form_type,
        n_filings=args.n_filings,
        db_path=str(args.db_path) if args.db_path else "default",
    )

    # Lazy import after path setup
    try:
        from vector_retail.data.embedder import FundamentalsEmbedder
    except ImportError as exc:
        log.error(
            "import_failed",
            error=str(exc),
            hint="Run: pip install chromadb sentence-transformers beautifulsoup4 lxml",
        )
        sys.exit(1)

    embedder = FundamentalsEmbedder(db_path=args.db_path)

    total_symbols = len(symbols) * len(args.form_type)
    processed = 0
    failed = 0
    t_start = time.time()

    for symbol in symbols:
        for form_type in args.form_type:
            log.info(
                "processing",
                symbol=symbol,
                form_type=form_type,
                progress=f"{processed + 1}/{total_symbols}",
            )
            try:
                n = embedder.ingest(symbol=symbol, form_type=form_type, n_filings=args.n_filings)
                if n == 0:
                    log.warning("no_chunks_ingested", symbol=symbol, form_type=form_type)
                    failed += 1
                else:
                    log.info("symbol_done", symbol=symbol, form_type=form_type, chunks=n)
                processed += 1
            except Exception as exc:
                log.error("symbol_failed", symbol=symbol, form_type=form_type, error=str(exc))
                failed += 1
                processed += 1

    elapsed = round(time.time() - t_start, 1)

    # Print summary
    succeeded = processed - failed
    print(
        f"\n{'=' * 60}\n"
        f"  Ingestion complete in {elapsed}s\n"
        f"  Symbols processed : {processed}/{total_symbols}\n"
        f"  Succeeded         : {succeeded}\n"
        f"  Failed / no data  : {failed}\n"
        f"{'=' * 60}\n"
    )

    if failed > 0 and succeeded == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
