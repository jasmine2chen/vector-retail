#!/usr/bin/env python3
"""
scripts/ingest_news.py
Real-time news ingestion pipeline for the SentimentAgent RAG layer.

Fetches recent news articles from yfinance, builds enriched text documents,
embeds with BAAI/bge-small-en-v1.5, and upserts into a local ChromaDB
collection at .chroma_db/market_news/.

Run this daily (or before each session) to keep the news corpus fresh.
Articles older than 7 days trigger a staleness confidence penalty in the
SentimentAnalysisAgent.

Usage:
    # Ingest news for specific symbols
    python scripts/ingest_news.py --symbols AAPL MSFT TSLA NVDA

    # Ingest from a holdings JSON file ({"holdings": [{"symbol": "AAPL"}, ...]})
    python scripts/ingest_news.py --from-holdings holdings.json

    # Custom DB path
    python scripts/ingest_news.py --symbols AAPL --db-path /data/chroma_news

Environment variables:
    NEWS_DB_PATH  — override default DB path (.chroma_db/market_news)
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
log = structlog.get_logger("ingest_news")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest recent news articles into ChromaDB for SentimentAgent RAG"
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
        "--db-path",
        type=Path,
        default=None,
        metavar="PATH",
        help="Override ChromaDB path (default: .chroma_db/market_news)",
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
        db_path=str(args.db_path) if args.db_path else "default",
    )

    try:
        from vector_retail.data.embedder import NewsEmbedder
    except ImportError as exc:
        log.error(
            "import_failed",
            error=str(exc),
            hint="Run: pip install chromadb sentence-transformers yfinance",
        )
        sys.exit(1)

    embedder = NewsEmbedder(db_path=args.db_path)

    processed = 0
    failed = 0
    t_start = time.time()

    for symbol in symbols:
        log.info("processing", symbol=symbol, progress=f"{processed + 1}/{len(symbols)}")
        try:
            n = embedder.ingest(symbol=symbol)
            if n == 0:
                log.warning("no_articles_ingested", symbol=symbol)
                failed += 1
            else:
                log.info("symbol_done", symbol=symbol, articles=n)
            processed += 1
        except Exception as exc:
            log.error("symbol_failed", symbol=symbol, error=str(exc))
            failed += 1
            processed += 1

    elapsed = round(time.time() - t_start, 1)
    succeeded = processed - failed
    print(
        f"\n{'=' * 60}\n"
        f"  News ingestion complete in {elapsed}s\n"
        f"  Symbols processed : {processed}/{len(symbols)}\n"
        f"  Succeeded         : {succeeded}\n"
        f"  Failed / no data  : {failed}\n"
        f"{'=' * 60}\n"
    )

    if failed > 0 and succeeded == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
