#!/usr/bin/env python3
"""
scripts/ingest_regulatory.py
Offline ingestion pipeline for the Canadian regulatory corpus.

Reads plain-text regulatory documents, chunks at ~400 words with 50-word
overlap, embeds with BAAI/bge-small-en-v1.5, and upserts into ChromaDB at
.chroma_db/regulatory/.

Run once per document update. Idempotent — safe to re-run; existing chunks
for the same source are replaced.

Supported regulators:
  OSFI     — B-20, E-21, advisories
  FINTRAC  — AML/ATF guidance
  CSA      — NI 31-103, client disclosure rules

Usage:
    # Ingest a single document
    python scripts/ingest_regulatory.py \\
        --source-dir data/regulatory/ \\
        --regulator OSFI \\
        --version-date 2024-11-01

    # Ingest all .txt files in a directory (uses filename as source ID)
    python scripts/ingest_regulatory.py \\
        --source-dir data/regulatory/fintrac/ \\
        --regulator FINTRAC \\
        --jurisdiction CA \\
        --version-date 2024-06-15

Environment variables:
    REGULATORY_DB_PATH  — override default DB path (.chroma_db/regulatory)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger("ingest_regulatory")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest regulatory documents into ChromaDB for Synthesizer RAG"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory containing .txt regulatory document files",
    )
    parser.add_argument(
        "--regulator",
        required=True,
        choices=["OSFI", "FINTRAC", "CSA"],
        help="Regulatory body that produced these documents",
    )
    parser.add_argument(
        "--jurisdiction",
        default="CA",
        metavar="CODE",
        help="Jurisdiction code (default: CA)",
    )
    parser.add_argument(
        "--version-date",
        required=True,
        metavar="YYYY-MM-DD",
        help="ISO-8601 date of this document version (e.g. 2024-11-01)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        metavar="PATH",
        help="Override ChromaDB path (default: .chroma_db/regulatory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_dir = args.source_dir
    if not source_dir.exists() or not source_dir.is_dir():
        log.error("source_dir_not_found", path=str(source_dir))
        sys.exit(1)

    txt_files = sorted(source_dir.glob("*.txt"))
    if not txt_files:
        log.error("no_txt_files_found", path=str(source_dir))
        sys.exit(1)

    log.info(
        "ingest_job_start",
        files=len(txt_files),
        regulator=args.regulator,
        jurisdiction=args.jurisdiction,
        version_date=args.version_date,
        db_path=str(args.db_path) if args.db_path else "default",
    )

    try:
        from vector_retail.data.regulatory_embedder import RegulatoryEmbedder
    except ImportError as exc:
        log.error(
            "import_failed",
            error=str(exc),
            hint="Run: pip install chromadb sentence-transformers",
        )
        sys.exit(1)

    embedder = RegulatoryEmbedder(db_path=args.db_path)

    processed = 0
    failed = 0
    t_start = time.time()

    for txt_file in txt_files:
        # Use filename stem as source ID (e.g. "osfi_b20" from "osfi_b20.txt")
        source_id = f"{args.regulator}_{txt_file.stem}".upper()
        log.info(
            "processing",
            file=txt_file.name,
            source=source_id,
            progress=f"{processed + 1}/{len(txt_files)}",
        )
        try:
            n = embedder.ingest(
                source_path=txt_file,
                source=source_id,
                regulator=args.regulator,
                jurisdiction=args.jurisdiction,
                version_date=args.version_date,
            )
            if n == 0:
                log.warning("no_chunks_ingested", file=txt_file.name)
                failed += 1
            else:
                log.info("file_done", file=txt_file.name, chunks=n)
            processed += 1
        except Exception as exc:
            log.error("file_failed", file=txt_file.name, error=str(exc))
            failed += 1
            processed += 1

    elapsed = round(time.time() - t_start, 1)
    succeeded = processed - failed
    print(
        f"\n{'=' * 60}\n"
        f"  Regulatory corpus ingestion complete in {elapsed}s\n"
        f"  Files processed : {processed}/{len(txt_files)}\n"
        f"  Succeeded       : {succeeded}\n"
        f"  Failed          : {failed}\n"
        f"{'=' * 60}\n"
    )

    if failed > 0 and succeeded == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
