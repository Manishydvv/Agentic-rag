"""
Enterprise Document Ingestion Pipeline
=======================================
Orchestrates the full pipeline:
  File -> Parse -> Chunk -> [Metadata] -> Embed -> Qdrant

Each stage uses LangChain Document objects so metadata
(source, page, section, slide...) is preserved through
every step and stored in Qdrant alongside the vector.

Usage:
    python -m app.ingestion.processor <path> [--wipe] [--no-metadata]

Examples:
    python -m app.ingestion.processor ./data/
    python -m app.ingestion.processor ./data/report.pdf --wipe
    python -m app.ingestion.processor ./data/ --no-metadata
"""

import hashlib
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from typing import List

from qdrant_client.http import models as qdrant_models
from langchain_core.documents import Document

from app.utils.logger import logger
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.docx import parse_docx, parse_xlsx, parse_pptx, parse_csv
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.chunking.splitter import chunk_documents
from app.ingestion.metadata.extractor import extract_metadata
from app.services.retrieval.embedding import embed_texts
from app.services.retrieval.qdrant_service import (
    get_qdrant_client,
    ensure_collection,
    COLLECTION_NAME,
)
from app.config import settings

# ──────────────────────────────────────────────────────────────────────────────
# Supported file extensions -> loader functions
# ──────────────────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".pdf":  parse_pdf,
    ".docx": parse_docx,
    ".xlsx": parse_xlsx,
    ".pptx": parse_pptx,
    ".csv":  parse_csv,
    ".html": parse_html,
    ".htm":  parse_html,
    ".txt":  parse_text,
    ".md":   parse_text,
}

# Max chunks per embedding API call (avoids OpenAI token limits)
EMBED_BATCH_SIZE = settings.EMBED_BATCH_SIZE


# ──────────────────────────────────────────────────────────────────────────────
# Text Cleaning
# ──────────────────────────────────────────────────────────────────────────────
def _clean_text(text: str) -> str:
    """
    Cleans raw text before chunking:
    - Collapses excessive whitespace/newlines
    - Removes null bytes and control characters
    - Strips leading/trailing whitespace
    """
    # Remove null bytes and most control chars (keep \n and \t)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse excessive spaces (but not newlines)
    text = re.sub(r'[ \t]{3,}', '  ', text)
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Duplicate Detection
# ──────────────────────────────────────────────────────────────────────────────
def _compute_file_hash(file_path: str) -> str:
    """Returns SHA-256 hash of a file's content."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()


def _is_already_ingested(client, file_path: str, file_hash: str) -> bool:
    """
    Checks if a file with the same hash already exists in Qdrant.
    This prevents duplicate ingestion of unchanged files.
    """
    try:
        results = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="file_hash",
                        match=qdrant_models.MatchValue(value=file_hash),
                    )
                ]
            ),
            limit=1,
        )
        return len(results[0]) > 0
    except Exception:
        return False


def _delete_file_chunks(client, file_path: str):
    """Deletes all existing chunks for a file (used during re-ingestion)."""
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="source",
                            match=qdrant_models.MatchValue(value=file_path),
                        )
                    ]
                )
            ),
        )
        logger.info(f"Deleted old chunks for '{os.path.basename(file_path)}'")
    except Exception as e:
        logger.warning(f"Could not delete old chunks: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Batch Embedding
# ──────────────────────────────────────────────────────────────────────────────
def _batch_embed(texts: List[str]) -> List[List[float]]:
    """Embeds texts in batches to avoid OpenAI token limits."""
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        logger.info(
            f"  Embedding batch {i // EMBED_BATCH_SIZE + 1} "
            f"({len(batch)} chunks)..."
        )
        embeddings = embed_texts(batch)
        all_embeddings.extend(embeddings)
    return all_embeddings


# ──────────────────────────────────────────────────────────────────────────────
# Core Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def process_file(
    file_path: str,
    client,
    extract_meta: bool = True,
    force: bool = False,
) -> dict:
    """
    Full pipeline for a single file:
      1. Hash Check  -> Skip if already ingested (unless --force)
      2. Parse       -> List[Document] with structural metadata
      3. Clean       -> Remove junk characters and excessive whitespace
      4. Chunk       -> Split with metadata preserved
      5. Metadata    -> LLM-extracted title/summary/keywords
      6. Embed       -> Batch embedding via OpenAI
      7. Upsert      -> Store in Qdrant with rich payloads
    """
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    logger.info(f"-" * 60)
    logger.info(f"Processing: {filename}")

    # ── Step 0: Check supported ───────────────────────────────────
    parser = SUPPORTED_EXTENSIONS.get(ext)
    if not parser:
        logger.warning(f"Skipping unsupported file: {filename}")
        return {"file": filename, "status": "skipped", "chunks": 0}

    # ── Step 1: Duplicate Detection ───────────────────────────────
    file_hash = _compute_file_hash(file_path)
    if not force and _is_already_ingested(client, file_path, file_hash):
        logger.info(f"SKIP: '{filename}' already ingested (hash match). Use --force to re-ingest.")
        return {"file": filename, "status": "already_ingested", "chunks": 0}

    # If file was previously ingested with a different hash, delete old chunks
    _delete_file_chunks(client, file_path)

    # ── Step 2: Parse ─────────────────────────────────────────────
    documents: List[Document] = parser(file_path)
    if not documents:
        logger.warning(f"No content extracted from: {filename}")
        return {"file": filename, "status": "empty", "chunks": 0}

    # ── Step 3: Clean Text ────────────────────────────────────────
    for doc in documents:
        doc.page_content = _clean_text(doc.page_content)
    # Remove docs that became empty after cleaning
    documents = [d for d in documents if d.page_content]

    # ── Step 4: Chunk ─────────────────────────────────────────────
    chunks: List[Document] = chunk_documents(documents)
    if not chunks:
        return {"file": filename, "status": "no_chunks", "chunks": 0}

    # ── Step 5: LLM Metadata Extraction ──────────────────────────
    sample_text = documents[0].page_content
    if extract_meta:
        doc_metadata = extract_metadata(sample_text, filename=filename)
    else:
        doc_metadata = {
            "title": filename,
            "summary": "",
            "keywords": [],
            "document_type": "general",
            "language": "en",
        }

    # ── Step 6: Embed (batched) ───────────────────────────────────
    texts = [chunk.page_content for chunk in chunks]
    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = _batch_embed(texts)

    if len(embeddings) != len(chunks):
        logger.error(f"Embedding count mismatch. Skipping '{filename}'.")
        return {"file": filename, "status": "embed_error", "chunks": 0}

    # ── Step 7: Upsert to Qdrant ──────────────────────────────────
    ingested_at = datetime.now(timezone.utc).isoformat()
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        payload = {
            # Content
            "text": chunk.page_content,
            "chunk_index": i,
            "total_chunks": len(chunks),
            # From loader (structural metadata)
            **chunk.metadata,
            # From LLM extractor (semantic metadata)
            "doc_title": doc_metadata["title"],
            "doc_summary": doc_metadata["summary"],
            "keywords": doc_metadata["keywords"],
            "document_type": doc_metadata["document_type"],
            "language": doc_metadata["language"],
            # Tracking metadata
            "file_hash": file_hash,
            "ingested_at": ingested_at,
        }

        points.append(qdrant_models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=payload,
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info(
        f"Indexed {len(points)} chunks from '{filename}' "
        f"(title: '{doc_metadata['title']}')"
    )

    return {"file": filename, "status": "success", "chunks": len(points)}


def process_directory(
    dir_path: str,
    client,
    extract_meta: bool = True,
    force: bool = False,
) -> List[dict]:
    """Scans a directory recursively and processes all supported files."""
    supported_files = []

    for root, _dirs, files in os.walk(dir_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                supported_files.append(os.path.join(root, f))

    logger.info(f"Found {len(supported_files)} supported files in '{dir_path}'")

    results = []
    for file_path in supported_files:
        result = process_file(file_path, client, extract_meta=extract_meta, force=force)
        results.append(result)
    return results


def run_ingestion(path: str, wipe: bool = False, extract_meta: bool = True, force: bool = False):
    """Main entry point. Accepts a file or directory path."""
    client = get_qdrant_client()
    ensure_collection(wipe=wipe)

    logger.info("=" * 60)
    logger.info("Ingestion Pipeline Started")
    logger.info(f"   Path         : {path}")
    logger.info(f"   Wipe         : {wipe}")
    logger.info(f"   LLM Metadata : {extract_meta}")
    logger.info(f"   Force        : {force}")
    logger.info("=" * 60)

    results = []
    if os.path.isfile(path):
        results.append(process_file(path, client, extract_meta=extract_meta, force=force))
    elif os.path.isdir(path):
        results = process_directory(path, client, extract_meta=extract_meta, force=force)
    else:
        logger.error(f"Path not found: {path}")
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────
    success = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if r["status"] == "already_ingested"]
    failed = [r for r in results if r["status"] not in ("success", "already_ingested")]
    total_chunks = sum(r["chunks"] for r in results)

    logger.info("=" * 60)
    logger.info("Ingestion Complete!")
    logger.info(f"   Processed  : {len(success)} file(s)")
    logger.info(f"   Skipped    : {len(skipped)} file(s) (already ingested)")
    logger.info(f"   Failed     : {len(failed)} file(s)")
    logger.info(f"   Total Chunks in Qdrant: {total_chunks}")
    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    wipe_flag = "--wipe" in sys.argv
    no_meta_flag = "--no-metadata" in sys.argv
    force_flag = "--force" in sys.argv
    clean_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    target_path = clean_args[0] if clean_args else "data"

    run_ingestion(
        path=target_path,
        wipe=wipe_flag,
        extract_meta=not no_meta_flag,
        force=force_flag,
    )
