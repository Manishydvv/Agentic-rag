"""
Enterprise Document Ingestion Pipeline
=======================================
Orchestrates the full pipeline:
  File → Parse → Chunk → [Metadata] → Embed → Qdrant

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

import os
import sys
import uuid
from typing import List

from qdrant_client import QdrantClient
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

# ──────────────────────────────────────────────────────────────────────────────
# Supported file extensions → loader functions
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

# ──────────────────────────────────────────────────────────────────────────────
# Qdrant config
# ──────────────────────────────────────────────────────────────────────────────
COLLECTION_NAME = "documents"
EMBEDDING_DIM = 1536  # text-embedding-3-small via OpenAI


def _get_qdrant_client() -> QdrantClient:
    return QdrantClient(path="./qdrant_storage")


def _ensure_collection(client: QdrantClient, wipe: bool = False):
    if wipe and client.collection_exists(COLLECTION_NAME):
        logger.warning(f"Wiping collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Collection '{COLLECTION_NAME}' deleted.")

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qdrant_models.VectorParams(
                size=EMBEDDING_DIM,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
        logger.info(f"Collection '{COLLECTION_NAME}' created.")


def process_file(
    file_path: str,
    client: QdrantClient,
    extract_meta: bool = True,
) -> dict:
    """
    Full pipeline for a single file:
      1. Parse  → List[Document] (with page/section/slide metadata)
      2. Chunk  → List[Document] (metadata preserved per chunk)
      3. Meta   → dict (LLM-extracted title/summary/keywords)
      4. Embed  → List[List[float]]
      5. Upsert → Qdrant
    """
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    logger.info(f"-" * 60)
    logger.info(f"Processing: {filename}")

    # ── Step 1: Parse ─────────────────────────────────────────────
    parser = SUPPORTED_EXTENSIONS.get(ext)
    if not parser:
        logger.warning(f"Skipping unsupported file: {filename}")
        return {"file": filename, "status": "skipped", "chunks": 0}

    documents: List[Document] = parser(file_path)
    if not documents:
        logger.warning(f"No content extracted from: {filename}")
        return {"file": filename, "status": "empty", "chunks": 0}

    # ── Step 2: Chunk ─────────────────────────────────────────────
    chunks: List[Document] = chunk_documents(documents)
    if not chunks:
        return {"file": filename, "status": "no_chunks", "chunks": 0}

    # ── Step 3: LLM Metadata Extraction ──────────────────────────
    # Combine first document's text for metadata extraction
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

    # ── Step 4: Embed ─────────────────────────────────────────────
    texts = [chunk.page_content for chunk in chunks]
    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = embed_texts(texts)

    if len(embeddings) != len(chunks):
        logger.error(f"Embedding count mismatch. Skipping '{filename}'.")
        return {"file": filename, "status": "embed_error", "chunks": 0}

    # ── Step 5: Upsert to Qdrant ──────────────────────────────────
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        # Merge chunk-level metadata + doc-level LLM metadata
        payload = {
            # Content
            "text": chunk.page_content,
            "chunk_index": i,
            "total_chunks": len(chunks),
            # From loader (structural — page, section, slide, etc.)
            **chunk.metadata,
            # From LLM extractor (semantic)
            "doc_title": doc_metadata["title"],
            "doc_summary": doc_metadata["summary"],
            "keywords": doc_metadata["keywords"],
            "document_type": doc_metadata["document_type"],
            "language": doc_metadata["language"],
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
    client: QdrantClient,
    extract_meta: bool = True,
) -> List[dict]:
    """Scans a directory and processes all supported files."""
    all_files = [
        f for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f))
    ]
    supported = [
        f for f in all_files
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    logger.info(
        f"Found {len(supported)} supported files "
        f"(of {len(all_files)} total) in '{dir_path}'"
    )

    results = []
    for filename in supported:
        file_path = os.path.join(dir_path, filename)
        result = process_file(file_path, client, extract_meta=extract_meta)
        results.append(result)
    return results


def run_ingestion(path: str, wipe: bool = False, extract_meta: bool = True):
    """Main entry point. Accepts a file or directory path."""
    client = _get_qdrant_client()
    _ensure_collection(client, wipe=wipe)

    logger.info(f"{'═'*60}")
    logger.info(f"Ingestion Pipeline Started")
    logger.info(f"   Path         : {path}")
    logger.info(f"   Wipe         : {wipe}")
    logger.info(f"   LLM Metadata : {extract_meta}")
    logger.info(f"{'═'*60}")

    results = []
    if os.path.isfile(path):
        results.append(process_file(path, client, extract_meta=extract_meta))
    elif os.path.isdir(path):
        results = process_directory(path, client, extract_meta=extract_meta)
    else:
        logger.error(f"Path not found: {path}")
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────
    success = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if r["status"] != "success"]
    total_chunks = sum(r["chunks"] for r in results)

    logger.info(f"{'═'*60}")
    logger.info(f"Ingestion Complete!")
    logger.info(f"   Processed  : {len(success)} file(s)")
    logger.info(f"   Skipped    : {len(skipped)} file(s)")
    logger.info(f"   Total Chunks in Qdrant: {total_chunks}")
    logger.info(f"{'═'*60}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    wipe_flag = "--wipe" in sys.argv
    no_meta_flag = "--no-metadata" in sys.argv
    clean_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    target_path = clean_args[0] if clean_args else "data"

    run_ingestion(
        path=target_path,
        wipe=wipe_flag,
        extract_meta=not no_meta_flag,
    )
