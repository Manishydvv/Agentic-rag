import sqlite3
import os
from datetime import datetime, timezone
from typing import List, Dict, Optional
from app.utils.logger import logger

DB_PATH = "metadata.sqlite"

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                upload_status TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                error_message TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()

# Ensure DB is initialized when module loads
init_db()

def create_document(doc_id: str, filename: str) -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        created_at = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            "INSERT INTO documents (id, filename, upload_status, created_at) VALUES (?, ?, ?, ?)",
            (doc_id, filename, "processing", created_at)
        )
        conn.commit()
        logger.info(f"Metadata DB: Created document {doc_id} ('processing')")

def update_document_status(doc_id: str, status: str, chunk_count: int = 0, error_message: Optional[str] = None) -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE documents SET upload_status = ?, chunk_count = ?, error_message = ? WHERE id = ?",
            (status, chunk_count, error_message, doc_id)
        )
        conn.commit()
        logger.info(f"Metadata DB: Updated document {doc_id} -> '{status}'")

def get_all_active_documents() -> List[Dict]:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # Return all except soft-deleted documents
        cursor.execute("SELECT * FROM documents WHERE upload_status != 'deleted' ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
