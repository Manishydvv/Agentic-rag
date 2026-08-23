import psycopg2
from psycopg2.extras import DictCursor
import os
from datetime import datetime, timezone
from typing import List, Dict, Optional
from app.utils.logger import logger
from app.config import settings

def get_connection():
    # Use the connection string from config
    return psycopg2.connect(settings.DATABASE_URL)

def init_db():
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS agentic_rag_documents (
                        id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        upload_status TEXT NOT NULL,
                        chunk_count INTEGER DEFAULT 0,
                        error_message TEXT,
                        created_at TEXT NOT NULL
                    )
                """)
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL metadata DB: {e}")

# Ensure DB is initialized when module loads
init_db()

def create_document(doc_id: str, filename: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            created_at = datetime.now(timezone.utc).isoformat()
            cursor.execute(
                "INSERT INTO agentic_rag_documents (id, filename, upload_status, created_at) VALUES (%s, %s, %s, %s)",
                (doc_id, filename, "processing", created_at)
            )
        conn.commit()
        logger.info(f"Metadata DB: Created document {doc_id} ('processing')")

def update_document_status(doc_id: str, status: str, chunk_count: int = 0, error_message: Optional[str] = None) -> None:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE agentic_rag_documents SET upload_status = %s, chunk_count = %s, error_message = %s WHERE id = %s",
                (status, chunk_count, error_message, doc_id)
            )
        conn.commit()
        logger.info(f"Metadata DB: Updated document {doc_id} -> '{status}'")

def get_all_active_documents() -> List[Dict]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=DictCursor) as cursor:
            # Return all except soft-deleted documents
            cursor.execute("SELECT * FROM agentic_rag_documents WHERE upload_status != 'deleted' ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
