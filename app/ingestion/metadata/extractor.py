import json
from typing import Optional
from openai import OpenAI
from app.config import settings
from app.utils.logger import logger

# Direct OpenAI client for metadata extraction (lightweight, fast)
_client = OpenAI(api_key=settings.OPENAI_API_KEY)

_METADATA_PROMPT = """You are a document analysis assistant. Analyze the following document excerpt and extract metadata.

Return ONLY a valid JSON object with exactly these fields:
- "title": A concise, descriptive title for the document (string)
- "summary": A 1-2 sentence summary of what the document is about (string)
- "keywords": A list of 3-7 relevant keywords or topics (array of strings)
- "document_type": One of: "technical_documentation", "research_paper", "legal", "business_report", "tutorial", "news_article", "general" (string)
- "language": ISO 639-1 language code, e.g. "en", "fr", "de" (string)

Document excerpt:
\"\"\"
{excerpt}
\"\"\"

Return ONLY the JSON object, no explanation or markdown."""


def extract_metadata(text: str, filename: str = "") -> dict:
    """
    Uses gpt-4o-mini to extract structured metadata from a document.
    Only the first ~1500 characters are sent to minimize cost (~$0.0003/doc).

    Args:
        text: The full document text.
        filename: Original filename (used as fallback for title).

    Returns:
        A dict with: title, summary, keywords, document_type, language.
        Falls back to safe defaults if the LLM call fails.
    """
    # Default fallback metadata
    fallback = {
        "title": filename or "Untitled Document",
        "summary": "No summary available.",
        "keywords": [],
        "document_type": "general",
        "language": "en",
    }

    if not text or not text.strip():
        return fallback

    # Use only first 1500 chars to minimize cost & latency
    excerpt = text[:1500].strip()

    try:
        logger.info(f"Metadata Extractor: Extracting metadata for '{filename}'...")

        response = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": _METADATA_PROMPT.format(excerpt=excerpt)}
            ],
            temperature=0,        # Deterministic output
            max_tokens=300,       # Metadata is always short
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        metadata = json.loads(raw)

        # Validate and fill any missing fields with fallbacks
        result = {
            "title": metadata.get("title") or fallback["title"],
            "summary": metadata.get("summary") or fallback["summary"],
            "keywords": metadata.get("keywords") or [],
            "document_type": metadata.get("document_type") or "general",
            "language": metadata.get("language") or "en",
        }

        logger.info(
            f"Metadata Extractor: Done. "
            f"Title='{result['title']}' | "
            f"Type='{result['document_type']}' | "
            f"Keywords={result['keywords']}"
        )
        return result

    except json.JSONDecodeError as e:
        logger.warning(f"Metadata Extractor: JSON parse failed: {e}. Using fallback.")
        return fallback

    except Exception as e:
        logger.warning(f"Metadata Extractor: LLM call failed: {e}. Using fallback.")
        return fallback
