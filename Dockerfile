# Stage 1: Build
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Force uv to install PyTorch CPU version to save ~3GB of CUDA bloat
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
# Increase timeout for slow connections when downloading heavy AI packages
ENV UV_HTTP_TIMEOUT=300

WORKDIR /app
COPY pyproject.toml uv.lock ./
# Sync dependencies (including PyTorch CPU due to the extra index URL)
RUN uv sync --frozen --no-install-project --no-dev

COPY . .
RUN uv sync --frozen --no-dev

# Stage 2: Run
FROM python:3.12-slim-bookworm

# Install Tesseract OCR for PDF ingestion
RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app /app

# Ensure we use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
