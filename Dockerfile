# =============================================================================
# Production Dockerfile for RAG Chatbot
# =============================================================================

# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set build arguments
ARG CONTAINER_USER=appuser
ARG CONTAINER_UID=1004

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies including LibreOffice for document processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    git \
    git-lfs \
    libreoffice \
    libreoffice-writer \
    libreoffice-calc \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd -m -u ${CONTAINER_UID} ${CONTAINER_USER} \
    && mkdir -p /app/logs /app/qdrant_storage /app/documents \
    && chown -R ${CONTAINER_USER}:${CONTAINER_USER} /app

# Copy requirements first for better caching
COPY --chown=${CONTAINER_USER}:${CONTAINER_USER} requirements.txt .

# Copy .env file
COPY --chown=${CONTAINER_USER}:${CONTAINER_USER} .env .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
        psycopg2-binary \
        sqlalchemy \
        alembic

# Copy application source code
COPY --chown=${CONTAINER_USER}:${CONTAINER_USER} fastapi_application.py .
COPY --chown=${CONTAINER_USER}:${CONTAINER_USER} application_runner.py .
COPY --chown=${CONTAINER_USER}:${CONTAINER_USER} app/ ./app/
COPY --chown=${CONTAINER_USER}:${CONTAINER_USER} database/ ./database/

# Create documents directory placeholder (no longer copying files)
RUN mkdir -p documents
RUN mkdir -p /app/models
RUN cd /app/models && git lfs install && git clone https://huggingface.co/intfloat/multilingual-e5-large-instruct && rm -rf multilingual-e5-large-instruct/.git


# Switch to non-root user
USER ${CONTAINER_USER}

ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Expose ports
EXPOSE 8081 7860

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Set default command
CMD ["python", "application_runner.py", "--mode", "both"] 