# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for PostgreSQL and missing packages
RUN pip install --no-cache-dir \
    psycopg2-binary \
    sqlalchemy \
    alembic \
    qdrant-client \
    langchain-openai

# Copy application files
COPY app_postgres.py .
COPY app/ ./app/
COPY database/ ./database/
COPY config.env .
COPY README.md .

# Copy startup script
COPY startup.sh .

# Copy documents directory (this will be the seed directory in the container)
COPY documents/ ./documents/

# Create necessary directories
RUN mkdir -p /app/data /app/qdrant_storage

# Make startup script executable
RUN chmod +x startup.sh

# Create non-root user for security (matching host user ID)
RUN useradd -m -u 1004 appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports
EXPOSE 8080 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || curl -f http://localhost:8081/health || exit 1

# Default command - use startup script
CMD ["./startup.sh"] 