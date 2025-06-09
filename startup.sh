#!/bin/bash

# Startup script for chatbot application with document persistence
# This script ensures that initial documents are copied to the persistent volume

set -e

echo "🚀 Starting chatbot application..."

# Define paths
DOCUMENTS_DIR="/app/documents"
DOCUMENTS_SEED_DIR="/app/documents_seed"

# Create documents directory if it doesn't exist
echo "📁 Ensuring documents directory exists..."
mkdir -p "$DOCUMENTS_DIR"

# Since documents and documents_seed now point to the same host directory,
# we just need to ensure the directory exists and has proper permissions
if [ -d "$DOCUMENTS_DIR" ]; then
    echo "📂 Documents directory exists and is mounted from host"
    echo "📄 Current documents in directory:"
    ls -la "$DOCUMENTS_DIR" 2>/dev/null || echo "Documents directory is empty"
else
    echo "📁 Creating documents directory..."
    mkdir -p "$DOCUMENTS_DIR"
fi

# Note: We don't change permissions on host-mounted directories
echo "🔧 Documents directory is host-mounted, skipping permission changes"

# List current documents
echo "📄 Current documents in volume:"
ls -la "$DOCUMENTS_DIR" || echo "Documents directory is empty"

echo "🎯 Starting main application..."

# Check if we're running the main app or gradio app based on command line arguments
if [ "$#" -eq 0 ]; then
    # Default: run main FastAPI app
    exec python app_postgres.py
else
    # Run with provided arguments (e.g., for Gradio app)
    exec "$@"
fi 