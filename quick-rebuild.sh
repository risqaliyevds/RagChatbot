#!/bin/bash

# Quick rebuild script for production deployment
set -e

echo "==================================="
echo "Quick Production Rebuild"
echo "==================================="

# Step 1: Stop all containers
echo "Stopping containers..."
docker-compose -f docker-compose.prod.yml down -v || true

# Step 2: Remove chatbot images
echo "Removing chatbot images..."
docker images | grep chatbot | awk '{print $3}' | xargs -r docker rmi -f || true

# Step 3: Rebuild with docker-compose
echo "Building fresh images..."
docker-compose -f docker-compose.prod.yml build --no-cache

# Step 4: Start services
echo "Starting services..."
docker-compose -f docker-compose.prod.yml up -d

# Step 5: Show logs
echo ""
echo "Services starting up. To view logs:"
echo "docker-compose -f docker-compose.prod.yml logs -f"
echo ""
echo "Service URLs:"
echo "- FastAPI: http://localhost:8081"
echo "- Gradio: http://localhost:7860"
echo "- Qdrant: http://localhost:6333/dashboard" 