#!/bin/bash

echo "Restarting RAG Chatbot with improvements..."

# Kill existing processes
echo "Stopping existing processes..."
pkill -f "python app_postgres.py"
pkill -f "python gradio_app.py"

# Wait a moment for processes to stop
sleep 3

# Remove old Qdrant storage to force recreation with new settings
echo "Cleaning Qdrant storage for recreation..."
rm -rf ./qdrant_storage

# Start the main application
echo "Starting main RAG application..."
nohup python app_postgres.py > app.log 2>&1 &

# Wait for the main app to initialize
echo "Waiting for main application to initialize..."
sleep 10

# Check if the main app is running
if curl -s http://localhost:8080/health > /dev/null; then
    echo "✅ Main application started successfully"
    
    # Start Gradio interface
    echo "Starting Gradio interface..."
    nohup python gradio_app.py > gradio.log 2>&1 &
    
    sleep 5
    echo "✅ Gradio interface started"
    echo ""
    echo "🚀 RAG Chatbot is now running with improvements!"
    echo "📊 Main API: http://localhost:8080"
    echo "🎨 Gradio UI: http://localhost:7860"
    echo "📋 Health Check: http://localhost:8080/health"
    echo ""
    echo "📝 Logs:"
    echo "   Main app: tail -f app.log"
    echo "   Gradio: tail -f gradio.log"
else
    echo "❌ Failed to start main application"
    echo "Check logs: tail -f app.log"
    exit 1
fi 