# 🤖 Advanced RAG Chatbot Platform

A sophisticated Retrieval-Augmented Generation (RAG) chatbot platform built with FastAPI, featuring intelligent document processing, user management, chat history tracking, and multi-language support. This platform provides both API endpoints and a user-friendly Gradio interface for seamless interaction.

## ✨ Key Features

### 🧠 Advanced RAG Capabilities
- **Multi-format Document Processing**: PDF, DOCX, TXT, MD, PY files
- **Intelligent Text Chunking**: Optimized document splitting for better retrieval
- **Vector Search**: Powered by Qdrant vector database with configurable similarity metrics
- **Context-Aware Responses**: Leverages chat history for coherent conversations

### 👥 User Management & Chat History
- **Unique User Identification**: Individual user sessions with persistent storage
- **Chat Session Management**: Multiple concurrent chats per user
- **Automatic Cleanup**: Expired sessions removed after 1 hour of inactivity
- **Conversation Context**: Maintains context across chat interactions

### 🔧 Flexible Configuration
- **Multiple Deployment Options**: Local file storage or Docker-based Qdrant
- **Configurable Vector Dimensions**: Auto-detection or manual configuration
- **Model Flexibility**: Support for various embedding and chat models
- **Environment-Based Configuration**: Easy setup through config files

### 🌐 Multi-Interface Support
- **RESTful API**: OpenAI-compatible chat completions endpoint
- **Gradio Web Interface**: User-friendly testing and interaction interface
- **Health Monitoring**: Built-in system health checks and diagnostics

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8+
- Docker (optional, for Qdrant)
- vLLM server running (for model inference)

### Installation

1. **Clone and Setup Environment**
   ```bash
   cd /mnt/mata/chatbot
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure the Application**
   
   Edit `config.env` to match your setup:
   ```env
   # Model Configuration
   EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1
   CHAT_MODEL=google/gemma-3-12b-it
   EMBEDDING_PORT=11445
   CHAT_MODEL_PORT=11444
   
   # Document Processing
   DOCUMENTS_PATH=/mnt/mata/chatbot/documents
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   TOP_K=3
   
   # Qdrant Vector Database
   QDRANT_URL=http://localhost:6333
   QDRANT_COLLECTION_NAME=rag_documents
   QDRANT_VECTOR_SIZE=0  # Auto-detect
   QDRANT_DISTANCE=COSINE
   ```

3. **Start Qdrant (Docker Method)**
   ```bash
   docker-compose up -d
   ```
   
   Or use local file storage by commenting out `QDRANT_URL` in config.env

4. **Launch the Application**
   ```bash
   # Start the main RAG server
   python app.py
   
   # In another terminal, start the Gradio interface (optional)
   python gradio_app.py
   ```

## 📡 API Reference

### Core Chat Endpoint

**POST** `/v1/chat`

Send a message and receive an intelligent response with context awareness.

```json
{
  "user_id": "user_123",
  "chat_id": "chat_abc123",  // Optional: leave empty for new chat
  "message": "How do I access the map page?"
}
```

**Response:**
```json
{
  "chat_id": "chat_abc123",
  "user_id": "user_123", 
  "message": "To access the map page, you can...",
  "timestamp": "2024-01-01T12:00:00"
}
```

### OpenAI-Compatible Chat Completions

**POST** `/v1/chat/completions`

Standard OpenAI-compatible endpoint for integration with existing tools.

```json
{
  "model": "google/gemma-3-12b-it",
  "messages": [
    {"role": "user", "content": "What is vLLM?"}
  ],
  "max_tokens": 150,
  "temperature": 0.7
}
```

### Chat History Management

**POST** `/v1/chat/history`
```json
{
  "user_id": "user_123",
  "chat_id": "chat_abc123"
}
```

**GET** `/v1/user/{user_id}/chats`

Retrieve all chat sessions for a specific user.

### System Monitoring

**GET** `/health`

Check system status, Qdrant connectivity, and model availability.

**GET** `/v1/collections`

View Qdrant collection information and statistics.

## 🎨 Gradio Web Interface

The included Gradio interface provides an intuitive way to interact with the chatbot:

### Features
- **Multi-tab Interface**: Chat, History, and User Management
- **Real-time Health Monitoring**: System status indicators
- **Sample Questions**: Pre-configured example queries
- **Chat Session Management**: Easy switching between conversations
- **User-friendly Design**: Clean, responsive interface

### Access
- **Main Interface**: http://localhost:7860
- **API Server**: http://localhost:8080
- **Qdrant UI**: http://localhost:3000 (when using Docker)

## 🔧 Advanced Configuration

### Vector Database Options

#### Local File Storage (Default)
```env
QDRANT_PATH=./qdrant_storage
# QDRANT_URL=  # Comment out for local storage
```

#### Docker Qdrant
```env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key_if_needed  # Optional
```

### Vector Configuration
```env
QDRANT_VECTOR_SIZE=0          # 0 = auto-detect, or specify dimension
QDRANT_DISTANCE=COSINE        # COSINE, EUCLID, or DOT
QDRANT_FORCE_RECREATE=true    # Recreate collection on config change
QDRANT_ON_DISK=false          # Store vectors on disk (saves RAM)
```

### Model Configuration
```env
# vLLM API Endpoints
VLLM_EMBEDDING_ENDPOINT=http://localhost:11445/v1
VLLM_CHAT_ENDPOINT=http://localhost:11444/v1
VLLM_API_KEY=EMPTY

# Model Selection
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1
CHAT_MODEL=google/gemma-3-12b-it
```

## 📁 Project Structure

```
chatbot/
├── app.py                    # Main FastAPI application
├── gradio_app.py            # Gradio web interface
├── config.env               # Configuration file
├── requirements.txt         # Python dependencies
├── docker-compose.yml       # Docker services
├── start.sh                 # Startup script
├── fix_permissions.sh       # Permission fix utility
├── documents/               # Document storage directory
├── qdrant_storage/          # Local Qdrant data (if using local storage)
├── chat_history.json        # Chat history storage (auto-generated)
└── README.md               # This file
```

## 🛠️ Development & Deployment

### Development Mode
```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 8080
```

### Production Deployment
```bash
# Use the provided startup script
chmod +x start.sh
./start.sh

# Or run directly
python app.py
```

### Docker Deployment
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📚 Document Management

### Supported Formats
- **PDF**: `.pdf` files using PyPDF loader
- **Word**: `.docx` files using Unstructured loader  
- **Text**: `.txt` files with UTF-8 encoding
- **Markdown**: `.md` files for documentation
- **Python**: `.py` files for code documentation

### Document Processing Pipeline
1. **Loading**: Documents loaded from configured directory
2. **Splitting**: Text split into optimized chunks
3. **Embedding**: Chunks converted to vector representations
4. **Storage**: Vectors stored in Qdrant for fast retrieval
5. **Indexing**: Automatic indexing for similarity search

### Adding Documents
Simply place supported files in the `documents/` directory and restart the application. The system will automatically process and index new documents.

## 🔍 Troubleshooting

### Common Issues

**Connection Errors**
- Verify vLLM servers are running on configured ports
- Check Qdrant connectivity with `curl http://localhost:6333/health`
- Ensure all required environment variables are set

**Vector Dimension Mismatches**
- Set `QDRANT_FORCE_RECREATE=true` to auto-fix collection issues
- Use `QDRANT_VECTOR_SIZE=0` for automatic dimension detection

**Performance Issues**
- Increase `TOP_K` value for more comprehensive retrieval
- Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` for better document processing
- Enable `QDRANT_ON_DISK=true` to reduce memory usage

### Health Checks
```bash
# Check API health
curl http://localhost:8080/health

# Check Qdrant status
curl http://localhost:6333/health

# View collections
curl http://localhost:8080/v1/collections
```

## 🤝 Contributing

This is a production-ready RAG chatbot platform. For issues or improvements:

1. Test thoroughly using the Gradio interface
2. Check logs for detailed error information
3. Verify configuration settings match your environment
4. Ensure all dependencies are properly installed

## 📄 License

This project is licensed under the Apache License 2.0. See the license header in source files for details.

---

**Built with ❤️ using FastAPI, LangChain, Qdrant, and Gradio** 