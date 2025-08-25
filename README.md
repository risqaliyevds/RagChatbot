# RAG Chatbot - Production Ready

A high-performance **Retrieval Augmented Generation (RAG)** chatbot system built with **FastAPI**, **PostgreSQL**, **Qdrant vector database**, and **Gradio interface**. Designed for enterprise deployment with Docker containerization and complete environment variable control.

[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue?logo=postgresql)](https://www.postgresql.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Latest-purple)](https://qdrant.tech/)

## 🚀 Features

- **OpenAI-Compatible API** - Direct integration with existing workflows
- **Advanced RAG Pipeline** - Document ingestion, chunking, and retrieval
- **PostgreSQL Database** - Persistent chat history and user sessions  
- **Qdrant Vector Store** - High-performance semantic search
- **Gradio Web Interface** - User-friendly chat interface
- **Production Ready** - Docker deployment with health checks
- **Environment Control** - All configuration via `.env` file
- **Auto Migration** - Database schema creation on startup
- **Security Hardened** - Non-root containers, secure defaults
- **Comprehensive Deployment** - One-script deployment for dev/prod
- **Multi-Format Support** - PDF, Word, Excel, Text files
- **Offline Deployment** - Full offline operation capability
- **Fresh Start System** - Complete data reset functionality

## 📁 Project Structure

```
chatbot/
├── .dockerignore              # Docker build ignore patterns
├── .env                       # Environment configuration (create from template)
├── .gitignore                # Git ignore patterns
├── Dockerfile                # Production Docker image definition
├── README.md                 # This documentation
├── requirements.txt          # Python dependencies
├── deploy.sh                 # Comprehensive deployment script ⭐
├── docker-compose.yml        # Development Docker setup
├── docker-compose.prod.yml   # Production Docker setup
├── application_runner.py     # Main application entry point
├── fastapi_application.py    # FastAPI web application
├── download_nltk_data.py     # NLTK data download for offline use
├── app/                      # Core application modules
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration management (50+ parameters)
│   ├── database_initializer.py  # Database setup and validation
│   ├── document_processing.py   # Document ingestion and processing
│   ├── embedding_manager.py     # Text embeddings generation
│   ├── gradio_app.py           # Gradio web interface
│   ├── models.py               # Pydantic data models
│   ├── nltk_setup.py           # NLTK offline configuration
│   ├── qdrant_manager.py       # Vector database operations
│   └── rag_pipeline_manager.py # RAG pipeline and chat logic
├── database/                 # Database management
│   ├── __init__.py          # Database package initialization
│   ├── postgresql_manager.py   # PostgreSQL operations
│   └── schema_initialization.sql # Database schema definition
├── documents/                # Document storage directory (runtime)
└── nltk_data/               # NLTK data for offline operation (122MB)
```

## ⚡ Quick Start

### Prerequisites
- **Docker** & **Docker Compose** (required)
- **vLLM server** running (for LLM inference)
- **4GB+ RAM** recommended

### 1. Clone & Setup
```bash
git clone <repository-url>
cd chatbot
```

### 2. Create Environment Configuration
Create `.env` file with your configuration:

```bash
# RAG Chatbot Configuration File
# ============================================================

# DATABASE CONFIGURATION
DATABASE_URL=postgresql://chatbot_user:chatbot_password@postgres:5432/chatbot_db

# POSTGRES CONFIGURATION  
POSTGRES_DB=chatbot_db
POSTGRES_USER=chatbot_user
POSTGRES_PASSWORD=chatbot_password
POSTGRES_INITDB_ARGS=--encoding=UTF8 --locale=C

# QDRANT CONFIGURATION
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION_NAME=rag_documents
QDRANT_VECTOR_SIZE=1024
QDRANT_DISTANCE=COSINE
QDRANT_FORCE_RECREATE=false
QDRANT_ON_DISK=true

# QDRANT SERVICE CONFIGURATION
QDRANT_SERVICE_HTTP_PORT=6333
QDRANT_SERVICE_GRPC_PORT=6334

# MODEL CONFIGURATION
EMBEDDING_MODEL=intfloat/multilingual-e5-large-instruct
CHAT_MODEL=google/gemma-3-12b-it

# vLLM CONFIGURATION
VLLM_API_KEY=EMPTY
VLLM_CHAT_ENDPOINT=http://host.docker.internal:8000/v1

# DOCUMENT CONFIGURATION
DOCUMENTS_PATH=/app/documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# RAG CONFIGURATION
TOP_K=3

# SERVER CONFIGURATION
HOST=0.0.0.0
PORT=8081

# GRADIO CONFIGURATION
GRADIO_PORT=7860
GRADIO_HOST=0.0.0.0
GRADIO_SHARE=false
API_BASE_URL=http://chatbot_app:8081

# LOGGING
LOG_LEVEL=INFO

# APPLICATION SETTINGS
ENVIRONMENT=production
DEBUG=false

# SECURITY SETTINGS (Production-Ready)
MAX_FILE_SIZE=52428800  # 50MB
MAX_QUERY_LENGTH=2000
MAX_DOCUMENTS_PER_USER=100
ALLOWED_FILE_EXTENSIONS=.pdf,.docx,.doc,.txt,.md,.py

# LLM SETTINGS (Configurable)
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2048
LLM_REQUEST_TIMEOUT=120
LLM_MAX_RETRIES=3
```

### 3. Deploy with One Command

The project includes a comprehensive deployment script for easy management:

```bash
# Development deployment (with hot reload)
./deploy.sh up dev

# Production deployment (optimized)
./deploy.sh up prod

# Check service status
./deploy.sh status

# View logs
./deploy.sh logs

# Clean shutdown
./deploy.sh down
```

### 4. Access Your Services

**Development Environment:**
- **FastAPI API**: http://localhost:8081
- **Gradio Interface**: http://localhost:7860
- **API Documentation**: http://localhost:8081/docs
- **Health Check**: http://localhost:8081/health
- **Qdrant Dashboard**: http://localhost:6333/dashboard

**Production Environment:**
- **Combined App**: http://localhost:8081 (FastAPI + Gradio)
- **API Documentation**: http://localhost:8081/docs
- **Health Check**: http://localhost:8081/health
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## 📋 Supported File Formats

✅ **Supported Document Types:**
- **PDF** (.pdf) - Multi-page document support
- **Word Documents** (.doc, .docx) - Microsoft Word files
- **Excel Files** (.xlsx, .xls) - All sheets processed separately
- **Text Files** (.txt) - Plain text with encoding detection
- **Markdown** (.md) - Markdown documents
- **Python** (.py) - Python source code

## 🏗️ Architecture

### System Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Gradio UI     │    │   FastAPI App   │    │   PostgreSQL    │
│   Port: 7860    │◄──►│   Port: 8081    │◄──►│   Port: 5432    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Qdrant DB     │    │   vLLM Server   │
                       │   Port: 6333    │    │   Port: 8000    │
                       └─────────────────┘    └─────────────────┘
```

### Core Components

#### 1. **Application Entry Points**

**Application Runner** (`application_runner.py`)
- Main entry point with multiple modes
- System initialization and health checks
- Service orchestration options:
  - `--mode api`: FastAPI server only
  - `--mode gradio`: Gradio interface only  
  - `--mode both`: Both services (production default)
  - `--check-db`: Database initialization and health check

**FastAPI Application** (`fastapi_application.py`)
- Complete web application with all endpoints
- OpenAI-compatible chat completions
- Document upload and management
- User session handling
- Health monitoring and system status

#### 2. **RAG Pipeline** (`app/rag_pipeline_manager.py`)
- Document processing and chunking
- Embedding generation and storage
- Vector similarity search
- Context retrieval and ranking
- LLM interaction and response generation

#### 3. **Database Management** (`database/postgresql_manager.py`)
- User and session management
- Chat history persistence
- Document metadata storage
- Connection pooling and optimization

#### 4. **Vector Store** (`app/qdrant_manager.py`)
- High-performance vector storage
- Semantic similarity search
- Batch processing and indexing
- Collection management

## 🔧 Offline Deployment

For deploying on servers without internet access:

### Step 1: Download NLTK Data (On Internet-Connected Machine)
```bash
# Install requirements
pip install -r requirements.txt

# Download NLTK data for offline use
python download_nltk_data.py
```

This creates a `nltk_data` folder (122MB) with all necessary tokenizer data.

### Step 2: Transfer and Deploy
```bash
# Transfer entire project to offline server
scp -r chatbot/ user@your-server:/deployment/path/

# Build and deploy
cd /deployment/path/chatbot
docker-compose up -d
```

### Environment Variables for Offline Operation
```bash
NLTK_DATA=/app/nltk_data
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

## 🧨 Fresh Start System

Complete system reset functionality for development and testing:

### Using Environment Variable (Automatic)
```bash
export FRESH_START=true
python application_runner.py
```

### Using API Endpoint (Manual)
```bash
curl -X POST http://localhost:8081/v1/system/fresh-init
```

### Using Gradio Interface
1. Go to http://localhost:7860
2. Navigate to "📄 Файл бошқаруви" → "📊 Статистика"
3. Click "🧨 ТИЗИМНИ ЯНГИ ҲОЛАТГА КЕЛТИРИШ"

**⚠️ Warning:** This permanently deletes ALL data (users, chats, documents, vectors)

## 🔒 Production Security Features

### Input Validation & Rate Limiting
- Maximum file size: 50MB (configurable)
- File extension whitelist (configurable)
- Query length limits (2000 chars)
- Maximum documents per user (100)
- Concurrent upload limits (5)

### Database Security
- Connection pooling with limits
- SQL injection prevention
- Session timeout management
- Automatic cleanup of old sessions

### API Security
- Request timeout controls
- Error message sanitization
- Resource usage limits
- Health check endpoints

## 🚀 API Endpoints

### Chat Completions (OpenAI Compatible)
```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "Your question here"}
  ],
  "model": "rag-chatbot",
  "stream": false
}
```

### Document Upload
```bash
POST /v1/documents/upload-with-progress
Content-Type: multipart/form-data

file: <your-document-file>
```

### System Health
```bash
GET /health
```

### Fresh System Reset
```bash
POST /v1/system/fresh-init
```

## 📊 Configuration Management

The system uses a comprehensive configuration system with 50+ parameters:

### Key Configuration Categories
- **Database Settings**: Pool size, timeouts, retry logic
- **LLM Parameters**: Temperature, max tokens, timeout, retries
- **Security Limits**: File sizes, query lengths, rate limits
- **Performance Tuning**: Batch sizes, chunk parameters
- **Monitoring**: Log levels, health check intervals

All hardcoded values have been eliminated and replaced with configurable environment variables.

## 🛠️ Development

### Running in Development Mode
```bash
# Start with hot reload
./deploy.sh up dev

# Run specific components
python application_runner.py --mode api
python application_runner.py --mode gradio
python application_runner.py --mode both
```

### Database Operations
```bash
# Check database health
python application_runner.py --check-db

# Fresh initialization
export FRESH_START=true
python application_runner.py
```

## 🔍 Troubleshooting

### Common Issues

**NLTK Token Errors:**
- Verify `nltk_data` folder exists in project root
- Check `NLTK_DATA` environment variable
- Ensure offline NLTK data is properly downloaded

**Database Connection Issues:**
- Check PostgreSQL service status
- Verify DATABASE_URL format
- Review connection pool settings

**Vector Store Issues:**
- Confirm Qdrant service is running
- Check QDRANT_URL configuration
- Verify collection creation

**File Upload Problems:**
- Check file size limits (MAX_FILE_SIZE)
- Verify file extension whitelist (ALLOWED_FILE_EXTENSIONS)
- Review upload timeout settings

### Health Checks
```bash
# System health
curl http://localhost:8081/health

# Service status
./deploy.sh status

# View logs
./deploy.sh logs
```

## 📈 Performance Optimization

### Database Optimization
- Connection pooling (configurable pool size)
- Query optimization with proper indexing
- Automatic session cleanup
- Batch processing for large operations

### Vector Store Optimization
- Configurable batch sizes for indexing
- On-disk storage for large collections
- Optimized search parameters
- Progress tracking for long operations

### Application Optimization
- Lazy loading of components
- Configurable timeouts and retries
- Resource usage monitoring
- Graceful error handling

---

## 🎯 Quick Commands

```bash
# Complete setup
git clone <repo> && cd chatbot && ./deploy.sh up prod

# Fresh start
export FRESH_START=true && ./deploy.sh restart

# Health check
curl http://localhost:8081/health

# Upload document
curl -X POST -F "file=@document.pdf" http://localhost:8081/v1/documents/upload-with-progress

# Chat query
curl -X POST -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Your question"}]}' http://localhost:8081/v1/chat/completions
```

**Ready for production deployment with comprehensive configuration control! 🚀** 