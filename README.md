# RAG Chatbot - Production Ready

A high-performance **Retrieval Augmented Generation (RAG)** chatbot system built with **FastAPI**, **PostgreSQL**, **Qdrant vector database**, and **Gradio interface**. Designed for enterprise deployment with Docker containerization and complete environment variable control.

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

## 📁 Project Structure

```
chatbot/
├── .dockerignore              # Docker build ignore patterns
├── .env                       # Environment configuration (not in git)
├── .gitignore                # Git ignore patterns
├── Dockerfile                # Production Docker image definition
├── README.md                 # Main project documentation
├── requirements.txt          # Python dependencies
├── deploy.sh                 # Deployment automation script
├── docker-compose.yml        # Development Docker setup
├── docker-compose.prod.yml   # Production Docker setup
├── application_runner.py     # Main application entry point
├── fastapi_application.py    # FastAPI web application
├── app/                      # Core application modules
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration management
│   ├── database_initializer.py  # Database setup and validation
│   ├── document_processing.py   # Document ingestion and processing
│   ├── embedding_manager.py     # Text embeddings generation
│   ├── gradio_app.py           # Gradio web interface
│   ├── models.py               # Pydantic data models
│   ├── qdrant_manager.py       # Vector database operations
│   └── rag_pipeline_manager.py # RAG pipeline and chat logic
├── database/                 # Database management
│   ├── __init__.py          # Database package initialization
│   ├── postgresql_manager.py   # PostgreSQL operations
│   └── schema_initialization.sql # Database schema definition
└── documents/                # Document storage directory (runtime)
```

## 📋 Quick Start

### Prerequisites
- Docker & Docker Compose
- vLLM server running (for LLM inference)

### 1. Clone & Configure
```bash
git clone <repository-url>
cd chatbot
```

### 2. Create Environment File
Create `.env` file with your configuration:

```bash
# RAG Chatbot Configuration File

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
API_BASE_URL=http://chatbot_app:8081

# LOGGING
LOG_LEVEL=INFO

# APPLICATION SETTINGS
ENVIRONMENT=production
DEBUG=false
```

### 3. Deploy
```bash
# Development deployment
./deploy.sh up

# Production deployment
./deploy.sh up prod

# Check status
./deploy.sh status

# View logs
./deploy.sh logs
```

### 4. Access Services
- **FastAPI**: http://localhost:8081
- **Gradio Interface**: http://localhost:7860
- **API Documentation**: http://localhost:8081/docs
- **Health Check**: http://localhost:8081/health

## 🏗️ Architecture

### Service Overview
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

#### 1. **FastAPI Application** (`fastapi_application.py`)
- RESTful API endpoints
- OpenAI-compatible chat completions
- Document upload and management
- User session handling
- Health monitoring and system status

#### 2. **Application Runner** (`application_runner.py`)
- Main entry point with multiple modes
- System initialization and health checks
- Service orchestration (API, Gradio, or both)
- Command-line options:
  - `--mode api`: FastAPI server only
  - `--mode gradio`: Gradio interface only  
  - `--mode both`: Both services (development)
  - `--check-db`: Database initialization and health check

#### 3. **App Package** (`app/`)
- **config.py**: Environment variable management and validation
- **database_initializer.py**: System setup and connection validation
- **document_processing.py**: File ingestion, chunking, and text extraction
- **embedding_manager.py**: Text-to-vector conversion using transformer models
- **gradio_app.py**: Web-based chat interface
- **models.py**: Pydantic schemas for API requests/responses
- **qdrant_manager.py**: Vector database operations and collection management
- **rag_pipeline_manager.py**: Chat logic, retrieval, and response generation

#### 4. **Database Package** (`database/`)
- **postgresql_manager.py**: Database connections, queries, and transactions
- **schema_initialization.sql**: SQL schema definition and migration scripts

#### 5. **PostgreSQL Database** (`postgres`)
- Chat history storage
- User session management
- Vector point tracking
- Automatic schema migration

#### 6. **Qdrant Vector Database** (`qdrant`)
- Document embeddings storage
- Semantic similarity search
- Collection management
- Persistent vector storage

#### 7. **Gradio Interface** (`gradio_app`)
- Web-based chat interface
- Document upload UI
- Real-time chat experience

## 📊 Data Flow

1. **Document Ingestion**
   - Upload via API → Document Processing → Text Extraction → Chunking
   - Generate Embeddings → Store in Qdrant → Metadata in PostgreSQL

2. **Chat Flow**
   - User Query → Embedding Generation → Vector Search (Qdrant)
   - Retrieve Context → LLM Generation (vLLM) → Response
   - Store Chat History (PostgreSQL)

3. **System Initialization**
   - Database Connection → Schema Creation → Qdrant Setup
   - Document Loading → Embedding Generation → Service Health Check

## 🔧 Configuration

### Environment Variables

All configuration is controlled via the `.env` file. Key sections:

#### Database Settings
- `DATABASE_URL`: PostgreSQL connection string
- `POSTGRES_DB/USER/PASSWORD`: Database credentials

#### Vector Store Settings  
- `QDRANT_URL`: Qdrant service endpoint
- `QDRANT_COLLECTION_NAME`: Vector collection name
- `QDRANT_VECTOR_SIZE`: Embedding dimensions

#### Model Configuration
- `EMBEDDING_MODEL`: HuggingFace embedding model
- `CHAT_MODEL`: Language model for chat
- `VLLM_CHAT_ENDPOINT`: vLLM server endpoint

#### Application Settings
- `HOST/PORT`: Server binding configuration
- `LOG_LEVEL`: Logging verbosity
- `ENVIRONMENT`: Runtime environment

### Production vs Development

Use `docker-compose.yml` for development and `docker-compose.prod.yml` for production:

**Development Features:**
- Hot reloading
- Debug logging
- Development-friendly settings

**Production Features:**
- Resource limits
- Health checks
- Optimized performance
- Security hardening

## 📖 API Reference

### Core Endpoints

#### Chat Completions (OpenAI Compatible)
```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "chatbot",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}
```

#### Create New Chat Session
```bash
POST /v1/chat/new
Content-Type: application/json

{
  "user_id": "user123"
}
```

#### Upload Document
```bash
POST /v1/documents/upload-with-progress
Content-Type: multipart/form-data

file: <document-file>
```

#### List Documents
```bash
GET /v1/documents/list
```

#### Delete Document (Deactivate)
```bash
DELETE /v1/documents/delete
Content-Type: application/json

{
  "filename": "document.pdf"
}
```

#### Document Statistics
```bash
GET /v1/documents/stats
```

#### Health Check
```bash
GET /health
```

#### System Initialization Status
```bash
GET /v1/system/init-status
```

#### Force Re-initialization (Admin)
```bash
POST /v1/system/reinitialize
```

### Response Formats

#### Chat Response
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "chatbot",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant", 
      "content": "Response text"
    },
    "finish_reason": "stop"
  }]
}
```

## 🚢 Deployment

### Using Deploy Script

The `deploy.sh` script provides comprehensive deployment management:

```bash
# Available commands
./deploy.sh build [prod]     # Build images
./deploy.sh up [prod]        # Start services  
./deploy.sh down [prod]      # Stop services
./deploy.sh restart [prod]   # Restart services
./deploy.sh logs [service]   # View logs
./deploy.sh status          # Check service status
./deploy.sh clean           # Clean up resources
```

### Manual Deployment

#### Development
```bash
docker-compose up -d
```

#### Production
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Verification Steps

1. **Check Service Health**
```bash
curl http://localhost:8081/health
```

2. **Test Chat API**
```bash
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatbot",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

3. **Access Web Interface**
- Open http://localhost:7860 in browser

## 🔍 Monitoring & Troubleshooting

### Health Checks

All services include health checks:
- **PostgreSQL**: `pg_isready` command
- **Qdrant**: Collections API endpoint  
- **FastAPI**: Custom health endpoint
- **Gradio**: HTTP response check

### Log Analysis

```bash
# View all logs
./deploy.sh logs

# View specific service logs
./deploy.sh logs chatbot_app
./deploy.sh logs postgres
./deploy.sh logs qdrant
./deploy.sh logs gradio_app

# Follow logs in real-time
docker-compose logs -f chatbot_app
```

### Common Issues

#### Database Connection Errors
```bash
# Check PostgreSQL status
./deploy.sh status
docker-compose exec postgres pg_isready -U chatbot_user

# Reset database
docker-compose down -v
docker-compose up -d
```

#### Qdrant Collection Issues
```bash
# Check collections
curl http://localhost:6333/collections

# Check specific collection
curl http://localhost:6333/collections/rag_documents
```

#### vLLM Connection Issues
```bash
# Test vLLM endpoint (outside Docker)
curl http://localhost:8000/v1/models

# Check Docker host connectivity
docker-compose exec chatbot_app curl http://host.docker.internal:8000/v1/models
```

## 🛡️ Security

### Production Security Features

- **Non-root containers**: All services run as non-privileged users
- **Network isolation**: Services communicate via internal Docker network  
- **Secure defaults**: Production configurations disable debug features
- **Environment isolation**: Sensitive data via environment variables
- **Input validation**: Pydantic models validate all API inputs
- **CORS configuration**: Controlled cross-origin access

### Security Checklist

- [ ] Change default database passwords
- [ ] Configure proper CORS origins
- [ ] Set up HTTPS/TLS in production
- [ ] Regular security updates
- [ ] Monitor access logs
- [ ] Backup encryption

## 🔄 Database Schema & Initialization

### Smart Initialization System

The application includes comprehensive initialization that works with any database:

1. **Connection Validation**: Checks PostgreSQL and Qdrant connectivity
2. **Database Creation**: Creates database if it doesn't exist 
3. **Schema Migration**: Runs `database/schema_initialization.sql`
4. **Qdrant Setup**: Creates vector collections if needed
5. **Component Initialization**: Sets up all application components

### Initialization Features

- **Production Ready**: Works with existing databases
- **Retry Logic**: Robust connection handling with retries
- **Health Monitoring**: Real-time initialization status
- **Error Recovery**: Graceful handling of partial failures
- **Manual Re-init**: Force re-initialization endpoint

### Document Handling Features

- **No Physical Storage**: Documents are processed and vectorized in-memory only
- **Database Metadata**: All document info stored in PostgreSQL with status tracking
- **Smart Deletion**: Documents are deactivated (is_active=False) instead of deleted
- **Embedding Cleanup**: Vector embeddings automatically removed when documents deactivated
- **Graceful Directory Handling**: Missing documents folder doesn't cause errors

### Schema Overview

```sql
-- Core tables
users (id, user_id, created_at, last_activity, metadata)
chat_sessions (id, chat_id, user_id, created_at, last_activity, metadata)  
chat_messages (id, chat_id, role, content, timestamp, metadata)
documents (id, filename, original_filename, file_size_bytes, is_active, processing_status, ...)
vector_points (id, document_id, filename, qdrant_point_id, chunk_index, ...)

-- Automatic features
- UUID primary keys
- Timestamp triggers  
- Cleanup functions
- Performance indexes
- Document status tracking
- Vector point relationships
```

## 📚 Development

### Local Development Setup

1. **Start dependencies only**
```bash
docker-compose up postgres qdrant -d
```

2. **Run FastAPI locally**
```bash
pip install -r requirements.txt
python application_runner.py
```

3. **Run Gradio locally**  
```bash
python application_runner.py --mode gradio
```

### Entry Points

- **Primary Entry Point**: `application_runner.py` - Main application launcher
- **Web Applications**: 
  - `fastapi_application.py` - FastAPI web server with RAG endpoints
  - `app/gradio_app.py` - Gradio chat interface (imported by runner)

### Key Features

- **Modular Architecture**: Clean separation of concerns
- **Production Ready**: Docker deployment with health checks
- **Database Integration**: PostgreSQL for persistence, Qdrant for vectors
- **API Compatibility**: OpenAI-compatible chat completions
- **Web Interface**: Gradio-based chat UI
- **Configuration Management**: Environment-based configuration
- **Health Monitoring**: Comprehensive health checks and status endpoints
- **Document Management**: Upload, process, and manage documents
- **RAG Pipeline**: Advanced retrieval-augmented generation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/name`
3. Make changes following existing patterns
4. Test with: `./deploy.sh build && ./deploy.sh up`
5. Submit pull request

## 📄 License

[Your License Here]

## 🆘 Support

For issues and questions:
- Create GitHub issues for bugs
- Check logs with `./deploy.sh logs`
- Review health status with `./deploy.sh status`

---

**Ready to deploy?** Start with `./deploy.sh up` and access your chatbot at http://localhost:7860! 