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

#### 2. **Application Package** (`app/`)

- **config.py**: Environment variable management and validation
- **database_initializer.py**: System setup and connection validation
- **document_processing.py**: File ingestion, chunking, and text extraction
- **embedding_manager.py**: Text-to-vector conversion using transformer models
- **gradio_app.py**: Web-based chat interface with multi-language support
- **models.py**: Pydantic schemas for API requests/responses
- **qdrant_manager.py**: Vector database operations and collection management
- **rag_pipeline_manager.py**: Chat logic, retrieval, and response generation

#### 3. **Database Package** (`database/`)

- **postgresql_manager.py**: Database connections, queries, and transactions
- **schema_initialization.sql**: SQL schema definition and migration scripts
- Automatic schema creation and validation
- Connection pooling and error handling

#### 4. **Vector & Storage Systems**

**PostgreSQL Database**
- Chat history storage with user sessions
- Conversation threading and context management
- User preference and session state tracking
- Automatic schema migration and validation

**Qdrant Vector Database**
- Document embeddings storage with metadata
- Semantic similarity search with filtering
- Collection management and optimization
- Persistent vector storage with backup

## 🚀 Deployment Options

### Development Mode
```bash
# Separate services for easier debugging
./deploy.sh up dev

# FastAPI: http://localhost:8081
# Gradio: http://localhost:7860 (separate container)
# Hot reload enabled
# Debug logging active
```

### Production Mode
```bash
# Optimized single container
./deploy.sh up prod

# Combined: http://localhost:8081 (FastAPI + Gradio)
# Resource optimized
# Security hardened
# Health checks enabled
```

### Deployment Script Commands

```bash
# Core Commands
./deploy.sh up [dev|prod]        # Start services
./deploy.sh down [dev|prod]      # Stop services
./deploy.sh restart [dev|prod]   # Restart services
./deploy.sh status [dev|prod]    # Show status
./deploy.sh logs [dev|prod] [service]  # View logs
./deploy.sh clean               # Clean everything

# Examples
./deploy.sh up prod             # Start production
./deploy.sh logs dev chatbot_app # Dev logs for specific service
./deploy.sh status prod         # Production status
```

## 📚 API Documentation

### OpenAI-Compatible Endpoints

**Chat Completions** (OpenAI Compatible)
```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "rag-chatbot",
  "messages": [
    {"role": "user", "content": "Your question here"}
  ],
  "user": "user_123"
}
```

**RAG Chat with History**
```bash
POST /v1/chat
Content-Type: application/json

{
  "user_id": "user_123",
  "message": "Your question here",
  "chat_id": "optional_chat_id"
}
```

**Document Management**
```bash
POST /v1/documents/upload-with-progress  # Upload documents
GET /v1/documents/list                   # List documents
DELETE /v1/documents/delete              # Delete documents
```

**User & Session Management**
```bash
POST /v1/chat/new                       # Create new chat
POST /v1/user/session-status           # Check user status
GET /v1/user/{user_id}/chats           # List user chats
```

**System Monitoring**
```bash
GET /health                            # System health
GET /v1/system/init-status            # Initialization status
POST /v1/system/reinitialize          # Force reinitialize
```

### Full API Documentation
- **Interactive Docs**: http://localhost:8081/docs
- **ReDoc**: http://localhost:8081/redoc

## 🔧 Configuration

### Environment Variables

All configuration is handled through the `.env` file. Key categories:

**Database Settings**
- `DATABASE_URL`: PostgreSQL connection string
- `POSTGRES_*`: Database credentials and settings

**Vector Database**
- `QDRANT_URL`: Qdrant connection URL
- `QDRANT_COLLECTION_NAME`: Collection for documents
- `QDRANT_VECTOR_SIZE`: Embedding dimensions

**Model Configuration**
- `EMBEDDING_MODEL`: HuggingFace embedding model
- `CHAT_MODEL`: LLM model identifier
- `VLLM_CHAT_ENDPOINT`: vLLM server endpoint

**Application Settings**
- `ENVIRONMENT`: development/production
- `LOG_LEVEL`: Logging verbosity
- `DEBUG`: Debug mode toggle

### Model Requirements

**Embedding Model**: `intfloat/multilingual-e5-large-instruct`
- Supports multiple languages including Uzbek
- 1024-dimensional embeddings
- Optimized for retrieval tasks

**Chat Model**: Compatible with vLLM
- Recommended: `google/gemma-3-12b-it`
- OpenAI API compatible endpoint
- Streaming response support

## 🧪 Development

### Local Development Setup

1. **Clone and setup**:
```bash
git clone <repo-url>
cd chatbot
cp .env.example .env  # Edit configuration
```

2. **Start development environment**:
```bash
./deploy.sh up dev
```

3. **Development features**:
- **Hot reload**: Code changes auto-reload
- **Volume mounts**: Local file editing
- **Separate services**: Independent debugging
- **Debug logging**: Detailed error information

### Adding Documents

```bash
# Copy documents to the documents/ folder
cp your-docs/* documents/

# Restart to reindex
./deploy.sh restart
```

### Monitoring & Debugging

```bash
# View all logs
./deploy.sh logs dev

# View specific service logs
./deploy.sh logs dev chatbot_app
./deploy.sh logs dev gradio_app

# Check system health
curl http://localhost:8081/health

# Monitor initialization
curl http://localhost:8081/v1/system/init-status
```

## 🛡️ Security Features

- **Non-root containers**: Enhanced security
- **Environment isolation**: Secure configuration
- **CORS configuration**: Controlled API access
- **Input validation**: Pydantic model validation
- **Health checks**: Automated monitoring
- **Resource limits**: Memory and CPU constraints

## 📊 Performance

### Resource Requirements

**Minimum Requirements**:
- 4GB RAM
- 2 CPU cores
- 10GB disk space

**Recommended Production**:
- 8GB RAM
- 4 CPU cores
- 50GB disk space
- SSD storage

### Optimization Features

- **Vector caching**: Faster similarity search
- **Connection pooling**: Efficient database usage
- **Lazy loading**: Reduced startup time
- **Async processing**: Non-blocking operations
- **Resource limits**: Controlled memory usage

## 🔄 Maintenance

### Backup & Recovery

```bash
# Backup database
docker exec chatbot_postgres_prod pg_dump -U chatbot_user chatbot_db > backup.sql

# Backup Qdrant (if using persistent storage)
docker cp qdrant_prod:/qdrant/storage ./qdrant_backup
```

### Updates & Upgrades

```bash
# Update application
git pull
./deploy.sh down prod
./deploy.sh up prod

# Clean rebuild
./deploy.sh clean
./deploy.sh up prod
```

### Monitoring

Monitor these key metrics:
- **Health endpoint**: `/health`
- **Database connections**: PostgreSQL metrics
- **Vector operations**: Qdrant dashboard
- **Memory usage**: Docker stats
- **Response times**: Application logs

## 🆘 Troubleshooting

### Common Issues

**Services won't start**:
```bash
# Check prerequisites
./deploy.sh status
docker info

# Check logs
./deploy.sh logs
```

**Database connection errors**:
```bash
# Verify PostgreSQL
docker exec -it chatbot_postgres_prod psql -U chatbot_user -d chatbot_db

# Check environment variables
grep DATABASE_URL .env
```

**Vector search issues**:
```bash
# Check Qdrant status
curl http://localhost:6333/collections

# Reinitialize system
curl -X POST http://localhost:8081/v1/system/reinitialize
```

**Performance issues**:
```bash
# Monitor resources
docker stats

# Check health status
curl http://localhost:8081/health
```

### Getting Help

1. **Check logs**: `./deploy.sh logs [environment] [service]`
2. **Verify health**: `curl http://localhost:8081/health`
3. **Review configuration**: Check `.env` file settings
4. **Monitor resources**: `docker stats`

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation at `/docs`

---

**Built with ❤️ for enterprise RAG applications** 