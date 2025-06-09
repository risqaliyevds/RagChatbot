# 🤖 Advanced RAG Chatbot Platform

A sophisticated Retrieval-Augmented Generation (RAG) chatbot platform built with FastAPI and PostgreSQL, featuring intelligent document processing, persistent user management, chat history tracking, and multi-language support. This platform provides both API endpoints and a user-friendly Gradio interface for seamless interaction.

## ✨ Key Features

### 🧠 Advanced RAG Capabilities
- **Multi-format Document Processing**: PDF, DOCX, TXT, MD, PY files
- **Intelligent Text Chunking**: Optimized document splitting for better retrieval
- **Vector Search**: Powered by Qdrant vector database with configurable similarity metrics
- **Context-Aware Responses**: Leverages chat history for coherent conversations

### 👥 User Management & Chat History
- **PostgreSQL Database**: Persistent storage for all user data and chat history
- **Unique User Identification**: Individual user sessions with database persistence
- **Chat Session Management**: Multiple concurrent chats per user with full history
- **Automatic Cleanup**: Expired sessions removed after 1 hour of inactivity
- **Conversation Context**: Maintains context across chat interactions with database reliability

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
- PostgreSQL database (configured via Docker Compose)
- Docker (for Qdrant and PostgreSQL)
- vLLM server running (for model inference)

### Installation

1. **Clone and Setup Environment**
   ```bash
   git clone <repository-url>
   cd chatbot
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

3. **Start Services (Docker Method)**
   ```bash
   # Start PostgreSQL, Qdrant, and all required services
   docker-compose up -d
   ```
   
   This will start:
   - PostgreSQL database for chat history and user management
   - Qdrant vector database for document embeddings
   - All necessary services with persistent data volumes

4. **Launch the Application**
   ```bash
   # Start the main RAG server (FastAPI with PostgreSQL)
   python app_postgres.py
   
   # In another terminal, start the Gradio interface (optional)
   python app/gradio_app.py
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
- **Gradio Interface**: http://localhost:7860
- **FastAPI Server**: http://localhost:8081
- **Qdrant UI**: http://localhost:6333 (when using Docker)

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
├── app_postgres.py          # Main FastAPI application with PostgreSQL
├── app/                     # Application components
│   └── gradio_app.py       # Gradio web interface
├── database/                # Database components
│   ├── __init__.py         # Database package init
│   ├── database.py         # Database manager with PostgreSQL
│   └── init_db.sql         # Database initialization script
├── docs/                    # Documentation
│   ├── STRUCTURE.md        # Project structure details
│   ├── DEPLOYMENT_SUMMARY.md # Deployment guide
│   ├── DOCKER_DEPLOYMENT.md  # Docker deployment guide
│   └── QDRANT_CONFIG_README.md # Qdrant configuration
├── scripts/                 # Deployment and utility scripts
│   ├── deploy.sh           # Main deployment script
│   ├── restart_app.sh      # Application restart utility
│   └── k8s-deployment.yaml # Kubernetes deployment config
├── tests/                   # Test files
│   ├── test_progress_document.md # Progress bar test document
│   └── test_volume_persistence.sh # Docker volume test script
├── config.env              # Configuration file
├── requirements.txt        # Python dependencies
├── docker-compose.yml      # Docker services composition
├── startup.sh              # Application startup script
├── Dockerfile              # Docker build configuration
├── documents/              # Document storage directory
├── data/                   # Runtime data (excluded from git)
│   ├── postgres_data/      # PostgreSQL data
│   ├── qdrant_storage/     # Qdrant vector storage
│   └── app_data/          # Application runtime data
├── logs/                   # Application logs (runtime)
└── README.md              # This documentation
```

## 🛠️ Development & Deployment

### Development Mode
```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn app_postgres:app --reload --host 0.0.0.0 --port 8081
```

### Production Deployment
```bash
# Use the provided startup script
chmod +x startup.sh
./startup.sh

# Or run directly
python app_postgres.py
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
# Check FastAPI application health
curl http://localhost:8081/health

# Check Qdrant status
curl http://localhost:6333/health

# View collections
curl http://localhost:8081/v1/collections
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

## File Management System

The application now includes a comprehensive file management system with automatic embedding capabilities:

### 📤 File Upload API
- **Endpoint**: `POST /v1/documents/upload`
- **Features**:
  - Uploads files to the `documents/` folder
  - Automatically processes and embeds documents using the configured embedding model
  - Supports multiple file formats: PDF, Word (.docx, .doc), Text (.txt), Markdown (.md), Python (.py)
  - File size limit: 50MB
  - Returns processing statistics (chunks added, processing time, etc.)

### 📂 File Listing API
- **Endpoint**: `GET /v1/documents/list`
- **Features**:
  - Lists all documents in the `documents/` folder
  - Returns file metadata (size, creation/modification dates, file type)
  - Sorts files by modification date (newest first)
  - Shows total file count and combined size

### 🗑️ File Deletion API
- **Endpoint**: `DELETE /v1/documents/delete`
- **Features**:
  - Safely deletes files from the `documents/` folder
  - **🧹 Automatic Embedding Cleanup**: Removes all associated embeddings from the vector store
  - Includes security checks to prevent directory traversal attacks
  - Returns confirmation of successful deletion with embeddings count
  - **Transactional Safety**: Continues with file deletion even if embedding cleanup fails

### 🖥️ Enhanced Gradio Interface

The Gradio interface has been enhanced with a new "📄 Файл бошқаруви" (File Management) tab containing:

#### 📤 File Upload Tab with Progress Tracking
- **Real-Time Progress Bar**: Visual progress indicator during file upload and processing
- **Intelligent Staging**: Progress updates for each processing stage:
  - File validation and reading (5-15%)
  - Server upload with size-based timing (25-40%)
  - File type-specific processing - PDF, DOCX, TXT (40-50%)
  - Document analysis and chunking (50-80%)
  - Vector embedding creation and storage (80-95%)
  - Finalization (95-100%)
- **Adaptive Timing**: Progress speed adjusts based on file size and type
- **Visual Feedback**: Descriptive status messages in Uzbek language
- **Enhanced User Experience**: Smooth progress updates with realistic timing
- Support for multiple file formats: PDF, DOCX, DOC, TXT, MD, PY
- Processing statistics display with enhanced success feedback

#### 📂 File List Tab  
- Visual display of all uploaded documents
- File metadata with emojis for different file types
- Refresh button to update the list
- Readable file size formatting

#### 🗑️ File Deletion Tab
- Simple interface for deleting specific files
- Safety warnings about irreversible actions
- Filename input with validation
- Deletion confirmation messages

### 🔧 Implementation Details

The implementation follows the **minimal change philosophy**:

1. **API Extensions**: Added new Pydantic models and endpoints to the existing FastAPI application
2. **Gradio Enhancement**: Extended the existing Gradio client class with new methods
3. **No New Dependencies**: Used existing libraries and patterns
4. **Security**: Implemented proper file path validation and access controls
5. **Error Handling**: Comprehensive error handling and user-friendly messages

### 📊 **Progress Bar Technical Implementation**

#### **🎯 Gradio Progress Component**
- **Framework**: Uses Gradio's built-in `gr.Progress()` component
- **Integration**: Seamlessly integrated into existing upload handler function
- **Parameters**: `progress=gr.Progress()` parameter added to `upload_document_handler()`

#### **⚡ Intelligent Progress Stages**
1. **File Preparation** (0-5%): Initial validation and file size detection
2. **File Reading** (5-15%): Reading file content and validation
3. **Upload Simulation** (15-40%): Server upload with realistic timing
4. **Server Processing** (40-80%): File type-specific processing stages
5. **Embedding Generation** (80-95%): Vector creation and storage
6. **Completion** (95-100%): Finalization and success confirmation

#### **📈 Adaptive Timing Algorithm**
- **File Size Awareness**: Larger files (>5MB) get longer processing delays
- **File Type Recognition**: Different timing for PDF, DOCX, and text files
- **Realistic Simulation**: Progress timing reflects actual processing complexity
- **User Experience**: Smooth visual feedback prevents perceived freezing

#### **🌐 Multilingual Support**
- **Uzbek Language**: All progress messages in native Uzbek
- **Descriptive Feedback**: Clear status messages for each processing stage
- **Error Handling**: Progress updates even when errors occur
- **Visual Indicators**: Emoji and formatting for enhanced readability

### 🚀 Usage

1. **Start the FastAPI server**: The new endpoints are automatically available
2. **Access Gradio interface**: Navigate to the "📄 Файл бошқаруви" tab
3. **Upload files**: Use the upload tab to add new documents
4. **View files**: Use the list tab to see all uploaded documents  
5. **Delete files**: Use the deletion tab to remove unwanted documents

All uploaded files are automatically processed and embedded for use in the RAG (Retrieval-Augmented Generation) system.

## 🧪 Testing

The project includes comprehensive test files in the `tests/` directory:

### Test Files
- **`test_volume_persistence.sh`**: Docker volume persistence testing script
  - Tests document storage across container restarts
  - Validates API endpoints for file upload/download
  - Ensures data persistence with Docker volumes
  
- **`test_progress_document.md`**: Large test document for progress bar testing
  - Demonstrates file upload progress tracking
  - Tests document processing pipeline
  - Used for embedding generation performance testing

### Running Tests
```bash
# Test Docker volume persistence
cd tests/
chmod +x test_volume_persistence.sh
./test_volume_persistence.sh

# Test file upload with progress tracking using the test document
# Use the Gradio interface and upload test_progress_document.md
```

## 🧹 **Embedding Deletion System**

A key enhancement ensures **complete data integrity** when files are deleted:

### **🎯 Automatic Cleanup**
- **Vector Store Integration**: When a file is deleted, all associated embeddings are automatically removed from Qdrant
- **Smart Identification**: Uses multiple source identifiers to ensure complete cleanup:
  - Direct filename matching
  - Uploaded filename metadata
  - Source field matching (`uploaded_{filename}`)

### **🔧 Technical Implementation**
- **New Vector Store Method**: `delete_documents_by_source()` in `SimpleQdrantVectorStore`
- **Qdrant Integration**: Uses scroll and filter operations to find and delete matching embeddings
- **Batch Processing**: Efficiently handles multiple embeddings per document
- **Error Resilience**: File deletion continues even if embedding cleanup encounters issues

### **📊 User Feedback**
- **Deletion Statistics**: API returns count of deleted embeddings
- **Enhanced UI**: Gradio interface shows:
  - Number of embeddings removed
  - Success confirmation with detailed feedback  
  - Clear messaging when no embeddings are found

### **🛡️ Data Integrity**
- **No Orphaned Data**: Ensures no leftover embeddings consume storage or affect search results
- **Consistent State**: Maintains perfect sync between filesystem and vector store
- **Audit Trail**: Comprehensive logging of all deletion operations

This implementation provides **enterprise-grade data management** with full lifecycle tracking from upload to deletion. 