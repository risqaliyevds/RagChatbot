# Chatbot Application - Docker Deployment Guide

This guide provides complete instructions for deploying the RAG-based chatbot application using Docker and Docker Compose.

## 🏗️ Architecture Overview

The application consists of the following services:

- **PostgreSQL Database**: Stores chat sessions and messages
- **Qdrant Vector Database**: Stores document embeddings for RAG
- **Main FastAPI Application**: Core chatbot API with PostgreSQL integration
- **Gradio Interface**: Web-based chat interface

## 📋 Prerequisites

### Required Software

1. **Docker** (version 20.10 or later)
   ```bash
   # Install Docker on Ubuntu/Debian
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   
   # Add user to docker group
   sudo usermod -aG docker $USER
   ```

2. **Docker Compose** (version 2.0 or later)
   ```bash
   # Install Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

### System Requirements

- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: Minimum 10GB free space
- **CPU**: 2+ cores recommended
- **Ports**: Ensure the following ports are available:
  - `8080` - Main API
  - `7860` - Gradio Interface
  - `5432` - PostgreSQL
  - `6333` - Qdrant

## 🚀 Quick Start

### 1. Clone and Prepare

```bash
# Navigate to your project directory
cd /path/to/your/chatbot/project

# Make deployment script executable
chmod +x deploy.sh
```

### 2. Deploy Everything

```bash
# Full deployment (builds image and starts all services)
./deploy.sh
```

This single command will:
- Check prerequisites
- Build the Docker image
- Start all services
- Wait for services to be ready
- Show access information

### 3. Access the Application

Once deployment is complete, you can access:

- **Main API**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs
- **Gradio Chat Interface**: http://localhost:7860
- **Health Check**: http://localhost:8080/health

## 🛠️ Deployment Script Commands

The `deploy.sh` script supports various commands:

```bash
# Full deployment
./deploy.sh

# Build Docker image only
./deploy.sh build

# Start services only (if image exists)
./deploy.sh start

# Stop all services
./deploy.sh stop

# Restart services
./deploy.sh restart

# View logs
./deploy.sh logs

# Check service status
./deploy.sh status

# Clean up everything (removes containers, images, volumes)
./deploy.sh clean

# Show help
./deploy.sh help
```

## 📁 Project Structure

```
chatbot/
├── app_postgres.py          # Main FastAPI application with PostgreSQL
├── database.py              # PostgreSQL database manager
├── gradio_app.py            # Gradio web interface
├── Dockerfile               # Docker image definition
├── docker-compose.yml       # Multi-service orchestration
├── init_db.sql             # Database initialization script
├── requirements.txt         # Python dependencies
├── config.env              # Environment configuration
├── deploy.sh               # Deployment script
├── documents/              # Document files for RAG
│   └── user_manual_mkbank_uz_uz.pdf
└── README.md               # This file
```

## ⚙️ Configuration

### Environment Variables

Key configuration options in `docker-compose.yml`:

```yaml
# Database Configuration
DATABASE_URL: postgresql://chatbot_user:chatbot_password@postgres:5432/chatbot_db

# Qdrant Configuration
QDRANT_URL: http://qdrant:6333
QDRANT_COLLECTION_NAME: rag_documents
QDRANT_VECTOR_SIZE: 1024
QDRANT_DISTANCE: COSINE

# Model Configuration
EMBEDDING_MODEL: intfloat/multilingual-e5-large-instruct
CHAT_MODEL: google/gemma-3-12b-it

# Document Configuration
DOCUMENTS_PATH: /app/documents
```

### Volumes and Persistence

The application uses Docker volumes for data persistence:

- `postgres_data`: PostgreSQL database files
- `qdrant_storage`: Qdrant vector database files
- `app_data`: Application data and logs

## 🔧 Manual Docker Commands

If you prefer manual control:

### Build Image
```bash
docker build -t chatbot:latest .
```

### Start Services
```bash
docker-compose up -d
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f chatbot_app
docker-compose logs -f postgres
docker-compose logs -f qdrant
```

### Stop Services
```bash
docker-compose down
```

### Access Database
```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U chatbot_user -d chatbot_db

# View tables
\dt

# View chat sessions
SELECT * FROM chat_sessions;
```

## 🧪 Testing the Deployment

### 1. Health Check
```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "healthy",
  "message": "RAG system is operational",
  "qdrant_status": "healthy (1 collections)",
  "database_status": "healthy"
}
```

### 2. Test Chat API
```bash
curl -X POST "http://localhost:8080/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "message": "Hello, how can you help me?"
  }'
```

### 3. Test Gradio Interface
Open http://localhost:7860 in your browser and start chatting.

## 🔍 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Check what's using the port
sudo netstat -tulpn | grep :8080

# Kill the process if needed
sudo kill -9 <PID>
```

#### 2. Docker Permission Denied
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and log back in, or run:
newgrp docker
```

#### 3. Services Not Starting
```bash
# Check logs
./deploy.sh logs

# Check specific service
docker-compose logs chatbot_app
```

#### 4. Database Connection Issues
```bash
# Check PostgreSQL logs
docker-compose logs postgres

# Verify database is ready
docker-compose exec postgres pg_isready -U chatbot_user -d chatbot_db
```

#### 5. Out of Memory
```bash
# Check Docker memory usage
docker stats

# Increase Docker memory limit in Docker Desktop settings
# Or add swap space on Linux
```

### Log Locations

- **Application logs**: `docker-compose logs chatbot_app`
- **Database logs**: `docker-compose logs postgres`
- **Qdrant logs**: `docker-compose logs qdrant`
- **Gradio logs**: `docker-compose logs gradio_app`

## 🔄 Updates and Maintenance

### Updating the Application
```bash
# Pull latest changes
git pull

# Rebuild and redeploy
./deploy.sh
```

### Database Backup
```bash
# Backup database
docker-compose exec postgres pg_dump -U chatbot_user chatbot_db > backup.sql

# Restore database
docker-compose exec -T postgres psql -U chatbot_user -d chatbot_db < backup.sql
```

### Cleaning Up
```bash
# Remove all containers and images
./deploy.sh clean

# Remove unused Docker resources
docker system prune -a
```

## 📊 Monitoring

### Service Status
```bash
./deploy.sh status
```

### Resource Usage
```bash
docker stats
```

### Database Queries
```bash
# Connect to database
docker-compose exec postgres psql -U chatbot_user -d chatbot_db

# Check chat statistics
SELECT 
    COUNT(*) as total_sessions,
    COUNT(DISTINCT user_id) as unique_users
FROM chat_sessions;

# Check message count
SELECT COUNT(*) as total_messages FROM chat_messages;
```

## 🔐 Security Considerations

1. **Change default passwords** in production
2. **Use environment files** for sensitive data
3. **Enable SSL/TLS** for production deployments
4. **Restrict network access** using Docker networks
5. **Regular security updates** for base images

## 📞 Support

If you encounter issues:

1. Check the logs: `./deploy.sh logs`
2. Verify prerequisites are met
3. Ensure ports are available
4. Check Docker and Docker Compose versions
5. Review the troubleshooting section above

## 🎯 Next Steps

After successful deployment:

1. **Test the chat functionality** via Gradio interface
2. **Explore the API documentation** at http://localhost:8080/docs
3. **Add your own documents** to the `documents/` folder
4. **Customize the configuration** in `docker-compose.yml`
5. **Set up monitoring** and logging for production use

---

**Note**: This deployment uses PostgreSQL instead of JSON files for chat history storage, providing better scalability and data integrity. 