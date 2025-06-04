# Project Structure

This document outlines the organized structure of the RAG Chatbot project.

## Root Directory Files

### Core Application Files
- `app_postgres.py` - Main FastAPI application with RAG functionality

### Configuration Files
- `.env` - Main environment configuration
- `config.env` - Docker-specific environment configuration
- `requirements.txt` - Python dependencies

### Docker Files
- `Dockerfile` - Docker image configuration
- `docker-compose.yml` - Multi-container orchestration
- `.dockerignore` - Docker build exclusions

### Documentation
- `README.md` - Main project documentation

### Development Files
- `.gitignore` - Git exclusions
- `.git/` - Git repository data
- `venv/` - Python virtual environment

## Directories

### `/app/`
Application modules and interfaces:
- `gradio_app.py` - Gradio web interface for testing

### `/database/`
Database-related modules and scripts:
- `database.py` - Database connection and management
- `init_db.sql` - Database initialization script
- `__init__.py` - Package initialization

### `/docs/`
Project documentation files:
- `DEPLOYMENT_SUMMARY.md` - Deployment overview
- `DOCKER_DEPLOYMENT.md` - Docker deployment guide
- `QDRANT_CONFIG_README.md` - Qdrant configuration guide
- `STRUCTURE.md` - This file (project structure overview)

### `/scripts/`
Deployment and utility scripts:
- `deploy.sh` - Deployment automation script
- `k8s-deployment.yaml` - Kubernetes deployment configuration
- `restart_app.sh` - Application restart script

### `/documents/`
RAG knowledge base documents:
- `user_manual_mkbank_uz_uz.pdf` - Bank user manual (Uzbek)

### `/data/`
Runtime data directories (created by Docker):
- `postgres_data/` - PostgreSQL data
- `qdrant_storage/` - Qdrant vector database storage
- `app_data/` - Application runtime data

## Cleaned Up Items

The following items were removed during cleanup:
- `*.log` files (app.log, app_new.log, gradio.log)
- `chat_history.json` (temporary test data)
- `__pycache__/` directories (Python cache)

These items are now prevented from being tracked by the `.gitignore` file.

## File Organization Principles

1. **Root Level**: Only essential application entry point and configuration
2. **Application Code**: Organized in `/app/` directory
3. **Database**: All database-related code and scripts in `/database/`
4. **Documentation**: Centralized in `/docs/` directory
5. **Scripts**: Deployment and utility scripts in `/scripts/`
6. **Data**: Runtime data isolated in `/data/` (gitignored)
7. **Documents**: Knowledge base documents in `/documents/`
8. **Temporary Files**: Automatically excluded via `.gitignore`

This structure ensures a clean, maintainable, and well-organized codebase with clear separation of concerns. 