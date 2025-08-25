"""
Core application package for the RAG chatbot system.
Includes all major components: RAG pipeline, database management, 
document processing, embeddings, and configuration.
"""

from .config import get_config, load_config, validate_config
from .rag_pipeline_manager import ChatService
from .database.postgresql_manager import DatabaseManager
from .qdrant_manager import init_qdrant_client, init_vectorstore
from .document_processing import load_and_split_documents
from .embedding_manager import MultilingualE5Embeddings
from .models import *
from .database_initializer import initialize_system, initialize_system_fresh, check_connections

__all__ = [
    'get_config', 'load_config', 'validate_config',
    'ChatService',
    'DatabaseManager', 
    'init_qdrant_client', 'init_vectorstore',
    'load_and_split_documents',
    'MultilingualE5Embeddings',
    'initialize_system', 'initialize_system_fresh', 'check_connections'
]

__version__ = "2.0.0" 