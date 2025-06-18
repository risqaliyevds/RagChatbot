"""
Configuration Management for RAG Application
==========================================

Handles loading and validation of environment variables and application configuration.
"""

import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Global configuration instance
_config_instance = None

class Config:
    """Configuration class with attribute access"""
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            # Convert to uppercase for environment variable style
            setattr(self, key.upper(), value)
    
    def get(self, key: str, default: Any = None):
        """Get config value with default"""
        return getattr(self, key.upper(), default)

def get_config() -> Config:
    """Get configuration instance (cached)"""
    global _config_instance
    if _config_instance is None:
        config_dict = load_config()
        _config_instance = Config(config_dict)
    return _config_instance


def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables"""
    
    # Load environment variables from .env file (if exists)
    # This allows for easy local development and deployment
    try:
        load_dotenv(".env", override=True)  # Override existing env vars with .env values
        logger.info("Loaded configuration from .env file")
    except Exception as e:
        logger.warning(f"Could not load .env file: {e}")
    
    # Legacy support - try to load config.env if it exists
    try:
        load_dotenv("config.env", override=False)  # Don't override if already set
        logger.info("Loaded additional configuration from config.env file")
    except Exception as e:
        logger.debug(f"No config.env file found: {e}")
    
    config = {
        # Application Configuration
        "environment": os.getenv("ENVIRONMENT", "development"),
        "debug": os.getenv("DEBUG", "false").lower() == "true",
        "reload": os.getenv("RELOAD", "false").lower() == "true",
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        
        # Server Configuration
        "host": os.getenv("HOST", "0.0.0.0"),
        "port": int(os.getenv("PORT", "8081")),
        
        # PostgreSQL Database Settings
        "postgres_host": os.getenv("POSTGRES_HOST", "localhost"),
        "postgres_port": int(os.getenv("POSTGRES_PORT", "5432")),
        "postgres_db": os.getenv("POSTGRES_DB", "chatbot_db"),
        "postgres_user": os.getenv("POSTGRES_USER", "chatbot_user"),
        "postgres_password": os.getenv("POSTGRES_PASSWORD", "chatbot_password"),
        "postgres_pool_size": int(os.getenv("POSTGRES_POOL_SIZE", "10")),
        "postgres_max_overflow": int(os.getenv("POSTGRES_MAX_OVERFLOW", "20")),
        "postgres_pool_timeout": int(os.getenv("POSTGRES_POOL_TIMEOUT", "30")),
        "postgres_pool_recycle": int(os.getenv("POSTGRES_POOL_RECYCLE", "3600")),
        
        # Database URL and Migration Settings
        "database_url": os.getenv("DATABASE_URL", 
            f"postgresql://{os.getenv('POSTGRES_USER', 'chatbot_user')}:{os.getenv('POSTGRES_PASSWORD', 'chatbot_password')}@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'chatbot_db')}"),
        "db_auto_migrate": os.getenv("DB_AUTO_MIGRATE", "true").lower() == "true",
        "db_drop_all": os.getenv("DB_DROP_ALL", "false").lower() == "true",
        
        # vLLM API Configuration - Chat Model
        "vllm_api_key": os.getenv("VLLM_API_KEY", "EMPTY"),
        "vllm_chat_endpoint": os.getenv("VLLM_CHAT_ENDPOINT", "http://localhost:8000/v1"),
        "chat_model": os.getenv("CHAT_MODEL", "google/gemma-3-12b-it"),
        
        # vLLM API Configuration - Embedding Model
        "vllm_embedding_key": os.getenv("VLLM_EMBEDDING_KEY", "EMPTY"),
        "vllm_embedding_endpoint": os.getenv("VLLM_EMBEDDING_ENDPOINT"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct"),
        
        # Qdrant Vector Database Configuration
        "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
        "qdrant_path": os.getenv("QDRANT_PATH", "./qdrant_storage"),
        "qdrant_collection_name": os.getenv("QDRANT_COLLECTION_NAME", "rag_documents"),
        "qdrant_vector_size": int(os.getenv("QDRANT_VECTOR_SIZE", "1024")),
        "qdrant_distance": os.getenv("QDRANT_DISTANCE", "COSINE"),
        "qdrant_force_recreate": os.getenv("QDRANT_FORCE_RECREATE", "false").lower() == "true",
        "qdrant_on_disk": os.getenv("QDRANT_ON_DISK", "true").lower() == "true",
        
        # Document Processing Configuration
        "document_url": os.getenv("DOCUMENT_URL", ""),
        "documents_path": os.getenv("DOCUMENTS_PATH", "./documents"),
        
        # RAG Configuration
        "top_k": int(os.getenv("TOP_K", "3")),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
        
        # Gradio UI Configuration
        "gradio_host": os.getenv("GRADIO_HOST", "0.0.0.0"),
        "gradio_port": int(os.getenv("GRADIO_PORT", "7860")),
        "gradio_share": os.getenv("GRADIO_SHARE", "false").lower() == "true",
        "api_base_url": os.getenv("API_BASE_URL", "http://localhost:8081"),
        
        # CORS Configuration
        "cors_origins": os.getenv("CORS_ORIGINS", '["*"]'),
        "cors_methods": os.getenv("CORS_METHODS", '["GET", "POST", "PUT", "DELETE"]'),
        "cors_headers": os.getenv("CORS_HEADERS", '["*"]'),
        
        # Health Check Configuration
        "health_check_interval": int(os.getenv("HEALTH_CHECK_INTERVAL", "30")),
        "db_health_check_timeout": int(os.getenv("DB_HEALTH_CHECK_TIMEOUT", "5")),
        "vector_db_health_check_timeout": int(os.getenv("VECTOR_DB_HEALTH_CHECK_TIMEOUT", "5")),
    }
    
    # Log configuration summary (without sensitive data)
    safe_config = {k: v for k, v in config.items() 
                   if not any(sensitive in k.lower() for sensitive in ['password', 'key', 'secret', 'token'])}
    logger.info(f"Configuration loaded successfully. Environment: {config['environment']}")
    logger.debug(f"Config summary: {safe_config}")
    
    return config


def validate_config(config) -> bool:
    """Validate required configuration parameters"""
    required_keys = [
        "embedding_model",
        "chat_model", 
        "qdrant_collection_name",
        "documents_path",
        "database_url",
        "postgres_host",
        "postgres_port",
        "postgres_db",
        "postgres_user"
    ]
    
    # Handle both dict and Config object
    get_func = config.get if hasattr(config, 'get') else lambda k, d=None: getattr(config, k.upper(), d)
    
    missing_keys = [key for key in required_keys if not get_func(key)]
    
    if missing_keys:
        logger.error(f"Missing required configuration keys: {missing_keys}")
        return False
    
    # Validate numeric values
    try:
        port = get_func("port", 8081)
        postgres_port = get_func("postgres_port", 5432)
        qdrant_vector_size = get_func("qdrant_vector_size", 1024)
        top_k = get_func("top_k", 3)
        chunk_size = get_func("chunk_size", 1000)
        chunk_overlap = get_func("chunk_overlap", 200)
        postgres_pool_size = get_func("postgres_pool_size", 10)
        postgres_max_overflow = get_func("postgres_max_overflow", 20)
        postgres_pool_timeout = get_func("postgres_pool_timeout", 30)
        gradio_port = get_func("gradio_port", 7860)
        health_check_interval = get_func("health_check_interval", 30)
        
        # Port validations
        assert 1 <= port <= 65535, f"Invalid port number: {port}"
        assert 1 <= postgres_port <= 65535, f"Invalid postgres port: {postgres_port}"
        assert 1 <= gradio_port <= 65535, f"Invalid gradio port: {gradio_port}"
        
        # Vector and processing validations
        assert qdrant_vector_size > 0, f"Invalid vector size: {qdrant_vector_size}"
        assert top_k > 0, f"Invalid top_k value: {top_k}"
        assert chunk_size > 0, f"Invalid chunk_size: {chunk_size}"
        assert chunk_overlap >= 0, f"Invalid chunk_overlap: {chunk_overlap}"
        
        # Database pool validations
        assert postgres_pool_size > 0, f"Invalid postgres pool size: {postgres_pool_size}"
        assert postgres_max_overflow >= 0, f"Invalid postgres max overflow: {postgres_max_overflow}"
        assert postgres_pool_timeout > 0, f"Invalid postgres pool timeout: {postgres_pool_timeout}"
        
        # Health check validations
        assert health_check_interval > 0, f"Invalid health check interval: {health_check_interval}"
        
    except (ValueError, AssertionError) as e:
        logger.error(f"Configuration validation error: {e}")
        return False
    
    logger.info("Configuration validation passed")
    return True 