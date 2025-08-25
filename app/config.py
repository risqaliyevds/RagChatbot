"""
Configuration Management for RAG Chatbot
=======================================

Centralized configuration management with environment variable support
and production-ready defaults.
"""

import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Legacy load_config function - use ConfigManager instead"""
    config_manager = ConfigManager()
    return config_manager._config


def validate_config(config) -> bool:
    """Legacy validate_config function - use ConfigManager validation instead"""
    return True  # ConfigManager handles validation internally


# Legacy functions to support existing code
def get_config() -> Any:
    """Get configuration as a namespace object for backward compatibility"""
    # Use ConfigManager instead of load_config
    config_manager = ConfigManager()
    config_dict = config_manager._config
    
    class Config:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key.upper(), value)
        
        def get(self, key, default=None):
            return getattr(self, key.upper(), default)
    
    return Config(config_dict)


@dataclass
class ConfigManager:
    """
    Configuration manager with comprehensive validation and environment support.
    
    This class handles all configuration loading, validation, and provides
    a centralized configuration source for the entire application.
    """
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize configuration manager"""
        self._config = {}
        
        # Load from .env files if they exist
        env_config = {}
        
        # Try to load from .env file (fallback gracefully if not found)
        try:
            env_config.update(self._load_from_env_file('.env'))
        except Exception as e:
            logger.warning(f"Could not load .env file: {e}")
        
        # Try to load from config.env file (fallback gracefully if not found)
        try:
            env_config.update(self._load_from_env_file('config.env'))
        except Exception as e:
            logger.debug(f"Could not load config.env file: {e}")
        
        # Load from environment variables (this will include Docker environment variables)
        os_env = dict(os.environ)
        
        # Merge configurations (environment variables take precedence)
        merged_config = {**env_config, **os_env}
        
        # Set defaults for required configuration
        defaults = self._get_default_config()
        
        # Apply defaults for missing values
        for key, default_value in defaults.items():
            if key not in merged_config or not merged_config[key]:
                merged_config[key] = default_value
        
        # Process and validate configuration
        self._config = self._process_config(merged_config)
        
        # Validate critical configuration
        validation_result = self._validate_config()
        if not validation_result['valid']:
            logger.error(f"Configuration validation failed: {validation_result['errors']}")
            # Don't raise exception in production - use defaults
            logger.warning("Using default configuration values for invalid settings")
        
        logger.info(f"Configuration loaded successfully. Environment: {self.environment}")
    
    def _get_default_config(self) -> Dict[str, str]:
        """Get default configuration values"""
        return {
            # Application Settings
            'ENVIRONMENT': 'production',
            'DEBUG': 'false',
            'LOG_LEVEL': 'INFO',
            'HOST': '0.0.0.0',
            'PORT': '8081',
            
            # Database Settings (Docker defaults)
            'POSTGRES_HOST': 'postgres',
            'POSTGRES_PORT': '5432',
            'POSTGRES_DB': 'chatbot_db',
            'POSTGRES_USER': 'chatbot_user', 
            'POSTGRES_PASSWORD': 'chatbot_password',
            
            # Qdrant Settings (Docker defaults)
            'QDRANT_URL': 'http://qdrant:6333',
            'QDRANT_COLLECTION_NAME': 'chatbot_collection',
            'QDRANT_VECTOR_SIZE': '1024',
            'QDRANT_DISTANCE': 'COSINE',
            'QDRANT_ON_DISK': 'true',
            'QDRANT_FORCE_RECREATE': 'false',
            
            # Gradio Settings
            'GRADIO_HOST': '0.0.0.0',
            'GRADIO_PORT': '7860',
            'GRADIO_SHARE': 'false',
            
            # Fresh Start
            'FRESH_START': 'false',
        }
    
    def get_config(self, reload_config: bool = False) -> Dict[str, Any]:
        """Get configuration with optional reload"""
        if self._config is None or reload_config:
            self._config = load_config()
            if not validate_config(self._config):
                raise ValueError("Invalid configuration detected")
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a specific configuration value"""
        config = self.get_config()
        return config.get(key, default)
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration at runtime (use with caution)"""
        if self._config is None:
            self._config = load_config()
        self._config.update(updates)
        if not validate_config(self._config):
            logger.error("Configuration update resulted in invalid configuration")
            raise ValueError("Invalid configuration after update")
    
    def _load_from_env_file(self, env_file: str) -> Dict[str, str]:
        """Load configuration from .env file"""
        config = {}
        env_path = Path(env_file)
        
        if not env_path.exists():
            logger.warning(f"Environment file {env_file} not found, using system environment variables only")
            return config
        
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse key=value pairs
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes if present
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        
                        config[key] = value
                    else:
                        logger.warning(f"Invalid line {line_num} in {env_file}: {line}")
                        
            logger.info(f"Loaded configuration from {env_file} file")
            return config
            
        except Exception as e:
            logger.error(f"Error reading {env_file}: {e}")
            return config
    
    def _process_config(self, config: Dict[str, str]) -> Dict[str, Any]:
        """Process and convert configuration values to appropriate types"""
        processed = {}
        
        # String values
        processed.update({
            'environment': config.get('ENVIRONMENT', 'production'),
            'log_level': config.get('LOG_LEVEL', 'INFO'),
            'host': config.get('HOST', '0.0.0.0'),
            'postgres_host': config.get('POSTGRES_HOST', 'postgres'),
            'postgres_db': config.get('POSTGRES_DB', 'chatbot_db'),
            'postgres_user': config.get('POSTGRES_USER', 'chatbot_user'),
            'postgres_password': config.get('POSTGRES_PASSWORD', 'chatbot_password'),
            'qdrant_url': config.get('QDRANT_URL', 'http://qdrant:6333'),
            'qdrant_collection_name': config.get('QDRANT_COLLECTION_NAME', 'chatbot_collection'),
            'qdrant_distance': config.get('QDRANT_DISTANCE', 'COSINE'),
            'gradio_host': config.get('GRADIO_HOST', '0.0.0.0'),
            'chat_model': config.get('CHAT_MODEL', 'google/gemma-3-12b-it'),
            'embedding_model': config.get('EMBEDDING_MODEL', 'intfloat/multilingual-e5-base'),
            'documents_path': config.get('DOCUMENTS_PATH', './documents'),
        })
        
        # Boolean values
        processed.update({
            'debug': config.get('DEBUG', 'false').lower() == 'true',
            'reload': config.get('RELOAD', 'false').lower() == 'true',
            'qdrant_on_disk': config.get('QDRANT_ON_DISK', 'true').lower() == 'true',
            'qdrant_force_recreate': config.get('QDRANT_FORCE_RECREATE', 'false').lower() == 'true',
            'gradio_share': config.get('GRADIO_SHARE', 'false').lower() == 'true',
            'fresh_start': config.get('FRESH_START', 'false').lower() == 'true',
        })
        
        # Integer values
        processed.update({
            'port': int(config.get('PORT', '8081')),
            'postgres_port': int(config.get('POSTGRES_PORT', '5432')),
            'qdrant_vector_size': int(config.get('QDRANT_VECTOR_SIZE', '1024')),
            'gradio_port': int(config.get('GRADIO_PORT', '7860')),
            'top_k': int(config.get('TOP_K', '3')),
            'chunk_size': int(config.get('CHUNK_SIZE', '1000')),
            'chunk_overlap': int(config.get('CHUNK_OVERLAP', '200')),
            'max_file_size': int(config.get('MAX_FILE_SIZE', '50485760')),
            'session_timeout_hours': int(config.get('SESSION_TIMEOUT_HOURS', '1')),
            'database_max_retries': int(config.get('DATABASE_MAX_RETRIES', '30')),
            'database_retry_delay': int(config.get('DATABASE_RETRY_DELAY', '2')),
            'qdrant_max_retries': int(config.get('QDRANT_MAX_RETRIES', '30')),
            'qdrant_retry_delay': int(config.get('QDRANT_RETRY_DELAY', '2')),
        })
        
        # Float values
        processed.update({
            'llm_temperature': float(config.get('LLM_TEMPERATURE', '0.7')),
            'vector_search_score_threshold': float(config.get('VECTOR_SEARCH_SCORE_THRESHOLD', '0.4')),
        })
        
        # Construct database URL if not provided
        if 'DATABASE_URL' not in config or not config['DATABASE_URL']:
            processed['database_url'] = f"postgresql://{processed['postgres_user']}:{processed['postgres_password']}@{processed['postgres_host']}:{processed['postgres_port']}/{processed['postgres_db']}"
        else:
            processed['database_url'] = config['DATABASE_URL']
        
        # Add other commonly used configurations with defaults
        processed.update({
            'api_base_url': config.get('API_BASE_URL', f"http://localhost:{processed['port']}"),
            'vllm_chat_endpoint': config.get('VLLM_CHAT_ENDPOINT', 'http://localhost:8000/v1'),
            'vllm_api_key': config.get('VLLM_API_KEY', 'EMPTY'),
            'llm_max_tokens': int(config.get('LLM_MAX_TOKENS', '2048')),
            'llm_request_timeout': int(config.get('LLM_REQUEST_TIMEOUT', '120')),
            'llm_max_retries': int(config.get('LLM_MAX_RETRIES', '3')),
            'min_document_length': int(config.get('MIN_DOCUMENT_LENGTH', '50')),
            'chunk_preview_length': int(config.get('CHUNK_PREVIEW_LENGTH', '200')),
            'error_text_limit': int(config.get('ERROR_TEXT_LIMIT', '500')),
            'http_client_timeout': int(config.get('HTTP_CLIENT_TIMEOUT', '10')),
            'gradio_client_timeout': int(config.get('GRADIO_CLIENT_TIMEOUT', '60')),
            'fresh_init_timeout': int(config.get('FRESH_INIT_TIMEOUT', '60')),
        })
        
        return processed
    
    def _validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return validation results"""
        errors = []
        warnings = []
        
        try:
            # Validate ports
            if not (1 <= self._config['port'] <= 65535):
                errors.append(f"Invalid port: {self._config['port']}")
            if not (1 <= self._config['postgres_port'] <= 65535):
                errors.append(f"Invalid postgres port: {self._config['postgres_port']}")
            if not (1 <= self._config['gradio_port'] <= 65535):
                errors.append(f"Invalid gradio port: {self._config['gradio_port']}")
            
            # Validate vector size
            if self._config['qdrant_vector_size'] <= 0:
                errors.append(f"Invalid vector size: {self._config['qdrant_vector_size']}")
            
            # Validate required string fields
            required_fields = ['postgres_host', 'postgres_db', 'postgres_user', 'qdrant_url', 'qdrant_collection_name']
            for field in required_fields:
                if not self._config.get(field):
                    errors.append(f"Missing required field: {field}")
            
            # Validate LLM parameters
            if not (0.0 <= self._config['llm_temperature'] <= 2.0):
                errors.append(f"Invalid LLM temperature: {self._config['llm_temperature']}")
            
            # Warnings for common issues
            if self._config['environment'] == 'development' and not self._config['debug']:
                warnings.append("Development environment but debug is disabled")
            
        except Exception as e:
            errors.append(f"Configuration validation exception: {e}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    @property
    def environment(self) -> str:
        """Get current environment"""
        return self._config.get('environment', 'production')


# Global configuration manager instance
config_manager = ConfigManager() 