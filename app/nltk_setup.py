"""
NLTK Setup for Offline Deployment
=================================

Handles NLTK data path configuration for offline environments.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_nltk_offline():
    """Setup NLTK for offline use"""
    try:
        import nltk
        
        # Check for local nltk_data directory
        current_dir = Path(__file__).parent.parent  # Go up to project root
        nltk_data_paths = [
            current_dir / "nltk_data",           # In project root
            Path("/app/nltk_data"),              # In Docker container
            Path.home() / "nltk_data",           # In user home
            Path("/usr/share/nltk_data"),        # System-wide
            Path("/usr/local/share/nltk_data"),  # Alternative system-wide
        ]
        
        # Add existing paths first (don't override system defaults completely)
        existing_paths = nltk.data.path.copy()
        
        # Add our custom paths at the beginning (higher priority)
        for nltk_path in nltk_data_paths:
            if nltk_path.exists():
                path_str = str(nltk_path)
                if path_str not in nltk.data.path:
                    nltk.data.path.insert(0, path_str)
                    logger.info(f"Added NLTK data path: {nltk_path}")
        
        # Check if NLTK_DATA environment variable is set
        env_nltk_data = os.getenv("NLTK_DATA")
        if env_nltk_data:
            env_path = Path(env_nltk_data)
            if env_path.exists():
                if str(env_path) not in nltk.data.path:
                    nltk.data.path.insert(0, str(env_path))
                    logger.info(f"Added NLTK data from environment: {env_path}")
        
        # Test if punkt tokenizer is available
        try:
            nltk.data.find('tokenizers/punkt')
            logger.info("✅ NLTK punkt tokenizer found")
        except LookupError:
            logger.warning("⚠️ NLTK punkt tokenizer not found - some document processing may be limited")
            
        # Test if punkt_tab tokenizer is available (newer NLTK)
        try:
            nltk.data.find('tokenizers/punkt_tab')
            logger.info("✅ NLTK punkt_tab tokenizer found")
        except LookupError:
            logger.warning("⚠️ NLTK punkt_tab tokenizer not found - using fallback")
        
        logger.info(f"NLTK data paths configured: {nltk.data.path[:3]}...")  # Show first 3 paths
        
    except ImportError:
        logger.warning("NLTK not installed - document processing may be limited")
    except Exception as e:
        logger.error(f"Failed to setup NLTK: {e}")


def ensure_nltk_downloads():
    """Ensure required NLTK downloads are available (with fallback)"""
    try:
        import nltk
        
        required_packages = ['punkt', 'punkt_tab']
        
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                try:
                    # Try to download if we have internet
                    logger.info(f"Attempting to download NLTK {package}...")
                    nltk.download(package, quiet=True)
                    logger.info(f"✅ Downloaded NLTK {package}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not download NLTK {package}: {e}")
                    
    except Exception as e:
        logger.error(f"NLTK package check failed: {e}")


# Initialize NLTK setup when module is imported
setup_nltk_offline() 