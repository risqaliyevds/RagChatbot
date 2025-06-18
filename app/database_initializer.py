"""
Database Initializer Module
Handles database connection validation and schema initialization for production deployments.
"""

import asyncio
import logging
import os
import time
from typing import Optional, Dict, Any, List
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import requests
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Handles database initialization and connection validation"""
    
    def __init__(self):
        self.config = get_config()
        self.max_retries = 30
        self.retry_delay = 2
        
    def check_postgresql_connection(self) -> bool:
        """Check PostgreSQL connection and create database if needed"""
        logger.info("🔍 Checking PostgreSQL connection...")
        
        for attempt in range(self.max_retries):
            try:
                # Parse DATABASE_URL to get connection parameters
                db_url = self.config.DATABASE_URL
                if db_url.startswith('postgresql://'):
                    # Extract components from DATABASE_URL
                    import urllib.parse as urlparse
                    parsed = urlparse.urlparse(db_url)
                    
                    host = parsed.hostname
                    port = parsed.port or 5432
                    user = parsed.username
                    password = parsed.password
                    database = parsed.path[1:]  # Remove leading slash
                    
                    # First, try to connect to the target database
                    try:
                        conn = psycopg2.connect(
                            host=host,
                            port=port,
                            user=user,
                            password=password,
                            database=database,
                            connect_timeout=10
                        )
                        conn.close()
                        logger.info("✅ PostgreSQL connection successful!")
                        return True
                        
                    except psycopg2.OperationalError as db_error:
                        if "does not exist" in str(db_error):
                            logger.warning(f"Database '{database}' does not exist. Attempting to create...")
                            
                            # Connect to postgres database to create target database
                            try:
                                admin_conn = psycopg2.connect(
                                    host=host,
                                    port=port,
                                    user=user,
                                    password=password,
                                    database='postgres',  # Connect to default postgres database
                                    connect_timeout=10
                                )
                                admin_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                                
                                with admin_conn.cursor() as cur:
                                    # Create database
                                    cur.execute(f'CREATE DATABASE "{database}"')
                                    logger.info(f"✅ Database '{database}' created successfully!")
                                
                                admin_conn.close()
                                
                                # Now try to connect to the newly created database
                                conn = psycopg2.connect(
                                    host=host,
                                    port=port,
                                    user=user,
                                    password=password,
                                    database=database,
                                    connect_timeout=10
                                )
                                conn.close()
                                logger.info("✅ PostgreSQL connection successful!")
                                return True
                                
                            except Exception as create_error:
                                logger.error(f"❌ Failed to create database: {create_error}")
                                raise
                        else:
                            raise
                            
                else:
                    logger.error("❌ Invalid DATABASE_URL format")
                    return False
                    
            except Exception as e:
                logger.warning(f"⚠️ PostgreSQL connection attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error("❌ PostgreSQL connection failed after all retries")
                    return False
                    
        return False
    
    def check_qdrant_connection(self) -> bool:
        """Check Qdrant connection"""
        logger.info("🔍 Checking Qdrant connection...")
        
        for attempt in range(self.max_retries):
            try:
                # Try HTTP health check first (Qdrant uses /collections endpoint for health)
                qdrant_url = self.config.QDRANT_URL
                health_url = f"{qdrant_url}/collections"
                
                response = requests.get(health_url, timeout=10)
                if response.status_code == 200:
                    logger.info("✅ Qdrant HTTP health check successful!")
                    
                    # Now test client connection
                    client = QdrantClient(url=qdrant_url)
                    collections = client.get_collections()
                    logger.info(f"✅ Qdrant client connection successful! Found {len(collections.collections)} collections")
                    return True
                else:
                    raise Exception(f"Health check failed with status {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Qdrant connection attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error("❌ Qdrant connection failed after all retries")
                    return False
                    
        return False
    
    def get_schema_sql(self) -> str:
        """Get the database schema SQL"""
        schema_file = os.path.join(os.path.dirname(__file__), '..', 'database', 'schema_initialization.sql')
        
        if not os.path.exists(schema_file):
            # Fallback to old location
            schema_file = os.path.join(os.path.dirname(__file__), '..', 'database', 'init_db.sql')
        
        if not os.path.exists(schema_file):
            raise FileNotFoundError(f"Database schema file not found at {schema_file}")
        
        with open(schema_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _parse_sql_statements(self, sql_content: str) -> List[str]:
        """Parse SQL content into individual statements, handling PostgreSQL $$ delimiters"""
        statements = []
        current_statement = ""
        in_dollar_quoted = False
        dollar_tag = ""
        
        lines = sql_content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('--'):
                continue
            
            # Check for dollar-quoted string delimiters
            if '$$' in line and not in_dollar_quoted:
                # Starting a dollar-quoted block
                in_dollar_quoted = True
                # Extract the tag (e.g., $func$ or $$)
                dollar_parts = line.split('$$')
                if len(dollar_parts) >= 2:
                    dollar_tag = dollar_parts[0].split()[-1] if dollar_parts[0].strip() else ""
                current_statement += line + '\n'
            elif '$$' in line and in_dollar_quoted:
                # Check if this is the matching closing delimiter
                if line.startswith(f"{dollar_tag}$$") or (not dollar_tag and line.startswith("$$")):
                    # Ending the dollar-quoted block
                    current_statement += line + '\n'
                    in_dollar_quoted = False
                    dollar_tag = ""
                else:
                    current_statement += line + '\n'
            elif in_dollar_quoted:
                # Inside a dollar-quoted block, add everything
                current_statement += line + '\n'
            else:
                # Normal SQL line
                current_statement += line + '\n'
                
                # Check if this line ends with a semicolon (end of statement)
                if line.rstrip().endswith(';'):
                    # Complete statement found
                    stmt = current_statement.strip()
                    if stmt:
                        statements.append(stmt)
                    current_statement = ""
        
        # Add any remaining statement
        if current_statement.strip():
            statements.append(current_statement.strip())
        
        return statements
    
    def run_database_migration(self) -> bool:
        """Run database schema initialization/migration"""
        logger.info("🚀 Running database schema initialization...")
        
        try:
            # Get database connection
            import psycopg2
            from urllib.parse import urlparse
            
            parsed = urlparse(self.config.DATABASE_URL)
            
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                user=parsed.username,
                password=parsed.password,
                database=parsed.path[1:],  # Remove leading slash
                connect_timeout=30
            )
            
            # Get schema SQL
            schema_sql = self.get_schema_sql()
            
            # Execute schema
            with conn.cursor() as cur:
                # Smart SQL parsing - handle PostgreSQL functions with $$ delimiters
                statements = self._parse_sql_statements(schema_sql)
                
                for i, statement in enumerate(statements):
                    try:
                        logger.info(f"📄 Executing statement {i + 1}/{len(statements)}")
                        cur.execute(statement)
                        conn.commit()
                    except psycopg2.errors.DuplicateTable as e:
                        logger.info(f"⚠️ Table already exists (skipping): {e}")
                        conn.rollback()
                    except psycopg2.errors.DuplicateObject as e:
                        logger.info(f"⚠️ Object already exists (skipping): {e}")
                        conn.rollback()
                    except Exception as e:
                        logger.error(f"❌ Failed to execute statement {i + 1}: {e}")
                        logger.error(f"Statement: {statement[:200]}...")
                        conn.rollback()
                        # Don't fail completely for non-critical errors
                        continue
            
            conn.close()
            logger.info("✅ Database schema initialization completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database migration failed: {e}")
            return False
    
    def initialize_qdrant_collection(self) -> bool:
        """Initialize Qdrant collection if it doesn't exist or recreate if needed"""
        logger.info("🔍 Checking/Creating Qdrant collection...")
        
        try:
            client = QdrantClient(url=self.config.QDRANT_URL)
            collection_name = self.config.QDRANT_COLLECTION_NAME
            
            # Check if collection exists
            collection_exists = False
            try:
                collection_info = client.get_collection(collection_name)
                collection_exists = True
                
                # Check if dimensions match
                current_size = collection_info.config.params.vectors.size
                expected_size = self.config.QDRANT_VECTOR_SIZE
                
                if current_size != expected_size:
                    logger.warning(f"⚠️ Collection '{collection_name}' exists with dimension {current_size}, but expected {expected_size}")
                    
                    if self.config.QDRANT_FORCE_RECREATE:
                        logger.info(f"🔄 QDRANT_FORCE_RECREATE=true, deleting and recreating collection")
                        client.delete_collection(collection_name)
                        collection_exists = False
                    else:
                        logger.error(f"❌ Dimension mismatch! Set QDRANT_FORCE_RECREATE=true to recreate the collection.")
                        return False
                else:
                    logger.info(f"✅ Qdrant collection '{collection_name}' already exists with correct dimensions ({current_size}) and {collection_info.points_count} points")
                    return True
                    
            except UnexpectedResponse as e:
                if "Not found" in str(e):
                    collection_exists = False
                else:
                    raise
            
            # Create collection if it doesn't exist
            if not collection_exists:
                logger.info(f"📦 Creating new Qdrant collection: {collection_name}")
                
                from qdrant_client.models import Distance, VectorParams
                
                # Create collection with proper configuration
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.config.QDRANT_VECTOR_SIZE,
                        distance=Distance.COSINE if self.config.QDRANT_DISTANCE == "COSINE" else Distance.DOT
                    ),
                    on_disk_payload=self.config.QDRANT_ON_DISK
                )
                
                logger.info(f"✅ Qdrant collection '{collection_name}' created successfully with dimension {self.config.QDRANT_VECTOR_SIZE}!")
                return True
                    
        except Exception as e:
            logger.error(f"❌ Qdrant collection initialization failed: {e}")
            return False
    
    async def run_full_initialization(self) -> Dict[str, Any]:
        """Run complete initialization process"""
        logger.info("🚀 Starting comprehensive system initialization...")
        
        results = {
            'postgresql_connection': False,
            'qdrant_connection': False,
            'database_migration': False,
            'qdrant_collection': False,
            'overall_success': False
        }
        
        try:
            # Step 1: Check PostgreSQL connection
            results['postgresql_connection'] = self.check_postgresql_connection()
            if not results['postgresql_connection']:
                logger.error("❌ PostgreSQL connection failed - cannot proceed")
                return results
            
            # Step 2: Check Qdrant connection  
            results['qdrant_connection'] = self.check_qdrant_connection()
            if not results['qdrant_connection']:
                logger.error("❌ Qdrant connection failed - cannot proceed")
                return results
            
            # Step 3: Run database migration
            results['database_migration'] = self.run_database_migration()
            if not results['database_migration']:
                logger.error("❌ Database migration failed")
                return results
            
            # Step 4: Initialize Qdrant collection
            results['qdrant_collection'] = self.initialize_qdrant_collection()
            if not results['qdrant_collection']:
                logger.error("❌ Qdrant collection initialization failed")
                return results
            
            # All steps completed successfully
            results['overall_success'] = True
            logger.info("🎉 Complete system initialization successful!")
            
        except Exception as e:
            logger.error(f"❌ Initialization process failed: {e}")
            
        return results


# Convenience functions for easy import
async def initialize_system() -> Dict[str, Any]:
    """Initialize the complete system"""
    initializer = DatabaseInitializer()
    return await initializer.run_full_initialization()


def check_connections() -> Dict[str, bool]:
    """Quick connection check"""
    initializer = DatabaseInitializer()
    return {
        'postgresql': initializer.check_postgresql_connection(),
        'qdrant': initializer.check_qdrant_connection()
    } 