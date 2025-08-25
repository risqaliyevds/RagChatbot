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
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config if config else {}
        self.max_retries = self.config.get("database_max_retries", 30)
        self.retry_delay = self.config.get("database_retry_delay", 2)
        self.connect_timeout = self.config.get("database_connect_timeout", 10)
        self.migration_timeout = self.config.get("database_migration_timeout", 30)
        
        # Use config for database connection if available
        if config:
            self.config = config
        else:
            from app.config import get_config
            self.config = get_config()
        
    def check_postgresql_connection(self) -> bool:
        """Check PostgreSQL connection and create database if needed"""
        logger.info("🔍 Checking PostgreSQL connection...")
        
        for attempt in range(self.max_retries):
            try:
                # Parse DATABASE_URL to get connection parameters
                # Handle both dict and object config
                if hasattr(self.config, 'DATABASE_URL'):
                    db_url = self.config.DATABASE_URL
                elif isinstance(self.config, dict):
                    db_url = self.config.get('database_url')
                else:
                    db_url = getattr(self.config, 'database_url', None)
                
                if not db_url:
                    # Fallback to environment variables
                    import os
                    postgres_host = os.getenv("POSTGRES_HOST", "localhost")
                    postgres_port = os.getenv("POSTGRES_PORT", "5432")
                    postgres_db = os.getenv("POSTGRES_DB", "chatbot_db")
                    postgres_user = os.getenv("POSTGRES_USER", "chatbot_user")
                    postgres_password = os.getenv("POSTGRES_PASSWORD", "chatbot_password")
                    db_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
                
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
                            connect_timeout=self.connect_timeout
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
                                    connect_timeout=self.connect_timeout
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
                                    connect_timeout=self.connect_timeout
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
        
        # Get Qdrant configuration values
        qdrant_max_retries = self.config.get("qdrant_max_retries", 30)
        qdrant_retry_delay = self.config.get("qdrant_retry_delay", 2)
        qdrant_timeout = self.config.get("qdrant_connect_timeout", 10)
        
        for attempt in range(qdrant_max_retries):
            try:
                # Try HTTP health check first (Qdrant uses /collections endpoint for health)
                # Handle both dict and object config
                if hasattr(self.config, 'QDRANT_URL'):
                    qdrant_url = self.config.QDRANT_URL
                elif isinstance(self.config, dict):
                    qdrant_url = self.config.get('qdrant_url', 'http://qdrant:6333')
                else:
                    qdrant_url = getattr(self.config, 'qdrant_url', 'http://qdrant:6333')
                
                health_url = f"{qdrant_url}/collections"
                
                response = requests.get(health_url, timeout=qdrant_timeout)
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
                logger.warning(f"⚠️ Qdrant connection attempt {attempt + 1}/{qdrant_max_retries} failed: {e}")
                if attempt < qdrant_max_retries - 1:
                    time.sleep(qdrant_retry_delay)
                else:
                    logger.error("❌ Qdrant connection failed after all retries")
                    return False
                    
        return False
    
    def get_schema_sql(self) -> str:
        """Get the database schema SQL"""
        schema_file = os.path.join(os.path.dirname(__file__), 'database', 'schema_initialization.sql')
        
        if not os.path.exists(schema_file):
            # Fallback to alternative path
            schema_file = os.path.join(os.path.dirname(__file__), '..', 'database', 'schema_initialization.sql')
        
        if not os.path.exists(schema_file):
            # Second fallback to old location
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
            import psycopg2
            from urllib.parse import urlparse
            
            # Handle both dict and object config
            if hasattr(self.config, 'DATABASE_URL'):
                db_url = self.config.DATABASE_URL
            elif isinstance(self.config, dict):
                db_url = self.config.get('database_url')
            else:
                db_url = getattr(self.config, 'database_url', None)
            
            if not db_url:
                # Fallback to environment variables
                import os
                postgres_host = os.getenv("POSTGRES_HOST", "localhost")
                postgres_port = os.getenv("POSTGRES_PORT", "5432")
                postgres_db = os.getenv("POSTGRES_DB", "chatbot_db")
                postgres_user = os.getenv("POSTGRES_USER", "chatbot_user")
                postgres_password = os.getenv("POSTGRES_PASSWORD", "chatbot_password")
                db_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
            
            parsed = urlparse(db_url)
            
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                user=parsed.username,
                password=parsed.password,
                database=parsed.path[1:],  # Remove leading slash
                connect_timeout=self.migration_timeout
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
    
    def clean_postgresql_data(self) -> bool:
        """Clean all data from PostgreSQL tables while preserving structure"""
        logger.info("🧹 Cleaning PostgreSQL data...")
        
        try:
            import psycopg2
            from urllib.parse import urlparse
            
            # Handle both dict and object config
            if hasattr(self.config, 'DATABASE_URL'):
                db_url = self.config.DATABASE_URL
            elif isinstance(self.config, dict):
                db_url = self.config.get('database_url')
            else:
                db_url = getattr(self.config, 'database_url', None)
            
            if not db_url:
                # Fallback to environment variables
                import os
                postgres_host = os.getenv("POSTGRES_HOST", "localhost")
                postgres_port = os.getenv("POSTGRES_PORT", "5432")
                postgres_db = os.getenv("POSTGRES_DB", "chatbot_db")
                postgres_user = os.getenv("POSTGRES_USER", "chatbot_user")
                postgres_password = os.getenv("POSTGRES_PASSWORD", "chatbot_password")
                db_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
            
            parsed = urlparse(db_url)
            
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                user=parsed.username,
                password=parsed.password,
                database=parsed.path[1:],  # Remove leading slash
                connect_timeout=self.connect_timeout
            )
            
            with conn.cursor() as cur:
                # Check if tables exist and clean them
                tables_to_clean = [
                    'vector_points',      # Clean vector points first (has foreign keys)
                    'chat_messages',      # Clean messages next (has foreign keys)
                    'chat_sessions',      # Clean sessions next (has foreign keys)
                    'documents',          # Clean documents
                    'users'              # Clean users last
                ]
                
                for table in tables_to_clean:
                    try:
                        # Check if table exists
                        cur.execute("""
                            SELECT EXISTS (
                                SELECT 1 FROM information_schema.tables 
                                WHERE table_name = %s
                            )
                        """, (table,))
                        
                        table_exists = cur.fetchone()[0]
                        
                        if table_exists:
                            # Clean the table data
                            cur.execute(f"TRUNCATE TABLE {table} CASCADE")
                            logger.info(f"  🗑️ Cleaned table: {table}")
                        else:
                            logger.info(f"  ⚠️ Table {table} does not exist (will be created)")
                        
                    except Exception as e:
                        logger.warning(f"  ⚠️ Failed to clean table {table}: {e}")
                        conn.rollback()
                        continue
                
                # Reset sequences if they exist
                sequences_to_reset = [
                    'users_id_seq',
                    'chat_sessions_id_seq', 
                    'chat_messages_id_seq',
                    'documents_id_seq',
                    'vector_points_id_seq'
                ]
                
                for sequence in sequences_to_reset:
                    try:
                        cur.execute(f"ALTER SEQUENCE IF EXISTS {sequence} RESTART WITH 1")
                    except Exception as e:
                        logger.debug(f"  ⚠️ Could not reset sequence {sequence}: {e}")
                
                conn.commit()
            
            conn.close()
            logger.info("✅ PostgreSQL data cleaned successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ PostgreSQL cleanup failed: {e}")
            return False

    def clean_qdrant_collection(self) -> bool:
        """Clean Qdrant collection by deleting and recreating it"""
        logger.info("🧹 Cleaning Qdrant collection...")
        
        try:
            # Handle both dict and object config
            if hasattr(self.config, 'QDRANT_URL'):
                qdrant_url = self.config.QDRANT_URL
                collection_name = self.config.QDRANT_COLLECTION_NAME
                vector_size = self.config.QDRANT_VECTOR_SIZE
                distance = self.config.QDRANT_DISTANCE
                on_disk = self.config.QDRANT_ON_DISK
            elif isinstance(self.config, dict):
                qdrant_url = self.config.get('qdrant_url', 'http://qdrant:6333')
                collection_name = self.config.get('qdrant_collection_name', 'chatbot_collection')
                vector_size = self.config.get('qdrant_vector_size', 1024)
                distance = self.config.get('qdrant_distance', 'COSINE')
                on_disk = self.config.get('qdrant_on_disk', True)
            else:
                qdrant_url = getattr(self.config, 'qdrant_url', 'http://qdrant:6333')
                collection_name = getattr(self.config, 'qdrant_collection_name', 'chatbot_collection')
                vector_size = getattr(self.config, 'qdrant_vector_size', 1024)
                distance = getattr(self.config, 'qdrant_distance', 'COSINE')
                on_disk = getattr(self.config, 'qdrant_on_disk', True)
            
            client = QdrantClient(url=qdrant_url)
            
            # Check if collection exists and delete it
            try:
                collection_info = client.get_collection(collection_name)
                points_count = collection_info.points_count
                logger.info(f"  🗑️ Deleting existing collection '{collection_name}' with {points_count} points")
                client.delete_collection(collection_name)
                logger.info("  ✅ Collection deleted successfully!")
                
            except UnexpectedResponse as e:
                if "Not found" in str(e):
                    logger.info(f"  ⚠️ Collection '{collection_name}' does not exist (will be created)")
                else:
                    raise
            
            # Create fresh collection
            logger.info(f"📦 Creating fresh Qdrant collection: {collection_name}")
            
            from qdrant_client.models import Distance, VectorParams
            
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE if distance == "COSINE" else Distance.DOT
                ),
                on_disk_payload=on_disk
            )
            
            logger.info(f"✅ Fresh Qdrant collection '{collection_name}' created successfully with dimension {vector_size}!")
            return True
                    
        except Exception as e:
            logger.error(f"❌ Qdrant cleanup failed: {e}")
            return False

    def initialize_qdrant_collection(self) -> bool:
        """Initialize Qdrant collection if it doesn't exist or recreate if needed"""
        logger.info("🔍 Checking/Creating Qdrant collection...")
        
        try:
            # Handle both dict and object config
            if hasattr(self.config, 'QDRANT_URL'):
                qdrant_url = self.config.QDRANT_URL
                collection_name = self.config.QDRANT_COLLECTION_NAME
                vector_size = self.config.QDRANT_VECTOR_SIZE
                distance = self.config.QDRANT_DISTANCE
                on_disk = self.config.QDRANT_ON_DISK
                force_recreate = self.config.QDRANT_FORCE_RECREATE
            elif isinstance(self.config, dict):
                qdrant_url = self.config.get('qdrant_url', 'http://qdrant:6333')
                collection_name = self.config.get('qdrant_collection_name', 'chatbot_collection')
                vector_size = self.config.get('qdrant_vector_size', 1024)
                distance = self.config.get('qdrant_distance', 'COSINE')
                on_disk = self.config.get('qdrant_on_disk', True)
                force_recreate = self.config.get('qdrant_force_recreate', False)
            else:
                qdrant_url = getattr(self.config, 'qdrant_url', 'http://qdrant:6333')
                collection_name = getattr(self.config, 'qdrant_collection_name', 'chatbot_collection')
                vector_size = getattr(self.config, 'qdrant_vector_size', 1024)
                distance = getattr(self.config, 'qdrant_distance', 'COSINE')
                on_disk = getattr(self.config, 'qdrant_on_disk', True)
                force_recreate = getattr(self.config, 'qdrant_force_recreate', False)
            
            client = QdrantClient(url=qdrant_url)
            
            # Check if collection exists
            collection_exists = False
            try:
                collection_info = client.get_collection(collection_name)
                collection_exists = True
                
                # Check if dimensions match
                current_size = collection_info.config.params.vectors.size
                expected_size = vector_size
                
                if current_size != expected_size:
                    logger.warning(f"⚠️ Collection '{collection_name}' exists with dimension {current_size}, but expected {expected_size}")
                    
                    if force_recreate:
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
                        size=vector_size,
                        distance=Distance.COSINE if distance == "COSINE" else Distance.DOT
                    ),
                    on_disk_payload=on_disk
                )
                
                logger.info(f"✅ Qdrant collection '{collection_name}' created successfully with dimension {vector_size}!")
                return True
                    
        except Exception as e:
            logger.error(f"❌ Qdrant collection initialization failed: {e}")
            return False
    
    async def run_fresh_initialization(self) -> Dict[str, Any]:
        """Run fresh initialization with complete data cleanup"""
        logger.info("🧨 Starting FRESH system initialization with data cleanup...")
        
        results = {
            'postgresql_connection': False,
            'qdrant_connection': False,
            'postgresql_cleanup': False,
            'qdrant_cleanup': False,
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
            
            # Step 3: Clean PostgreSQL data
            results['postgresql_cleanup'] = self.clean_postgresql_data()
            if not results['postgresql_cleanup']:
                logger.error("❌ PostgreSQL cleanup failed")
                return results
            
            # Step 4: Clean Qdrant collection
            results['qdrant_cleanup'] = self.clean_qdrant_collection()
            if not results['qdrant_cleanup']:
                logger.error("❌ Qdrant cleanup failed")
                return results
            
            # Step 5: Run database migration (recreate schema)
            results['database_migration'] = self.run_database_migration()
            if not results['database_migration']:
                logger.error("❌ Database migration failed")
                return results
            
            # Step 6: Initialize fresh Qdrant collection (already done in cleanup)
            results['qdrant_collection'] = True
            
            # All steps completed successfully
            results['overall_success'] = True
            logger.info("🎉 Fresh system initialization successful! All old data cleared.")
            
        except Exception as e:
            logger.error(f"❌ Fresh initialization process failed: {e}")
            
        return results

    async def run_full_initialization(self) -> Dict[str, Any]:
        """Run complete initialization process (preserves existing data)"""
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
    
    # Check if fresh start is requested via environment variable
    fresh_start = os.getenv('FRESH_START', 'false').lower() in ('true', '1', 'yes', 'on')
    
    if fresh_start:
        logger.info("🧨 FRESH_START environment variable detected - performing clean initialization")
        return await initializer.run_fresh_initialization()
    else:
        logger.info("🚀 Performing normal initialization (preserving existing data)")
        return await initializer.run_full_initialization()


async def initialize_system_fresh() -> Dict[str, Any]:
    """Force fresh initialization with data cleanup"""
    initializer = DatabaseInitializer()
    return await initializer.run_fresh_initialization()


def check_connections() -> Dict[str, bool]:
    """Quick connection check"""
    initializer = DatabaseInitializer()
    return {
        'postgresql': initializer.check_postgresql_connection(),
        'qdrant': initializer.check_qdrant_connection()
    } 