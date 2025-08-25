"""
PostgreSQL Database Manager for Chatbot Application
==================================================

This module handles all database operations for chat sessions and messages,
replacing the JSON file-based storage with PostgreSQL.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import uuid

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
import json

logger = logging.getLogger(__name__)

class DatabaseManager:
    """PostgreSQL database manager for chat operations"""
    
    def __init__(self, database_url: str = None, min_connections: int = None, max_connections: int = None, config: Dict[str, Any] = None):
        """Initialize database manager with connection pool"""
        # Build database URL from environment variables if not provided
        if not database_url:
            postgres_host = os.getenv("POSTGRES_HOST", "localhost")
            postgres_port = os.getenv("POSTGRES_PORT", "5432")
            postgres_db = os.getenv("POSTGRES_DB", "chatbot_db")
            postgres_user = os.getenv("POSTGRES_USER", "chatbot_user")
            postgres_password = os.getenv("POSTGRES_PASSWORD", "chatbot_password")
            database_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
        
        self.database_url = database_url
        self.config = config or {}
        
        # Use new config fields for pool settings if available
        if min_connections is None:
            min_connections = int(os.getenv("POSTGRES_POOL_SIZE", "10")) // 2  # Half for min
        if max_connections is None:
            max_connections = int(os.getenv("POSTGRES_POOL_SIZE", "10")) + int(os.getenv("POSTGRES_MAX_OVERFLOW", "20"))
        
        # Connection pool configuration
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.pool_timeout = int(os.getenv("POSTGRES_POOL_TIMEOUT", "30"))
        self.pool_recycle = int(os.getenv("POSTGRES_POOL_RECYCLE", "3600"))
        
        logger.info(f"Initializing PostgreSQL connection pool: min={min_connections}, max={max_connections}")
        
        try:
            # Create connection pool
            self.pool = SimpleConnectionPool(
                minconn=min_connections,
                maxconn=max_connections,
                dsn=database_url
            )
            logger.info("✅ PostgreSQL connection pool created successfully")
            
            # Test connection
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    logger.info("Database connection test successful")
                    
        except Exception as e:
            logger.error(f"❌ Failed to create PostgreSQL connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get database connection from pool"""
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def ensure_user_exists(self, user_id: str) -> bool:
        """Ensure user exists in database"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO users (user_id) VALUES (%s) ON CONFLICT (user_id) DO NOTHING",
                        (user_id,)
                    )
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Failed to ensure user exists: {e}")
            return False
    
    def create_chat_session(self, user_id: str, chat_id: str = None) -> Optional[str]:
        """Create a new chat session"""
        if not chat_id:
            chat_id = f"chat_{uuid.uuid4().hex[:8]}"
        
        try:
            # Ensure user exists
            self.ensure_user_exists(user_id)
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO chat_sessions (chat_id, user_id) 
                        VALUES (%s, %s) 
                        ON CONFLICT (chat_id) DO NOTHING
                        RETURNING chat_id
                        """,
                        (chat_id, user_id)
                    )
                    result = cur.fetchone()
                    conn.commit()
                    
                    if result:
                        logger.info(f"Created new chat session: {chat_id} for user: {user_id}")
                        return chat_id
                    else:
                        # Chat ID already exists, return it
                        return chat_id
                        
        except Exception as e:
            logger.error(f"Failed to create chat session: {e}")
            return None
    
    def add_message(self, chat_id: str, role: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Add a message to a chat session"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO chat_messages (chat_id, role, content, metadata) 
                        VALUES (%s, %s, %s, %s)
                        """,
                        (chat_id, role, content, json.dumps(metadata or {}))
                    )
                    conn.commit()
                    logger.debug(f"Added message to chat {chat_id}: {role}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            return False
    
    def get_chat_messages(self, chat_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """Get messages for a chat session"""
        try:
            # Use configurable limit or fall back to parameter or default
            if limit is None:
                limit = self.config.get("max_chat_messages_limit", 100)
            
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT role, content, timestamp, metadata
                        FROM chat_messages 
                        WHERE chat_id = %s 
                        ORDER BY timestamp ASC 
                        LIMIT %s
                        """,
                        (chat_id, limit)
                    )
                    messages = []
                    for row in cur.fetchall():
                        message = {
                            "role": row["role"],
                            "content": row["content"],
                            "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None
                        }
                        if row["metadata"]:
                            try:
                                metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
                                message.update(metadata)
                            except:
                                pass
                        messages.append(message)
                    
                    return messages
                    
        except Exception as e:
            logger.error(f"Failed to get chat messages: {e}")
            return []
    
    def get_chat_session(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Get chat session details"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT chat_id, user_id, created_at, last_activity, metadata
                        FROM chat_sessions 
                        WHERE chat_id = %s
                        """,
                        (chat_id,)
                    )
                    row = cur.fetchone()
                    if row:
                        return {
                            "chat_id": row["chat_id"],
                            "user_id": row["user_id"],
                            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                            "last_activity": row["last_activity"].isoformat() if row["last_activity"] else None,
                            "messages": self.get_chat_messages(chat_id)
                        }
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to get chat session: {e}")
            return None
    
    def get_user_chats(self, user_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """Get all chat sessions for a user"""
        try:
            # Use configurable limit or fall back to parameter or default
            if limit is None:
                limit = self.config.get("max_user_chats_limit", 50)
            
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT cs.chat_id, cs.user_id, cs.created_at, cs.last_activity,
                               COUNT(cm.id) as message_count
                        FROM chat_sessions cs
                        LEFT JOIN chat_messages cm ON cs.chat_id = cm.chat_id
                        WHERE cs.user_id = %s
                        GROUP BY cs.chat_id, cs.user_id, cs.created_at, cs.last_activity
                        ORDER BY cs.last_activity DESC
                        LIMIT %s
                        """,
                        (user_id, limit)
                    )
                    chats = []
                    for row in cur.fetchall():
                        chats.append({
                            "chat_id": row["chat_id"],
                            "user_id": row["user_id"],
                            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                            "last_activity": row["last_activity"].isoformat() if row["last_activity"] else None,
                            "message_count": row["message_count"] or 0
                        })
                    
                    return chats
                    
        except Exception as e:
            logger.error(f"Failed to get user chats: {e}")
            return []
    
    def cleanup_expired_sessions(self, hours_threshold: int = None) -> int:
        """Clean up expired chat sessions"""
        try:
            # Use configurable threshold or fall back to parameter or default
            if hours_threshold is None:
                hours_threshold = self.config.get("session_cleanup_threshold_hours", 24)
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT cleanup_expired_sessions(%s)",
                        (hours_threshold,)
                    )
                    result = cur.fetchone()
                    conn.commit()
                    deleted_count = result[0] if result else 0
                    logger.info(f"Cleaned up {deleted_count} expired sessions")
                    return deleted_count
                    
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
    
    def get_or_create_session(self, user_id: str, chat_id: str = None) -> Optional[Dict[str, Any]]:
        """Get existing session or create new one with time-based chat ID management"""
        # If no chat_id provided, check for active session within 1 hour
        if not chat_id:
            active_session = self.get_active_user_session(user_id)
            if active_session:
                return active_session
        else:
            # Try to get existing session
            session = self.get_chat_session(chat_id)
            if session and session["user_id"] == user_id:
                # Check if session is still active (within 1 hour)
                if self.is_session_active(chat_id):
                    return session
                else:
                    # Session expired, create new one
                    logger.info(f"Session {chat_id} expired for user {user_id}, creating new session")
        
        # Create new session
        new_chat_id = self.create_chat_session(user_id, chat_id)
        if new_chat_id:
            return self.get_chat_session(new_chat_id)
        
        return None

    def get_active_user_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user's most recent active session (within 1 hour)"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT chat_id, user_id, created_at, last_activity, metadata
                        FROM chat_sessions 
                        WHERE user_id = %s 
                        AND last_activity > CURRENT_TIMESTAMP - INTERVAL '1 hour'
                        ORDER BY last_activity DESC
                        LIMIT 1
                        """,
                        (user_id,)
                    )
                    row = cur.fetchone()
                    if row:
                        return {
                            "chat_id": row["chat_id"],
                            "user_id": row["user_id"],
                            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                            "last_activity": row["last_activity"].isoformat() if row["last_activity"] else None,
                            "messages": self.get_chat_messages(row["chat_id"])
                        }
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to get active user session: {e}")
            return None

    def is_session_active(self, chat_id: str) -> bool:
        """Check if a chat session is still active (within 1 hour)"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT COUNT(*) FROM chat_sessions 
                        WHERE chat_id = %s 
                        AND last_activity > CURRENT_TIMESTAMP - INTERVAL '1 hour'
                        """,
                        (chat_id,)
                    )
                    result = cur.fetchone()
                    return result[0] > 0 if result else False
                    
        except Exception as e:
            logger.error(f"Failed to check session activity: {e}")
            return False

    def create_new_chat_for_user(self, user_id: str) -> Optional[str]:
        """Create a new chat session for user (for New Chat button functionality)"""
        # This is now just a wrapper around create_chat_session
        return self.create_chat_session(user_id)

    def get_user_last_activity(self, user_id: str) -> Optional[datetime]:
        """Get user's last activity timestamp"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT last_activity FROM users 
                        WHERE user_id = %s
                        """,
                        (user_id,)
                    )
                    result = cur.fetchone()
                    return result[0] if result else None
                    
        except Exception as e:
            logger.error(f"Failed to get user last activity: {e}")
            return None
    
    def migrate_from_json(self, json_file_path: str) -> bool:
        """Migrate data from JSON file to PostgreSQL"""
        try:
            if not os.path.exists(json_file_path):
                logger.warning(f"JSON file not found: {json_file_path}")
                return True
            
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            sessions = data.get('sessions', [])
            migrated_count = 0
            
            for session in sessions:
                chat_id = session.get('chat_id')
                user_id = session.get('user_id')
                messages = session.get('messages', [])
                
                if not chat_id or not user_id:
                    continue
                
                # Create session
                if self.create_chat_session(user_id, chat_id):
                    # Add messages
                    for message in messages:
                        role = message.get('role')
                        content = message.get('content')
                        timestamp = message.get('timestamp')
                        
                        if role and content:
                            # Convert timestamp if needed
                            metadata = {}
                            if timestamp:
                                metadata['original_timestamp'] = timestamp
                            
                            self.add_message(chat_id, role, content, metadata)
                    
                    migrated_count += 1
            
            logger.info(f"Successfully migrated {migrated_count} sessions from JSON")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate from JSON: {e}")
            return False
    
    def create_document_record(self, filename: str, original_filename: str, file_size: int = 0, 
                              file_type: str = None, metadata: Dict[str, Any] = None) -> Optional[str]:
        """Create a document record and return document ID"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    logger.info(f"Creating document record: filename={filename}, original={original_filename}, size={file_size}, type={file_type}")
                    cur.execute(
                        """
                        INSERT INTO documents 
                        (filename, original_filename, file_size_bytes, file_type, metadata)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (filename, original_filename, file_size, file_type, json.dumps(metadata or {}))
                    )
                    document_id = cur.fetchone()[0]
                    
                    # Verify the record was created with correct values
                    cur.execute(
                        """
                        SELECT filename, original_filename, file_size_bytes, file_type, chunks_count 
                        FROM documents WHERE id = %s
                        """,
                        (document_id,)
                    )
                    result = cur.fetchone()
                    if result:
                        stored_filename, stored_original, stored_size, stored_type, stored_chunks = result
                        logger.info(f"✅ Document record created and verified: ID={document_id}")
                        logger.info(f"   Stored values: filename={stored_filename}, size={stored_size}, chunks={stored_chunks}")
                    
                    conn.commit()
                    return str(document_id)
                    
        except Exception as e:
            logger.error(f"❌ Failed to create document record: {e}")
            logger.error(f"   Input values: filename={filename}, size={file_size}, type={file_type}")
            return None

    def store_vector_points(self, document_id: str, filename: str, point_ids: List[int], 
                           collection_name: str = "rag_documents", chunks_data: List[Dict] = None) -> bool:
        """Store vector point IDs for a document"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Insert vector points
                    for i, point_id in enumerate(point_ids):
                        chunk_preview = ""
                        if chunks_data and i < len(chunks_data):
                            # Debug logging with configurable preview length
                            if logger.isEnabledFor(logging.DEBUG):
                                preview_length = self.config.get("chunk_preview_length", 200)
                                chunk_preview = chunks_data[i].get("content", "")[:preview_length]  # Configurable preview length
                                logger.debug(f"Storing chunk {i+1}/{len(chunks_data)}: filename={filename}, preview='{chunk_preview}...'")
                        
                        cur.execute(
                            """
                            INSERT INTO vector_points 
                            (document_id, filename, qdrant_point_id, chunk_index, chunk_content_preview, collection_name)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            (document_id, filename, point_id, i, chunk_preview, collection_name)
                        )
                    
                    # Update document with chunks count and status
                    logger.info(f"Updating document {document_id} with chunks_count = {len(point_ids)}")
                    cur.execute(
                        """
                        UPDATE documents 
                        SET chunks_count = %s, processing_status = 'completed'
                        WHERE id = %s
                        """,
                        (len(point_ids), document_id)
                    )
                    
                    # Verify the update worked
                    cur.execute(
                        """
                        SELECT chunks_count, processing_status FROM documents WHERE id = %s
                        """,
                        (document_id,)
                    )
                    result = cur.fetchone()
                    if result:
                        updated_chunks_count, updated_status = result
                        logger.info(f"✅ Document {document_id} updated: chunks_count={updated_chunks_count}, status={updated_status}")
                    else:
                        logger.warning(f"⚠️ Could not verify document update for ID: {document_id}")
                    
                    conn.commit()
                    logger.info(f"✅ Stored {len(point_ids)} vector point IDs for document: {filename}")
                    return True
                    
        except Exception as e:
            logger.error(f"❌ Failed to store vector points: {e}")
            logger.error(f"❌ Error details: document_id={document_id}, filename={filename}, point_ids_count={len(point_ids) if point_ids else 0}")
            return False
    
    def get_vector_points_for_file(self, filename: str, collection_name: str = "rag_documents") -> List[int]:
        """Get all vector point IDs for a specific file"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT qdrant_point_id FROM vector_points 
                        WHERE filename = %s AND collection_name = %s
                        ORDER BY chunk_index
                        """,
                        (filename, collection_name)
                    )
                    results = cur.fetchall()
                    point_ids = [row[0] for row in results]
                    logger.info(f"🔍 Found {len(point_ids)} vector points for file: {filename}")
                    return point_ids
                    
        except Exception as e:
            logger.error(f"❌ Failed to get vector points for file: {e}")
            return []
    
    def deactivate_document(self, filename: str) -> bool:
        """Deactivate a document (set is_active = False) and get vector points for deletion"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Update document status
                    cur.execute(
                        """
                        UPDATE documents 
                        SET is_active = FALSE, processing_status = 'deleted'
                        WHERE filename = %s AND is_active = TRUE
                        """,
                        (filename,)
                    )
                    
                    if cur.rowcount > 0:
                        conn.commit()
                        logger.info(f"✅ Deactivated document: {filename}")
                        return True
                    else:
                        logger.warning(f"⚠️ Document not found or already inactive: {filename}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Failed to deactivate document: {e}")
            return False

    def get_vector_points_for_document(self, filename: str, collection_name: str = "rag_documents") -> List[int]:
        """Get all vector point IDs for a document (only active documents)"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT vp.qdrant_point_id 
                        FROM vector_points vp
                        JOIN documents d ON vp.document_id = d.id
                        WHERE vp.filename = %s AND vp.collection_name = %s AND d.is_active = TRUE
                        ORDER BY vp.chunk_index
                        """,
                        (filename, collection_name)
                    )
                    results = cur.fetchall()
                    point_ids = [row[0] for row in results]
                    logger.info(f"🔍 Found {len(point_ids)} vector points for active document: {filename}")
                    return point_ids
                    
        except Exception as e:
            logger.error(f"❌ Failed to get vector points for document: {e}")
            return []

    def delete_vector_points_for_file(self, filename: str, collection_name: str = "rag_documents") -> int:
        """Delete vector point tracking records for a file (legacy method)"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        DELETE FROM vector_points 
                        WHERE filename = %s AND collection_name = %s
                        """,
                        (filename, collection_name)
                    )
                    deleted_count = cur.rowcount
                    conn.commit()
                    logger.info(f"🗑️ Deleted {deleted_count} vector point records for file: {filename}")
                    return deleted_count
                    
        except Exception as e:
            logger.error(f"❌ Failed to delete vector points for file: {e}")
            return 0
    
    def get_documents_list(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get list of documents with metadata"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    query = """
                        SELECT 
                            id,
                            filename,
                            original_filename,
                            file_size_bytes,
                            file_type,
                            upload_timestamp,
                            is_active,
                            chunks_count,
                            processing_status,
                            error_message,
                            metadata
                        FROM documents
                    """
                    
                    if active_only:
                        query += " WHERE is_active = TRUE"
                    
                    query += " ORDER BY upload_timestamp DESC"
                    
                    cur.execute(query)
                    results = cur.fetchall()
                    return [dict(row) for row in results]
                    
        except Exception as e:
            logger.error(f"❌ Failed to get documents list: {e}")
            return []

    def get_document_stats(self) -> Dict[str, Any]:
        """Get overall document statistics"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT 
                            COUNT(*) as total_documents,
                            COUNT(*) FILTER (WHERE is_active = TRUE) as active_documents,
                            COUNT(*) FILTER (WHERE is_active = FALSE) as inactive_documents,
                            SUM(file_size_bytes) as total_size,
                            SUM(chunks_count) as total_chunks,
                            MIN(upload_timestamp) as first_upload,
                            MAX(upload_timestamp) as last_upload
                        FROM documents
                        """
                    )
                    
                    result = cur.fetchone()
                    return dict(result) if result else {}
                    
        except Exception as e:
            logger.error(f"❌ Failed to get document stats: {e}")
            return {}

    def get_file_vector_stats(self, filename: str = None) -> Dict[str, Any]:
        """Get statistics about vector points (legacy method)"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if filename:
                        # Stats for specific file
                        cur.execute(
                            """
                            SELECT 
                                COUNT(*) as total_points,
                                MIN(vp.created_at) as first_created,
                                MAX(vp.created_at) as last_created
                            FROM vector_points vp
                            JOIN documents d ON vp.document_id = d.id
                            WHERE vp.filename = %s AND d.is_active = TRUE
                            """,
                            (filename,)
                        )
                    else:
                        # Overall stats
                        cur.execute(
                            """
                            SELECT 
                                COUNT(*) as total_points,
                                COUNT(DISTINCT vp.filename) as total_files,
                                MIN(vp.created_at) as first_created,
                                MAX(vp.created_at) as last_created
                            FROM vector_points vp
                            JOIN documents d ON vp.document_id = d.id
                            WHERE d.is_active = TRUE
                            """
                        )
                    
                    result = cur.fetchone()
                    return dict(result) if result else {}
                    
        except Exception as e:
            logger.error(f"❌ Failed to get vector stats: {e}")
            return {}
    
    def close(self):
        """Close database connection pool"""
        if hasattr(self, 'pool') and self.pool:
            self.pool.closeall()
            logger.info("Database connection pool closed")


# Global database manager instance
db_manager = None

def get_db_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager

def init_database() -> DatabaseManager:
    """Initialize database manager"""
    global db_manager
    db_manager = DatabaseManager()
    return db_manager 