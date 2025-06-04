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
    
    def __init__(self, database_url: str = None, min_connections: int = 1, max_connections: int = 10):
        """Initialize database manager with connection pool"""
        self.database_url = database_url or os.getenv("DATABASE_URL", 
            "postgresql://chatbot_user:chatbot_password@localhost:5432/chatbot_db")
        
        try:
            # Create connection pool
            self.pool = SimpleConnectionPool(
                min_connections, max_connections, self.database_url
            )
            logger.info("Database connection pool created successfully")
            
            # Test connection
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    logger.info("Database connection test successful")
                    
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
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
    
    def get_chat_messages(self, chat_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get messages for a chat session"""
        try:
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
    
    def get_user_chats(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all chat sessions for a user"""
        try:
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
    
    def cleanup_expired_sessions(self, hours_threshold: int = 24) -> int:
        """Clean up expired chat sessions"""
        try:
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
        try:
            # Ensure user exists
            self.ensure_user_exists(user_id)
            
            # Generate new chat ID
            new_chat_id = f"chat_{uuid.uuid4().hex[:8]}"
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO chat_sessions (chat_id, user_id) 
                        VALUES (%s, %s) 
                        RETURNING chat_id
                        """,
                        (new_chat_id, user_id)
                    )
                    result = cur.fetchone()
                    conn.commit()
                    
                    if result:
                        logger.info(f"Created new chat session: {new_chat_id} for user: {user_id}")
                        return result[0]
                    return None
                        
        except Exception as e:
            logger.error(f"Failed to create new chat for user: {e}")
            return None

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