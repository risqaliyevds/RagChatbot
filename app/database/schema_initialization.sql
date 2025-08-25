-- Database initialization script for chatbot application
-- This script creates the necessary tables for storing chat sessions and messages

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create chat sessions table
CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chat_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- Create chat messages table
CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chat_id VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    FOREIGN KEY (chat_id) REFERENCES chat_sessions(chat_id) ON DELETE CASCADE
);

-- Create documents table for tracking uploaded documents
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(500) NOT NULL UNIQUE,
    original_filename VARCHAR(500) NOT NULL,
    file_size_bytes BIGINT DEFAULT 0,
    file_type VARCHAR(100),
    upload_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    chunks_count INTEGER DEFAULT 0,
    processing_status VARCHAR(50) DEFAULT 'processing',
    error_message TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create vector points table for tracking Qdrant point IDs  
CREATE TABLE IF NOT EXISTS vector_points (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL,
    filename VARCHAR(500) NOT NULL,
    qdrant_point_id BIGINT NOT NULL,
    chunk_index INTEGER DEFAULT 0,
    chunk_content_preview TEXT,
    collection_name VARCHAR(255) DEFAULT 'rag_documents',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_user_id ON users(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_chat_id ON chat_sessions(chat_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_last_activity ON chat_sessions(last_activity);
CREATE INDEX IF NOT EXISTS idx_chat_messages_chat_id ON chat_messages(chat_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_timestamp ON chat_messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_chat_messages_role ON chat_messages(role);
-- Document indexes
CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
CREATE INDEX IF NOT EXISTS idx_documents_is_active ON documents(is_active);
CREATE INDEX IF NOT EXISTS idx_documents_upload_timestamp ON documents(upload_timestamp);
CREATE INDEX IF NOT EXISTS idx_documents_processing_status ON documents(processing_status);

-- Vector points indexes
CREATE INDEX IF NOT EXISTS idx_vector_points_document_id ON vector_points(document_id);
CREATE INDEX IF NOT EXISTS idx_vector_points_filename ON vector_points(filename);
CREATE INDEX IF NOT EXISTS idx_vector_points_qdrant_id ON vector_points(qdrant_point_id);
CREATE INDEX IF NOT EXISTS idx_vector_points_collection ON vector_points(collection_name);
CREATE INDEX IF NOT EXISTS idx_vector_points_created_at ON vector_points(created_at);

-- Create function to update last_activity automatically
CREATE OR REPLACE FUNCTION update_last_activity()
RETURNS TRIGGER AS $$
BEGIN
    -- Update last_activity in chat_sessions when a new message is added
    UPDATE chat_sessions 
    SET last_activity = CURRENT_TIMESTAMP 
    WHERE chat_id = NEW.chat_id;
    
    -- Update last_activity in users table
    UPDATE users 
    SET last_activity = CURRENT_TIMESTAMP 
    WHERE user_id = (SELECT user_id FROM chat_sessions WHERE chat_id = NEW.chat_id);
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically update last_activity
CREATE TRIGGER trigger_update_last_activity
    AFTER INSERT ON chat_messages
    FOR EACH ROW
    EXECUTE FUNCTION update_last_activity();

-- Create function to clean up expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions(hours_threshold INTEGER DEFAULT 24)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Delete expired chat sessions and their messages (CASCADE will handle messages)
    DELETE FROM chat_sessions 
    WHERE last_activity < CURRENT_TIMESTAMP - INTERVAL '1 hour' * hours_threshold;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Clean up users with no active sessions
    DELETE FROM users 
    WHERE user_id NOT IN (SELECT DISTINCT user_id FROM chat_sessions);
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Insert default user if not exists
INSERT INTO users (user_id) 
VALUES ('test_user_001') 
ON CONFLICT (user_id) DO NOTHING;

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO chatbot_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO chatbot_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO chatbot_user; 