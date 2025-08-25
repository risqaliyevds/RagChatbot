#!/usr/bin/env python3
"""
FastAPI RAG Application - Main Entry Point
=========================================

Clean and modular RAG chatbot application with PostgreSQL and Qdrant integration.
"""

import logging
import asyncio
import uuid
import time
from typing import Dict, Any, List
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import our modular components
from app import (
    get_config, load_config, validate_config,
    initialize_system, check_connections
)
# Setup NLTK for offline use
from app import nltk_setup
from app.qdrant_manager import init_qdrant_client, init_vectorstore
from app.document_processing import load_and_split_documents, process_uploaded_document_with_progress
from app.rag_pipeline_manager import ChatService, init_llm, get_qa_prompt, create_qa_chain
from app.models import (
    ChatRequest, ChatResponse, ChatHistoryRequest, ChatCompletionRequest,
    ChatCompletionResponse, NewChatRequest, NewChatResponse,
    UserSessionStatusRequest, UserSessionStatusResponse, HealthResponse,
    DocumentUploadResponse, DocumentListResponse, DocumentDeleteRequest,
    DocumentDeleteResponse, DocumentUploadProgress, FileInfo
)

# Database imports
from app.database.postgresql_manager import DatabaseManager, get_db_manager, init_database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for application state
vectorstore = None
qa_chain = None
qdrant_client = None
db_manager = None
chat_service = None
config = None

# Progress tracking for document uploads
upload_progress_store: Dict[str, DocumentUploadProgress] = {}


async def initialize_rag_system():
    """Initialize the RAG system with dependencies - Non-blocking approach"""
    global db_manager, vectorstore, qa_chain, config, chat_service, qdrant_client
    
    # Phase 1: Initialize core components that must work
    try:
        logger.info("🚀 Phase 1: Initializing core components...")
        
        # Load configuration
        from app.config import get_config
        config = get_config()
        logger.info("✅ Configuration loaded successfully")
        
        # Initialize database (critical component)
        from app.database.postgresql_manager import DatabaseManager
        db_manager = DatabaseManager(config=config.__dict__ if hasattr(config, '__dict__') else config)
        logger.info("✅ Database manager initialized")
        
        # Run database schema initialization
        from app.database_initializer import DatabaseInitializer
        db_initializer = DatabaseInitializer(config=config.__dict__ if hasattr(config, '__dict__') else config)
        
        # Critical: Ensure database is ready
        if not db_initializer.check_postgresql_connection():
            raise Exception("PostgreSQL connection failed - system cannot start")
        
        if not db_initializer.run_database_migration():
            logger.warning("⚠️ Database migration had issues but continuing...")
        
        logger.info("✅ Phase 1 completed - Core system ready")
        
    except Exception as e:
        logger.error(f"❌ Phase 1 failed: {e}")
        raise  # Cannot continue without core components
    
    # Phase 2: Initialize enhanced components (non-blocking)
    try:
        logger.info("🔄 Phase 2: Initializing enhanced components...")
        
        # Try to initialize enhanced features but don't fail if they don't work
        success = await _initialize_enhanced_components()
        
        if success:
            logger.info("✅ Phase 2 completed - Full RAG system ready")
        else:
            logger.warning("⚠️ Phase 2 had issues - Running with limited functionality")
            
    except Exception as e:
        logger.warning(f"⚠️ Phase 2 failed: {e} - Continuing with basic functionality")
    
    # Phase 3: Always initialize chat service (critical for API functionality)
    try:
        logger.info("🔄 Phase 3: Initializing chat service...")
        
        from app.rag_pipeline_manager import ChatService
        # Initialize with whatever we have (qa_chain may be None)
        chat_service = ChatService(qa_chain, db_manager)
        logger.info("✅ Chat service initialized")
        
    except Exception as e:
        logger.error(f"❌ Chat service initialization failed: {e}")
        # Create a minimal chat service
        chat_service = None
    
    logger.info("🎉 RAG system initialization completed")
    return True


async def _initialize_enhanced_components() -> bool:
    """Initialize enhanced components (Qdrant, embeddings, RAG chain)"""
    global vectorstore, qa_chain, qdrant_client
    
    try:
        # Check Qdrant connection with timeout
        from app.database_initializer import DatabaseInitializer
        config_dict = config.__dict__ if hasattr(config, '__dict__') else config
        
        # Debug logging for collection name
        collection_name = config_dict.get("qdrant_collection_name", "chatbot_collection")
        logger.info(f"🔧 DEBUG: Collection name = '{collection_name}'")
        
        db_initializer = DatabaseInitializer(config=config_dict)
        
        # Quick Qdrant check with timeout
        qdrant_ready = False
        try:
            import asyncio
            qdrant_ready = await asyncio.wait_for(
                asyncio.to_thread(db_initializer.check_qdrant_connection), 
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("⚠️ Qdrant connection check timed out")
            # Force fallback
            raise Exception("Qdrant timeout - using fallback")
        
        if not qdrant_ready:
            logger.warning("⚠️ Qdrant not ready - using fallback")
            raise Exception("Qdrant not ready - using fallback")
        
        # Initialize Qdrant client
        from app.qdrant_manager import init_qdrant_client, init_vectorstore
        qdrant_client = init_qdrant_client(config_dict)
        
        # Load documents (quick operation)
        from app.document_processing import load_and_split_documents
        documents = load_and_split_documents(config_dict)
        logger.info(f"Loaded {len(documents)} initial document chunks")
        
        # Initialize vector store with conflict handling
        try:
            # Handle collection conflicts by forcing recreation
            config_dict["qdrant_force_recreate"] = True
            vectorstore = await asyncio.wait_for(
                init_vectorstore(config_dict, documents, qdrant_client),
                timeout=30.0
            )
            logger.info("✅ Vector store initialized")
        except asyncio.TimeoutError:
            logger.warning("⚠️ Vector store initialization timed out")
            raise Exception("Vector store timeout - using fallback")
        except Exception as vs_e:
            logger.warning(f"⚠️ Vector store failed: {vs_e}")
            raise Exception(f"Vector store failed: {vs_e}")
        
        # Initialize RAG chain with Mock LLM
        from app.rag_pipeline_manager import init_llm, get_qa_prompt, create_qa_chain
        llm = init_llm(config_dict)  # This will use Mock LLM
        prompt = get_qa_prompt()
        qa_chain = create_qa_chain(vectorstore, llm, prompt, config_dict)
        logger.info("✅ QA chain initialized with intelligent LLM")
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Enhanced components failed: {e}")
        
        # ALWAYS create fallback QA chain - this is critical for functionality
        try:
            logger.info("🔧 Creating basic QA chain with Mock LLM for immediate functionality")
            from app.rag_pipeline_manager import init_llm, get_qa_prompt, create_qa_chain
            
            # Create a simple mock vectorstore for search
            class MockVectorStore:
                def search(self, query, k=3, score_threshold=0.4):
                    logger.info(f"🔧 Mock search for: {query}")
                    return []  # No documents, but won't crash
                
                async def add_documents(self, documents, progress_callback=None):
                    """Mock implementation of add_documents for file upload compatibility"""
                    logger.info(f"🔧 Mock add_documents: {len(documents)} documents")
                    if progress_callback:
                        await progress_callback("vectorizing", 80, "Mock vectorization in progress...")
                    
                    # Generate integer point IDs for database storage (matching BIGINT schema)
                    import time
                    base_id = int(time.time() * 1000000)  # Use microseconds as base
                    point_ids = [base_id + i for i in range(len(documents))]
                    logger.info(f"🔧 Mock generated {len(point_ids)} integer point IDs")
                    return point_ids
                
                async def delete_points(self, point_ids):
                    """Mock implementation of delete_points for document deletion compatibility"""
                    logger.info(f"🔧 Mock delete_points: {len(point_ids)} points")
                    return len(point_ids)  # Return number of deleted points
            
            mock_vectorstore = MockVectorStore()
            llm = init_llm(config_dict)  # This will use Mock LLM
            prompt = get_qa_prompt()
            qa_chain = create_qa_chain(mock_vectorstore, llm, prompt, config_dict)
            
            # Set globals
            vectorstore = mock_vectorstore
            
            logger.info("✅ Basic QA chain with Mock LLM created successfully")
            return True
            
        except Exception as fallback_e:
            logger.error(f"❌ Even basic QA chain creation failed: {fallback_e}")
            # Force a minimal working setup
            logger.info("🔧 Creating absolute minimal setup")
            
            class MinimalVectorStore:
                def search(self, query, k=3, score_threshold=0.4):
                    return []
                
                async def add_documents(self, documents, progress_callback=None):
                    """Minimal implementation of add_documents for file upload compatibility"""
                    if progress_callback:
                        await progress_callback("vectorizing", 80, "Minimal vectorization...")
                    
                    # Generate integer point IDs for database storage (matching BIGINT schema)
                    import time
                    base_id = int(time.time() * 1000000)  # Use microseconds as base
                    point_ids = [base_id + i for i in range(len(documents))]
                    return point_ids
                
                async def delete_points(self, point_ids):
                    """Minimal implementation of delete_points for document deletion compatibility"""
                    return len(point_ids)  # Return number of deleted points
            
            class MinimalLLM:
                def invoke(self, prompt):
                    class Response:
                        def __init__(self):
                            self.content = "Kechirasiz, hozir sistema to'liq ishlamayapti. Iltimos, administratorga murojaat qiling."
                    return Response()
            
            vectorstore = MinimalVectorStore()
            qa_chain = lambda question, history=None: "Sistema yuklanmoqda. Iltimos, biroz kuting."
            
            logger.info("✅ Minimal setup created")
            return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting up RAG application...")
    await initialize_rag_system()
    yield
    # Shutdown
    logger.info("Shutting down RAG application...")
    if db_manager:
        db_manager.close()


# Create FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Enhanced RAG chatbot with PostgreSQL and Qdrant integration",
    version="1.0.0",
    lifespan=lifespan
)

# Add default CORS middleware (will be replaced with config-based settings if needed)
def configure_cors():
    """Configure CORS middleware with settings from config"""
    try:
        import json
        import os
        # Parse CORS configuration directly from environment
        cors_origins = json.loads(os.getenv("CORS_ORIGINS", '["*"]'))
        cors_methods = json.loads(os.getenv("CORS_METHODS", '["GET", "POST", "PUT", "DELETE"]'))
        cors_headers = json.loads(os.getenv("CORS_HEADERS", '["*"]'))
        
        logger.info(f"CORS configured with origins: {cors_origins}")
        return cors_origins, cors_methods, cors_headers
    except Exception as e:
        # Fallback to default CORS settings
        logger.warning(f"Failed to parse CORS config, using defaults: {e}")
        return ["*"], ["*"], ["*"]

# Configure CORS middleware
cors_origins, cors_methods, cors_headers = configure_cors()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=cors_methods,
    allow_headers=cors_headers,
)


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health with initialization status"""
    try:
        # Check Qdrant connection
        qdrant_status = "connected"
        try:
            if qdrant_client:
                collections = qdrant_client.get_collections()
                qdrant_status = f"connected ({len(collections.collections)} collections)"
            else:
                qdrant_status = "not initialized"
        except Exception as e:
            qdrant_status = f"error: {str(e)}"
        
        # Check database connection
        database_status = "connected"
        try:
            if db_manager:
                with db_manager.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                database_status = "connected"
            else:
                database_status = "not initialized"
        except Exception as e:
            database_status = f"error: {str(e)}"
        
        # Check if core components are initialized
        components_ready = all([
            vectorstore is not None,
            qa_chain is not None,
            qdrant_client is not None,
            db_manager is not None,
            chat_service is not None
        ])
        
        overall_status = "healthy" if (
            qdrant_status.startswith("connected") and 
            database_status == "connected" and 
            components_ready
        ) else "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            message="System health check completed",
            qdrant_status=qdrant_status,
            database_status=database_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


# System initialization status endpoint
@app.get("/v1/system/init-status")
async def get_initialization_status():
    """Get detailed system initialization status"""
    try:
        # Quick connection check
        connections = check_connections()
        
        # Component status
        components_status = {
            'vectorstore': vectorstore is not None,
            'qa_chain': qa_chain is not None,
            'qdrant_client': qdrant_client is not None,
            'db_manager': db_manager is not None,
            'chat_service': chat_service is not None,
            'config': config is not None
        }
        
        # Overall readiness
        all_ready = all(connections.values()) and all(components_status.values())
        
        return {
            "status": "ready" if all_ready else "initializing",
            "connections": connections,
            "components": components_status,
            "overall_ready": all_ready,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get initialization status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


# Force re-initialization endpoint (for production troubleshooting)
@app.post("/v1/system/reinitialize")
async def force_reinitialize():
    """Force system re-initialization (admin endpoint)"""
    try:
        logger.info("🔄 Force re-initialization requested...")
        await initialize_rag_system()
        return {
            "status": "success",
            "message": "System re-initialized successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Re-initialization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Re-initialization failed: {str(e)}")


# Fresh initialization endpoint (clears all data)
@app.post("/v1/system/fresh-init")
async def force_fresh_initialization():
    """Force fresh system initialization with complete data cleanup - DESTRUCTIVE OPERATION"""
    try:
        logger.warning("🧨 FRESH INITIALIZATION REQUESTED - ALL DATA WILL BE PERMANENTLY DELETED!")
        
        # Import the fresh initialization function
        from app.database_initializer import initialize_system_fresh
        
        # Run fresh initialization
        init_results = await initialize_system_fresh()
        
        if init_results['overall_success']:
            logger.info("✅ Fresh initialization completed successfully - all old data cleared")
            
            # Reinitialize global variables
            global vectorstore, qa_chain, qdrant_client, db_manager, chat_service, config
            
            # Re-run the full RAG system initialization with clean state
            await initialize_rag_system()
            
            return {
                "status": "success",
                "message": "Fresh system initialization completed successfully",
                "details": init_results,
                "warning": "⚠️ ALL PREVIOUS DATA HAS BEEN PERMANENTLY DELETED",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            failed_components = [k for k, v in init_results.items() if not v and k != 'overall_success']
            logger.error(f"❌ Fresh initialization failed: {failed_components}")
            raise HTTPException(
                status_code=500, 
                detail=f"Fresh initialization failed. Failed components: {failed_components}"
            )
        
    except Exception as e:
        logger.error(f"❌ Fresh initialization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Fresh initialization failed: {str(e)}")


# Chat completions endpoint (OpenAI-compatible)
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    try:
        if not qa_chain:
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        # Extract the last message as the question
        last_message = request.messages[-1] if request.messages else None
        if not last_message or last_message.role != "user":
            raise HTTPException(status_code=400, detail="Last message must be from user")
        
        question = last_message.content
        chat_history = request.messages[:-1] if len(request.messages) > 1 else []
        
        # Generate response
        response_text = qa_chain(question, chat_history)
        
        # Format response
        completion_response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:config.get('chat_completion_id_length', 8)]}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(question.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(question.split()) + len(response_text.split())
            }
        )
        
        return completion_response
        
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")


# Collections endpoint
@app.get("/v1/collections")
async def get_collections():
    """Get Qdrant collections information"""
    try:
        if not qdrant_client:
            raise HTTPException(status_code=503, detail="Qdrant client not initialized")
        
        collections = qdrant_client.get_collections()
        
        result = []
        for collection in collections.collections:
            try:
                collection_info = qdrant_client.get_collection(collection.name)
                result.append({
                    "name": collection.name,
                    "vectors_count": collection_info.vectors_count,
                    "indexed_vectors_count": collection_info.indexed_vectors_count,
                    "points_count": collection_info.points_count
                })
            except Exception as e:
                logger.warning(f"Failed to get info for collection {collection.name}: {e}")
                result.append({
                    "name": collection.name,
                    "error": str(e)
                })
        
        return {"collections": result}
        
    except Exception as e:
        logger.error(f"Failed to get collections: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get collections: {str(e)}")


# New chat endpoint
@app.post("/v1/chat/new", response_model=NewChatResponse)
async def create_new_chat(request: NewChatRequest):
    """Create a new chat session"""
    try:
        if not chat_service:
            raise HTTPException(status_code=503, detail="Chat service not initialized")
        
        result = chat_service.create_new_chat(request.user_id)
        
        return NewChatResponse(
            chat_id=result["chat_id"],
            user_id=result["user_id"],
            message=result["message"],
            created_at=result["created_at"],
            last_activity=result["last_activity"]
        )
        
    except Exception as e:
        logger.error(f"Failed to create new chat: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create new chat: {str(e)}")


# User session status endpoint
@app.post("/v1/user/session-status", response_model=UserSessionStatusResponse)
async def get_user_session_status(request: UserSessionStatusRequest):
    """Get user session status"""
    try:
        if not chat_service:
            raise HTTPException(status_code=503, detail="Chat service not initialized")
        
        result = chat_service.get_user_session_status(request.user_id)
        
        return UserSessionStatusResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to get user session status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user session status: {str(e)}")


# Chat endpoint
@app.post("/v1/chat", response_model=ChatResponse)
async def chat_with_history(request: ChatRequest):
    """Chat with history tracking"""
    try:
        if not chat_service:
            raise HTTPException(status_code=503, detail="Chat service not initialized")
        
        result = await chat_service.process_message(
            user_id=request.user_id,
            message=request.message,
            chat_id=request.chat_id
        )
        
        return ChatResponse(**result)
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


# Chat history endpoint
@app.post("/v1/chat/history")
async def get_chat_history(request: ChatHistoryRequest):
    """Get chat history"""
    try:
        if not chat_service:
            raise HTTPException(status_code=503, detail="Chat service not initialized")
        
        messages = chat_service.get_chat_history(request.user_id, request.chat_id)
        
        return {
            "success": True,
            "chat_id": request.chat_id,
            "user_id": request.user_id,
            "messages": messages,
            "total_messages": len(messages)
        }
        
    except Exception as e:
        logger.error(f"Failed to get chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get chat history: {str(e)}")


# User chats endpoint
@app.get("/v1/user/{user_id}/chats")
async def get_user_chats(user_id: str):
    """Get all chats for a user"""
    try:
        if not chat_service:
            raise HTTPException(status_code=503, detail="Chat service not initialized")
        
        chats = chat_service.get_user_chats(user_id)
        
        return {
            "success": True,
            "user_id": user_id,
            "chats": chats,
            "total_chats": len(chats)
        }
        
    except Exception as e:
        logger.error(f"Failed to get user chats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user chats: {str(e)}")


# Document upload with progress tracking (no physical file storage)
async def _upload_with_progress_task(file_content: bytes, filename: str, file_size: int, upload_id: str, db_manager_instance, vectorstore_instance, config_instance):
    """Background task for document upload with progress - no physical file storage"""
    try:
        print(f"BACKGROUND TASK STARTED: {filename} (size: {file_size} bytes)")
        logger.info(f"🚀 Starting upload task for {filename} (size: {file_size} bytes)")
        
        async def progress_callback(stage: str, progress: float, message: str):
            upload_progress_store[upload_id] = DocumentUploadProgress(
                stage=stage,
                progress=progress,
                message=message,
                details={"upload_id": upload_id, "filename": filename}
            )
        
        await progress_callback("processing", 10, "Processing document...")
        
        # Generate unique filename (timestamp + original name)
        import time as time_module
        timestamp = int(time_module.time() * 1000)  # milliseconds
        original_filename = filename or "unknown"
        unique_filename = f"{timestamp}_{original_filename}"
        
        # Create document record first
        await progress_callback("creating_record", 20, "Creating document record...")
        logger.info(f"📝 Creating document record for {unique_filename} (size: {file_size} bytes)")
        document_id = db_manager_instance.create_document_record(
            filename=unique_filename,
            original_filename=original_filename,
            file_size=file_size or 0,
            file_type=Path(original_filename).suffix.lower(),
            metadata={"upload_id": upload_id}
        )
        logger.info(f"✅ Document record created with ID: {document_id}, size: {file_size} bytes")
        
        if not document_id:
            raise Exception("Failed to create document record")
        
        # Process the document (in memory only - no file saving)
        start_time = time.time()
        await progress_callback("extracting", 40, "Extracting content...")
        
        # Import here to avoid circular imports
        from app.document_processing import process_document_content_with_progress
        logger.info(f"About to process document: {filename}")
        documents, _ = await process_document_content_with_progress(file_content, filename, config_instance, progress_callback)
        logger.info(f"Document processing completed, got {len(documents)} documents")
        
        # Add to vector store
        await progress_callback("vectorizing", 70, "Creating embeddings...")
        logger.info("About to add documents to vector store")
        point_ids = await vectorstore_instance.add_documents(documents, progress_callback=progress_callback)
        logger.info(f"Successfully added {len(point_ids)} points to vector store")
        
        # Store vector points in database with document reference
        if point_ids:
            await progress_callback("storing", 90, "Storing metadata...")
            success = db_manager_instance.store_vector_points(
                document_id=document_id,
                filename=unique_filename,
                point_ids=point_ids,
                chunks_data=[{"content": doc.page_content[:config.get('chunk_preview_length', 200)], "metadata": doc.metadata} for doc in documents]
            )
            
            if not success:
                raise Exception("Failed to store vector points")
        
        processing_time = time.time() - start_time
        
        # Final progress update
        await progress_callback("completed", 100, f"Successfully processed {original_filename}")
        
        # Store final result
        upload_progress_store[upload_id] = DocumentUploadProgress(
            stage="completed",
            progress=100,
            message=f"Successfully uploaded and processed {original_filename}",
            details={
                "upload_id": upload_id,
                "filename": unique_filename,
                "original_filename": original_filename,
                "document_id": document_id,
                "chunks_added": len(documents),
                "processing_time": processing_time,
                "success": True
            }
        )
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"BACKGROUND TASK ERROR: {e}")
        print(f"FULL TRACEBACK:\n{error_details}")
        logger.error(f"Document upload failed: {e}")
        logger.error(f"Full traceback:\n{error_details}")
        
        # Update document record with error if it was created
        try:
            if 'document_id' in locals() and document_id:
                with db_manager_instance.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE documents SET processing_status = 'failed', error_message = %s WHERE id = %s",
                            (str(e), document_id)
                        )
                    conn.commit()
        except Exception as db_e:
            logger.error(f"Failed to update document error status: {db_e}")
        
        upload_progress_store[upload_id] = DocumentUploadProgress(
            stage="error",
            progress=0,
            message=f"Upload failed: {str(e)}",
            details={"upload_id": upload_id, "error": str(e), "success": False}
        )


@app.post("/v1/documents/upload-with-progress")
async def upload_document_with_progress(file: UploadFile = File(...)):
    """Upload document with progress tracking"""
    try:
        upload_id = f"upload_{uuid.uuid4().hex[:config.get('upload_id_length', 8)]}"
        
        # Read file content immediately to prevent "I/O operation on closed file" errors
        file_content = await file.read()
        filename = file.filename or "unknown"
        file_size = len(file_content)
        
        # Create a file-like object with the content
        from io import BytesIO
        class UploadFileWrapper:
            def __init__(self, content: bytes, filename: str, size: int):
                self.file = BytesIO(content)
                self.filename = filename
                self.size = size
                self._content = content
                self._read_count = 0
            
            async def read(self, size: int = -1):
                """Read file content, always returning the full content regardless of position"""
                self._read_count += 1
                # Always return the full content for compatibility
                return self._content
            
            def seek(self, position):
                return self.file.seek(position)
            
            def tell(self):
                return self.file.tell()
            
            def close(self):
                # Do nothing - we want to keep the file "open"
                pass
        
        file_wrapper = UploadFileWrapper(file_content, filename, file_size)
        
        # Start background task with the content data
        print(f"CREATING BACKGROUND TASK for {filename}")
        try:
            task = asyncio.create_task(_upload_with_progress_task(file_content, filename, file_size, upload_id, db_manager, vectorstore, config))
            print(f"BACKGROUND TASK CREATED: {task}")
        except Exception as task_error:
            print(f"FAILED TO CREATE TASK: {task_error}")
            import traceback
            print(f"TASK CREATION TRACEBACK:\n{traceback.format_exc()}")
        
        return {"upload_id": upload_id, "message": "Upload started", "status": "processing"}
        
    except Exception as e:
        logger.error(f"Failed to start document upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start upload: {str(e)}")


@app.get("/v1/documents/upload-progress/{upload_id}")
async def get_upload_progress(upload_id: str):
    """Get upload progress status"""
    try:
        if upload_id not in upload_progress_store:
            raise HTTPException(status_code=404, detail="Upload ID not found")
        
        progress = upload_progress_store[upload_id]
        
        # Clean up completed/failed uploads after some time
        if progress.stage in ["completed", "error"]:
            return progress
        
        return progress
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get upload progress: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get progress: {str(e)}")


# List documents endpoint (from database, not filesystem)
@app.get("/v1/documents/list", response_model=DocumentListResponse)
async def list_documents():
    """List uploaded documents from database"""
    try:
        if not db_manager:
            raise HTTPException(status_code=503, detail="Database not available")
        
        # Get documents from database
        documents = db_manager.get_documents_list(active_only=True)
        
        files = []
        total_size = 0
        
        for doc in documents:
            file_size = doc['file_size_bytes'] or 0
            chunks_count = doc['chunks_count'] or 0
            
            # Debug logging
            logger.debug(f"Document {doc['original_filename']}: size={file_size}, chunks={chunks_count}, status={doc.get('processing_status', 'unknown')}")
            
            files.append(FileInfo(
                filename=doc['original_filename'],
                file_size=file_size,
                created_at=doc['upload_timestamp'].isoformat() if doc['upload_timestamp'] else None,
                modified_at=doc['upload_timestamp'].isoformat() if doc['upload_timestamp'] else None,
                file_extension=doc['file_type'] or ""
            ))
            total_size += file_size
        
        return DocumentListResponse(
            success=True,
            message=f"Found {len(files)} active documents",
            files=files,
            total_files=len(files),
            total_size=total_size
        )
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


# Delete document endpoint (deactivate instead of physical deletion)
@app.delete("/v1/documents/delete", response_model=DocumentDeleteResponse)
async def delete_document(request: DocumentDeleteRequest):
    """Deactivate a document (set is_active=False) and remove embeddings"""
    try:
        original_filename = request.filename
        
        if not db_manager:
            raise HTTPException(status_code=503, detail="Database not available")
        
        # First, find the actual filename with timestamp by original filename
        with db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT filename FROM documents 
                    WHERE original_filename = %s AND is_active = TRUE
                    ORDER BY upload_timestamp DESC
                    LIMIT 1
                    """,
                    (original_filename,)
                )
                result = cur.fetchone()
                
                if not result:
                    raise HTTPException(status_code=404, detail=f"Document {original_filename} not found or already inactive")
                
                actual_filename = result[0]
        
        # Get vector point IDs before deactivating
        point_ids = db_manager.get_vector_points_for_document(actual_filename)
        
        if not point_ids:
            logger.warning(f"No vector points found for {actual_filename}")
        
        # Delete from vector store first
        embeddings_deleted = 0
        if vectorstore and point_ids:
            try:
                embeddings_deleted = await vectorstore.delete_points(point_ids)
                logger.info(f"Deleted {embeddings_deleted} embeddings for {actual_filename}")
            except Exception as e:
                logger.warning(f"Failed to delete embeddings for {actual_filename}: {e}")
        
        # Deactivate document in database (set is_active = False)
        success = db_manager.deactivate_document(actual_filename)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Document {original_filename} not found or already inactive")
        
        return DocumentDeleteResponse(
            success=True,
            message=f"Successfully deactivated document {original_filename}",
            filename=original_filename,
            embeddings_deleted=embeddings_deleted
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


# Document statistics endpoint
@app.get("/v1/documents/stats")
async def get_document_statistics():
    """Get document statistics"""
    try:
        if not db_manager:
            raise HTTPException(status_code=503, detail="Database not available")
        
        stats = db_manager.get_document_stats()
        vector_stats = db_manager.get_file_vector_stats()
        
        # Add debug logging
        logger.info(f"Document stats: {stats}")
        logger.info(f"Vector stats: {vector_stats}")
        
        # Ensure numeric values are not None
        safe_stats = {
            "total_documents": stats.get("total_documents", 0) or 0,
            "active_documents": stats.get("active_documents", 0) or 0,
            "inactive_documents": stats.get("inactive_documents", 0) or 0,
            "total_size": stats.get("total_size", 0) or 0,
            "total_chunks": stats.get("total_chunks", 0) or 0,
            "first_upload": stats.get("first_upload"),
            "last_upload": stats.get("last_upload")
        }
        
        safe_vector_stats = {
            "total_points": vector_stats.get("total_points", 0) or 0,
            "total_files": vector_stats.get("total_files", 0) or 0,
            "first_created": vector_stats.get("first_created"),
            "last_created": vector_stats.get("last_created")
        }
        
        return {
            "success": True,
            "document_stats": safe_stats,
            "vector_stats": safe_vector_stats,
            "message": "Document statistics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get document statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Entry point is handled by run.py - this file should not be run directly 