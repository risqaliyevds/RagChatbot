"""
Qdrant Vector Store Implementation
================================

Custom vector store implementation for Qdrant with document management
and similarity search capabilities.
"""

import logging
import uuid
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from .embedding_manager import MultilingualE5Embeddings

logger = logging.getLogger(__name__)


class SimpleQdrantVectorStore:
    """Simple Qdrant vector store implementation with document tracking"""
    
    def __init__(self, client: QdrantClient, collection_name: str, embeddings, config):
        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.config = config
        
    async def add_documents(self, documents: List[Document], embedding_dim: int = None, progress_callback=None):
        """Add documents to the vector store with progress tracking"""
        if not documents:
            logger.warning("No documents to add")
            return []
        
        logger.info(f"Adding {len(documents)} documents to collection '{self.collection_name}'")
        
        # Generate embeddings with progress tracking
        if progress_callback:
            await progress_callback("embedding", 10, "Generating embeddings...")
        
        texts = [doc.page_content for doc in documents]
        embeddings = self.embeddings.embed_documents(texts)
        
        if progress_callback:
            await progress_callback("embedding", 50, f"Generated {len(embeddings)} embeddings")
        
        # Prepare points for insertion
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point_id = int(str(uuid.uuid4().int)[:18])  # Convert to int for Qdrant
            
            # Prepare metadata, ensuring all values are JSON serializable
            metadata = {
                "content": doc.page_content[:500],  # Truncate for metadata
                "source": str(doc.metadata.get("source", "unknown")),
                "page": doc.metadata.get("page", 0),
                "chunk_index": i
            }
            
            # Add other metadata fields if they exist and are serializable
            for key, value in doc.metadata.items():
                if key not in metadata and isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                elif key not in metadata:
                    metadata[key] = str(value)  # Convert complex types to string
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=metadata
            )
            points.append(point)
            
            if progress_callback and i % 10 == 0:
                progress = 50 + (i / len(documents)) * 40
                await progress_callback("indexing", progress, f"Prepared {i+1}/{len(documents)} points")
        
        if progress_callback:
            await progress_callback("indexing", 90, "Uploading to Qdrant...")
        
        # Insert points in batches
        batch_size = 100
        total_inserted = 0
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                total_inserted += len(batch)
                logger.debug(f"Inserted batch of {len(batch)} points")
                
                if progress_callback:
                    progress = 90 + (total_inserted / len(points)) * 10
                    await progress_callback("indexing", progress, f"Inserted {total_inserted}/{len(points)} points")
                    
            except Exception as e:
                logger.error(f"Failed to insert batch: {e}")
                raise
        
        if progress_callback:
            await progress_callback("complete", 100, f"Successfully added {total_inserted} documents")
        
        logger.info(f"Successfully added {total_inserted} documents to collection '{self.collection_name}'")
        return [point.id for point in points]
    
    def search(self, query: str, k: int = 3, score_threshold: float = 0.4) -> List[Document]:
        """Search for similar documents"""
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                score_threshold=score_threshold
            )
            
            # Convert results to Documents
            documents = []
            for result in search_results:
                content = result.payload.get("content", "")
                metadata = {
                    "source": result.payload.get("source", "unknown"),
                    "page": result.payload.get("page", 0),
                    "score": result.score,
                    "chunk_index": result.payload.get("chunk_index", 0)
                }
                
                # Add other metadata fields
                for key, value in result.payload.items():
                    if key not in metadata:
                        metadata[key] = value
                
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            
            logger.debug(f"Found {len(documents)} similar documents for query: {query[:50]}...")
            return documents
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def delete_documents_by_source(self, source_identifier: str) -> int:
        """Delete documents by source identifier"""
        try:
            # Create filter for the source
            source_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source_identifier)
                    )
                ]
            )
            
            # First, get the points to be deleted to count them
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=source_filter,
                limit=10000,  # Adjust as needed
                with_payload=False,
                with_vectors=False
            )
            
            points_to_delete = [point.id for point in search_result[0]]
            
            if points_to_delete:
                # Delete the points
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=points_to_delete
                )
                
                logger.info(f"Deleted {len(points_to_delete)} points with source: {source_identifier}")
                return len(points_to_delete)
            else:
                logger.info(f"No points found with source: {source_identifier}")
                return 0
                
        except Exception as e:
            logger.error(f"Failed to delete documents by source {source_identifier}: {e}")
            return 0
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "config": {
                    "params": collection_info.config.params.__dict__ if collection_info.config.params else {},
                    "hnsw_config": collection_info.config.hnsw_config.__dict__ if collection_info.config.hnsw_config else {},
                    "optimizer_config": collection_info.config.optimizer_config.__dict__ if collection_info.config.optimizer_config else {},
                }
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}


def init_qdrant_client(config) -> QdrantClient:
    """Initialize Qdrant client"""
    try:
        qdrant_url = config.get("qdrant_url", "http://localhost:6333")
        client = QdrantClient(
            url=qdrant_url,
            api_key=config.get("qdrant_api_key")
        )
        logger.info(f"Connected to Qdrant at {qdrant_url}")
        return client
        
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant client: {e}")
        raise


async def init_vectorstore(config, documents: List[Document], client: QdrantClient) -> SimpleQdrantVectorStore:
    """Initialize vector store with documents"""
    
    # Initialize embeddings
    try:
        vllm_endpoint = config.get("vllm_embedding_endpoint")
        if vllm_endpoint and vllm_endpoint.strip():
            # Use vLLM endpoint
            embeddings_model = OpenAIEmbeddings(
                openai_api_base=vllm_endpoint,
                openai_api_key=config.get("vllm_embedding_key", "EMPTY"),
                model=config.get("embedding_model")
            )
            logger.info(f"Using vLLM embeddings: {config.get('embedding_model')}")
        else:
            # Use local multilingual E5 model
            embeddings_model = MultilingualE5Embeddings(
                model_name=config.get("embedding_model")
            )
            logger.info(f"Using local embeddings: {config.get('embedding_model')}")
        
        # Test embeddings
        test_embedding = embeddings_model.embed_query("test")
        embedding_dim = len(test_embedding)
        logger.info(f"Embedding dimension: {embedding_dim}")
        
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}")
        raise
    
    # Initialize collection
    collection_name = config.get("qdrant_collection_name")
    
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_exists = any(col.name == collection_name for col in collections)
        
        if collection_exists:
            collection_info = client.get_collection(collection_name)
            existing_dim = collection_info.config.params.vectors.size
            
            if existing_dim != embedding_dim:
                if config.get("qdrant_force_recreate", False):
                    logger.warning(f"Dimension mismatch. Recreating collection {collection_name}")
                    client.delete_collection(collection_name)
                    collection_exists = False
                else:
                    raise ValueError(
                        f"Collection {collection_name} exists with dimension {existing_dim}, "
                        f"but embedding model produces {embedding_dim} dimensions. "
                        f"Set QDRANT_FORCE_RECREATE=true to recreate the collection."
                    )
        
        if not collection_exists:
            # Create collection
            distance = Distance.COSINE
            if config.get("qdrant_distance", "").upper() == "EUCLID":
                distance = Distance.EUCLID
            elif config.get("qdrant_distance", "").upper() == "DOT":
                distance = Distance.DOT
            
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=distance,
                    on_disk=config.get("qdrant_on_disk", True)
                )
            )
            logger.info(f"Created Qdrant collection: {collection_name}")
        
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant collection: {e}")
        raise
    
    # Initialize vector store
    vectorstore = SimpleQdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings_model,
        config=config
    )
    
    # Add documents if provided
    if documents:
        logger.info(f"Adding {len(documents)} documents to vector store")
        await vectorstore.add_documents(documents, embedding_dim)
    
    return vectorstore 