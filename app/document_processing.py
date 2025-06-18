"""
Document Processing Utilities
===========================

Document loading, chunking, and preprocessing utilities for the RAG system.
"""

import logging
import os
import asyncio
from typing import List, Dict, Any, Tuple
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader, DirectoryLoader, TextLoader, 
    PyPDFLoader, UnstructuredWordDocumentLoader
)
from fastapi import UploadFile

logger = logging.getLogger(__name__)


def load_and_split_documents(config: Dict[str, Any]) -> List[Document]:
    """Load and split documents from various sources (gracefully handle missing directories)"""
    documents = []
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.get("chunk_size", 1000),
        chunk_overlap=config.get("chunk_overlap", 200),
        length_function=len,
        add_start_index=True,
    )
    
    logger.info("Loading documents from configured sources...")
    
    # Load from directory if specified and exists (don't error if missing)
    documents_path = config.get("documents_path", "./documents")
    if documents_path:
        documents_dir = Path(documents_path)
        if documents_dir.exists() and documents_dir.is_dir():
            logger.info(f"Loading documents from directory: {documents_path}")
            
            # Load text files
            try:
                txt_loader = DirectoryLoader(
                    documents_path,
                    glob="*.txt",
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"}
                )
                txt_docs = txt_loader.load()
                if txt_docs:
                    logger.info(f"Loaded {len(txt_docs)} text files")
                    documents.extend(txt_docs)
            except Exception as e:
                logger.warning(f"Failed to load text files: {e}")
            
            # Load PDF files
            try:
                pdf_files = list(documents_dir.glob("*.pdf"))
                for pdf_file in pdf_files:
                    pdf_loader = PyPDFLoader(str(pdf_file))
                    pdf_docs = pdf_loader.load()
                    if pdf_docs:
                        logger.info(f"Loaded {len(pdf_docs)} pages from {pdf_file.name}")
                        documents.extend(pdf_docs)
            except Exception as e:
                logger.warning(f"Failed to load PDF files: {e}")
            
            # Load Word documents
            try:
                doc_files = list(documents_dir.glob("*.doc*"))
                for doc_file in doc_files:
                    word_loader = UnstructuredWordDocumentLoader(str(doc_file))
                    word_docs = word_loader.load()
                    if word_docs:
                        logger.info(f"Loaded {len(word_docs)} sections from {doc_file.name}")
                        documents.extend(word_docs)
            except Exception as e:
                logger.warning(f"Failed to load Word documents: {e}")
                
            if not documents:
                logger.info(f"No documents found in directory: {documents_path}")
        else:
            logger.info(f"Documents directory not found or not accessible: {documents_path} (this is okay)")
    
    # Load from URL if specified
    document_url = config.get("document_url")
    if document_url:
        try:
            logger.info(f"Loading document from URL: {document_url}")
            web_loader = WebBaseLoader(document_url)
            web_docs = web_loader.load()
            documents.extend(web_docs)
            logger.info(f"Loaded {len(web_docs)} documents from URL")
        except Exception as e:
            logger.error(f"Failed to load document from URL {document_url}: {e}")
    
    # Preprocess and split documents
    if documents:
        logger.info(f"Preprocessing {len(documents)} documents...")
        documents = preprocess_documents(documents)
        
        logger.info("Splitting documents into chunks...")
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(split_docs)} chunks")
        
        return split_docs
    else:
        logger.info("No documents loaded - starting with empty knowledge base (documents can be uploaded via API)")
        return []


def preprocess_documents(documents: List[Document]) -> List[Document]:
    """Preprocess documents to clean and normalize content"""
    processed_docs = []
    
    for i, doc in enumerate(documents):
        try:
            # Clean content
            content = doc.page_content.strip()
            
            # Remove excessive whitespace
            content = " ".join(content.split())
            
            # Skip very short documents
            if len(content) < 50:
                logger.debug(f"Skipping short document {i}: {len(content)} characters")
                continue
            
            # Update metadata
            metadata = doc.metadata.copy()
            metadata["original_length"] = len(doc.page_content)
            metadata["processed_length"] = len(content)
            metadata["chunk_id"] = i
            
            # Create processed document
            processed_doc = Document(
                page_content=content,
                metadata=metadata
            )
            processed_docs.append(processed_doc)
            
        except Exception as e:
            logger.error(f"Failed to preprocess document {i}: {e}")
            continue
    
    logger.info(f"Preprocessed {len(processed_docs)}/{len(documents)} documents")
    return processed_docs


async def process_uploaded_document(file: UploadFile, config: Dict[str, Any]) -> Tuple[List[Document], str]:
    """Process an uploaded document and return chunks"""
    
    # Read file content
    content = await file.read()
    filename = file.filename or "unknown"
    
    # Determine file type and process accordingly
    file_extension = Path(filename).suffix.lower()
    
    documents = []
    
    if file_extension == ".pdf":
        documents = await process_pdf_with_progress(content, filename)
    elif file_extension in [".doc", ".docx"]:
        documents = await process_docx_with_progress(content, filename)
    elif file_extension == ".txt":
        documents = await process_text_with_progress(content, filename)
    else:
        # Try to process as text
        try:
            text_content = content.decode('utf-8')
            doc = Document(
                page_content=text_content,
                metadata={
                    "source": filename,
                    "file_type": file_extension,
                    "file_size": len(content)
                }
            )
            documents = [doc]
        except Exception as e:
            raise ValueError(f"Unsupported file type {file_extension}: {e}")
    
    if not documents:
        raise ValueError(f"No content extracted from {filename}")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.get("chunk_size", 1000),
        chunk_overlap=config.get("chunk_overlap", 200),
        length_function=len,
        add_start_index=True,
    )
    
    split_docs = text_splitter.split_documents(documents)
    
    # Update metadata with chunk information
    for i, doc in enumerate(split_docs):
        doc.metadata.update({
            "chunk_index": i,
            "total_chunks": len(split_docs),
            "source": filename
        })
    
    return split_docs, filename


async def process_uploaded_document_with_progress(file: UploadFile, config: Dict[str, Any], progress_callback=None) -> Tuple[List[Document], str]:
    """Process uploaded document with progress reporting"""
    
    if progress_callback:
        await progress_callback("reading", 10, "Reading file...")
    
    # Read file content
    content = await file.read()
    filename = file.filename or "unknown"
    
    return await process_document_content_with_progress(content, filename, config, progress_callback)


async def process_document_content_with_progress(content: bytes, filename: str, config: Dict[str, Any], progress_callback=None) -> Tuple[List[Document], str]:
    """Process document content with progress reporting"""
    
    if progress_callback:
        await progress_callback("processing", 30, f"Processing {filename}...")
    
    # Determine file type and process accordingly
    file_extension = Path(filename).suffix.lower()
    
    documents = []
    
    if file_extension == ".pdf":
        documents = await process_pdf_with_progress(content, filename, progress_callback)
    elif file_extension in [".doc", ".docx"]:
        documents = await process_docx_with_progress(content, filename, progress_callback)
    elif file_extension == ".txt":
        documents = await process_text_with_progress(content, filename, progress_callback)
    else:
        # Try to process as text
        try:
            text_content = content.decode('utf-8')
            doc = Document(
                page_content=text_content,
                metadata={
                    "source": filename,
                    "file_type": file_extension,
                    "file_size": len(content)
                }
            )
            documents = [doc]
        except Exception as e:
            raise ValueError(f"Unsupported file type {file_extension}: {e}")
    
    if not documents:
        raise ValueError(f"No content extracted from {filename}")
    
    if progress_callback:
        await progress_callback("chunking", 70, "Splitting into chunks...")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.get("chunk_size", 1000),
        chunk_overlap=config.get("chunk_overlap", 200),
        length_function=len,
        add_start_index=True,
    )
    
    split_docs = text_splitter.split_documents(documents)
    
    # Update metadata with chunk information
    for i, doc in enumerate(split_docs):
        doc.metadata.update({
            "chunk_index": i,
            "total_chunks": len(split_docs),
            "source": filename
        })
    
    if progress_callback:
        await progress_callback("complete", 80, f"Created {len(split_docs)} chunks")
    
    return split_docs, filename


async def process_pdf_with_progress(content: bytes, filename: str, progress_callback=None) -> List[Document]:
    """Process PDF content with progress reporting"""
    import tempfile
    from langchain_community.document_loaders import PyPDFLoader
    
    # Save content to temporary file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        if progress_callback:
            await progress_callback("pdf_processing", 40, "Extracting text from PDF...")
        
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        
        # Update metadata
        for i, doc in enumerate(documents):
            doc.metadata.update({
                "source": filename,
                "file_type": "pdf",
                "file_size": len(content),
                "page_number": i + 1
            })
        
        if progress_callback:
            await progress_callback("pdf_processing", 60, f"Extracted {len(documents)} pages")
        
        return documents
        
    finally:
        # Clean up temporary file
        os.unlink(temp_path)


async def process_docx_with_progress(content: bytes, filename: str, progress_callback=None) -> List[Document]:
    """Process DOCX content with progress reporting"""
    import tempfile
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader
    
    # Save content to temporary file
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temp_file:
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        if progress_callback:
            await progress_callback("docx_processing", 40, "Extracting text from Word document...")
        
        loader = UnstructuredWordDocumentLoader(temp_path)
        documents = loader.load()
        
        # Update metadata
        for doc in documents:
            doc.metadata.update({
                "source": filename,
                "file_type": "docx",
                "file_size": len(content)
            })
        
        if progress_callback:
            await progress_callback("docx_processing", 60, f"Extracted {len(documents)} sections")
        
        return documents
        
    finally:
        # Clean up temporary file
        os.unlink(temp_path)


async def process_text_with_progress(content: bytes, filename: str, progress_callback=None) -> List[Document]:
    """Process text content with progress reporting"""
    
    if progress_callback:
        await progress_callback("text_processing", 40, "Processing text file...")
    
    try:
        # Try UTF-8 first
        text_content = content.decode('utf-8')
    except UnicodeDecodeError:
        try:
            # Try latin-1 as fallback
            text_content = content.decode('latin-1')
        except UnicodeDecodeError:
            # Try with error handling
            text_content = content.decode('utf-8', errors='replace')
    
    doc = Document(
        page_content=text_content,
        metadata={
            "source": filename,
            "file_type": "txt",
            "file_size": len(content)
        }
    )
    
    if progress_callback:
        await progress_callback("text_processing", 60, "Text file processed")
    
    return [doc]


def check_available_models(endpoint: str, api_key: str = "EMPTY") -> List[str]:
    """Check available models at the vLLM endpoint"""
    import httpx
    
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(
                f"{endpoint}/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            
            if response.status_code == 200:
                data = response.json()
                models = [model["id"] for model in data.get("data", [])]
                logger.info(f"Available models at {endpoint}: {models}")
                return models
            else:
                logger.warning(f"Failed to get models from {endpoint}: {response.status_code}")
                return []
                
    except Exception as e:
        logger.error(f"Failed to check available models: {e}")
        return [] 