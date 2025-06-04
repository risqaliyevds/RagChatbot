# Qdrant Vector Store Configuration Guide

This guide explains how to configure Qdrant vector store dimensions and other parameters using environment variables in your RAG chatbot application.

## Overview

The application now supports configurable Qdrant vector store parameters through environment variables, allowing you to:
- Set custom vector dimensions for different embedding models
- Choose distance metrics (COSINE, EUCLID, DOT)
- Control collection recreation behavior
- Configure vector storage options

## Environment Variables

### Basic Qdrant Configuration

```bash
# Qdrant connection settings
QDRANT_URL=http://localhost:6333          # Qdrant server URL (for Docker/remote)
QDRANT_API_KEY=your_api_key               # API key for Qdrant Cloud (optional)
QDRANT_PATH=./qdrant_storage              # Local storage path (for local mode)
QDRANT_COLLECTION_NAME=rag_documents      # Collection name
```

### Vector Configuration

```bash
# Vector dimension - set to 0 for auto-detection from embedding model
QDRANT_VECTOR_SIZE=768                    # Vector dimension (0 = auto-detect)

# Distance metric for similarity search
QDRANT_DISTANCE=COSINE                    # Options: COSINE, EUCLID, DOT

# Collection management
QDRANT_FORCE_RECREATE=true                # Recreate collection if config mismatch
QDRANT_ON_DISK=false                      # Store vectors on disk (saves RAM)
```

## Common Vector Dimensions by Model

| Embedding Model | Dimension | QDRANT_VECTOR_SIZE |
|----------------|-----------|-------------------|
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 384 |
| sentence-transformers/all-mpnet-base-v2 | 768 | 768 |
| text-embedding-ada-002 (OpenAI) | 1536 | 1536 |
| Alibaba-NLP/gte-modernbert-base | 768 | 768 |
| BAAI/bge-small-en-v1.5 | 384 | 384 |
| BAAI/bge-base-en-v1.5 | 768 | 768 |
| BAAI/bge-large-en-v1.5 | 1024 | 1024 |

## Configuration Files

The application loads environment variables from two files in order:

1. `.env` (if exists) - Standard environment file
2. `config.env` - Application-specific configuration (overrides .env)

### Example config.env

```bash
# Ports
EMBEDDING_PORT=11445
CHAT_MODEL_PORT=11444

# vLLM API Configuration
VLLM_API_KEY=EMPTY
VLLM_EMBEDDING_ENDPOINT=http://localhost:11445/v1
VLLM_CHAT_ENDPOINT=http://localhost:11444/v1

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=rag_documents

# Qdrant Vector Configuration
QDRANT_VECTOR_SIZE=768                    # Set to match your embedding model
QDRANT_DISTANCE=COSINE                    # Distance metric
QDRANT_FORCE_RECREATE=true                # Auto-recreate on mismatch
QDRANT_ON_DISK=false                      # Memory vs disk storage

# Model Configuration
EMBEDDING_MODEL=Alibaba-NLP/gte-modernbert-base
CHAT_MODEL=google/gemma-3-12b-it

# Document and RAG Configuration
DOCUMENTS_PATH=/mnt/mata/chatbot/documents
TOP_K=3
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Server Configuration
HOST=0.0.0.0
PORT=8080
```

## Usage Scenarios

### Scenario 1: Auto-Detection (Recommended)

Set `QDRANT_VECTOR_SIZE=0` to automatically detect the dimension from your embedding model:

```bash
QDRANT_VECTOR_SIZE=0
EMBEDDING_MODEL=Alibaba-NLP/gte-modernbert-base
```

The application will:
1. Load the embedding model
2. Generate a sample embedding
3. Detect the dimension automatically
4. Create the collection with the correct dimension

### Scenario 2: Explicit Dimension

Set a specific dimension if you know your model's output size:

```bash
QDRANT_VECTOR_SIZE=768
EMBEDDING_MODEL=Alibaba-NLP/gte-modernbert-base
```

### Scenario 3: Switching Models

When switching to a model with different dimensions:

```bash
# Old configuration
QDRANT_VECTOR_SIZE=384
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# New configuration
QDRANT_VECTOR_SIZE=768
EMBEDDING_MODEL=Alibaba-NLP/gte-modernbert-base
QDRANT_FORCE_RECREATE=true  # Important: recreate collection
```

### Scenario 4: Different Distance Metrics

Choose the appropriate distance metric for your use case:

```bash
# For normalized embeddings (most common)
QDRANT_DISTANCE=COSINE

# For non-normalized embeddings
QDRANT_DISTANCE=EUCLID

# For dot product similarity
QDRANT_DISTANCE=DOT
```

## Error Handling

### Dimension Mismatch Error

If you see an error like:
```
Collection 'rag_documents' exists with different vector configuration:
  Current: size=384, distance=COSINE
  Required: size=768, distance=COSINE
Set QDRANT_FORCE_RECREATE=true to automatically recreate the collection.
```

**Solution**: Set `QDRANT_FORCE_RECREATE=true` in your configuration.

### Invalid Distance Metric

If you specify an invalid distance metric, the application will:
1. Log a warning
2. Fall back to COSINE distance
3. Continue operation

### Embedding Dimension Validation

The application validates that document embeddings match the configured dimension and will:
1. Log warnings for mismatched documents
2. Skip documents with incorrect dimensions
3. Continue processing valid documents

## Performance Considerations

### Memory vs Disk Storage

```bash
# Store vectors in memory (faster access, more RAM usage)
QDRANT_ON_DISK=false

# Store vectors on disk (slower access, less RAM usage)
QDRANT_ON_DISK=true
```

### Distance Metrics Performance

- **COSINE**: Best for normalized embeddings, good performance
- **EUCLID**: Good for non-normalized embeddings, moderate performance
- **DOT**: Fastest, but requires careful consideration of embedding properties

## Troubleshooting

### 1. Check Configuration Loading

The application logs the loaded configuration:
```
DEBUG: Qdrant vector config - size: 768, distance: COSINE
```

### 2. Verify Embedding Model

Check that your embedding model is accessible:
```
Testing embedding service...
Detected embedding dimension: 768
```

### 3. Collection Status

Monitor collection creation/recreation:
```
Creating collection 'rag_documents' with dimension 768 and distance COSINE
```

### 4. Document Processing

Watch for dimension validation warnings:
```
Document 5: embedding dimension mismatch (384 vs 768)
```

## Best Practices

1. **Use Auto-Detection**: Set `QDRANT_VECTOR_SIZE=0` for automatic dimension detection
2. **Enable Force Recreate**: Set `QDRANT_FORCE_RECREATE=true` when experimenting with different models
3. **Match Distance to Model**: Use COSINE for most modern embedding models
4. **Monitor Logs**: Check application logs for configuration and processing status
5. **Test Configuration**: Verify your setup with a small document set first

## Migration Guide

### From Fixed Dimension to Configurable

If you're upgrading from a version with fixed dimensions:

1. **Backup your data** (if important)
2. **Set current dimension** in config:
   ```bash
   QDRANT_VECTOR_SIZE=768  # or whatever your current dimension is
   ```
3. **Enable force recreate** for future changes:
   ```bash
   QDRANT_FORCE_RECREATE=true
   ```
4. **Test with new models** by changing `EMBEDDING_MODEL` and `QDRANT_VECTOR_SIZE`

### Switching Between Models

1. **Update model and dimension**:
   ```bash
   EMBEDDING_MODEL=new-model-name
   QDRANT_VECTOR_SIZE=new-dimension  # or 0 for auto-detect
   ```
2. **Ensure force recreate is enabled**:
   ```bash
   QDRANT_FORCE_RECREATE=true
   ```
3. **Restart the application**

The collection will be automatically recreated with the new configuration. 