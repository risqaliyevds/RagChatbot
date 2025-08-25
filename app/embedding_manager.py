"""
Custom Embedding Models for the RAG Application
==============================================

Custom embedding implementations including multilingual E5 model
for better multilingual support.
"""

import logging
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List

logger = logging.getLogger(__name__)


class MultilingualE5Embeddings:
    """Custom embedding class for intfloat/multilingual-e5-large-instruct model"""
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large-instruct", device: str = None):
        # Handle None model_name by setting default
        if model_name is None:
            model_name = "/app/models/multilingual-e5-large-instruct"
            logger.warning(f"model_name was None, using default: {model_name}")
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading multilingual E5 model: {model_name} on {self.device}")
        
        # Check if model_name is a local path or HuggingFace model name
        is_local_path = model_name.startswith('/') or model_name.startswith('./') or model_name.startswith('../')
        
        try:
            # Load tokenizer and model
            if is_local_path:
                # Local path - use offline mode
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
            else:
                # HuggingFace model - try offline first, then online
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                    self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
                except Exception as e:
                    logger.warning(f"Failed to load model offline: {e}")
                    logger.info("Trying to load model online...")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModel.from_pretrained(model_name)
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Successfully loaded {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def _get_detailed_instruct(self, task_description: str, query: str) -> str:
        """Format query with task description for E5 instruct model"""
        return f"Instruct: {task_description}\nQuery: {query}"
    
    def _encode_text(self, texts: List[str], task_description: str = "Given a question, retrieve passages that answer the question") -> np.ndarray:
        """Encode texts using the E5 model"""
        # Format texts with instruction
        formatted_texts = [self._get_detailed_instruct(task_description, text) for text in texts]
        
        # Tokenize
        batch_dict = self.tokenizer(
            formatted_texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self._average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def _average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Average pooling with attention mask"""
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = self._encode_text(texts, "Given a question, retrieve passages that answer the question")
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embeddings = self._encode_text([text], "Given a question, retrieve passages that answer the question")
        return embeddings[0].tolist() 