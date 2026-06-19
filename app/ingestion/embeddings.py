# Responsability: Generate embeddings from the chunked documents.

from sentence_transformers import SentenceTransformer
import numpy as np

# This module defines the EmbeddingGenerator class, 
# which uses a pre-trained SentenceTransformer model to generate embeddings for chunks of text and queries. 
# The embeddings are normalized and returned as numpy arrays.
class EmbeddingGenerator:
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2'
    ):
        self.model = SentenceTransformer(model_name)

    # This method takes a list of chunks (dictionaries with 'content' keys) and generates embeddings for each chunk using the SentenceTransformer model.
    def embed_chunks(self, chunks: list[dict]) -> np.ndarray:
        texts = [chunk["content"] for chunk in chunks]
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    
    # This method takes a query string and generates an embedding for it using the same SentenceTransformer model, returning a normalized numpy array.
    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)

