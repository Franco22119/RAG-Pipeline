# Responsibility: Generate embeddings from the chunked documents.

from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingGenerator:
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2'
    ):
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: list[dict]) -> np.ndarray:
        texts = [chunk["content"] for chunk in chunks]
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)

