import re
import numpy as np
from rank_bm25 import BM25Okapi

class SparseRetriever:
    def __init__(self):
        self.bm25 = None
        self.documents = []

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        return re.findall(r'\w+', text)

    def build_index(self, chunks: list[dict]):
        if not chunks:
            raise ValueError("Chunks list cannot be empty.")

        self.documents = chunks
        tokenized = [self._tokenize(chunk["content"]) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")

        tokenized_query = self._tokenize(query)
        scores = np.array(self.bm25.get_scores(tokenized_query))
        top_k = min(top_k, len(self.documents))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "score": float(scores[idx])
            })
        return results