import faiss
import numpy as np

class DenseRetriever:
    """
    Dense retrieval FAISS
    """

    def __init__(self):
        self.index = None
        self.documents = []

    def build_index(self, chunks: list[dict], embeddings: np.ndarray):
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must be the same.")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))
        self.documents = chunks

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        if self.index is None:
            raise ValueError("Index has not been built yet.")

        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            top_k
        )

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                break
            results.append({
                "document": self.documents[idx],
                "score": float(score)
            })

        return results