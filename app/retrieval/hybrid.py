from collections import defaultdict

class HybridRetrieval:
    """
    Hybrid retrieval with Reciprocal Rank Fusion (RRF)
    """

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    @staticmethod
    def _doc_key(result: dict) -> str:
        doc = result["document"]
        meta = doc.get("metadata", {})
        source = meta.get("source", "")
        chunk_id = meta.get("chunk_id", "")
        return f"{source}#{chunk_id}"

    def fuse(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        if not dense_results and not sparse_results:
            raise ValueError("Both dense and sparse results cannot be empty.")

        combined_scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, dict] = {}

        for rank, result in enumerate(dense_results):
            key = self._doc_key(result)
            combined_scores[key] += 1 / (rank + 1 + self.rrf_k)
            doc_map[key] = result["document"]

        for rank, result in enumerate(sparse_results):
            key = self._doc_key(result)
            combined_scores[key] += 1 / (rank + 1 + self.rrf_k)
            doc_map[key] = result["document"]

        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        top_k = min(top_k, len(sorted_docs))
        top_results = []
        for key, score in sorted_docs[:top_k]:
            top_results.append({
                "document": doc_map[key],
                "score": score,
            })

        return top_results