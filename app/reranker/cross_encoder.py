from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[dict], top_k: int = None) -> list[dict]:
        if not results:
            return []
        pairs = [(query, r["document"]["content"]) for r in results]
        scores = self.model.predict(pairs)

        scored = list(zip(results, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        return [
            {"document": r["document"], "score": float(s)}
            for r, s in scored
        ]