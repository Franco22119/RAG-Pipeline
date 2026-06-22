import ollama

class LLMGenerator:
    def __init__(self, model: str = "llama3.2"):
        self.model = model

    def _build_prompt(self, query: str, results: list[dict]) -> str:
        context = "\n\n".join(
            f"Documento {i+1}: {r['document']['content']}"
            for i, r in enumerate(results)
        )
        return (
            f"Eres un asistente util. Responde la pregunta usando SOLO el contexto proporcionado.\n\n"
            f"Contexto:\n{context}\n\n"
            f"Pregunta: {query}\n\n"
            f"Respuesta:"
        )

    def generate(self, query: str, results: list[dict]) -> str:
        prompt = self._build_prompt(query, results)
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
