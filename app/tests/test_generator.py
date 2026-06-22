from app.generation.generator import LLMGenerator

def _make_result(content: str):
    return {
        "document": {"content": content, "metadata": {"source": "doc.txt", "chunk_id": 0}},
        "score": 0.9,
    }


def test_build_prompt_single_doc():
    generator = LLMGenerator()
    results = [_make_result("Python es un lenguaje de programacion.")]

    prompt = generator._build_prompt("Que es Python?", results)

    assert "Python es un lenguaje de programacion." in prompt
    assert "Que es Python?" in prompt
    assert "Contexto:" in prompt
    assert "Documento 1:" in prompt


def test_build_prompt_multiple_docs():
    generator = LLMGenerator()
    results = [
        _make_result("Python es un lenguaje."),
        _make_result("Python se usa en data science."),
    ]

    prompt = generator._build_prompt("Que es Python?", results)

    assert "Documento 1:" in prompt
    assert "Documento 2:" in prompt
    assert "Python es un lenguaje." in prompt
    assert "Python se usa en data science." in prompt


def test_build_prompt_empty_results():
    generator = LLMGenerator()
    prompt = generator._build_prompt("Que es Python?", [])

    assert "Pregunta: Que es Python?" in prompt
    assert "Contexto:" in prompt


def test_generate_returns_string(mocker):
    generator = LLMGenerator()

    mock_response = {
        "message": {"content": "Python es un lenguaje de programacion."}
    }
    mocker.patch("ollama.chat", return_value=mock_response)

    results = [_make_result("Python es un lenguaje de programacion.")]
    answer = generator.generate("Que es Python?", results)

    assert answer == "Python es un lenguaje de programacion."
