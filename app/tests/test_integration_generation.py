import pytest
from app.ingestion.loader import load_documents
from app.ingestion.chunking import chunk_documents
from app.ingestion.embeddings import EmbeddingGenerator
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever
from app.retrieval.hybrid import HybridRetrieval
from app.reranker.cross_encoder import CrossEncoderReranker
from app.generation.generator import LLMGenerator


@pytest.fixture(scope="module")
def embedding_generator():
    return EmbeddingGenerator()


@pytest.fixture(scope="module")
def reranker():
    return CrossEncoderReranker()


@pytest.fixture(scope="module")
def generator():
    return LLMGenerator()


def test_end_to_end_generation(tmp_path, embedding_generator, reranker, generator):
    doc1 = tmp_path / "python.txt"
    doc1.write_text(
        "Python es un lenguaje de programacion de alto nivel creado por Guido van Rossum en 1991. "
        "Es conocido por su sintaxis clara y legible. "
        "Se utiliza ampliamente en ciencia de datos, desarrollo web y automatizacion."
    )

    doc2 = tmp_path / "basketball.txt"
    doc2.write_text(
        "El baloncesto es un deporte de equipo inventado por James Naismith en 1891. "
        "Se juega entre dos equipos de cinco jugadores cada uno. "
        "El objetivo es anotar puntos introduciendo un balon en el aro del equipo contrario."
    )

    documents = load_documents(str(tmp_path))
    chunks = chunk_documents(documents, chunk_size=500, overlap=50)

    dense = DenseRetriever()
    embeddings = embedding_generator.embed_chunks(chunks)
    dense.build_index(chunks, embeddings)

    sparse = SparseRetriever()
    sparse.build_index(chunks)

    query = "Quien creo Python y para que se usa?"
    query_embedding = embedding_generator.embed_query(query)

    dense_results = dense.search(query_embedding, top_k=3)
    sparse_results = sparse.search(query, top_k=3)

    hybrid = HybridRetrieval(rrf_k=60)
    fused = hybrid.fuse(dense_results, sparse_results, top_k=3)

    reranked = reranker.rerank(query, fused, top_k=2)

    answer = generator.generate(query, reranked)

    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "Guido" in answer or "van Rossum" in answer
