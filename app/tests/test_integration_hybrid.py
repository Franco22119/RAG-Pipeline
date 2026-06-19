"""Integration test: full pipeline with HybridRetrieval (dense + sparse + RRF)."""

import pytest
from pathlib import Path

from app.ingestion.loader import load_documents
from app.ingestion.chunking import chunk_documents
from app.ingestion.embeddings import EmbeddingGenerator
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever
from app.retrieval.hybrid import HybridRetrieval


@pytest.fixture(scope="module")
def embedding_generator():
    """Shared EmbeddingGenerator fixture, loaded once per module."""
    return EmbeddingGenerator()


def test_end_to_end_hybrid(tmp_path, embedding_generator):
    """Load, chunk, embed, retrieve from both systems, and fuse with RRF."""
    doc1 = tmp_path / "python.txt"
    doc1.write_text(
        "Python is a high-level programming language known for its readability and simplicity. "
        "It supports multiple programming paradigms including object-oriented and functional programming. "
        "Python has a large standard library and a vast ecosystem of third-party packages. "
        "It is widely used in data science, web development, and automation."
    )

    doc2 = tmp_path / "basketball.txt"
    doc2.write_text(
        "Basketball is a team sport played between two teams of five players each. "
        "The objective is to score points by shooting a ball through the opponent's hoop. "
        "The game was invented by James Naismith in 1891. "
        "Major leagues include the NBA in North America and the EuroLeague in Europe."
    )

    doc3 = tmp_path / "cooking.txt"
    doc3.write_text(
        "Cooking is the art and science of preparing food for consumption. "
        "It involves a variety of techniques such as boiling, frying, baking, and grilling. "
        "Different cuisines around the world use distinct combinations of ingredients and spices. "
        "Good cooking requires practice, creativity, and attention to detail."
    )

    documents = load_documents(str(tmp_path))
    assert len(documents) == 3

    chunks = chunk_documents(documents, chunk_size=100, overlap=20)
    assert len(chunks) >= 3

    dense_retriever = DenseRetriever()
    embeddings = embedding_generator.embed_chunks(chunks)
    dense_retriever.build_index(chunks, embeddings)

    sparse_retriever = SparseRetriever()
    sparse_retriever.build_index(chunks)

    query = "Python programming language"
    query_embedding = embedding_generator.embed_query(query)

    dense_results = dense_retriever.search(query_embedding, top_k=5)
    assert len(dense_results) >= 1

    sparse_results = sparse_retriever.search(query, top_k=5)
    assert len(sparse_results) >= 1

    hybrid = HybridRetrieval(rrf_k=60)
    fused = hybrid.fuse(dense_results, sparse_results, top_k=5)
    assert len(fused) >= 1

    assert fused[0]["score"] >= fused[-1]["score"]
    scores = [r["score"] for r in fused]
    assert scores == sorted(scores, reverse=True)

    assert any(
        result["document"]["metadata"]["source"] == str(doc1)
        for result in fused
    )
