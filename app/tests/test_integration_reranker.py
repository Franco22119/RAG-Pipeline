"""Integration test: full pipeline with CrossEncoderReranker."""

import pytest
from app.ingestion.loader import load_documents
from app.ingestion.chunking import chunk_documents
from app.ingestion.embeddings import EmbeddingGenerator
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever
from app.retrieval.hybrid import HybridRetrieval
from app.reranker.cross_encoder import CrossEncoderReranker


@pytest.fixture(scope="module")
def embedding_generator():
    return EmbeddingGenerator()


@pytest.fixture(scope="module")
def reranker():
    return CrossEncoderReranker()


def test_end_to_end_reranker(tmp_path, embedding_generator, reranker):
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
    chunks = chunk_documents(documents, chunk_size=100, overlap=20)

    dense = DenseRetriever()
    embeddings = embedding_generator.embed_chunks(chunks)
    dense.build_index(chunks, embeddings)

    sparse = SparseRetriever()
    sparse.build_index(chunks)

    query = "Python programming language"
    query_embedding = embedding_generator.embed_query(query)

    dense_results = dense.search(query_embedding, top_k=5)
    assert len(dense_results) >= 1

    sparse_results = sparse.search(query, top_k=5)
    assert len(sparse_results) >= 1

    hybrid = HybridRetrieval(rrf_k=60)
    fused = hybrid.fuse(dense_results, sparse_results, top_k=5)
    assert len(fused) >= 1

    reranked = reranker.rerank(query, fused, top_k=3)
    assert len(reranked) <= 3
    assert len(reranked) >= 1

    assert reranked[0]["score"] >= reranked[-1]["score"]
    scores = [r["score"] for r in reranked]
    assert scores == sorted(scores, reverse=True)

    assert any(
        result["document"]["metadata"]["source"] == str(doc1)
        for result in reranked
    )
