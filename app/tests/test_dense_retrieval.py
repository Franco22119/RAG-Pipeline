"""Unit tests for DenseRetriever."""

import numpy as np
import pytest

from app.retrieval.dense import DenseRetriever


def test_build_index():
    """Build a FAISS index and verify it stores the correct number of documents."""
    retriever = DenseRetriever()
    chunks = [
        {'content': 'First document', 'metadata': {'source': 'doc1.txt', 'chunk_id': 0}},
        {'content': 'Second document', 'metadata': {'source': 'doc2.txt', 'chunk_id': 0}},
    ]
    embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

    retriever.build_index(chunks, embeddings)

    assert retriever.index is not None
    assert retriever.index.ntotal == 2
    assert retriever.documents == chunks


def test_search():
    """Search returns results in correct format with the most similar document first."""
    retriever = DenseRetriever()
    chunks = [
        {'content': 'Python is a programming language', 'metadata': {'source': 'doc1.txt', 'chunk_id': 0}},
        {'content': 'Java is also a programming language', 'metadata': {'source': 'doc2.txt', 'chunk_id': 0}},
        {'content': 'Cats are fluffy animals', 'metadata': {'source': 'doc3.txt', 'chunk_id': 0}},
    ]
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)

    retriever.build_index(chunks, embeddings)
    query_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = retriever.search(query_embedding, top_k=2)

    assert len(results) == 2
    assert results[0]['document']['content'] == 'Python is a programming language'
    assert results[0]['score'] > 0.0
    assert 'document' in results[0]
    assert 'score' in results[0]


def test_search_top_k():
    """top_k parameter is respected and capped at the number of indexed documents."""
    retriever = DenseRetriever()
    chunks = [{'content': f'Doc {i}', 'metadata': {}} for i in range(10)]
    embeddings = np.random.rand(10, 4).astype(np.float32)
    retriever.build_index(chunks, embeddings)

    query_embedding = np.random.rand(4).astype(np.float32)

    results_3 = retriever.search(query_embedding, top_k=3)
    assert len(results_3) == 3

    results_10 = retriever.search(query_embedding, top_k=10)
    assert len(results_10) == 10

    results_20 = retriever.search(query_embedding, top_k=20)
    assert len(results_20) <= 10


def test_build_index_dimension_mismatch():
    """ValueError is raised when chunk count does not match embedding count."""
    chunks = [
        {'content': 'a', 'metadata': {}},
        {'content': 'b', 'metadata': {}},
    ]
    embeddings = np.random.rand(3, 4).astype(np.float32)

    retriever = DenseRetriever()
    with pytest.raises(ValueError):
        retriever.build_index(chunks, embeddings)


def test_search_without_index():
    """ValueError is raised when search is called before building the index."""
    retriever = DenseRetriever()
    query_embedding = np.random.rand(4).astype(np.float32)

    with pytest.raises(ValueError):
        retriever.search(query_embedding)
