"""Unit tests for SparseRetriever."""

import pytest

from app.retrieval.sparse import SparseRetriever


def test_build_index():
    """Build a BM25 index and verify it stores the chunks correctly."""
    retriever = SparseRetriever()
    chunks = [
        {'content': 'Python is a programming language', 'metadata': {'source': 'doc1.txt', 'chunk_id': 0}},
        {'content': 'Java is also a programming language', 'metadata': {'source': 'doc2.txt', 'chunk_id': 0}},
    ]

    retriever.build_index(chunks)

    assert retriever.bm25 is not None
    assert retriever.documents == chunks


def test_tokenize():
    """Tokenization lowercases text and extracts word tokens, discarding punctuation."""
    retriever = SparseRetriever()
    text = "Hello, World! This is a test."
    tokens = retriever._tokenize(text)
    assert tokens == ['hello', 'world', 'this', 'is', 'a', 'test']


def test_search():
    """Search returns results in the correct format with the most relevant document first."""
    retriever = SparseRetriever()
    chunks = [
        {'content': 'Python is a programming language used in data science', 'metadata': {'source': 'doc1.txt', 'chunk_id': 0}},
        {'content': 'Cats are fluffy animals that make great pets', 'metadata': {'source': 'doc2.txt', 'chunk_id': 0}},
        {'content': 'Cooking involves techniques like boiling and frying food', 'metadata': {'source': 'doc3.txt', 'chunk_id': 0}},
    ]

    retriever.build_index(chunks)
    results = retriever.search('programming language', top_k=2)

    assert len(results) == 2
    assert results[0]['document']['content'] == 'Python is a programming language used in data science'
    assert results[0]['score'] > 0.0
    assert 'document' in results[0]
    assert 'score' in results[0]


def test_search_top_k():
    """top_k is capped at the number of indexed documents, never returns more than available."""
    retriever = SparseRetriever()
    chunks = [
        {'content': f'Document number {i} contains unique vocabulary about topic A', 'metadata': {}}
        for i in range(10)
    ]

    retriever.build_index(chunks)

    results_3 = retriever.search('unique vocabulary', top_k=3)
    assert len(results_3) == 3

    results_10 = retriever.search('unique vocabulary', top_k=10)
    assert len(results_10) == 10

    results_20 = retriever.search('unique vocabulary', top_k=20)
    assert len(results_20) <= 10


def test_build_index_empty():
    """ValueError is raised when building an index with an empty chunk list."""
    retriever = SparseRetriever()

    with pytest.raises(ValueError):
        retriever.build_index([])


def test_search_without_index():
    """ValueError is raised when search is called before building the index."""
    retriever = SparseRetriever()

    with pytest.raises(ValueError):
        retriever.search('some query')


def test_search_relevance():
    """The document with the most matching keywords appears first in the results."""
    retriever = SparseRetriever()
    chunks = [
        {'content': 'Python programming language for data science and machine learning', 'metadata': {'source': 'doc1.txt'}},
        {'content': 'Basketball sport game played with ball and hoop points', 'metadata': {'source': 'doc2.txt'}},
        {'content': 'Cooking food recipes ingredients kitchen techniques boiling frying', 'metadata': {'source': 'doc3.txt'}},
    ]

    retriever.build_index(chunks)
    results = retriever.search('python data science programming', top_k=3)

    assert results[0]['document']['metadata']['source'] == 'doc1.txt'
    assert results[0]['score'] >= results[1]['score']
    assert results[1]['score'] >= results[2]['score']


def test_search_scores_descending():
    """Results are always returned in descending score order."""
    retriever = SparseRetriever()
    chunks = [
        {'content': 'Python programming for data science and machine learning applications', 'metadata': {}},
        {'content': 'Java programming for enterprise applications and backend services', 'metadata': {}},
        {'content': 'Rust programming for systems and performance critical software', 'metadata': {}},
    ]

    retriever.build_index(chunks)
    results = retriever.search('programming', top_k=3)

    scores = [r['score'] for r in results]
    assert scores == sorted(scores, reverse=True)
