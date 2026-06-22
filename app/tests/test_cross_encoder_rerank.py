"""Unit tests for CrossEncoderReranker."""

import pytest

from app.reranker.cross_encoder import CrossEncoderReranker

@pytest.fixture(scope="module")
def reranker():
    return CrossEncoderReranker()


def _make_result(content: str, score: float):
    return {
        "document": {"content": content, "metadata": {"source": "doc.txt", "chunk_id": 0}},
        "score": score,
    }


def test_rerank_basic(reranker):
    results = [
        _make_result("I love programming in Python", 0.5),
        _make_result("Cats are very cute animals", 0.3),
        _make_result("Cooking requires fresh ingredients", 0.4),
    ]

    reranked = reranker.rerank("programming", results, top_k=3)

    assert len(reranked) == 3
    assert reranked[0]["document"]["content"] == "I love programming in Python"


def test_rerank_top_k(reranker):
    results = [
        _make_result("Python is a programming language", 0.5),
        _make_result("Java is also a programming language", 0.4),
        _make_result("Rust systems programming", 0.3),
        _make_result("Cooking recipes and techniques", 0.2),
    ]

    reranked = reranker.rerank("programming", results, top_k=2)

    assert len(reranked) == 2


def test_rerank_empty(reranker):
    reranked = reranker.rerank("test", [])

    assert reranked == []


def test_rerank_scores_descending(reranker):
    results = [
        _make_result("Dogs are loyal animals", 0.3),
        _make_result("Cats are independent pets", 0.4),
        _make_result("Birds can fly in the sky", 0.2),
    ]

    reranked = reranker.rerank("animals", results)

    scores = [r["score"] for r in reranked]
    assert scores == sorted(scores, reverse=True)


def test_rerank_output_format(reranker):
    results = [
        _make_result("Python programming language", 0.5),
        _make_result("Basketball sport game", 0.3),
    ]

    reranked = reranker.rerank("programming", results, top_k=1)

    assert len(reranked) == 1
    assert "document" in reranked[0]
    assert "score" in reranked[0]
    assert "content" in reranked[0]["document"]
    assert "metadata" in reranked[0]["document"]
