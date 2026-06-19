import pytest

from app.retrieval.hybrid import HybridRetrieval


def _make_chunk(source: str, chunk_id: int):
    return {
        "content": f"content from {source}",
        "metadata": {"source": source, "chunk_id": chunk_id},
    }


def _dense_result(source: str, chunk_id: int, score: float):
    return {"document": _make_chunk(source, chunk_id), "score": score}


def _sparse_result(source: str, chunk_id: int, score: float):
    return {"document": _make_chunk(source, chunk_id), "score": score}


def test_fuse_basic():
    dense_results = [
        _dense_result("doc1.txt", 0, 0.9),
        _dense_result("doc2.txt", 0, 0.8),
        _dense_result("doc3.txt", 0, 0.7),
    ]
    sparse_results = [
        _sparse_result("doc2.txt", 0, 10.0),
        _sparse_result("doc1.txt", 0, 5.0),
        _sparse_result("doc3.txt", 0, 3.0),
    ]

    hybrid = HybridRetrieval(rrf_k=60)
    results = hybrid.fuse(dense_results, sparse_results, top_k=3)

    assert len(results) == 3
    for r in results:
        assert "document" in r
        assert "score" in r
    assert results[0]["score"] >= results[1]["score"]
    assert results[1]["score"] >= results[2]["score"]


def test_fuse_dense_only():
    dense_results = [
        _dense_result("doc1.txt", 0, 0.9),
        _dense_result("doc2.txt", 0, 0.8),
    ]

    hybrid = HybridRetrieval()
    results = hybrid.fuse(dense_results, [], top_k=2)

    assert len(results) == 2


def test_fuse_sparse_only():
    sparse_results = [
        _sparse_result("doc1.txt", 0, 5.0),
        _sparse_result("doc2.txt", 0, 3.0),
    ]

    hybrid = HybridRetrieval()
    results = hybrid.fuse([], sparse_results, top_k=2)

    assert len(results) == 2


def test_fuse_empty_both():
    hybrid = HybridRetrieval()

    with pytest.raises(ValueError):
        hybrid.fuse([], [])


def test_fuse_top_k():
    dense_results = [
        _dense_result(f"doc{i}.txt", 0, 0.5) for i in range(10)
    ]
    sparse_results = [
        _sparse_result(f"doc{i}.txt", 0, 5.0) for i in range(10)
    ]

    hybrid = HybridRetrieval()
    results = hybrid.fuse(dense_results, sparse_results, top_k=3)

    assert len(results) == 3


def test_fuse_top_k_greater_than_available():
    dense_results = [
        _dense_result("doc1.txt", 0, 0.9),
        _dense_result("doc2.txt", 0, 0.8),
    ]

    hybrid = HybridRetrieval()
    results = hybrid.fuse(dense_results, [], top_k=100)

    assert len(results) == 2


def test_fuse_duplicate_across_systems():
    shared = _make_chunk("doc1.txt", 0)
    dense_results = [
        {"document": shared, "score": 0.9},
    ]
    sparse_results = [
        {"document": shared, "score": 10.0},
    ]

    hybrid = HybridRetrieval(rrf_k=60)
    results = hybrid.fuse(dense_results, sparse_results, top_k=1)

    assert len(results) == 1


def test_fuse_scores_descending():
    dense_results = [
        _dense_result("doc1.txt", 0, 0.9),
        _dense_result("doc2.txt", 0, 0.8),
        _dense_result("doc3.txt", 0, 0.7),
        _dense_result("doc4.txt", 0, 0.6),
    ]
    sparse_results = [
        _sparse_result("doc4.txt", 0, 8.0),
        _sparse_result("doc3.txt", 0, 6.0),
        _sparse_result("doc2.txt", 0, 4.0),
        _sparse_result("doc1.txt", 0, 2.0),
    ]

    hybrid = HybridRetrieval(rrf_k=60)
    results = hybrid.fuse(dense_results, sparse_results, top_k=4)

    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_fuse_rrf_k_effect():
    doc1 = _make_chunk("doc1.txt", 0)
    doc2 = _make_chunk("doc2.txt", 0)

    dense_results = [{"document": doc1, "score": 0.9}]
    sparse_results = [{"document": doc2, "score": 10.0}]

    hybrid = HybridRetrieval(rrf_k=60)
    results = hybrid.fuse(dense_results, sparse_results, top_k=2)
    dense_rrf = 1 / (0 + 1 + 60)
    sparse_rrf = 1 / (0 + 1 + 60)

    for r in results:
        if r["document"]["metadata"]["source"] == "doc1.txt":
            assert r["score"] == pytest.approx(dense_rrf)
        else:
            assert r["score"] == pytest.approx(sparse_rrf)

def test_rrf_score_accumulation():

    shared = _make_chunk("doc1.txt", 0)
    dense = [
        {
            "document": shared,
            "score": 0.9
        }
    ]
    sparse = [
        {
            "document": shared,
            "score": 10.0
        }
    ]

    hybrid = HybridRetrieval(rrf_k=60)

    results = hybrid.fuse(
        dense,
        sparse,
        top_k=1
    )
    expected = (
        1/(60+1)
        +
        1/(60+1)
    )

    assert results[0]["score"] == pytest.approx(expected)