import pytest
from pathlib import Path

from app.ingestion.loader import load_documents
from app.ingestion.chunking import chunk_documents
from app.retrieval.sparse import SparseRetriever

def test_end_to_end_sparse_retrieval(tmp_path):
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

    retriever = SparseRetriever()
    retriever.build_index(chunks)

    query1 = "Python programming language"
    results1 = retriever.search(query1, top_k=3)
    assert len(results1) >= 1
    assert results1[0]["score"] > 0.0
    assert any(
        result["document"]["metadata"]["source"] == str(doc1)
        for result in results1
    )
    scores1 = [r["score"] for r in results1]
    assert scores1 == sorted(scores1, reverse=True)

    query2 = "basketball sport"
    results2 = retriever.search(query2, top_k=3)
    assert len(results2) >= 1
    assert results2[0]["score"] > 0.0
    assert any(
        result["document"]["metadata"]["source"] == str(doc2)
        for result in results2
    )
    scores2 = [r["score"] for r in results2]
    assert scores2 == sorted(scores2, reverse=True)

    query3 = "cooking food techniques"
    results3 = retriever.search(query3, top_k=3)
    assert len(results3) >= 1
    assert results3[0]["score"] > 0.0
    assert any(
        result["document"]["metadata"]["source"] == str(doc3)
        for result in results3
    )
    scores3 = [r["score"] for r in results3]
    assert scores3 == sorted(scores3, reverse=True)
