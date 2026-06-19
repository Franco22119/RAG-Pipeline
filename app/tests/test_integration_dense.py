import pytest
from pathlib import Path

from app.ingestion.loader import load_documents
from app.ingestion.chunking import chunk_documents
from app.ingestion.embeddings import EmbeddingGenerator
from app.retrieval.dense import DenseRetriever


@pytest.fixture(scope="module")
def embedding_generator():
    return EmbeddingGenerator()

# Test integration of loading, chunking, embedding, and retrieval
def test_end_to_end_dense(tmp_path, embedding_generator):
    # Create a examples documents in the temporary directory
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

    # Load
    documents = load_documents(str(tmp_path))
    assert len(documents) == 3

    # Chunk
    chunks = chunk_documents(documents, chunk_size=100, overlap=20)
    assert len(chunks) >= 3

    # Embed
    embeddings = embedding_generator.embed_chunks(chunks)
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] == 384

    # Retrieve
    retriever = DenseRetriever()
    retriever.build_index(chunks, embeddings)

    # Test retrieval with a query related to Python programming

    # First query about Python programming
    query = "What is Python used for?"
    query_embedding = embedding_generator.embed_query(query)

    results = retriever.search(query_embedding, top_k=3)
    assert len(results) >= 1

    assert results[0]["score"] > 0.0
    assert any(
        result["document"]["metadata"]["source"] == str(doc1)
        for result in results
    )
    scores = [
        result["score"]
        for result in results
    ]
    assert scores == sorted(
        scores,
        reverse=True
    )

    # Second query about basketball
    query2 = "Who invented basketball?"
    query_embedding2 = embedding_generator.embed_query(query2)
    results2 = retriever.search(query_embedding2, top_k=3)
    assert len(results2) >= 1
    assert results2[0]["score"] > 0.0
    assert any(
        result["document"]["metadata"]["source"] == str(doc2)
        for result in results2
    )
    scores2 = [
        result["score"]
        for result in results2
    ]
    assert scores2 == sorted(
        scores2,
        reverse=True
    )

    # Third query about cooking
    query3 = "What are some cooking techniques?"
    query_embedding3 = embedding_generator.embed_query(query3)
    results3 = retriever.search(query_embedding3, top_k=3)
    assert len(results3) >= 1
    assert results3[0]["score"] > 0.0
    assert any(
        result["document"]["metadata"]["source"] == str(doc3)
        for result in results3  
    )
    scores3 = [
        result["score"]
        for result in results3
    ]
    assert scores3 == sorted(
        scores3,
        reverse=True
    )
