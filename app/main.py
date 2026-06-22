import argparse
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"

from app.ingestion.loader import load_documents
from app.ingestion.chunking import chunk_documents
from app.ingestion.embeddings import EmbeddingGenerator
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever
from app.retrieval.hybrid import HybridRetrieval
from app.reranker.cross_encoder import CrossEncoderReranker
from app.generation.generator import LLMGenerator


def main():
    parser = argparse.ArgumentParser(
        description="RAG Pipeline: answer questions from documents in a directory."
    )
    parser.add_argument(
        "directory",
        help="Path to directory containing .txt or .md documents.",
    )
    parser.add_argument(
        "query",
        help="Question to answer based on the documents.",
    )
    args = parser.parse_args()

    documents = load_documents(args.directory)
    if not documents:
        print("No documents found in the specified directory.")
        sys.exit(1)

    print(f"Loaded {len(documents)} document(s).")

    chunks = chunk_documents(documents)
    print(f"Split into {len(chunks)} chunk(s).")

    print("Generating embeddings...")
    embedder = EmbeddingGenerator()
    chunk_embeddings = embedder.embed_chunks(chunks)

    print("Building dense index...")
    dense = DenseRetriever()
    dense.build_index(chunks, chunk_embeddings)

    print("Building sparse index...")
    sparse = SparseRetriever()
    sparse.build_index(chunks)

    print(f"Searching for: {args.query}")
    query_embedding = embedder.embed_query(args.query)

    dense_results = dense.search(query_embedding, top_k=5)
    sparse_results = sparse.search(args.query, top_k=5)

    hybrid = HybridRetrieval()
    fused = hybrid.fuse(dense_results, sparse_results, top_k=5)

    print("Reranking results...")
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(args.query, fused, top_k=3)

    if not reranked:
        print("No relevant results found. Cannot generate an answer.")
        sys.exit(1)

    print("Generating answer...")
    generator = LLMGenerator()
    answer = generator.generate(args.query, reranked)

    print(f"\n{'='*60}")
    print(f"Answer: {answer}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
