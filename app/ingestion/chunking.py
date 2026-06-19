# Responsability: Transform data into smaller chunks to be processed by the embedding model

# This module provides functions to chunk documents into smaller pieces and an embedding generator class to create embeddings from those chunks.
def chunk_document(document: dict, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    chunks = []
    content = document['content']
    
    start = 0
    chunk_id = 0

    while start + overlap < len(content):
        end = start + chunk_size
        chunk_content = content[start:end]
        chunks.append({
            'content': chunk_content,
            'metadata': {
                'source': document['metadata']['source'],
                'chunk_id': chunk_id
            }
        })
        start += chunk_size - overlap
        chunk_id += 1

    return chunks

# This function takes a list of documents and applies the chunking process to each document, returning a combined list of all chunks.
def chunk_documents(documents: list, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    all_chunks = []
    for document in documents:
        chunks = chunk_document(document, chunk_size, overlap)
        all_chunks.extend(chunks)
    return all_chunks