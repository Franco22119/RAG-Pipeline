# Responsibility: Transform data into smaller chunks to be processed by the embedding model

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

def chunk_documents(documents: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    all_chunks = []
    for document in documents:
        chunks = chunk_document(document, chunk_size, overlap)
        all_chunks.extend(chunks)
    return all_chunks