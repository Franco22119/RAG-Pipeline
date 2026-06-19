from pathlib import Path

from app.ingestion.loader import load_documents
from app.ingestion.chunking import chunk_document, chunk_documents
from app.ingestion.embeddings import EmbeddingGenerator

def test_load_documents():
    test_dir = Path('test_data')
    test_dir.mkdir(exist_ok=True)
    (test_dir / 'doc1.txt').write_text('This is the first document.')
    (test_dir / 'doc2.md').write_text('This is the second document.')

    documents = load_documents(str(test_dir))
    
    assert len(documents) == 2
    assert documents[0]['content'] == 'This is the first document.'
    assert documents[1]['content'] == 'This is the second document.'

    for file in test_dir.iterdir():
        file.unlink()
    test_dir.rmdir()

def test_chunk_document():
    document = {
        'content': 'This is a test document that will be chunked into smaller pieces.',
        'metadata': {'source': 'test_doc.txt'}
    }
    chunks = chunk_document(document, chunk_size=20, overlap=5)
    
    assert len(chunks) == 4
    assert chunks[0]['content'] == 'This is a test docum'
    assert chunks[1]['content'] == 'document that will b'
    assert chunks[2]['content'] == 'ill be chunked into '
    assert chunks[3]['content'] == 'into smaller pieces.'
    
def test_chunk_documents():
    documents = [
        {
            'content': 'This is the first document that will be chunked.',
            'metadata': {'source': 'doc1.txt'}
        },
        {
            'content': 'This is the second document that will also be chunked.',
            'metadata': {'source': 'doc2.txt'}
        }
    ]
    chunks = chunk_documents(documents, chunk_size=20, overlap=5)
    
    assert len(chunks) == 7
    assert chunks[0]['content'] == 'This is the first do'
    assert chunks[4]['content'] == 'ond document that wi'

def test_embedding_generator():
    generator = EmbeddingGenerator()
    chunks = [
        {'content': 'This is the first chunk.'},
        {'content': 'This is the second chunk.'}
    ]
    embeddings = generator.embed_chunks(chunks)
    
    assert embeddings.shape == (2, 384)  # Assuming the model produces 384-dimensional embeddings

    query_embedding = generator.embed_query('What is the first chunk?')
    assert query_embedding.shape == (384,)