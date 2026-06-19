# Responsibility: Read data from a source and return it in uniform format

from pathlib import Path

def load_documents(source: str) -> list[dict]:
    documents = []
    data_dir = Path(source)
    for file_path in data_dir.iterdir():
        if file_path.is_file() and file_path.suffix in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    'content': content,
                    'metadata': {
                        'source': str(file_path)
                    }
                })

    return documents
