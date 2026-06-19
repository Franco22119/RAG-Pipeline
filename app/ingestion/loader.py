# Responsability: Read data from a source and return it in uniform format

from pathlib import Path

# This module defines the load_documents function, which reads text files from a specified directory 
# and returns a list of documents in a uniform format (dictionaries with 'content' and 'metadata' keys). 
# The function filters for files with .txt and .md extensions and reads their content into memory.
def load_documents(source: str):
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
