# Hybrid RAG Pipeline

Sistema RAG híbrido que combina búsqueda densa (dense) y dispersa (sparse) con fusión y reranking cross-encoder.

## Objetivo

Construir un sistema Retrieval-Augmented Generation(RAG), con la finalidad de recuperar información de una base de datos acotada mediante una estrategia híbrida combinando búsqueda densa y dispersa, seguida de reranking y evaluación cuantitativa.

## Arquitectura

```
┌─────────────┐     ┌─────────────┐     ┌──────────┐     ┌──────────┐
│   Query     │────▶│  Dense      │    │          │     │  Top-K   │
│             │     │  Retrieval  │     │  Hybrid  │     │  Context │
│             │────▶│  (FAISS)    │────▶│  Fusion │────▶│ Selection│
│             │     │             │     │ (RRF)│   │     │     │    │ 
│             │────▶│  Sparse     │    │          │      │    │     │
│             │     │  Retrieval  │     │          │     │    ▼     │
│             │     │  (BM25)     │     │          │     │          │
└─────────────┘     └─────────────┘     └──────────┘     │  Cross   │
                                                         │  Encoder │
                                                         │  Rerank  │
                                                         │          │
                                                         └────┬─────┘
                                                              │
                                                              ▼
                                                     ┌──────────────┐
                                                     │    LLM       │
                                                     │  Generation  │
                                                     └──────────────┘
```

## Estructura del Proyecto

```
rag-pipeline/
├── app/
│   ├── retrieval/
│   │   ├── dense.py         # Dense retrieval con FAISS
│   │   ├── sparse.py        # Sparse retrieval con BM25
│   │   └── hybrid.py        # Fusión híbrida (RRF)
│   ├── ingestion/
│   │   ├── chunking.py      # Chunking de documentos
│   │   ├── loader.py        # Carga de .txt y .md
│   │   └── embeddings.py    # Embeddings con Sentence Transformers
│   ├── reranker/
│   │   └── cross_encoder.py # Cross-encoder reranker
│   ├── generation/
│   │   └── generator.py     # Generación con Ollama + llama3.2
│   ├── tests/               # Tests unitarios e integración
│   └── main.py              # CLI Simple
├── data_samples/            # Documentos de ejemplo
├── requirements.txt
└── README.md
```

## Pre-requisitos

### Ollama

Instalar [Ollama](https://ollama.com/) y descargar el modelo local:

```bash
ollama pull llama3.2
```

### Dependencias Python

```bash
pip install -r requirements.txt
```

### Modelos de Hugging Face

Los modelos `all-MiniLM-L6-v2` y `cross-encoder/ms-marco-MiniLM-L-6-v2` se descargan automáticamente la primera vez desde Hugging Face Hub. Para que se descarguen, **comentá temporalmente** `os.environ["HF_HUB_OFFLINE"] = "1"` en `app/main.py`, o alternativamente seteá un token:

```bash
$env:HF_TOKEN = "hf_..."
```

Una vez cacheados en `~/.cache/huggingface/`, ya no necesitan internet y el pipeline corre 100% offline.

### Tests

```bash
python -m pytest app/tests/ -v
```

## Uso

El pipeline acepta documentos en formato `.txt` y `.md`. Pasá un directorio con los archivos y la consulta:

```bash
python -m app.main "ruta/a/mis/documentos" "¿Quién creó Python?"
```

Ejemplo real:

```bash
python -m app.main "ruta/con/archivos" "¿Cuál es la capital de Chile?"
# Answer: Santiago.
```

## Pipeline RAG Híbrido

1. **Dense Retrieval**: Embeddings densos con Sentence Transformers + FAISS
2. **Sparse Retrieval**: BM25 con rank-bm25
3. **Hybrid Fusion**: Reciprocal Rank Fusion (RRF)
4. **Cross-Encoder Reranker**: Re-ranking con Cross-Encoder (sentence-transformers)
5. **LLM Generation**: Los fragmentos rerankeados se pasan como contexto a un prompt. Ollama ejecuta `llama3.2` en local (`localhost:11434`) y genera una respuesta en español, completamente offline únicamente basada en la database proporcionada.
