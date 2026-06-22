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
│   │   ├── dense.py      # Dense retrieval con FAISS + Sentence Transformers
│   │   ├── sparse.py     # Sparse retrieval con BM25 (rank-bm25)
│   │   └── hybrid.py     # Fusión híbrida (RRF, weighted sum)
|   ├── ingestion/
|   |   ├── chunking.py    # Transforma la data en chunks para el procesado
|   |   ├── loader.py      # Carga la informacion desde documentos .txt y .md
|   |   ├── embeddings.py  # Genera el embeddings de los chunks
│   ├── reranker/
│   │   └── cross_encoder.py  # Cross-encoder reranker (sentence-transformers)
│   └── tests/            # Tests unitarios
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

### Primera ejecución

Los modelos de Hugging Face (`all-MiniLM-L6-v2`, `cross-encoder/ms-marco-MiniLM-L-6-v2`) utilizados necesitan descargarse la primera vez ejecutados. Para ello, comentá temporalmente `os.environ["HF_HUB_OFFLINE"] = "1"` en `app/main.py` o seteá un `HF_TOKEN`:

```bash
$env:HF_TOKEN = "hf_..."
```

Una vez descargados ya son cacheados para luego correr completamente offline.

## Uso

```bash
python -m app.main "ruta/a/documentos" "tu pregunta?"
```

Ejemplos:

```bash
python -m app.main "./data_samples" "Quien creo Python?"
# Output: Answer: Guido van Rossum.

python -m app.main "./data_samples" "Cual es la capital de Chile?"
# Output: Answer: Santiago.
```

El pipeline carga documentos `.txt` y `.md` del directorio, los procesa localmente (chunking → embeddings → FAISS + BM25 → fusión híbrida → cross-encoder reranking → generación con Ollama/llama3.2) y devuelve la respuesta por terminal.

## Pipeline RAG Híbrido

1. **Dense Retrieval**: Embeddings densos con Sentence Transformers + FAISS
2. **Sparse Retrieval**: BM25 con rank-bm25
3. **Hybrid Fusion**: Reciprocal Rank Fusion (RRF) o Weighted Sum
4. **Cross-Encoder Reranker**: Re-ranking con Cross-Encoder (sentence-transformers)
3. **Top-K Context Selection**: Selección del top context y generación con LLM
