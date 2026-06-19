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
│   ├── reranker/
│   │   └── cross_encoder.py  # Cross-encoder reranker (sentence-transformers)
│   ├── evaluation/
│   │   ├── metrics.py    # Métricas: Recall@k, MRR, NDCG, MAP
    │   └── benchmark.py  # Benchmarking y evaluación
│   ├── api/
│   │   └── routes.py     # FastAPI routes
│   ├── notebooks/        # Jupyter notebooks para experimentación (no fijo)
│   └── tests/            # Tests unitarios
├── data/
│   ├── raw/              # Datos crudos (PDFs, txt, etc.)
│   └── processed/        # Datos procesados (chunks, embeddings, índices)
├── requirements.txt
└── README.md
```

## Instalación

```bash
pip install -r requirements.txt
```

## Pipeline RAG Híbrido

1. **Dense Retrieval**: Embeddings densos con Sentence Transformers + FAISS
2. **Sparse Retrieval**: BM25 con rank-bm25
3. **Hybrid Fusion**: Reciprocal Rank Fusion (RRF) o Weighted Sum
4. **Cross-Encoder Reranker**: Re-ranking con Cross-Encoder (sentence-transformers)
3. **Top-K Context Selection**: Selección del top context y generación con LLM