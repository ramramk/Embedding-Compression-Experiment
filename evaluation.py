from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import time

import numpy as np
import pandas as pd

from .chunking import Chunk, TextChunker
from .chroma_store import ChromaVectorStore
from .config import RetrievalConfig
from .embeddings import Embedder
from .pan11 import Document


@dataclass(frozen=True)
class RetrievalOutput:
    metrics: dict[str, float]
    per_document: pd.DataFrame
    query_time_seconds: float


def aggregate_results(
    chunk_results: list[list[dict[str, object]]],
    strategy: str,
    top_n: int,
) -> list[tuple[str, float]]:
    scores: dict[str, list[float]] = defaultdict(list)
    for row in chunk_results:
        for hit in row:
            scores[str(hit["source_document_id"])].append(float(hit["score"]))
    aggregate_scores: dict[str, float] = {}
    for source_id, values in scores.items():
        values = sorted(values, reverse=True)
        if strategy == "max":
            aggregate_scores[source_id] = values[0]
        elif strategy == "sum_top_n":
            aggregate_scores[source_id] = sum(values[:top_n])
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")
    return sorted(aggregate_scores.items(), key=lambda item: item[1], reverse=True)


def evaluate_retrieval(
    suspicious_documents: list[Document],
    ground_truth: dict[str, set[str]],
    chunker: TextChunker,
    embedder: Embedder,
    transformer,
    store: ChromaVectorStore,
    retrieval_config: RetrievalConfig,
) -> RetrievalOutput:
    rows: list[dict[str, object]] = []
    hits = {k: 0 for k in retrieval_config.hit_ks}
    total_query_time = 0.0

    for document in suspicious_documents:
        chunks = chunker.chunk_document(document)
        if not chunks:
            continue
        query_embeddings = transformer.transform_for_index(embedder.encode([chunk.text for chunk in chunks]))
        start = time.perf_counter()
        chunk_results = store.query(
            query_embeddings,
            retrieval_config.top_k_chunks,
            retrieval_config.query_batch_size,
        )
        total_query_time += time.perf_counter() - start
        ranking = aggregate_results(
            chunk_results,
            retrieval_config.aggregate,
            retrieval_config.aggregate_top_n,
        )
        ranked_ids = [source_id for source_id, _ in ranking]
        true_sources = ground_truth.get(document.doc_id, set())
        row = {
            "suspicious_document_id": document.doc_id,
            "true_source_document_ids": "|".join(sorted(true_sources)),
            "ranked_source_document_ids": "|".join(ranked_ids),
            "top_score": ranking[0][1] if ranking else np.nan,
        }
        for k in retrieval_config.hit_ks:
            hit = bool(true_sources.intersection(ranked_ids[:k]))
            hits[k] += int(hit)
            row[f"hit@{k}"] = hit
        rows.append(row)

    denominator = max(len(rows), 1)
    metrics = {f"hit@{k}": hits[k] / denominator for k in retrieval_config.hit_ks}
    metrics["num_evaluated_suspicious_documents"] = float(len(rows))
    return RetrievalOutput(metrics, pd.DataFrame(rows), total_query_time)
