from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd

from .chroma_store import ChromaVectorStore
from .chunking import TextChunker
from .compression import VectorCondition, build_transformer
from .config import ExperimentConfig, set_reproducible_seed
from .embeddings import Embedder
from .evaluation import evaluate_retrieval
from .memory import compute_memory_stats
from .pan11 import load_pan11_dataset
from .plotting import plot_results, write_markdown_summary


def default_conditions(original_dim: int, config: ExperimentConfig, baseline_only: bool) -> list[VectorCondition]:
    conditions = [VectorCondition("full", "full")]
    if baseline_only:
        return conditions
    pca_dims = sorted({dim for dim in config.pca_dims if dim <= original_dim}, reverse=True)
    trunc_dims = sorted({dim for dim in config.truncation_dims if dim <= original_dim}, reverse=True)
    conditions.extend(VectorCondition("pca", f"pca_{dim}", dim) for dim in pca_dims)
    conditions.extend(VectorCondition("truncation", f"truncation_{dim}", dim) for dim in trunc_dims)
    conditions.append(VectorCondition("int8", "int8_full_dim", original_dim))
    return conditions


def _collection_name(condition: VectorCondition, original_dim: int) -> str:
    if condition.method == "truncation" and condition.target_dim == original_dim:
        return "full"
    return condition.name.replace("-", "_")


def _source_embedding_cache_path(config: ExperimentConfig) -> Path:
    payload = {
        "data_root": str(config.data_root.resolve()),
        "model_name": config.model_name,
        "debug_limit": config.debug_limit,
        "debug_source_limit": config.debug_source_limit,
        "sample_fraction": config.sample_fraction,
        "seed": config.seed,
        "chunk_size": config.chunking.chunk_size,
        "overlap": config.chunking.overlap,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return config.output_dir / "cache" / f"source_embeddings_{digest}.npy"


def _load_or_encode_source_embeddings(
    embedder: Embedder,
    source_chunks,
    cache_path: Path,
) -> np.ndarray:
    if cache_path.exists():
        embeddings = np.load(cache_path)
        if embeddings.shape[0] == len(source_chunks):
            return embeddings.astype(np.float32, copy=False)

    embeddings = embedder.encode([chunk.text for chunk in source_chunks])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings.astype(np.float32, copy=False))
    return embeddings


def run_experiment(config: ExperimentConfig, baseline_only: bool = False) -> pd.DataFrame:
    set_reproducible_seed(config.seed)
    output_dir = config.output_dir
    tables_dir = output_dir / "tables"
    per_doc_dir = output_dir / "per_document"
    manifests_dir = output_dir / "manifests"
    for directory in (tables_dir, per_doc_dir, manifests_dir, config.chroma_dir):
        directory.mkdir(parents=True, exist_ok=True)

    config.save(manifests_dir / "config.json")
    dataset = load_pan11_dataset(
        config.data_root,
        config.debug_limit,
        config.debug_source_limit,
        config.sample_fraction,
        config.seed,
    )
    chunker = TextChunker(config.chunking.chunk_size, config.chunking.overlap)
    source_chunks = chunker.chunk_documents(dataset.source_documents)
    embedder = Embedder(config.model_name)

    source_embeddings = _load_or_encode_source_embeddings(
        embedder,
        source_chunks,
        _source_embedding_cache_path(config),
    )
    original_dim = source_embeddings.shape[1]
    baseline_bytes_per_vector = original_dim * 4
    metrics_rows: list[dict[str, object]] = []

    for condition in default_conditions(original_dim, config, baseline_only):
        transformer = build_transformer(condition, original_dim, config.seed).fit(source_embeddings)
        index_embeddings = transformer.transform_for_index(source_embeddings)
        effective_dim = transformer.storage_dimension(original_dim)
        collection_name = _collection_name(condition, original_dim)
        is_full_dim_truncation = condition.method == "truncation" and condition.target_dim == original_dim
        store = ChromaVectorStore(config.chroma_dir, collection_name, rebuild=False)
        existing_count = store.count()
        if (config.reuse_existing_chroma or is_full_dim_truncation) and existing_count == len(source_chunks):
            indexing_time = 0.0
        else:
            store = ChromaVectorStore(config.chroma_dir, collection_name, rebuild=True)
            start = time.perf_counter()
            store.add_chunks(source_chunks, index_embeddings)
            indexing_time = time.perf_counter() - start

        result = evaluate_retrieval(
            dataset.suspicious_documents,
            dataset.ground_truth,
            chunker,
            embedder,
            transformer,
            store,
            config.retrieval,
        )
        memory = compute_memory_stats(
            method=condition.method,
            condition=condition.name,
            embedding_dim=effective_dim,
            storage_dtype=transformer.storage_dtype,
            bytes_per_scalar=transformer.bytes_per_scalar,
            num_source_vectors=len(source_chunks),
            baseline_bytes_per_vector=baseline_bytes_per_vector,
        )
        row = {
            "method": condition.method,
            "condition": condition.name,
            **result.metrics,
            **memory.to_dict(),
            "indexing_time_seconds": indexing_time,
            "query_time_seconds": result.query_time_seconds,
            "num_source_documents": len(dataset.source_documents),
            "num_source_chunks": len(source_chunks),
            "num_suspicious_documents": len(dataset.suspicious_documents),
        }
        metrics_rows.append(row)
        result.per_document.to_csv(per_doc_dir / f"{condition.name}_retrieval_results.csv", index=False)
        (manifests_dir / f"{condition.name}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    metrics = pd.DataFrame(metrics_rows)
    metrics_path = tables_dir / "compression_comparison.csv"
    metrics.to_csv(metrics_path, index=False)
    metrics[metrics["method"] == "full"].to_csv(tables_dir / "baseline_results.csv", index=False)
    metrics[
        ["condition", "method", "embedding_dim", "bytes_per_vector", "estimated_total_vector_bytes", "hit@1", "hit@5", "hit@10"]
    ].to_csv(tables_dir / "memory_vs_hit.csv", index=False)
    metrics[
        ["condition", "method", "indexing_time_seconds", "query_time_seconds", "hit@1", "hit@5", "hit@10"]
    ].to_csv(tables_dir / "latency_vs_hit.csv", index=False)
    metrics[
        [
            "condition",
            "method",
            "embedding_dim",
            "storage_dtype",
            "bytes_per_scalar",
            "bytes_per_vector",
            "num_source_vectors",
            "estimated_total_vector_bytes",
            "compression_ratio_vs_full",
        ]
    ].to_csv(tables_dir / "vector_size_comparison.csv", index=False)

    plot_results(metrics_path, output_dir)
    write_markdown_summary(metrics_path, output_dir / "RESULTS.md")
    return metrics
