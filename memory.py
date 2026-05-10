from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class MemoryStats:
    method: str
    condition: str
    embedding_dim: int
    storage_dtype: str
    bytes_per_scalar: int
    bytes_per_vector: int
    num_source_vectors: int
    estimated_total_vector_bytes: int
    compression_ratio_vs_full: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_memory_stats(
    method: str,
    condition: str,
    embedding_dim: int,
    storage_dtype: str,
    bytes_per_scalar: int,
    num_source_vectors: int,
    baseline_bytes_per_vector: int,
) -> MemoryStats:
    bytes_per_vector = embedding_dim * bytes_per_scalar
    total = bytes_per_vector * num_source_vectors
    ratio = baseline_bytes_per_vector / bytes_per_vector
    return MemoryStats(
        method=method,
        condition=condition,
        embedding_dim=embedding_dim,
        storage_dtype=storage_dtype,
        bytes_per_scalar=bytes_per_scalar,
        bytes_per_vector=bytes_per_vector,
        num_source_vectors=num_source_vectors,
        estimated_total_vector_bytes=total,
        compression_ratio_vs_full=ratio,
    )

