from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from .embeddings import l2_normalize


@dataclass(frozen=True)
class VectorCondition:
    method: str
    name: str
    target_dim: int | None = None


class VectorTransformer:
    storage_dtype = "float32"
    bytes_per_scalar = 4

    def fit(self, source_embeddings: np.ndarray) -> "VectorTransformer":
        return self

    def transform_for_index(self, embeddings: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def storage_dimension(self, original_dim: int) -> int:
        return original_dim


class FullTransformer(VectorTransformer):
    def transform_for_index(self, embeddings: np.ndarray) -> np.ndarray:
        return l2_normalize(embeddings.astype(np.float32, copy=False))


class TruncationTransformer(VectorTransformer):
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def transform_for_index(self, embeddings: np.ndarray) -> np.ndarray:
        return l2_normalize(embeddings[:, : self.dim].astype(np.float32, copy=False))

    def storage_dimension(self, original_dim: int) -> int:
        return min(self.dim, original_dim)


class PCATransformer(VectorTransformer):
    def __init__(self, dim: int, seed: int) -> None:
        self.requested_dim = dim
        self.seed = seed
        self.pca: PCA | None = None
        self.dim = dim

    def fit(self, source_embeddings: np.ndarray) -> "PCATransformer":
        max_dim = min(source_embeddings.shape[0], source_embeddings.shape[1], self.requested_dim)
        self.dim = max_dim
        self.pca = PCA(n_components=max_dim, random_state=self.seed)
        self.pca.fit(source_embeddings)
        return self

    def transform_for_index(self, embeddings: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("PCA transformer must be fitted before transform")
        return l2_normalize(self.pca.transform(embeddings).astype(np.float32, copy=False))

    def storage_dimension(self, original_dim: int) -> int:
        return min(self.dim, original_dim)


class ScalarInt8Quantizer(VectorTransformer):
    storage_dtype = "int8"
    bytes_per_scalar = 1

    def __init__(self) -> None:
        self.scale: np.ndarray | None = None

    def fit(self, source_embeddings: np.ndarray) -> "ScalarInt8Quantizer":
        max_abs = np.max(np.abs(source_embeddings), axis=0)
        self.scale = np.where(max_abs == 0, 1.0, max_abs / 127.0).astype(np.float32)
        return self

    def quantize(self, embeddings: np.ndarray) -> np.ndarray:
        if self.scale is None:
            raise RuntimeError("Quantizer must be fitted before quantize")
        quantized = np.round(embeddings / self.scale)
        return np.clip(quantized, -127, 127).astype(np.int8)

    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        if self.scale is None:
            raise RuntimeError("Quantizer must be fitted before dequantize")
        return (quantized.astype(np.float32) * self.scale).astype(np.float32)

    def transform_for_index(self, embeddings: np.ndarray) -> np.ndarray:
        # Chroma expects float embeddings. We store/account int8 vectors, then
        # dequantize for Chroma retrieval to keep the vector DB path consistent.
        return l2_normalize(self.dequantize(self.quantize(embeddings)))


def build_transformer(condition: VectorCondition, original_dim: int, seed: int) -> VectorTransformer:
    if condition.method == "full":
        return FullTransformer()
    if condition.method == "pca":
        if condition.target_dim is None:
            raise ValueError("PCA condition requires target_dim")
        return PCATransformer(min(condition.target_dim, original_dim), seed)
    if condition.method == "truncation":
        if condition.target_dim is None:
            raise ValueError("Truncation condition requires target_dim")
        return TruncationTransformer(min(condition.target_dim, original_dim))
    if condition.method == "int8":
        return ScalarInt8Quantizer()
    raise ValueError(f"Unknown method: {condition.method}")

