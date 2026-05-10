from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal
import json
import random

import numpy as np


AggregationName = Literal["max", "sum_top_n"]


@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int = 1200
    overlap: int = 200


@dataclass(frozen=True)
class RetrievalConfig:
    top_k_chunks: int = 20
    query_batch_size: int = 8
    aggregate: AggregationName = "sum_top_n"
    aggregate_top_n: int = 3
    hit_ks: tuple[int, ...] = (1, 5, 10)


@dataclass(frozen=True)
class ExperimentConfig:
    data_root: Path
    output_dir: Path = Path("outputs")
    chroma_dir: Path = Path("chroma_db")
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    seed: int = 13
    debug_limit: int | None = None
    debug_source_limit: int | None = None
    sample_fraction: float | None = None
    reuse_existing_chroma: bool = True
    pca_dims: tuple[int, ...] = (384, 256, 128, 64)
    truncation_dims: tuple[int, ...] = (384, 256, 128, 64)
    chunking: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["data_root"] = str(self.data_root)
        payload["output_dir"] = str(self.output_dir)
        payload["chroma_dir"] = str(self.chroma_dir)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
