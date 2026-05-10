from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from .chunking import Chunk


class ChromaVectorStore:
    def __init__(self, persist_dir: Path, collection_name: str, rebuild: bool = True) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        if rebuild:
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray, batch_size: int = 1000) -> None:
        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            batch_chunks = chunks[start:end]
            self.collection.add(
                ids=[chunk.chunk_id for chunk in batch_chunks],
                embeddings=embeddings[start:end].astype(float).tolist(),
                documents=[chunk.text for chunk in batch_chunks],
                metadatas=[
                    {
                        "source_document_id": chunk.document_id,
                        "chunk_id": chunk.chunk_id,
                        "start": chunk.start,
                        "end": chunk.end,
                    }
                    for chunk in batch_chunks
                ],
            )

    def count(self) -> int:
        return int(self.collection.count())

    def query(
        self,
        embeddings: np.ndarray,
        n_results: int,
        batch_size: int = 8,
    ) -> list[list[dict[str, Any]]]:
        out: list[list[dict[str, Any]]] = []
        safe_batch_size = max(1, batch_size)
        for start in range(0, len(embeddings), safe_batch_size):
            batch = embeddings[start : start + safe_batch_size]
            results = self.collection.query(
                query_embeddings=batch.astype(float).tolist(),
                n_results=n_results,
                include=["metadatas", "distances"],
            )
            ids = results.get("ids", [])
            metadatas = results.get("metadatas", [])
            distances = results.get("distances", [])
            for row_ids, row_metas, row_dists in zip(ids, metadatas, distances):
                row: list[dict[str, Any]] = []
                for item_id, metadata, distance in zip(row_ids, row_metas, row_dists):
                    row.append(
                        {
                            "chunk_id": item_id,
                            "source_document_id": metadata["source_document_id"],
                            "score": 1.0 - float(distance),
                            "distance": float(distance),
                        }
                    )
                out.append(row)
        return out
