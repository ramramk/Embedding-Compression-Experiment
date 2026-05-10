from __future__ import annotations

from dataclasses import dataclass

from .pan11 import Document


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    start: int
    end: int


class TextChunker:
    def __init__(self, chunk_size: int = 1200, overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document: Document) -> list[Chunk]:
        text = " ".join(document.text.split())
        if not text:
            return []
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        start = 0
        index = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}::chunk-{index:05d}",
                        document_id=document.doc_id,
                        text=chunk_text,
                        start=start,
                        end=end,
                    )
                )
            if end == len(text):
                break
            start += step
            index += 1
        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks

