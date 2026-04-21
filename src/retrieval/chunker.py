import os
import hashlib
import time
from dataclasses import dataclass
from typing import Optional


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@dataclass
class Chunk:
    text: str
    metadata: dict

    def to_dict(self) -> dict:
        return {"text": self.text, "metadata": self.metadata}

    @property
    def chunk_id(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]


class HybridChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict, file_type: str) -> list[Chunk]:
        if file_type == ".pdf":
            return self._chunk_pdf(text, metadata)
        elif file_type in (".html", ".htm", ".md", ".txt"):
            return self._chunk_structured(text, metadata)
        elif file_type == ".csv":
            return self._chunk_csv(text, metadata)
        else:
            return self._chunk_recursive(text, metadata)

    def _chunk_recursive(self, text: str, metadata: dict) -> list[Chunk]:
        chunks = []
        start = 0
        chunk_index = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            chunk_metadata = {**metadata, "chunk_start": start, "chunk_end": end, "chunk_index": chunk_index}
            chunks.append(Chunk(text=chunk_text, metadata=chunk_metadata))
            start += self.chunk_size - self.overlap
            chunk_index += 1
        return chunks

    def _chunk_pdf(self, text: str, metadata: dict) -> list[Chunk]:
        return self._chunk_recursive(text, metadata)

    def _chunk_structured(self, text: str, metadata: dict) -> list[Chunk]:
        lines = text.split("\n")
        sections = []
        current_section = []
        current_heading = ""

        for line in lines:
            stripped = line.strip()
            is_heading = (
                stripped.startswith("# ")
                or stripped.isupper()
                or (len(stripped) < 60 and stripped and not stripped[0].isdigit())
                or (":" in stripped and len(stripped) < 80 and stripped[-1] in ".:")
            )
            if is_heading and not any(c in stripped for c in ",.!?;"):
                if current_section:
                    sections.append((current_heading, "\n".join(current_section)))
                current_heading = stripped
                current_section = []
            else:
                current_section.append(line)

        if current_section:
            sections.append((current_heading, "\n".join(current_section)))

        if not sections:
            return self._chunk_recursive(text, metadata)

        chunks = []
        for heading, content in sections:
            content_chunks = self._chunk_recursive(content, metadata)
            for c in content_chunks:
                if heading:
                    c.text = f"[Section: {heading}]\n{c.text}"
            chunks.extend(content_chunks)
        return chunks

    def _chunk_csv(self, text: str, metadata: dict) -> list[Chunk]:
        lines = [l for l in text.strip().split("\n") if l.strip()]
        if not lines:
            return []
        chunks = []
        for i, row in enumerate(lines, 1):
            chunk_text = row.strip()
            if not chunk_text:
                continue
            chunk_metadata = {
                **metadata,
                "row_index": i,
                "source_file": metadata.get("source", "unknown"),
                "source_page": str(i),
            }
            chunks.append(Chunk(text=chunk_text, metadata=chunk_metadata))
        return chunks
