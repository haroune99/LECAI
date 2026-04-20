import os
import re
import hashlib
from typing import Optional
from src.retrieval.chunker import HybridChunker
from src.retrieval.embedder import Embedder
from src.retrieval.bm25_index import BM25Index
from src.retrieval.vector_store import VectorStore
from src.retrieval.reranker import Reranker


DEFAULT_INDEX_DIR = "data/indexes"
DEFAULT_DATA_DIR = "data/raw"


def split_pdf_by_language(text: str) -> str:
    lines = text.split("\n")
    english_lines = []
    for line in lines:
        latin_chars = len(re.findall(r"[a-zA-Z]", line))
        total_chars = len(re.findall(r"[a-zA-Z\u4e00-\u9fff]", line))
        if total_chars > 0 and latin_chars / total_chars > 0.5:
            english_lines.append(line)
    return "\n".join(english_lines)


def parse_pdf(file_path: str, extract_english_only: bool = True) -> tuple[str, dict]:
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    text_parts = []
    metadata = {
        "source": os.path.basename(file_path),
        "file_type": ".pdf",
        "pages": len(reader.pages),
    }

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if extract_english_only:
            text = split_pdf_by_language(text)
        text_parts.append(text)
        metadata[f"page_{i}_text_len"] = len(text)

    return "\n".join(text_parts), metadata


def parse_csv(file_path: str) -> tuple[str, dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    metadata = {
        "source": os.path.basename(file_path),
        "file_type": ".csv",
        "rows": len(content.split("\n")) - 1,
    }
    return content, metadata


def parse_html(file_path: str) -> tuple[str, dict]:
    from bs4 import BeautifulSoup

    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    metadata = {
        "source": os.path.basename(file_path),
        "file_type": ".html",
        "url": "unknown",
    }
    return text, metadata


def parse_text(file_path: str) -> tuple[str, dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    metadata = {
        "source": os.path.basename(file_path),
        "file_type": os.path.splitext(file_path)[1],
    }
    return content, metadata


class DocumentIngestor:
    def __init__(
        self,
        index_dir: str = DEFAULT_INDEX_DIR,
        data_dir: str = DEFAULT_DATA_DIR,
    ):
        self.index_dir = index_dir
        self.data_dir = data_dir
        self.chunker = HybridChunker(chunk_size=512, overlap=64)
        self.embedder = Embedder()
        self.vector_store = VectorStore(index_dir=index_dir)
        self.bm25_index = BM25Index(index_dir=index_dir)
        self.reranker = Reranker()

    def ingest_file(self, file_path: str) -> int:
        file_hash = self._file_hash(file_path)
        metadata_hash_file = os.path.join(self.index_dir, "processed_hashes.txt")

        existing = set()
        if os.path.exists(metadata_hash_file):
            with open(metadata_hash_file, "r") as f:
                existing = set(f.read().splitlines())

        if file_hash in existing:
            return 0

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            text, base_metadata = parse_pdf(file_path)
        elif ext == ".csv":
            text, base_metadata = parse_csv(file_path)
        elif ext in (".html", ".htm"):
            text, base_metadata = parse_html(file_path)
        else:
            text, base_metadata = parse_text(file_path)

        base_metadata["document_hash"] = file_hash
        chunks = self.chunker.chunk(text, base_metadata, ext)

        if not chunks:
            return 0

        chunk_dicts = [c.to_dict() for c in chunks]
        embeddings = self.embedder.embed([c["text"] for c in chunk_dicts])

        self.vector_store.add_chunks(chunk_dicts, embeddings.tolist())
        self.bm25_index.build(chunk_dicts)
        self.bm25_index.save(name="bm25_main")
        self.vector_store.client.persist()

        with open(metadata_hash_file, "a") as f:
            f.write(file_hash + "\n")

        return len(chunks)

    def ingest_directory(self, directory: str) -> int:
        total = 0
        for root, _, files in os.walk(directory):
            for file in files:
                if file.startswith("."):
                    continue
                ext = os.path.splitext(file)[1].lower()
                if ext in (".pdf", ".csv", ".html", ".htm", ".txt", ".md"):
                    path = os.path.join(root, file)
                    try:
                        count = self.ingest_file(path)
                        total += count
                    except Exception as e:
                        print(f"Error ingesting {path}: {e}")
        return total

    def _file_hash(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
