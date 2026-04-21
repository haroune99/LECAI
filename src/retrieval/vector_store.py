import os
import chromadb
from chromadb.config import Settings
from typing import Optional


INDEX_DIR = "data/indexes"
os.makedirs(INDEX_DIR, exist_ok=True)


class VectorStore:
    def __init__(self, collection_name: str = "lec_documents", index_dir: str = INDEX_DIR):
        self.index_dir = index_dir
        self.client = chromadb.PersistentClient(path=index_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]):
        ids = []
        for c in chunks:
            source = c["metadata"].get("source", "unk")
            chunk_idx = c["metadata"].get("chunk_index")
            row_idx = c["metadata"].get("row_index")
            page_idx = c["metadata"].get("page_index")
            if chunk_idx is not None:
                unique_part = str(chunk_idx)
            elif row_idx is not None:
                unique_part = f"r{row_idx}"
            elif page_idx is not None:
                unique_part = f"p{page_idx}"
            else:
                unique_part = c["text"][:32]
            ids.append(f"{source}_{unique_part}")
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        filters: Optional[dict] = None,
        top_k: int = 10,
    ) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters,
        )

        output = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                output.append({
                    "text": results["documents"][0][i],
                    "chunk_id": results["ids"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "semantic_score": float(1 - results["distances"][0][i]) if results["distances"] else 0.0,
                })
        return output

    def delete_collection(self, name: str = "lec_documents"):
        self.client.delete_collection(name=name)
