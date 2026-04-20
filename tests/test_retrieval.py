import pytest
from src.retrieval.chunker import HybridChunker, Chunk


def test_chunker_recursive():
    chunker = HybridChunker(chunk_size=100, overlap=20)
    text = "A" * 300
    chunks = chunker.chunk(text, {"source": "test"}, ".txt")
    assert len(chunks) > 1


def test_chunker_csv():
    chunker = HybridChunker()
    csv_text = "col1,col2,col3\nval1,val2,val3\nval4,val5,val6"
    chunks = chunker.chunk(csv_text, {"source": "test.csv"}, ".csv")
    assert len(chunks) == 2
    assert "col1" in chunks[0].text


def test_chunk_to_dict():
    chunk = Chunk(text="test text", metadata={"source": "test"})
    d = chunk.to_dict()
    assert d["text"] == "test text"
    assert d["metadata"]["source"] == "test"
