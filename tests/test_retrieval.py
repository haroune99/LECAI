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
    assert len(chunks) == 3
    assert chunks[0].text == "col1,col2,col3"
    assert chunks[1].text == "val1,val2,val3"
    assert chunks[2].text == "val4,val5,val6"
    assert chunks[0].metadata["row_index"] == 1
    assert chunks[2].metadata["row_index"] == 3


def test_chunk_to_dict():
    chunk = Chunk(text="test text", metadata={"source": "test"})
    d = chunk.to_dict()
    assert d["text"] == "test text"
    assert d["metadata"]["source"] == "test"
