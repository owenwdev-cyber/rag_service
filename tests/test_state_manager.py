"""
StateManager Test
Test saving, loading, and integrity of FAISS index + metadata
"""

import common
import faiss
import numpy as np
from src.rag_service.storage.state_manager import StateManager
from src.rag_service.core.settings import settings


def test_state_manager_save_and_load():
    """Test saving and loading knowledge base state"""
    print("=" * 60)
    print("Testing StateManager save & load")
    print("=" * 60)

    dimension = 1024
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))

    vectors = np.random.rand(3, dimension).astype("float32")
    ids = np.array([1, 2, 3])
    index.add_with_ids(vectors, ids)

    chunks = [
        "这是第一段测试文本",
        "这是第二段测试文本",
        "这是第三段测试文本"
    ]
    sources = ["test1.pdf", "test1.pdf", "test2.pdf"]
    chunk_ids = [1, 2, 3]
    file_chunk_map = {
        "test1.pdf": [1, 2],
        "test2.pdf": [3]
    }
    file_md5 = {
        "test1.pdf": "abcd1234",
        "test2.pdf": "efgh5678"
    }
    current_max_id = 3

    print("\nSaving state...")
    StateManager.save(
        index=index,
        chunks=chunks,
        sources=sources,
        chunk_ids=chunk_ids,
        file_chunk_map=file_chunk_map,
        file_md5=file_md5,
        current_max_id=current_max_id
    )
    print("State saved successfully!")

    print("\nLoading state...")
    state = StateManager.load()
    assert state is not None, "Load failed, state is None"
    print("State loaded successfully!")

    loaded_index = state["index"]
    loaded_chunks = state["chunks"]
    loaded_sources = state["sources"]
    loaded_chunk_ids = state["chunk_ids"]
    loaded_file_map = state["file_chunk_map"]
    loaded_md5 = state["file_md5"]
    loaded_max_id = state["current_max_id"]

    assert loaded_index.ntotal == 3, "Vector count mismatch"
    assert loaded_chunks == chunks, "Chunks mismatch"
    assert loaded_sources == sources, "Sources mismatch"
    assert loaded_chunk_ids == chunk_ids, "Chunk IDs mismatch"
    assert loaded_file_map == file_chunk_map, "File map mismatch"
    assert loaded_md5 == file_md5, "MD5 mismatch"
    assert loaded_max_id == current_max_id, "Max ID mismatch"

    print("\nAll validations passed!")
    print(f"Index vector count: {loaded_index.ntotal}")
    print(f"Chunk count: {len(loaded_chunks)}")
    print(f"Current max ID: {loaded_max_id}")
    print(f"File map keys: {list(loaded_file_map.keys())}")

    print("\nStateManager test completed successfully!")
    print("=" * 60)


def test_state_manager_empty():
    """Test returning None when no files exist"""
    print("\nTesting empty state...")
    import os
    if os.path.exists(settings.faiss.index_file):
        os.remove(settings.faiss.index_file)
    if os.path.exists(settings.faiss.chunks_file):
        os.remove(settings.faiss.chunks_file)

    state = StateManager.load()
    assert state is None, "Should return None when no files exist"
    print("Returned None correctly with no existing files!")


if __name__ == "__main__":
    test_state_manager_save_and_load()
    test_state_manager_empty()