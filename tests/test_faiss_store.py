"""
Test FAISSStore class
Functions: initialization, add vectors, remove vectors, search vectors
"""
import common
import numpy as np
from src.rag_service.storage.faiss_store import FAISSStore


def test_faiss_store():
    dim = 4
    faiss_store = FAISSStore(dim)
    print("FAISS initialized successfully")

    vectors = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype="float32")

    ids = [100, 200, 300, 400]

    faiss_store.add_vectors(vectors, ids)
    print("Added 4 vectors successfully")

    query = np.array([0, 1, 0, 0])
    scores, result_ids = faiss_store.search(query, top_k=2)

    print("\nSearch results:")
    print("Scores:", scores)
    print("IDs:", result_ids)
    print("Search completed")

    faiss_store.remove_ids([200])
    print("\nRemoved ID=200")

    scores, result_ids = faiss_store.search(query, top_k=2)
    print("\nSearch after removal:")
    print("Scores:", scores)
    print("IDs:", result_ids)

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_faiss_store()