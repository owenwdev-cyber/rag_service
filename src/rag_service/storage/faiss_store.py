"""
FAISS Vector Storage Module
Responsible for vector index creation, addition, deletion, and search
"""
import numpy as np
import faiss


class FAISSStore:
    """FAISS Vector Library Management Class"""

    def __init__(self, dimension: int):
        """
        Initialize FAISS storage
        
        Args:
            dimension: dimension of embeddings
        """
        self.dimension = dimension
        self.index = None
        self._init_index()

    def _init_index(self) -> None:
        """Create inner product index and wrap as IDMap"""
        base = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap(base)

    def add_vectors(self, vectors: np.ndarray, ids: list) -> None:
        """
        Add vectors in batch with custom IDs
        
        Args:
            vectors: vector array
            ids: list of custom IDs
        """
        vectors = np.ascontiguousarray(vectors).astype('float32')
        ids = np.ascontiguousarray(ids).astype('int64')
        self.index.add_with_ids(vectors, ids)

    def remove_ids(self, ids: list) -> None:
        """
        Remove vectors by ID
        
        Args:
            ids: list of IDs to remove
        """
        if hasattr(self.index, "remove_ids"):
            self.index.remove_ids(np.array(ids, dtype='int64'))

    def search(self, query_vec: np.ndarray, top_k: int) -> tuple:
        """
        Vector similarity search
        
        Args:
            query_vec: query vector
            top_k: number of results to return
            
        Returns:
            (scores, indices)
        """
        return self.index.search(query_vec.reshape(1, -1), top_k)