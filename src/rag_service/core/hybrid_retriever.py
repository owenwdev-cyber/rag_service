"""
Hybrid Retrieval Module
Combines FAISS Vector Search + BM25 Keyword Search
"""
import numpy as np
from rank_bm25 import BM25Okapi


class HybridRetriever:
    """Hybrid retriever: Vector + Keyword"""

    def __init__(self, faiss_store, tokenizer_func):
        """
        Initialize hybrid retriever
        
        Args:
            faiss_store: FAISS store instance
            tokenizer_func: tokenization function
        """
        self.faiss = faiss_store
        self.tokenizer = tokenizer_func
        self.bm25 = None
        self.chunks = []

    def build_bm25(self, chunks: list) -> None:
        """
        Build BM25 keyword index
        
        Args:
            chunks: list of text chunks
        """
        self.chunks = chunks
        tokenized = [self.tokenizer(c) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int, query_vector: np.ndarray) -> list:
        """
        Hybrid search: Vector + BM25 fusion and deduplication
        
        Args:
            query: user query
            top_k: number of items to recall
            query_vector: query vector
            
        Returns:
            deduplicated candidate index list
        """
        _, idx_faiss = self.faiss.search(query_vector, top_k)
        tq = self.tokenizer(query)
        scores = self.bm25.get_scores(tq)
        idx_bm25 = np.argsort(scores)[::-1][:top_k]
        union = list(set(idx_faiss[0].tolist() + idx_bm25.tolist()))
        return [i for i in union if 0 <= i < len(self.chunks)]