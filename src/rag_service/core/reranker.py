"""
Reranking Module
Perform precise ranking on candidate texts
"""


class Reranker:
    """Reranking execution class"""

    def __init__(self, rerank_model):
        """
        Initialize reranker
        
        Args:
            rerank_model: Preloaded RerankModel
        """
        self.model = rerank_model

    def rerank(self, query: str, chunks: list, candidate_indices: list) -> list:
        """
        Perform precise ranking on candidate indices
        
        Args:
            query: User query
            chunks: All text chunks
            candidate_indices: Candidate indices
            
        Returns:
            Ranked results [(index, score), ...]
        """
        pairs = [[query, chunks[i]] for i in candidate_indices]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidate_indices, scores), key=lambda x: x[1], reverse=True)
        return ranked