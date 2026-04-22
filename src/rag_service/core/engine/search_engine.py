"""Search Engine: Hybrid Search + Reranking + Prompt Construction"""
from src.rag_service.core.settings import settings
from src.rag_service.models.cross_encoder import RerankModel
from src.rag_service.core.reranker import Reranker
import logging
from typing import List, Dict, Union

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def encode(self, text: str):
        ...

class HybridRetriever:
    def search(self, query: str, top_k: int, query_vector):
        ...

class SearchEngine:
    def __init__(self):
        """Initialize the search engine and load reranking models"""
        self.rerank_model = RerankModel()
        self.reranker = Reranker(self.rerank_model)
        self.enable_language_filter = False

        self.embedding: Union[EmbeddingModel, None] = None
        self.hybrid: Union[HybridRetriever, None] = None
        self.chunks: List[str] = []
        self.sources: List[str] = []
        self.chunk_languages: List[str] = []

    def sync_state(
        self,
        embedding: EmbeddingModel,
        hybrid_retriever: HybridRetriever,
        chunks: List[str],
        chunk_sources: List[str],
        chunk_languages: List[str]
    ):
        """Synchronize the knowledge base state from the builder"""
        self.embedding = embedding
        self.hybrid = hybrid_retriever
        self.chunks = chunks
        self.sources = chunk_sources
        self.chunk_languages = chunk_languages

    def search(self, query: str, top_k: Union[int, None] = None):
        """Execute full search: vector search + BM25 + hybrid recall + reranking + language filtering"""
        if top_k is None:
            top_k = settings.search.default_top_k

        if not self.chunks:
            raise Exception("Please build or load the knowledge base first")

        query_lang = self._detect_language(query)
        top_k = min(top_k, len(self.chunks), settings.search.max_top_k)
        query_vector = self.embedding.encode(query).reshape(1, -1)
        candidate_indices = self.hybrid.search(query, top_k, query_vector)
        
        if not candidate_indices:
            logger.warning(f"No relevant content found: {query}")
            return [], "No reference available"

        if self.enable_language_filter:
            filtered_indices = []
            for idx in candidate_indices:
                if self.chunk_languages[idx] == query_lang:
                    filtered_indices.append(idx)

            if not filtered_indices:
                logger.warning(f"No [{query_lang}] content found: {query}")
                return [], "No reference available"
        else:
            filtered_indices = candidate_indices

        ranked_results = self.reranker.rerank(query, self.chunks, filtered_indices)
        final_results = ranked_results[:top_k]

        results: List[Dict[str, Union[str, float]]] = []
        for idx, score in final_results:
            results.append({
                "score": round(float(score), 4),
                "text": self.chunks[idx],
                "filename": self.sources[idx]
            })

        prompt = self.generate_rag_prompt(query, results)
        return results, prompt

    def _detect_language(self, text: str) -> str:
        """
        Detect if the text is Chinese or English
        If any Chinese character exists → Chinese
        Otherwise → English
        """
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return 'zh'
        return 'en'

    def generate_rag_prompt(self, query: str, contents: List[Dict]) -> str:
        """Generate standard RAG prompt from search results"""
        context_text = "\n".join([
            f"【来自文档: {c['filename']}】{c['text']}"
            for c in contents
        ])

        return f"""
请根据下面的参考资料回答用户问题，不要编造答案。
如果参考资料中没有答案，请说明"根据现有资料无法回答"

参考资料：
{context_text}

用户问题：{query}

请给出你的回答：
""".strip()