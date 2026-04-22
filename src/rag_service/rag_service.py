"""
Core RAG Service Module
Provides document indexing, vector retrieval, and hybrid search capabilities
"""
import logging
from src.rag_service.core.engine.kb_builder import KBBuilder
from src.rag_service.core.engine.search_engine import SearchEngine

logger = logging.getLogger(__name__)


class RAGService:
    """
    Main RAG Service Class
    Integrates document processing, vector storage, hybrid search, reranking, and state management
    """

    def __init__(self):
        """Initialize RAG service, load models and initialize core components"""
        logger.info("Initializing RAG service...")

        self.builder = KBBuilder()
        self.searcher = SearchEngine()

        logger.info("RAG service initialization completed")

    def auto_build_kb(self, folder=None, force_rebuild=False):
        """Automatically build or incrementally update knowledge base"""
        self.builder.auto_build(folder, force_rebuild)
        mgr = self.builder.manager
        self.searcher.sync_state(
            embedding=mgr.embedding_model,
            hybrid_retriever=mgr.hybrid_retriever,
            chunks=mgr.chunks,
            chunk_sources=mgr.chunk_sources,
            chunk_languages=mgr.chunk_languages
        )

    def search(self, query, top_k=None):
        """Perform retrieval and return results with prompt"""
        return self.searcher.search(query, top_k)

    def generate_rag_prompt(self, query, contents):
        """Generate RAG prompt based on retrieval results"""
        return self.searcher.generate_rag_prompt(query, contents)
    
