"""Knowledge Base Management Engine: File Management, Index Maintenance, State Persistence"""
import os
import logging
import numpy as np

from src.rag_service.core.settings import settings
from src.rag_service.utils.text_utils import TextProcessor
from src.rag_service.utils.document_utils import DocumentLoader
from src.rag_service.models.embedding import EmbeddingModel
from src.rag_service.storage.faiss_store import FAISSStore
from src.rag_service.storage.state_manager import StateManager
from src.rag_service.core.doc2query import Doc2Query

logger = logging.getLogger(__name__)


class KBManager:
    def __init__(self):
        """Initialize knowledge base manager and load core components"""
        self.text_processor = TextProcessor()
        self.embedding_model = EmbeddingModel()
        self.faiss_store = None
        self._init_faiss()

        self.chunks = []
        self.chunk_sources = []
        self.chunk_ids = []
        self.file_chunk_map = {}
        self.file_md5 = {}
        self.file_doc2query_status = {}
        self.current_max_id = 0
        self.chunk_languages = []

    def _init_faiss(self):
        """Initialize FAISS vector store based on model dimension"""
        dim = self.embedding_model.get_dimension()
        self.faiss_store = FAISSStore(dim)

    def reset(self):
        """Reset the entire knowledge base and clear all data"""
        self.chunks.clear()
        self.chunk_sources.clear()
        self.chunk_ids.clear()
        self.file_chunk_map.clear()
        self.file_md5.clear()
        self.file_doc2query_status.clear()
        self.current_max_id = 0
        self._init_faiss()

    def add_file(self, fname, content, md5, enable_doc2query=True, language='zh'):
        """
        Add a single file to the knowledge base
        :param fname: File name
        :param content: File text content
        :param md5: MD5 hash of file content
        :param enable_doc2query: Enable query expansion
        :param language: Text language zh/en
        """
        cached_chunks = DocumentLoader.load_chunks_from_cache(fname, md5)

        if cached_chunks is not None:
            chunks = cached_chunks
        else:
            chunks = self.text_processor.chunk_text(content, language=language)
            DocumentLoader.save_chunks_to_cache(fname, md5, chunks)

        total_chunks = len(chunks)
        new_ids = list(range(self.current_max_id, self.current_max_id + total_chunks))
        self.current_max_id += total_chunks

        self.chunks.extend(chunks)
        self.chunk_sources.extend([fname] * total_chunks)
        self.chunk_ids.extend(new_ids)
        self.chunk_languages.extend([language] * total_chunks)
        self.file_chunk_map[fname] = new_ids
        self.file_md5[fname] = md5
        self.file_doc2query_status[fname] = enable_doc2query

        vectors = []
        for chunk in chunks:
            if enable_doc2query:
                queries = Doc2Query.generate(chunk)
                combined_text = chunk + " " + " ".join(queries)
                vec = self.embedding_model.encode(combined_text)
            else:
                vec = self.embedding_model.encode(chunk)
            vectors.append(vec)

        if vectors:
            self.faiss_store.add_vectors(np.array(vectors), new_ids)

    def remove_file(self, fname):
        """Remove a file and all its chunks from the knowledge base"""
        if fname not in self.file_chunk_map:
            return

        old_ids = self.file_chunk_map[fname]
        self.faiss_store.remove_ids(old_ids)

        for chunk_id in reversed(old_ids):
            if chunk_id in self.chunk_ids:
                pos = self.chunk_ids.index(chunk_id)
                del self.chunks[pos]
                del self.chunk_sources[pos]
                del self.chunk_ids[pos]

        del self.file_chunk_map[fname]
        self.file_md5.pop(fname, None)
        self.file_doc2query_status.pop(fname, None)

    def build_retriever(self):
        """Build hybrid retriever (Vector + BM25)"""
        from src.rag_service.core.hybrid_retriever import HybridRetriever
        self.hybrid_retriever = HybridRetriever(
            faiss_store=self.faiss_store,
            tokenizer_func=self.text_processor.tokenize
        )
        self.hybrid_retriever.build_bm25(self.chunks)

    def save(self):
        """Save knowledge base state to local files"""
        StateManager.save(
            index=self.faiss_store.index,
            chunks=self.chunks,
            sources=self.chunk_sources,
            chunk_ids=self.chunk_ids,
            file_chunk_map=self.file_chunk_map,
            file_md5=self.file_md5,
            current_max_id=self.current_max_id
        )

    def load(self):
        """Load knowledge base state from local files"""
        state = StateManager.load()
        if not state:
            return False

        self.faiss_store.index = state["index"]
        self.chunks = state["chunks"]
        self.chunk_sources = state["sources"]
        self.chunk_ids = state["chunk_ids"]
        self.file_chunk_map = state["file_chunk_map"]
        self.file_md5 = state["file_md5"]
        self.current_max_id = state["current_max_id"]

        self.build_retriever()
        return True

    def state_exist(self):
        """Check if knowledge base index exists locally"""
        return os.path.exists(settings.faiss.index_file)