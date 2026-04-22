"""
Knowledge Base State Management Module
Responsible for persistence and loading of FAISS index and chunk metadata
"""
import json
import os
import faiss
from src.rag_service.core.settings import settings


class StateManager:
    """Static utility class: save/load index, chunks, and file mappings"""

    @staticmethod
    def save(index, chunks, sources, chunk_ids, file_chunk_map, file_md5, current_max_id) -> None:
        """
        Save the entire knowledge base state to disk
        
        Args:
            index: FAISS index
            chunks: list of text chunks
            sources: list of chunk source files
            chunk_ids: list of chunk IDs
            file_chunk_map: mapping from files to chunk IDs
            file_md5: file MD5 dictionary
            current_max_id: current maximum ID
        """
        faiss.write_index(index, settings.faiss.index_file)

        data = {
            "chunks": chunks,
            "sources": sources,
            "chunk_ids": chunk_ids,
            "file_chunk_map": file_chunk_map,
            "file_md5": file_md5,
            "current_max_id": current_max_id,
        }

        with open(settings.faiss.chunks_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load() -> dict | None:
        """
        Load knowledge base state from disk
        
        Returns:
            state dictionary or None
        """
        if not os.path.exists(settings.faiss.index_file):
            return None

        try:
            index = faiss.read_index(settings.faiss.index_file)
            if not isinstance(index, faiss.IndexIDMap):
                index = faiss.IndexIDMap(index)

            with open(settings.faiss.chunks_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            return {
                "index": index,
                "chunks": data.get("chunks", []),
                "sources": data.get("sources", []),
                "chunk_ids": data.get("chunk_ids", []),
                "file_chunk_map": data.get("file_chunk_map", {}),
                "file_md5": data.get("file_md5", {}),
                "current_max_id": data.get("current_max_id", 0),
            }
        except Exception as e:
            return None