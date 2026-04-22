"""
Project Configuration Module
Manages all application settings and configuration items
"""
import os
from dataclasses import dataclass, field
from typing import List
import nltk

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

@dataclass
class ModelConfig:
    """Model Configuration"""
    embedding_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-base"
    cache_folder: str = "./model_cache"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "deepseek-r1:7b"
    ollama_timeout: int = 60
    enable_doc2query: bool = True
    enable_rerank: bool = True
    enable_query_rewrite: bool = True

@dataclass
class TextProcessingConfig:
    """Text Processing Configuration"""
    chunk_size: int = 1024
    chunk_overlap: int = 200
    zh_separators: List[str] = field(default_factory=lambda: [
        "\n\n",
        "\n",
        "。", "！", "？", "；",
        " ",
        "，", "、",
        ""
    ])
    en_separators: List[str] = field(default_factory=lambda: [
        "\n\n",
        "\n",
        ". ", "! ", "? ", "; ", ", ",
        ".", "!", "?", ";", ",",
        " ",
        ""
    ])
    stop_words_languages: list[str] = field(default_factory=lambda: ["zh", "en"])

@dataclass
class FAISSConfig:
    """FAISS Index Storage Configuration"""
    index_file: str = "vector_db/faiss_index.index"
    chunks_file: str = "cache/chunks/chunks.json"
    md5_record_file: str = "records/md5_record.json"

@dataclass
class DocumentConfig:
    """Document Storage Configuration"""
    documents_folder: str = "documents"
    en_disable_doc2query: str = "documents/disable_doc2query/en"
    zh_disable_doc2query: str = "documents/disable_doc2query/zh"
    en_enable_doc2query: str = "documents/enable_doc2query/en"
    zh_enable_doc2query: str = "documents/enable_doc2query/zh"

    supported_extensions: List[str] = field(default_factory=lambda: [".txt", ".md", ".pdf", ".docx", ".epub"])

@dataclass
class ServerConfig:
    """Server Runtime Configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

@dataclass
class SearchConfig:
    """Search and Retrieval Configuration"""
    default_top_k: int = 5
    max_top_k: int = 15
    doc2query_num_queries: int = 3

@dataclass
class Settings:
    """Global Application Configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    text_processing: TextProcessingConfig = field(default_factory=TextProcessingConfig)
    faiss: FAISSConfig = field(default_factory=FAISSConfig)
    document: DocumentConfig = field(default_factory=DocumentConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    
    def __post_init__(self):
        """Initialize required directories"""
        os.makedirs(os.path.dirname(self.faiss.index_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.faiss.chunks_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.faiss.md5_record_file), exist_ok=True)

        os.makedirs(self.document.documents_folder, exist_ok=True)
        os.makedirs(self.document.en_disable_doc2query, exist_ok=True)
        os.makedirs(self.document.zh_disable_doc2query, exist_ok=True)
        os.makedirs(self.document.en_enable_doc2query, exist_ok=True)
        os.makedirs(self.document.zh_enable_doc2query, exist_ok=True)

# Global configuration instance
settings = Settings()