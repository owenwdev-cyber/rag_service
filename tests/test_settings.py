"""
Test settings.py configuration
"""

import common
from src.rag_service.core.settings import settings

print("Configuration loaded successfully!")
print("Embedding model:", settings.model.embedding_model)
print("Service port:", settings.server.port)
print("Document directory:", settings.document.documents_folder)
print("FAISS index path:", settings.faiss.index_file)
print("Chunks file path:", settings.faiss.chunks_file)
print("MD5 record path:", settings.faiss.md5_record_file)
print("Configuration validation passed!")