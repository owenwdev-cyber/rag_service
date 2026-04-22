"""
Document Processing Utilities
Supported formats: .txt, .md, .pdf, .docx, .epub
Includes text chunking cache optimization.
"""

import os
import json
import hashlib
import logging
from typing import List, Dict, Optional
from pathlib import Path

from src.rag_service.core.settings import settings

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = "cache/chunks"


class DocumentLoader:
    """Document Loader"""

    @staticmethod
    def load_documents_from_folder(folder: str = None) -> List[Dict[str, str]]:
        """
        Recursively load all supported documents in the specified folder.

        Args:
            folder: Path to the document folder

        Returns:
            List of document dicts containing filename, content, and metadata

        Raises:
            FileNotFoundError: If the folder does not exist
            ValueError: If no readable documents are found
        """
        if folder is None:
            folder = settings.document.documents_folder

        if not os.path.exists(folder):
            raise FileNotFoundError(f"Document folder '{folder}' does not exist")

        docs = []

        for root, _, files in os.walk(folder):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                ext = os.path.splitext(fname)[1].lower()

                if ext in settings.document.supported_extensions:
                    try:
                        rel_path = os.path.relpath(root, folder)
                        enable_doc2query = 'disable_doc2query' not in rel_path.split(os.sep)

                        path_parts = Path(fpath).parts
                        language = 'zh' if 'zh' in path_parts else 'en'

                        content = DocumentLoader._read_file(fpath, ext)

                        if content:
                            docs.append({
                                "filename": fname,
                                "content": content,
                                "enable_doc2query": enable_doc2query,
                                "filepath": fpath,
                                "language": language
                            })

                            status = "enabled" if enable_doc2query else "disabled"
                            logger.info(f"Loaded document: {fname} (Doc2Query: {status}, Language: {language})")
                    except Exception as e:
                        logger.error(f"Failed to read file {fname}: {str(e)}")

        if not docs:
            raise ValueError(f"No readable documents found in '{folder}'")

        logger.info(f"Total documents loaded: {len(docs)}")
        return docs

    @staticmethod
    def _read_file(filepath: str, ext: str) -> str:
        """
        Read content based on file type.

        Args:
            filepath: Path to the file
            ext: File extension

        Returns:
            Extracted text content
        """
        try:
            if ext in ['.txt', '.md']:
                return DocumentLoader._read_text_file(filepath)
            elif ext == '.pdf':
                return DocumentLoader._read_pdf_file(filepath)
            elif ext == '.docx':
                return DocumentLoader._read_docx_file(filepath)
            elif ext == '.epub':
                return DocumentLoader._read_epub_file(filepath)
            else:
                logger.warning(f"Unsupported file type: {ext}")
                return ""
        except Exception as e:
            logger.error(f"Failed to read file {filepath}: {str(e)}")
            return ""

    @staticmethod
    def _read_text_file(filepath: str) -> str:
        """Read text file"""
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()

    @staticmethod
    def _read_pdf_file(filepath: str) -> str:
        """Read PDF file"""
        try:
            import fitz
            doc = fitz.open(filepath)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text.strip()
        except ImportError:
            logger.error("PyMuPDF is required: pip install PyMuPDF")
            raise

    @staticmethod
    def _read_docx_file(filepath: str) -> str:
        """Read Word document"""
        try:
            from docx import Document
            doc = Document(filepath)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        except ImportError:
            logger.error("python-docx is required: pip install python-docx")
            raise

    @staticmethod
    def _read_epub_file(filepath: str) -> str:
        """Read EPUB file"""
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup

            book = epub.read_epub(filepath)
            text = ""

            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text += soup.get_text() + "\n"

            return text.strip()
        except ImportError:
            logger.error("EbookLib and beautifulsoup4 are required: pip install EbookLib beautifulsoup4")
            raise

    @staticmethod
    def get_file_md5(content: str) -> str:
        """
        Compute MD5 hash of file content.

        Args:
            content: File content string

        Returns:
            MD5 hash string
        """
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    @staticmethod
    def get_chunk_cache_path(filename: str, md5: str) -> str:
        """
        Get cache file path for chunks.

        Args:
            filename: Original filename
            md5: MD5 hash of file content

        Returns:
            Full cache file path
        """
        safe_name = hashlib.md5(filename.encode()).hexdigest()[:16]
        return os.path.join(CACHE_DIR, f"{safe_name}_{md5}.json")

    @staticmethod
    def load_chunks_from_cache(filename: str, md5: str) -> Optional[List[str]]:
        """
        Load chunks from cache if available.

        Args:
            filename: Original filename
            md5: MD5 hash of file content

        Returns:
            Cached list of text chunks, or None if not found
        """
        cache_path = DocumentLoader.get_chunk_cache_path(filename, md5)

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"Cache hit: {filename} (loaded {len(data)} chunks from cache)")
                    return data
            except Exception as e:
                logger.warning(f"Failed to load cache for {filename}: {e}")

        return None

    @staticmethod
    def save_chunks_to_cache(filename: str, md5: str, chunks: List[str]):
        """
        Save text chunks to cache.

        Args:
            filename: Original filename
            md5: MD5 hash of file content
            chunks: List of text chunks
        """
        cache_path = DocumentLoader.get_chunk_cache_path(filename, md5)

        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            logger.info(f"Cache saved: {filename} ({len(chunks)} chunks)")
        except Exception as e:
            logger.warning(f"Failed to save cache for {filename}: {e}")

    @staticmethod
    def clear_chunk_cache():
        """Clear all chunk cache files"""
        try:
            if os.path.exists(CACHE_DIR):
                for file in os.listdir(CACHE_DIR):
                    os.remove(os.path.join(CACHE_DIR, file))
                logger.info(f"Chunk cache cleared: {CACHE_DIR}")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")