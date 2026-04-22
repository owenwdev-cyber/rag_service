"""
Text Processing Utilities Module
"""
import re
import jieba
from typing import List
from stopwordsiso import stopwords
from src.rag_service.core.settings import settings
from nltk.tokenize import word_tokenize


class TextProcessor:
    """Text Processor"""

    def __init__(self):
        self.stop_words = set(stopwords(settings.text_processing.stop_words_languages))

    def clean_text(self, text: str) -> str:
        """
        Clean text: remove line breaks, extra spaces, and special symbols
        
        Args:
            text: Original input text
            
        Returns:
            Cleaned text
        """
        text = text.replace("\n", " ").replace("\t", " ")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]", "", text)
        return text.strip()

    def tokenize(self, text: str, language: str = 'zh') -> List[str]:
        """
        Professional tokenization:
            Chinese: jieba
            English: nltk word_tokenize
        Automatic stopword removal + whitespace cleaning
        
        Args:
            text: Text to be tokenized
            language: Language code ('zh' or 'en')
            
        Returns:
            List of tokens
        """
        if not text:
            return []

        if language == 'zh':
            words = jieba.lcut(text)
        else:
            words = word_tokenize(text.lower())

        return [w.strip() for w in words if w.strip() and w not in self.stop_words]

    def chunk_text(self, text: str,
                   language: str = 'zh',
                   chunk_size: int = settings.text_processing.chunk_size,
                   chunk_overlap: int = settings.text_processing.chunk_overlap
                   ) -> List[str]:
        """
        Semantic chunking for long documents
        
        Args:
            text: Original full text
            language: Language of the text
            chunk_size: Maximum length of each chunk
            chunk_overlap: Overlap length between adjacent chunks
            
        Returns:
            List of semantic text chunks
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        if not text:
            return []

        # Preprocess
        text = re.sub(r'\s+', ' ', text).strip()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=settings.text_processing.zh_separators if language == 'zh' else settings.text_processing.en_separators,
            length_function=len,
            strip_whitespace=True
        )

        chunks = splitter.split_text(text)
        filtered_chunks = []

        for chunk in chunks:
            cleaned_chunk = chunk.strip()

            # Filter empty or meaningless chunks
            if not cleaned_chunk or not any(c.isalnum() for c in cleaned_chunk):
                continue

            # Remove leading punctuation marks
            while cleaned_chunk and cleaned_chunk[0] in '。，、！？；,.!?;：""''[]()、':
                cleaned_chunk = cleaned_chunk[1:].strip()

            # Keep valid chunks
            if cleaned_chunk and any(c.isalnum() for c in cleaned_chunk):
                filtered_chunks.append(cleaned_chunk)

        return filtered_chunks