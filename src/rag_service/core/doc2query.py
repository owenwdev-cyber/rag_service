"""
Doc2Query Query Expansion Module
Generate pseudo-queries via LLM to improve retrieval recall
"""
import requests
from src.rag_service.core.settings import settings
import logging
import re

logger = logging.getLogger(__name__)


class Doc2Query:
    """Static utility class: generate pseudo queries"""

    @staticmethod
    def generate(doc_text: str, num_queries: int = None) -> list:
        """
        Generate user-style pseudo queries from text chunks
        
        Args:
            doc_text: document segment
            num_queries: number of queries to generate
            
        Returns:
            list of pseudo queries
        """
        num = num_queries or settings.search.doc2query_num_queries

        prompt = f"""
你是一个提问助手。
根据下面的文档内容，生成 {num} 个独立的、用户会问的查询问题。
规则：
1. 每个问题占一行
2. 不要编号
3. 不要解释
4. 不要空行
5. 必须生成 {num} 条，不能少

文档内容：
{doc_text}
""".strip()

        url = f"{settings.model.ollama_url}/api/generate"
        payload = {
            "model": settings.model.ollama_model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.7
        }

        try:
            r = requests.post(url, json=payload, timeout=30)
            r.raise_for_status()
            lines = [q.strip() for q in r.json()["response"].splitlines() if q.strip()]
            clean_lines = []
            for line in lines:
                clean_line = re.sub(r'^\d+\.?\s*', '', line).strip()
                if clean_line:
                    clean_lines.append(clean_line)

            return clean_lines[:num]
        
        except requests.exceptions.ConnectionError:
            print("\n" + "="*60)
            print("Error: Ollama is not running, please start Ollama first!")
            raise SystemExit(1) 

        except Exception as e:
            logger.warning(f"Doc2Query generation failed: {e}")
            return [doc_text[:15] + " related"]