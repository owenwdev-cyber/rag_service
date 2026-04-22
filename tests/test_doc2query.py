"""
Test Doc2Query Pseudo Query Generation
"""

import common
from src.rag_service.core.doc2query import Doc2Query


def test_doc2query():
    print("=" * 60)
    print("Start testing Doc2Query pseudo query generation")
    print("=" * 60)

    test_doc = """
    FAISS是Facebook开源的向量检索库，用于高效存储和搜索向量。
    支持内积、欧式距离检索，支持批量添加、删除、检索功能。
    常用于RAG系统的知识库检索。
    """

    queries = Doc2Query.generate(test_doc, num_queries=3)

    print("\nOriginal document snippet:\n", test_doc.strip())
    print("\nGenerated pseudo queries:")
    for i, q in enumerate(queries, 1):
        print(f"{i}. {q}")

    print("\nTest completed!")


if __name__ == "__main__":
    test_doc2query()