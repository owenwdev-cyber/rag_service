"""
Hybrid Retrieval Test (jieba tokenizer only used here)
"""
import common
import numpy as np
import jieba
import faiss
from src.rag_service.core.hybrid_retriever import HybridRetriever
from src.rag_service.models.embedding import EmbeddingModel

def jieba_tokenizer(text):
    return list(jieba.cut(text))


def test_hybrid_retriever_memory():
    print("=" * 60)

    chunks = [
        "机器学习是人工智能的重要分支，用于数据分析",
        "深度学习通过神经网络模拟人类的思考方式",
        "苹果含有丰富的维生素C和膳食纤维",
        "今天天气晴朗，非常适合户外运动",
        "苹果公司最新发布的手机售价较高"
    ]

    embed = EmbeddingModel()
    vectors = [embed.encode(c) for c in chunks]
    vectors = np.array(vectors, dtype=np.float32)

    dim = vectors.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(vectors)

    print(f"FAISS index created, vector dimension: {dim}, count: {faiss_index.ntotal}")

    retriever = HybridRetriever(
        faiss_store=faiss_index,
        tokenizer_func=jieba_tokenizer
    )
    retriever.build_bm25(chunks)

    query = "智能设备里的苹果"
    query_vec = embed.encode(query)
    query_vec = np.array(query_vec, dtype=np.float32).reshape(1, -1)

    result_indices = retriever.search(
        query=query,
        top_k=2,
        query_vector=query_vec
    )

    print("\nHybrid retrieval results:")
    for i in result_indices:
        print(f"[{i}] {chunks[i]}")


if __name__ == "__main__":
    test_hybrid_retriever_memory()