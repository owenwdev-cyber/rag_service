"""
Reranker Module Test
"""

import common
import numpy as np
from src.rag_service.models.cross_encoder import RerankModel
from src.rag_service.core.reranker import Reranker


def test_reranker():
    print("=" * 60)

    chunks = [
        "机器学习是人工智能的重要分支，用于数据分析",
        "深度学习通过神经网络模拟人类思考",
        "苹果含有丰富的维生素C和膳食纤维",
        "今天天气晴朗，适合户外运动",
        "苹果公司最新发布的手机售价较高"
    ]

    rerank_model = RerankModel()
    print("RerankModel loaded successfully")

    reranker = Reranker(rerank_model)

    query = "苹果的营养价值"
    candidate_indices = [2, 4, 0]

    ranked = reranker.rerank(
        query=query,
        chunks=chunks,
        candidate_indices=candidate_indices
    )

    print(f"\nQuery: {query}")
    print(f"Candidate indices: {candidate_indices}")
    print(f"\nReranked results (highest score first):")
    for idx, score in ranked:
        print(f"[{idx}] Score={score:.4f} | {chunks[idx]}")


if __name__ == "__main__":
    test_reranker()