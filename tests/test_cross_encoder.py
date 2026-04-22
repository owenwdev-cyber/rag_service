"""
CrossEncoder Reranking Model Test
"""
import common
from src.rag_service.models.cross_encoder import RerankModel


def test_rerank_model():
    """Test if the reranking model can load, predict, and sort correctly"""

    print("Initializing Rerank model...")
    rerank_model = RerankModel()
    print("Model initialized successfully!")

    query = "What is machine learning?"
    documents = [
        "Machine learning is a branch of artificial intelligence that enables computers to learn from data.",
        "The weather is nice today, perfect for going out.",
        "Deep learning is a subset of machine learning that uses neural networks for training.",
        "Apple released its latest smartphone product.",
        "Machine learning algorithms include decision trees, SVM, neural networks, and more."
    ]
    pairs = [[query, doc] for doc in documents]

    print("\nStarting relevance score prediction...")
    scores = rerank_model.predict(pairs)
    print(f"Prediction completed. Raw scores: {scores}")

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

    print("\n===== Reranked Results (Highest to Lowest) =====")
    for i, (doc, score) in enumerate(ranked):
        print(f"{i+1}. Score: {score:.4f} | {doc}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_rerank_model()