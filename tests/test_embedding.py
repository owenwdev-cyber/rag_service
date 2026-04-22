"""
Test Embedding Model
Functions: test initialization, encoding, dimension, normalization
"""
import common
import numpy as np

from src.rag_service.models.embedding import EmbeddingModel


def test_embedding_model():
    print("=" * 60)
    print("Testing Embedding Model")
    print("=" * 60)

    embed = EmbeddingModel()
    print("Model initialization successful")

    dim = embed.get_dimension()
    print("Embedding dimension:", dim)

    test_text = "青春是一场漫长的行路，我们在坚持中成长，在热爱中发光。"
    vec = embed.encode(test_text)
    print("Text encoding completed, vector shape:", vec.shape)

    norm = np.linalg.norm(vec)
    print("Vector norm (normalized):", f"{norm:.4f}")

    assert vec.ndim == 1, "Vector must be 1-dimensional"
    assert vec.shape[0] == dim, "Vector dimension mismatch"
    assert abs(norm - 1.0) < 1e-6, "Vector must be normalized"

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_embedding_model()