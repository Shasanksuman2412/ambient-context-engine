"""
Processing Layer — Embedding Generation

Generates 384-dimensional sentence embeddings using the all-MiniLM-L6-v2
model from sentence-transformers. Used for semantic search in the
storage layer (sqlite-vec).
"""

import logging
import time
from typing import Union

import numpy as np

from config import EMBEDDING_MODEL, EMBEDDING_DIM

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates normalized sentence embeddings using sentence-transformers.

    The model (~90MB) is loaded once at initialization and reused.
    Supports both single-text and batch embedding generation.
    """

    def __init__(self):
        logger.info(f"Loading embedding model '{EMBEDDING_MODEL}'...")
        load_start = time.time()

        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.dim = EMBEDDING_DIM

        load_time = time.time() - load_start
        logger.info(
            f"Embedding model loaded in {load_time:.1f}s "
            f"(dim={self.dim})"
        )

        self._total_embedded = 0

    def generate(self, text: str) -> np.ndarray:
        """
        Generate a normalized embedding for a single text string.

        Args:
            text: Input text to embed

        Returns:
            numpy array of shape (384,), dtype float32, L2-normalized
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        self._total_embedded += 1
        return embedding.astype(np.float32)

    def generate_batch(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts at once (more efficient
        than calling generate() in a loop).

        Args:
            texts: List of input texts

        Returns:
            numpy array of shape (len(texts), 384), dtype float32
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )

        self._total_embedded += len(texts)
        return embeddings.astype(np.float32)

    @property
    def stats(self) -> dict:
        return {
            "total_embedded": self._total_embedded,
            "model": EMBEDDING_MODEL,
            "dimension": self.dim,
        }


# ─── Standalone Test ─────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("Testing embedding generator...")
    embedder = EmbeddingGenerator()

    # Single embedding
    text1 = "The quick brown fox jumps over the lazy dog"
    emb1 = embedder.generate(text1)
    print(f"  Single: shape={emb1.shape}, dtype={emb1.dtype}, norm={np.linalg.norm(emb1):.4f}")

    # Batch embedding
    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language",
        "The weather today is sunny and warm",
    ]
    emb_batch = embedder.generate_batch(texts)
    print(f"  Batch: shape={emb_batch.shape}, dtype={emb_batch.dtype}")

    # Cosine similarity test (embeddings are normalized, so dot product = cosine sim)
    sim_12 = np.dot(emb_batch[0], emb_batch[1])
    sim_13 = np.dot(emb_batch[0], emb_batch[2])
    print(f"  Similarity ML↔Python: {sim_12:.4f}")
    print(f"  Similarity ML↔Weather: {sim_13:.4f}")
    print(f"  (ML↔Python should be higher than ML↔Weather)")

    print(f"  Stats: {embedder.stats}")
    print("Embedding test complete!")
