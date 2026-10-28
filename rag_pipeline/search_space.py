"""
================================================================================
RAG HYPERPARAMETER SEARCH SPACE & CONSTRAINTS
================================================================================
Defines the 5-dimensional search space representing critical tunable knobs in
advanced Retrieval-Augmented Generation (RAG) pipelines.

Decision Variables:
    x0: Chunk Size        [128, 1024] tokens  (Discretized step = 32)
    x1: Chunk Overlap     [0, 256] tokens     (Constraint: Overlap < 0.5 * Chunk Size)
    x2: Top-K Retrieved   [1, 20] documents   (Integer)
    x3: Hybrid Alpha      [0.0, 1.0]          (0.0 = BM25 sparse, 1.0 = Dense vector)
    x4: Rerank Threshold  [0.0, 0.90]         (Cross-encoder minimum confidence score)
================================================================================
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class RAGHyperparameters:
    """Decoded physical hyperparameter values for a RAG pipeline."""
    chunk_size: int
    chunk_overlap: int
    top_k: int
    hybrid_alpha: float
    rerank_threshold: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "hybrid_alpha": round(self.hybrid_alpha, 3),
            "rerank_threshold": round(self.rerank_threshold, 3),
        }

    def __str__(self) -> str:
        return (
            f"ChunkSize={self.chunk_size} | Overlap={self.chunk_overlap} | "
            f"TopK={self.top_k} | Alpha={self.hybrid_alpha:.2f} | "
            f"RerankThresh={self.rerank_threshold:.2f}"
        )


class RAGSearchSpace:
    """Manages continuous encoding, bounds, and decoding of the 5D search space."""

    DIMENSION: int = 5

    # Continuous lower and upper bounds for WOA search agents
    BOUNDS: List[Tuple[float, float]] = [
        (128.0, 1024.0),  # x0: chunk_size
        (0.0, 256.0),     # x1: chunk_overlap
        (1.0, 20.0),      # x2: top_k
        (0.0, 1.0),       # x3: hybrid_alpha
        (0.0, 0.90),      # x4: rerank_threshold
    ]

    @classmethod
    def get_bounds(cls) -> Tuple[np.ndarray, np.ndarray]:
        """Returns lower bounds array and upper bounds array."""
        lb = np.array([b[0] for b in cls.BOUNDS], dtype=np.float64)
        ub = np.array([b[1] for b in cls.BOUNDS], dtype=np.float64)
        return lb, ub

    @classmethod
    def decode(cls, vector: np.ndarray) -> RAGHyperparameters:
        """
        Converts a continuous continuous search agent vector into valid, constrained
        physical RAG hyperparameters.
        """
        raw_chunk_size = float(vector[0])
        raw_overlap = float(vector[1])
        raw_top_k = float(vector[2])
        raw_alpha = float(vector[3])
        raw_rerank = float(vector[4])

        # 1. Discretize chunk_size to nearest step of 32
        chunk_size = int(round(raw_chunk_size / 32.0) * 32)
        chunk_size = int(np.clip(chunk_size, 128, 1024))

        # 2. Overlap constraint: Overlap must not exceed 40% of chunk_size
        max_valid_overlap = int(0.40 * chunk_size)
        overlap = int(round(raw_overlap / 16.0) * 16)
        overlap = int(np.clip(overlap, 0, max_valid_overlap))

        # 3. Discretize top_k to integer in [1, 20]
        top_k = int(np.clip(int(round(raw_top_k)), 1, 20))

        # 4. Clamp alpha in [0.0, 1.0]
        hybrid_alpha = float(np.clip(raw_alpha, 0.0, 1.0))

        # 5. Clamp rerank threshold in [0.0, 0.90]
        rerank_threshold = float(np.clip(raw_rerank, 0.0, 0.90))

        return RAGHyperparameters(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            top_k=top_k,
            hybrid_alpha=hybrid_alpha,
            rerank_threshold=rerank_threshold,
        )

    @classmethod
    def encode(cls, hp: RAGHyperparameters) -> np.ndarray:
        """Encodes physical hyperparameters back into continuous vector representation."""
        return np.array(
            [
                float(hp.chunk_size),
                float(hp.chunk_overlap),
                float(hp.top_k),
                float(hp.hybrid_alpha),
                float(hp.rerank_threshold),
            ],
            dtype=np.float64,
        )
