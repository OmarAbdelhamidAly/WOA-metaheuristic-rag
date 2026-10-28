"""
================================================================================
RAG EVALUATION & MULTI-OBJECTIVE FITNESS ENGINE
================================================================================
Simulates end-to-end RAG retrieval, chunking, hybrid scoring, and reranking,
computing RAGAS-aligned metrics: Context Precision, Context Recall, and Latency.

Objective Function Formulation (Minimization):
    Fitness = 0.45 * (1.0 - Context_Precision) +
              0.45 * (1.0 - Context_Recall) +
              0.10 * Latency_Penalty
================================================================================
"""

import time
from typing import Dict, List, Tuple
import numpy as np

from .search_space import RAGHyperparameters, RAGSearchSpace
from .synthetic_corpus import get_domain_corpus, get_evaluation_queries


class RAGEvaluator:
    """
    Simulates a full RAG retrieval pipeline and evaluates hyperparameter sets.
    """

    def __init__(self):
        self.corpus = get_domain_corpus()
        self.queries = get_evaluation_queries()

    def _chunk_document(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Simulates splitting document text into overlapping token windows."""
        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        chunks = []
        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            window = words[i : i + chunk_size]
            chunks.append(" ".join(window))
            if i + chunk_size >= len(words):
                break
        return chunks

    def _compute_bm25_sim(self, query_terms: List[str], chunk_text: str) -> float:
        """Simulates BM25 term frequency / keyword match."""
        chunk_lower = chunk_text.lower()
        matched = sum(1 for term in query_terms if term.lower() in chunk_lower)
        return matched / max(1, len(query_terms))

    def _compute_dense_sim(self, relevant_doc_ids: List[str], parent_doc_id: str, chunk_text: str) -> float:
        """Simulates dense embedding semantic cosine similarity with realistic noise."""
        if parent_doc_id in relevant_doc_ids:
            # High semantic similarity for ground truth source
            base_score = 0.88
            # Add slight penalty if chunk is too small (fragmented) or too large (diluted)
            word_count = len(chunk_text.split())
            if word_count < 20:
                base_score -= 0.15
            elif word_count > 300:
                base_score -= 0.10
            return float(np.clip(base_score, 0.0, 1.0))
        # Non-relevant background text
        return float(np.random.uniform(0.10, 0.45))

    def evaluate(self, hp: RAGHyperparameters) -> Dict[str, float]:
        """
        Runs evaluation across all test queries using the specified hyperparameters.
        Returns a dictionary of precision, recall, latency, and composite fitness.
        """
        total_precision = 0.0
        total_recall = 0.0

        for q in self.queries:
            query_terms = q["key_terms"]
            relevant_ids = set(q["relevant_doc_ids"])

            # 1. Chunk corpus
            candidate_chunks = []
            for doc in self.corpus:
                chunks = self._chunk_document(doc["text"], hp.chunk_size, hp.chunk_overlap)
                for c in chunks:
                    candidate_chunks.append({"doc_id": doc["id"], "text": c})

            # 2. Score candidate chunks (Hybrid BM25 + Dense)
            scored_candidates = []
            for item in candidate_chunks:
                bm25_score = self._compute_bm25_sim(query_terms, item["text"])
                dense_score = self._compute_dense_sim(list(relevant_ids), item["doc_id"], item["text"])

                hybrid_score = hp.hybrid_alpha * dense_score + (1.0 - hp.hybrid_alpha) * bm25_score

                # 3. Cross-Encoder Rerank Filtering
                if hybrid_score >= hp.rerank_threshold:
                    scored_candidates.append((hybrid_score, item["doc_id"]))

            # Sort descending by hybrid relevance score
            scored_candidates.sort(key=lambda x: x[0], reverse=True)

            # 4. Truncate to top_k
            retrieved = scored_candidates[: hp.top_k]
            retrieved_doc_ids = set([x[1] for x in retrieved])

            # 5. Precision & Recall calculation
            if len(retrieved) > 0:
                hits = sum(1 for _, d_id in retrieved if d_id in relevant_ids)
                q_prec = hits / len(retrieved)
            else:
                q_prec = 0.0

            q_recall = len(retrieved_doc_ids.intersection(relevant_ids)) / max(1, len(relevant_ids))

            total_precision += q_prec
            total_recall += q_recall

        avg_precision = total_precision / len(self.queries)
        avg_recall = total_recall / len(self.queries)

        # 6. Latency Penalty
        # Larger top_k and larger chunk size consume more memory and LLM prompt tokens
        latency_penalty = (hp.top_k / 20.0) * 0.5 + (hp.chunk_size / 1024.0) * 0.5
        latency_penalty = float(np.clip(latency_penalty, 0.0, 1.0))

        # Composite multi-objective cost to minimize
        # Lower is better (0.0 = perfect retrieval with zero latency penalty)
        loss = (
            0.45 * (1.0 - avg_precision)
            + 0.45 * (1.0 - avg_recall)
            + 0.10 * latency_penalty
        )

        return {
            "loss": float(loss),
            "precision": float(avg_precision),
            "recall": float(avg_recall),
            "latency_penalty": float(latency_penalty),
            "f1_score": float(
                2.0 * avg_precision * avg_recall / max(1e-6, avg_precision + avg_recall)
            ),
        }

    def get_objective_function(self):
        """
        Returns a callable f(vector) -> float suitable for WhaleOptimizer.optimize().
        """
        def objective(vector: np.ndarray) -> float:
            hp = RAGSearchSpace.decode(vector)
            result = self.evaluate(hp)
            return result["loss"]

        return objective
