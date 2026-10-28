"""
================================================================================
SYNTHETIC TECHNICAL CORPUS & EVALUATION QUERY DATASET
================================================================================
Provides a realistic knowledge corpus covering Advanced AI, RAG architectures,
vector indexing, and prompt engineering, along with labeled evaluation queries.
================================================================================
"""

from typing import Dict, List


def get_domain_corpus() -> List[Dict[str, str]]:
    """Returns a curated corpus of technical documents across AI/RAG domains."""
    return [
        {
            "id": "doc_01",
            "title": "Humpback Whale Predatory Mechanics",
            "text": (
                "Humpback whales (Megaptera novaeangliae) exhibit complex cooperative hunting behaviors "
                "called bubble-net feeding. The whales dive 10 to 15 meters below a shoal of krill or small "
                "fish and exhale bubbles in a shrinking circular or upward spiral trajectory. This creates "
                "an acoustic and visual wall of bubbles that concentrates the prey into a tight cylinder. "
                "Once trapped, the whales lunge vertically upward with jaws wide open, engulfing thousands "
                "of prey in a single surge."
            ),
        },
        {
            "id": "doc_02",
            "title": "Mathematical Modeling of Swarm Metaheuristics",
            "text": (
                "The Whale Optimization Algorithm (WOA) mathematically models bubble-net foraging using two "
                "distinct operators: shrinking encircling and logarithmic spiral position updating. In WOA, "
                "the search agent switches between these behaviors based on a random probability p with a 50% "
                "threshold. When p < 0.5, vector A controls whether the agent exploits the current global best "
                "leader (|A| < 1) or explores the broader space by following a random peer (|A| >= 1)."
            ),
        },
        {
            "id": "doc_03",
            "title": "Chunking Strategies in Modern RAG Pipelines",
            "text": (
                "Document chunking is the foundational stage of retrieval-augmented generation. Choosing an "
                "inappropriate chunk size causes either severe context fragmentation or noisy dilution. If "
                "the chunk size is too small (e.g., 64 tokens), semantic coherence and complete concepts are "
                "severed. If the chunk size is excessively large (e.g., 2048 tokens), irrelevant sentences dilute "
                "the dense embedding vector, reducing retrieval precision. An overlap of 10% to 25% prevents "
                "boundary information loss."
            ),
        },
        {
            "id": "doc_04",
            "title": "Hybrid Search: Combining BM25 and Dense Embeddings",
            "text": (
                "Hybrid search integrates lexical keyword matching (BM25) with dense vector representations "
                "from embedding models like text-embedding-3 or BGE. BM25 excels at exact keyword lookups, "
                "acronyms, serial numbers, and rare domain terminology. In contrast, dense embeddings capture "
                "latent semantic similarity and paraphrasing. The balance is controlled by hybrid parameter alpha: "
                "score = alpha * dense_score + (1 - alpha) * bm25_score."
            ),
        },
        {
            "id": "doc_05",
            "title": "Two-Stage Retrieval and Cross-Encoder Reranking",
            "text": (
                "Modern high-accuracy RAG architectures adopt a two-stage retrieval pipeline. In stage one, a fast "
                "bi-encoder retrieves a broad candidate set of top-k documents (e.g., k=20). In stage two, a compute-heavy "
                "cross-encoder reranker performs full cross-attention between the query and each candidate chunk, outputting "
                "a calibrated relevance logit. Filtering out candidates below a confidence rerank threshold prevents "
                "hallucinatory context injection into LLM prompts."
            ),
        },
        {
            "id": "doc_06",
            "title": "Automated RAG Optimization (AutoRAG Framework)",
            "text": (
                "The AutoRAG framework (Kim et al., October 2024, arXiv:2410.20878) addresses the combinatorial "
                "explosion of RAG tuning. Manually tuning chunk size, overlap, retrieval top-k, vector similarity "
                "metrics, and prompt formats across millions of combinations is intractable. AutoRAG formulates pipeline "
                "optimization as an automated search problem across discrete modular nodes, measuring end-to-end generation "
                "quality and retrieval precision using RAGAS metrics."
            ),
        },
        {
            "id": "doc_07",
            "title": "Evaluation Metrics: RAGAS Framework",
            "text": (
                "RAGAS (Retrieval Augmented Generation Assessment, Es et al., 2023) defines four core automated metrics: "
                "Context Precision (evaluates whether all ground-truth relevant items are ranked at the top), Context "
                "Recall (measures whether all required evidence was successfully retrieved), Faithfulness (ensures the "
                "answer strictly adheres to retrieved facts), and Answer Relevance (evaluates query-answer alignment)."
            ),
        },
        {
            "id": "doc_08",
            "title": "Exploration versus Exploitation Trade-off",
            "text": (
                "Metaheuristic search algorithms must balance exploration (discovering new unvisited regions of the search "
                "space to avoid premature convergence) and exploitation (refining solutions around promising local basins). "
                "In WOA, exploration is driven by vector A when |A| >= 1, whereas exploitation is driven by shrinking "
                "encircling when |A| < 1 and logarithmic spiral lunging. Linear decay of parameter 'a' guarantees smooth "
                "transition from exploration to exploitation."
            ),
        },
    ]


def get_evaluation_queries() -> List[Dict]:
    """Returns evaluation queries with expected relevant document IDs and keywords."""
    return [
        {
            "query": "How do humpback whales trap prey using bubble nets?",
            "relevant_doc_ids": ["doc_01"],
            "key_terms": ["bubble-net", "bubbles", "cylinder", "krill", "vertical"],
        },
        {
            "query": "What are the mathematical equations governing WOA position updates?",
            "relevant_doc_ids": ["doc_02", "doc_08"],
            "key_terms": ["shrinking", "spiral", "vector A", "probability p", "exploration"],
        },
        {
            "query": "Why does chunk size impact retrieval precision and how should overlap be set?",
            "relevant_doc_ids": ["doc_03"],
            "key_terms": ["chunk size", "fragmentation", "dilution", "overlap", "boundary"],
        },
        {
            "query": "How does hybrid search combine BM25 and dense embeddings with parameter alpha?",
            "relevant_doc_ids": ["doc_04"],
            "key_terms": ["hybrid", "bm25", "dense", "alpha", "acronyms", "paraphrasing"],
        },
        {
            "query": "What is the role of a cross-encoder reranker in two-stage RAG?",
            "relevant_doc_ids": ["doc_05"],
            "key_terms": ["cross-encoder", "reranking", "two-stage", "threshold", "candidate"],
        },
        {
            "query": "How does AutoRAG solve combinatorial hyperparameter tuning?",
            "relevant_doc_ids": ["doc_06", "doc_07"],
            "key_terms": ["autorag", "combinatorial", "ragas", "precision", "recall"],
        },
    ]
