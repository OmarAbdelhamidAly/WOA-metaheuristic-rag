"""
================================================================================
EXPERIMENT VISUALIZER & PLOTTING MODULE
================================================================================
Generates high-resolution academic visualization figures:
    1. Convergence comparison curves across all algorithms.
    2. Radar chart of the optimal RAG hyperparameter profile.
    3. Mathematical dynamics of parameter a and vector A decay.
================================================================================
"""

import os
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(
    benchmark_results: Dict[str, Dict],
    save_path: str = "convergence_comparison.png",
):
    """Plots optimization convergence curves for all algorithms."""
    plt.figure(figsize=(10, 6), dpi=300)

    # Style palette
    palette = {
        "Whale Optimization (WOA)": ("#2563EB", "-", 2.5),
        "Particle Swarm (PSO)": ("#059669", "--", 1.8),
        "Genetic Algorithm (GA)": ("#D97706", "-.", 1.8),
        "Random Search": ("#7C3AED", ":", 1.5),
        "Grid Search": ("#DC2626", ":", 1.5),
    }

    for name, data in benchmark_results.items():
        hist = data["history"]
        color, style, width = palette.get(name, ("#6B7280", "-", 1.5))
        # Subsample if history is too long
        if len(hist) > 50:
            indices = np.linspace(0, len(hist) - 1, 50, dtype=int)
            x_vals = np.arange(1, len(indices) + 1)
            y_vals = hist[indices]
        else:
            x_vals = np.arange(1, len(hist) + 1)
            y_vals = hist

        plt.plot(x_vals, y_vals, label=name, color=color, linestyle=style, linewidth=width)

    plt.title("Convergence Velocity Comparison on 5D RAG Tuning", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Normalized Optimization Progress / Epochs", fontsize=12)
    plt.ylabel("Multi-Objective Cost (Loss)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10, loc="upper right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved convergence plot: {save_path}")


def plot_parameter_radar(
    best_hp,
    save_path: str = "hyperparameter_radar.png",
):
    """Generates a normalized radar plot of the optimal RAG hyperparameters."""
    labels = ["Chunk Size", "Overlap", "Top-K", "Hybrid Alpha", "Rerank Thresh"]
    
    # Normalized values to [0, 1] for radar plotting
    values = [
        (best_hp.chunk_size - 128) / (1024 - 128),
        best_hp.chunk_overlap / 256.0,
        (best_hp.top_k - 1) / (20 - 1),
        best_hp.hybrid_alpha,
        best_hp.rerank_threshold / 0.90,
    ]

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True), dpi=300)
    ax.fill(angles, values, color="#2563EB", alpha=0.25)
    ax.plot(angles, values, color="#1D4ED8", linewidth=2.5)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_title("WOA Optimal RAG Hyperparameter Profile", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved radar plot: {save_path}")
