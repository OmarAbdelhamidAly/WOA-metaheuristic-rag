"""
================================================================================
COMPARATIVE BENCHMARK RUNNER (WOA vs BASELINES)
================================================================================
Executes a fair, standardized benchmark comparing 5 optimization strategies
on the 5D RAG Hyperparameter Optimization task:
    1. Grid Search (Discretized uniform grid)
    2. Random Search (Uniform stochastic sampling)
    3. Genetic Algorithm (GA) (Continuous crossover + mutation)
    4. Particle Swarm Optimization (PSO) (Velocity-based swarm)
    5. Whale Optimization Algorithm (WOA) (Bubble-net dual hunting)

All metaheuristic algorithms are allocated the EXACT same evaluation budget.
================================================================================
"""

import time
from typing import Dict, List
import numpy as np
import pandas as pd
from tabulate import tabulate

from core.woa import WhaleOptimizer, WOAConfig
from core.baseline_solvers import (
    GridSearchOptimizer,
    RandomSearchOptimizer,
    ParticleSwarmOptimizer,
    GeneticAlgorithmOptimizer,
)
from rag_pipeline.search_space import RAGSearchSpace
from rag_pipeline.evaluator import RAGEvaluator


def run_comparative_benchmark(
    pop_size: int = 25,
    max_iter: int = 30,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Runs all 5 algorithms on the RAG pipeline objective.
    Total evaluation budget = pop_size * max_iter evaluations.
    """
    evaluator = RAGEvaluator()
    obj_func = evaluator.get_objective_function()
    lb, ub = RAGSearchSpace.get_bounds()
    dim = RAGSearchSpace.DIMENSION
    total_evals = pop_size * max_iter

    if verbose:
        print(f"\n{'='*80}")
        print(f" BENCHMARK: RAG HYPERPARAMETER OPTIMIZATION (5D SEARCH SPACE)")
        print(f"{'='*80}")
        print(f" Population Size: {pop_size} | Epochs: {max_iter} | Total Eval Budget: {total_evals}")
        print(f" Dimension: {dim} (Chunk Size, Overlap, Top-K, Alpha, Rerank Threshold)")
        print(f"{'-'*80}\n")

    results: Dict[str, Dict] = {}

    # 1. Grid Search
    if verbose:
        print("[1/5] Running Grid Search baseline...")
    grid_opt = GridSearchOptimizer(points_per_dim=4)
    g_pos, g_loss, g_hist, g_time = grid_opt.optimize(obj_func, lb, ub, dim, max_evals=total_evals)
    g_hp = RAGSearchSpace.decode(g_pos)
    g_eval = evaluator.evaluate(g_hp)
    results["Grid Search"] = {
        "loss": g_loss,
        "precision": g_eval["precision"],
        "recall": g_eval["recall"],
        "f1": g_eval["f1_score"],
        "latency_penalty": g_eval["latency_penalty"],
        "time_sec": g_time,
        "best_hp": g_hp,
        "history": g_hist,
    }

    # 2. Random Search
    if verbose:
        print("[2/5] Running Random Search baseline...")
    rnd_opt = RandomSearchOptimizer(seed=seed)
    r_pos, r_loss, r_hist, r_time = rnd_opt.optimize(obj_func, lb, ub, dim, max_evals=total_evals)
    r_hp = RAGSearchSpace.decode(r_pos)
    r_eval = evaluator.evaluate(r_hp)
    results["Random Search"] = {
        "loss": r_loss,
        "precision": r_eval["precision"],
        "recall": r_eval["recall"],
        "f1": r_eval["f1_score"],
        "latency_penalty": r_eval["latency_penalty"],
        "time_sec": r_time,
        "best_hp": r_hp,
        "history": r_hist,
    }

    # 3. Genetic Algorithm (GA)
    if verbose:
        print("[3/5] Running Genetic Algorithm (GA) baseline...")
    ga_opt = GeneticAlgorithmOptimizer(pop_size=pop_size, seed=seed)
    ga_pos, ga_loss, ga_hist, ga_time = ga_opt.optimize(obj_func, lb, ub, dim, max_evals=total_evals)
    ga_hp = RAGSearchSpace.decode(ga_pos)
    ga_eval = evaluator.evaluate(ga_hp)
    results["Genetic Algorithm (GA)"] = {
        "loss": ga_loss,
        "precision": ga_eval["precision"],
        "recall": ga_eval["recall"],
        "f1": ga_eval["f1_score"],
        "latency_penalty": ga_eval["latency_penalty"],
        "time_sec": ga_time,
        "best_hp": ga_hp,
        "history": ga_hist,
    }

    # 4. Particle Swarm Optimization (PSO)
    if verbose:
        print("[4/5] Running Particle Swarm Optimization (PSO) baseline...")
    pso_opt = ParticleSwarmOptimizer(pop_size=pop_size, seed=seed)
    pso_pos, pso_loss, pso_hist, pso_time = pso_opt.optimize(obj_func, lb, ub, dim, max_evals=total_evals)
    pso_hp = RAGSearchSpace.decode(pso_pos)
    pso_eval = evaluator.evaluate(pso_hp)
    results["Particle Swarm (PSO)"] = {
        "loss": pso_loss,
        "precision": pso_eval["precision"],
        "recall": pso_eval["recall"],
        "f1": pso_eval["f1_score"],
        "latency_penalty": pso_eval["latency_penalty"],
        "time_sec": pso_time,
        "best_hp": pso_hp,
        "history": pso_hist,
    }

    # 5. Whale Optimization Algorithm (WOA)
    if verbose:
        print("[5/5] Running Whale Optimization Algorithm (WOA)...")
    woa_config = WOAConfig(pop_size=pop_size, max_iter=max_iter, seed=seed, verbose=False)
    woa_opt = WhaleOptimizer(woa_config)
    woa_res = woa_opt.optimize(obj_func, lb, ub, dim)
    woa_hp = RAGSearchSpace.decode(woa_res.best_position)
    woa_eval = evaluator.evaluate(woa_hp)
    results["Whale Optimization (WOA)"] = {
        "loss": woa_res.best_fitness,
        "precision": woa_eval["precision"],
        "recall": woa_eval["recall"],
        "f1": woa_eval["f1_score"],
        "latency_penalty": woa_eval["latency_penalty"],
        "time_sec": woa_res.execution_time_sec,
        "best_hp": woa_hp,
        "history": woa_res.convergence_curve,
    }

    # Format comparative table
    table_rows = []
    baseline_loss = results["Grid Search"]["loss"]

    for name, data in results.items():
        imp_vs_grid = ((baseline_loss - data["loss"]) / baseline_loss) * 100.0
        table_rows.append(
            [
                name,
                f"{data['loss']:.4f}",
                f"{data['precision']*100:.1f}%",
                f"{data['recall']*100:.1f}%",
                f"{data['f1']*100:.1f}%",
                f"{data['latency_penalty']:.3f}",
                f"{imp_vs_grid:+.1f}%",
                f"{data['time_sec']:.2f}s",
            ]
        )

    headers = ["Algorithm", "Min Loss", "Precision", "Recall", "F1 Score", "Latency Pen.", "Gain vs Grid", "Runtime"]
    table_str = tabulate(table_rows, headers=headers, tablefmt="grid")

    if verbose:
        print(f"\n{'='*80}")
        print(" FINAL BENCHMARK COMPARATIVE RESULTS TABLE")
        print(f"{'='*80}")
        print(table_str)
        print(f"\nOptimal RAG Hyperparameters discovered by WOA:")
        print(f"  --> {results['Whale Optimization (WOA)']['best_hp']}")
        print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    run_comparative_benchmark()
