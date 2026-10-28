"""
================================================================================
WHALE OPTIMIZATION ALGORITHM (WOA) FOR ADVANCED RAG PIPELINES
MASTER CLI & EXECUTION ENGINE
================================================================================
Authors (Team 6 - FCAI):
    Omar Abdelhamid  |  Abdelmasih Ashraf  |  Mohamed Maher  |  Hussein Mohamed

Usage:
    python main.py                      # Interactive console menu
    python main.py --mode optimize      # Run single RAG optimization run with WOA
    python main.py --mode benchmark     # Run 5-algorithm comparative benchmark
    python main.py --mode exam          # Solve the 3 Exam Problems step-by-step
    python main.py --mode test-func     # Run standard benchmark functions (Sphere, etc.)
================================================================================
"""

import argparse
import sys
import os

from core.woa import WhaleOptimizer, WOAConfig
from core.test_functions import BENCHMARK_FUNCTIONS
from rag_pipeline.search_space import RAGSearchSpace
from rag_pipeline.evaluator import RAGEvaluator
from exam_problems.exam_solver import solve_all_exam_problems
from experiments.run_benchmark import run_comparative_benchmark
from experiments.visualizer import plot_convergence, plot_parameter_radar


def run_rag_optimization(pop_size: int = 30, max_iter: int = 35, seed: int = 42):
    """Executes a complete RAG hyperparameter optimization using WOA."""
    print("\n" + "=" * 80)
    print(" [MODE: RAG HYPERPARAMETER OPTIMIZATION USING WOA]")
    print("=" * 80)

    evaluator = RAGEvaluator()
    obj_func = evaluator.get_objective_function()
    lb, ub = RAGSearchSpace.get_bounds()
    dim = RAGSearchSpace.DIMENSION

    config = WOAConfig(pop_size=pop_size, max_iter=max_iter, seed=seed, verbose=True)
    optimizer = WhaleOptimizer(config)

    result = optimizer.optimize(obj_func, lb, ub, dim)
    best_hp = RAGSearchSpace.decode(result.best_position)
    evaluation_metrics = evaluator.evaluate(best_hp)

    print("\n" + "=" * 80)
    print(" DISCOVERED OPTIMAL RAG CONFIGURATION")
    print("=" * 80)
    print(f" 1. Chunk Size:            {best_hp.chunk_size} tokens")
    print(f" 2. Chunk Overlap:         {best_hp.chunk_overlap} tokens ({(best_hp.chunk_overlap/best_hp.chunk_size)*100:.1f}% overlap)")
    print(f" 3. Top-K Documents:       {best_hp.top_k}")
    print(f" 4. Hybrid Alpha:          {best_hp.hybrid_alpha:.3f} ({best_hp.hybrid_alpha*100:.1f}% Dense / {(1-best_hp.hybrid_alpha)*100:.1f}% BM25)")
    print(f" 5. Rerank Threshold:      {best_hp.rerank_threshold:.3f}")
    print("-" * 80)
    print(" RESULTING PERFORMANCE METRICS (RAGAS-ALIGNED):")
    print(f" - Context Precision:      {evaluation_metrics['precision']*100:.2f}%")
    print(f" - Context Recall:         {evaluation_metrics['recall']*100:.2f}%")
    print(f" - Retrieval F1 Score:     {evaluation_metrics['f1_score']*100:.2f}%")
    print(f" - Latency Penalty:        {evaluation_metrics['latency_penalty']:.4f}")
    print(f" - Final Composite Loss:   {result.best_fitness:.6f}")
    print("=" * 80 + "\n")

    # Generate radar visualization
    radar_path = os.path.join(os.path.dirname(__file__), "assets", "optimal_rag_radar.png")
    plot_parameter_radar(best_hp, save_path=radar_path)


def run_test_functions():
    """Runs WOA across standard unimodal and multimodal benchmark functions."""
    print("\n" + "=" * 80)
    print(" [MODE: STANDARD MATHEMATICAL BENCHMARK FUNCTIONS]")
    print("=" * 80)

    for name, meta in BENCHMARK_FUNCTIONS.items():
        print(f"\nEvaluating: {name.upper()} ({meta['type']})")
        print(f"  Description: {meta['description']}")
        print(f"  Theoretical Minimum: f(x*) = {meta['optimal_val']}")

        cfg = WOAConfig(pop_size=30, max_iter=40, seed=42, verbose=False)
        opt = WhaleOptimizer(cfg)
        res = opt.optimize(meta["func"], lb=meta["lb"], ub=meta["ub"], dim=5)

        print(f"  --> Discovered Best Fitness: {res.best_fitness:.6e}")
        print(f"  --> Solution Vector: {res.best_position.round(4)}")
        print(f"  --> Time: {res.execution_time_sec:.3f}s")


def run_benchmark_and_visualize():
    """Runs the 5-algorithm benchmark and saves convergence plots."""
    results = run_comparative_benchmark(pop_size=20, max_iter=25, seed=42, verbose=True)
    conv_path = os.path.join(os.path.dirname(__file__), "assets", "benchmark_convergence.png")
    plot_convergence(results, save_path=conv_path)
    radar_path = os.path.join(os.path.dirname(__file__), "assets", "benchmark_radar.png")
    plot_parameter_radar(results["Whale Optimization (WOA)"]["best_hp"], save_path=radar_path)


def interactive_menu():
    """Displays an interactive console menu."""
    while True:
        print("\n" + "=" * 70)
        print("  WHALE OPTIMIZATION ALGORITHM (WOA) - INTERACTIVE CONSOLE")
        print("  Team 6 • Faculty of Computers & Artificial Intelligence")
        print("=" * 70)
        print("  [1] Run Full RAG Hyperparameter Optimization (WOA)")
        print("  [2] Run Comparative Benchmark (WOA vs PSO vs GA vs Random vs Grid)")
        print("  [3] Solve the 3 Examination Problems Step-by-Step")
        print("  [4] Run Standard Benchmark Mathematical Functions (Sphere, Ackley, etc.)")
        print("  [5] Run Complete Pytest Suite")
        print("  [0] Exit")
        print("-" * 70)

        choice = input("Select an option (0-5): ").strip()
        if choice == "1":
            run_rag_optimization()
        elif choice == "2":
            run_benchmark_and_visualize()
        elif choice == "3":
            solve_all_exam_problems()
        elif choice == "4":
            run_test_functions()
        elif choice == "5":
            import pytest
            pytest.main(["tests/"])
        elif choice == "0":
            print("Exiting. Good luck with your presentation!")
            sys.exit(0)
        else:
            print("Invalid choice. Please select 0 to 5.")


def main():
    parser = argparse.ArgumentParser(description="Whale Optimization Algorithm (WOA) for RAG Pipelines")
    parser.add_argument(
        "--mode",
        choices=["interactive", "optimize", "benchmark", "exam", "test-func"],
        default="interactive",
        help="Execution mode (default: interactive)",
    )
    parser.add_argument("--pop", type=int, default=30, help="Population size")
    parser.add_argument("--iter", type=int, default=35, help="Maximum iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.mode == "interactive":
        interactive_menu()
    elif args.mode == "optimize":
        run_rag_optimization(pop_size=args.pop, max_iter=args.iter, seed=args.seed)
    elif args.mode == "benchmark":
        run_benchmark_and_visualize()
    elif args.mode == "exam":
        solve_all_exam_problems()
    elif args.mode == "test-func":
        run_test_functions()


if __name__ == "__main__":
    main()
