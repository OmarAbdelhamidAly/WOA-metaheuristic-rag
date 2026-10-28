"""
================================================================================
FASTAPI BACKEND SERVER FOR WOA & RAG SIMULATION
================================================================================
Exposes REST APIs for:
    - 2D Whale Swarm Dynamics & Simulation Replay
    - Automated 5D RAG Hyperparameter Optimization
    - 5-Algorithm Comparative Benchmarking
    - Step-by-Step Exam Problems Verification
    - Static File Delivery for the Single Page Web Application
================================================================================
"""

import os
import time
from typing import Dict, List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Core optimization imports
from core.woa import WhaleOptimizer, WOAConfig
from core.test_functions import BENCHMARK_FUNCTIONS
from core.baseline_solvers import (
    GridSearchOptimizer,
    RandomSearchOptimizer,
    ParticleSwarmOptimizer,
    GeneticAlgorithmOptimizer,
)
from rag_pipeline.search_space import RAGSearchSpace, RAGHyperparameters
from rag_pipeline.evaluator import RAGEvaluator
from exam_problems.exam_solver import solve_problem_1, solve_problem_2, solve_problem_3

# Initialize FastAPI application
app = FastAPI(
    title="Whale Optimization Algorithm & RAG Simulator",
    description="Interactive Web Simulator and Hyperparameter Optimization Engine",
    version="1.0.0",
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
os.makedirs(PUBLIC_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=PUBLIC_DIR), name="static")


# ------------------------------------------------------------------------------
# Pydantic Request Models
# ------------------------------------------------------------------------------
class SimulationRunRequest(BaseModel):
    function_name: str = Field(default="sphere", description="Target test function")
    pop_size: int = Field(default=25, ge=5, le=100)
    max_iter: int = Field(default=35, ge=5, le=100)
    seed: Optional[int] = Field(default=42)


class RAGOptimizationRequest(BaseModel):
    pop_size: int = Field(default=20, ge=5, le=60)
    max_iter: int = Field(default=25, ge=5, le=50)
    seed: Optional[int] = Field(default=42)


class BenchmarkRequest(BaseModel):
    pop_size: int = Field(default=15, ge=5, le=30)
    max_iter: int = Field(default=20, ge=5, le=40)
    seed: Optional[int] = Field(default=42)


class ExamSolveRequest(BaseModel):
    problem_id: int = Field(default=1, ge=1, le=3)
    # Optional overrides for problem 1
    p1_x1: Optional[float] = None
    p1_p: Optional[float] = None
    # Optional overrides for problem 2
    p2_p: Optional[float] = None
    p2_b: Optional[float] = None
    p2_l: Optional[float] = None
    # Optional overrides for problem 3
    p3_a: Optional[float] = None
    p3_r1: Optional[float] = None


# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------
@app.get("/")
async def serve_index():
    """Serves the main single-page application."""
    index_path = os.path.join(PUBLIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(
        status_code=404,
        content={"error": "Frontend index.html not yet built in public/"},
    )


@app.get("/health")
@app.get("/api/health")
async def health_check():
    """Render health-check endpoint."""
    return {
        "status": "healthy",
        "service": "woa-metaheuristic-rag",
        "timestamp": time.time(),
        "environment": "production" if os.environ.get("RENDER") else "development",
    }


@app.post("/api/simulation/run")
async def run_simulation(req: SimulationRunRequest):
    """
    Runs WOA on a 2D mathematical test function and records every swarm snapshot
    so the HTML5 Canvas can animate the movement smoothly.
    """
    func_key = req.function_name.lower()
    if func_key not in BENCHMARK_FUNCTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown function '{req.function_name}'. Available: {list(BENCHMARK_FUNCTIONS.keys())}",
        )

    meta = BENCHMARK_FUNCTIONS[func_key]
    func = meta["func"]
    lb, ub = meta["lb"], meta["ub"]
    dim = 2  # 2D for direct canvas projection

    rng = np.random.default_rng(req.seed)
    pop_size = req.pop_size
    max_iter = req.max_iter

    # Initialize population in 2D
    population = rng.uniform(lb, ub, size=(pop_size, dim))
    fitness = np.array([func(p) for p in population])
    best_idx = np.argmin(fitness)
    best_pos = population[best_idx].copy()
    best_fit = float(fitness[best_idx])

    frames = []

    # Record initial frame (epoch 0)
    frames.append({
        "epoch": 0,
        "a": 2.0,
        "avg_abs_A": 1.0,
        "p": 0.0,
        "active_phase": "Initialization",
        "whales": population.round(4).tolist(),
        "leader": best_pos.round(4).tolist(),
        "best_fitness": best_fit,
    })

    # Main iterative loop recording frames
    for t in range(1, max_iter + 1):
        a = 2.0 - 2.0 * (t / max_iter)
        last_phase = "Shrinking Encircling"
        last_p = 0.0
        A_magnitudes = []

        for i in range(pop_size):
            r1 = rng.random(dim)
            r2 = rng.random(dim)
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            l = rng.uniform(-1.0, 1.0)
            p = float(rng.random())
            last_p = p
            A_magnitudes.append(float(np.mean(np.abs(A))))

            if p < 0.5:
                if np.mean(np.abs(A)) < 1.0:
                    last_phase = "Shrinking Encircling (Exploitation)"
                    D = np.abs(C * best_pos - population[i])
                    new_pos = best_pos - A * D
                else:
                    last_phase = "Global Exploration (Search Prey)"
                    rand_idx = int(rng.integers(0, pop_size))
                    X_rand = population[rand_idx]
                    D = np.abs(C * X_rand - population[i])
                    new_pos = X_rand - A * D
            else:
                last_phase = "Logarithmic Spiral Attack"
                D_prime = np.abs(best_pos - population[i])
                spiral = np.exp(1.0 * l) * np.cos(2.0 * np.pi * l)
                new_pos = D_prime * spiral + best_pos

            new_pos = np.clip(new_pos, lb, ub)
            population[i] = new_pos

            fit = func(population[i])
            fitness[i] = fit
            if fit < best_fit:
                best_fit = float(fit)
                best_pos = population[i].copy()

        frames.append({
            "epoch": t,
            "a": round(float(a), 3),
            "avg_abs_A": round(float(np.mean(A_magnitudes)), 3),
            "p": round(float(last_p), 3),
            "active_phase": last_phase,
            "whales": population.round(4).tolist(),
            "leader": best_pos.round(4).tolist(),
            "best_fitness": best_fit,
        })

    return {
        "function_name": func_key,
        "type": meta["type"],
        "bounds": [lb, ub],
        "total_epochs": max_iter,
        "frames": frames,
        "final_leader": best_pos.round(4).tolist(),
        "final_fitness": best_fit,
    }


@app.post("/api/rag/optimize")
async def optimize_rag(req: RAGOptimizationRequest):
    """Runs automated WOA on the 5D RAG pipeline."""
    start_time = time.perf_counter()
    evaluator = RAGEvaluator()
    obj_func = evaluator.get_objective_function()
    lb, ub = RAGSearchSpace.get_bounds()
    dim = RAGSearchSpace.DIMENSION

    # Baseline unoptimized RAG configuration (typical naive defaults)
    naive_hp = RAGHyperparameters(
        chunk_size=512,
        chunk_overlap=50,
        top_k=10,
        hybrid_alpha=0.50,
        rerank_threshold=0.20,
    )
    naive_metrics = evaluator.evaluate(naive_hp)

    # Run WOA optimizer
    config = WOAConfig(
        pop_size=req.pop_size,
        max_iter=req.max_iter,
        seed=req.seed,
        verbose=False,
    )
    optimizer = WhaleOptimizer(config)
    res = optimizer.optimize(obj_func, lb, ub, dim)

    best_hp = RAGSearchSpace.decode(res.best_position)
    optimized_metrics = evaluator.evaluate(best_hp)

    duration = time.perf_counter() - start_time
    gain_pct = (
        (naive_metrics["loss"] - optimized_metrics["loss"]) / max(1e-6, naive_metrics["loss"])
    ) * 100.0

    return {
        "execution_time_sec": round(duration, 3),
        "total_evaluations": res.total_evaluations,
        "convergence_curve": res.convergence_curve.tolist(),
        "optimized_hyperparameters": best_hp.to_dict(),
        "optimized_metrics": {
            "precision_pct": round(optimized_metrics["precision"] * 100.0, 2),
            "recall_pct": round(optimized_metrics["recall"] * 100.0, 2),
            "f1_pct": round(optimized_metrics["f1_score"] * 100.0, 2),
            "latency_penalty": round(optimized_metrics["latency_penalty"], 4),
            "composite_loss": round(optimized_metrics["loss"], 4),
        },
        "baseline_metrics": {
            "hyperparameters": naive_hp.to_dict(),
            "precision_pct": round(naive_metrics["precision"] * 100.0, 2),
            "recall_pct": round(naive_metrics["recall"] * 100.0, 2),
            "f1_pct": round(naive_metrics["f1_score"] * 100.0, 2),
            "latency_penalty": round(naive_metrics["latency_penalty"], 4),
            "composite_loss": round(naive_metrics["loss"], 4),
        },
        "gain_vs_baseline_pct": round(gain_pct, 2),
    }


@app.post("/api/benchmark/run")
async def run_benchmark(req: BenchmarkRequest):
    """Runs the 5-algorithm benchmark and returns comparison data."""
    evaluator = RAGEvaluator()
    obj_func = evaluator.get_objective_function()
    lb, ub = RAGSearchSpace.get_bounds()
    dim = RAGSearchSpace.DIMENSION
    total_evals = req.pop_size * req.max_iter

    algorithms = {}

    # 1. Grid Search
    g_opt = GridSearchOptimizer(points_per_dim=4)
    g_pos, g_loss, g_hist, g_time = g_opt.optimize(obj_func, lb, ub, dim, max_evals=total_evals)
    g_hp = RAGSearchSpace.decode(g_pos)
    g_eval = evaluator.evaluate(g_hp)
    algorithms["Grid Search"] = {
        "loss": round(g_loss, 4),
        "precision": round(g_eval["precision"] * 100, 1),
        "recall": round(g_eval["recall"] * 100, 1),
        "f1": round(g_eval["f1_score"] * 100, 1),
        "latency": round(g_eval["latency_penalty"], 3),
        "runtime_sec": round(g_time, 2),
        "hp": g_hp.to_dict(),
        "history": g_hist[:40].tolist(),
    }

    # 2. Random Search
    r_opt = RandomSearchOptimizer(seed=req.seed)
    r_pos, r_loss, r_hist, r_time = r_opt.optimize(obj_func, lb, ub, dim, max_evals=total_evals)
    r_hp = RAGSearchSpace.decode(r_pos)
    r_eval = evaluator.evaluate(r_hp)
    algorithms["Random Search"] = {
        "loss": round(r_loss, 4),
        "precision": round(r_eval["precision"] * 100, 1),
        "recall": round(r_eval["recall"] * 100, 1),
        "f1": round(r_eval["f1_score"] * 100, 1),
        "latency": round(r_eval["latency_penalty"], 3),
        "runtime_sec": round(r_time, 2),
        "hp": r_hp.to_dict(),
        "history": r_hist[:40].tolist(),
    }

    # 3. Genetic Algorithm (GA)
    ga_opt = GeneticAlgorithmOptimizer(pop_size=req.pop_size, seed=req.seed)
    ga_pos, ga_loss, ga_hist, ga_time = ga_opt.optimize(obj_func, lb, ub, dim, max_evals=total_evals)
    ga_hp = RAGSearchSpace.decode(ga_pos)
    ga_eval = evaluator.evaluate(ga_hp)
    algorithms["Genetic Algorithm"] = {
        "loss": round(ga_loss, 4),
        "precision": round(ga_eval["precision"] * 100, 1),
        "recall": round(ga_eval["recall"] * 100, 1),
        "f1": round(ga_eval["f1_score"] * 100, 1),
        "latency": round(ga_eval["latency_penalty"], 3),
        "runtime_sec": round(ga_time, 2),
        "hp": ga_hp.to_dict(),
        "history": ga_hist[:40].tolist(),
    }

    # 4. Particle Swarm Optimization (PSO)
    pso_opt = ParticleSwarmOptimizer(pop_size=req.pop_size, seed=req.seed)
    pso_pos, pso_loss, pso_hist, pso_time = pso_opt.optimize(obj_func, lb, ub, dim, max_evals=total_evals)
    pso_hp = RAGSearchSpace.decode(pso_pos)
    pso_eval = evaluator.evaluate(pso_hp)
    algorithms["Particle Swarm (PSO)"] = {
        "loss": round(pso_loss, 4),
        "precision": round(pso_eval["precision"] * 100, 1),
        "recall": round(pso_eval["recall"] * 100, 1),
        "f1": round(pso_eval["f1_score"] * 100, 1),
        "latency": round(pso_eval["latency_penalty"], 3),
        "runtime_sec": round(pso_time, 2),
        "hp": pso_hp.to_dict(),
        "history": pso_hist[:40].tolist(),
    }

    # 5. Whale Optimization Algorithm (WOA)
    woa_cfg = WOAConfig(pop_size=req.pop_size, max_iter=req.max_iter, seed=req.seed, verbose=False)
    woa_opt = WhaleOptimizer(woa_cfg)
    woa_res = woa_opt.optimize(obj_func, lb, ub, dim)
    woa_hp = RAGSearchSpace.decode(woa_res.best_position)
    woa_eval = evaluator.evaluate(woa_hp)
    algorithms["Whale Optimization (WOA)"] = {
        "loss": round(woa_res.best_fitness, 4),
        "precision": round(woa_eval["precision"] * 100, 1),
        "recall": round(woa_eval["recall"] * 100, 1),
        "f1": round(woa_eval["f1_score"] * 100, 1),
        "latency": round(woa_eval["latency_penalty"], 3),
        "runtime_sec": round(woa_res.execution_time_sec, 2),
        "hp": woa_hp.to_dict(),
        "history": woa_res.convergence_curve[:40].tolist(),
    }

    # Compute gain vs grid
    base_loss = algorithms["Grid Search"]["loss"]
    for k in algorithms:
        algorithms[k]["gain_vs_grid_pct"] = round(
            ((base_loss - algorithms[k]["loss"]) / max(1e-6, base_loss)) * 100.0, 1
        )

    return {"benchmark": algorithms, "total_eval_budget": total_evals}


@app.post("/api/exam/solve")
async def solve_exam(req: ExamSolveRequest):
    """Returns step-by-step mathematical breakdown for the selected exam problem."""
    if req.problem_id == 1:
        # Default or overridden values
        X1 = req.p1_x1 if req.p1_x1 is not None else 4.0
        p = req.p1_p if req.p1_p is not None else 0.20
        X2, X3 = -2.0, 5.0
        t, T_max = 1, 10
        r1, r2 = 0.30, 0.60

        f_X1, f_X2, f_X3 = X1**2, X2**2, X3**2
        X_star = X2
        a = 2.0 - 2.0 * (t / T_max)
        A = 2.0 * a * r1 - a
        C = 2.0 * r2
        D = abs(C * X_star - X1)
        X1_new = X_star - (A * D)
        f_new = X1_new**2

        return {
            "problem_id": 1,
            "title": "Problem 1: 1D Shrinking Encircling Step",
            "given": {
                "objective": "Minimize f(x) = x^2",
                "whales": {"X1": X1, "X2": X2, "X3": X3},
                "parameters": {"t": t, "T_max": T_max, "r1": r1, "r2": r2, "p": p},
            },
            "steps": [
                {
                    "step": 1,
                    "title": "Identify Best Solution (X*)",
                    "formula": "X* = argmin [f(X1), f(X2), f(X3)]",
                    "values": f"f({X1}) = {f_X1:.1f}, f({X2}) = {f_X2:.1f} (Lowest Cost!), f({X3}) = {f_X3:.1f}",
                    "result": f"Leader X* = {X_star}",
                },
                {
                    "step": 2,
                    "title": "Compute Decay Parameter a",
                    "formula": "a = 2 - 2 * (t / T_max)",
                    "values": f"a = 2 - 2 * ({t} / {T_max}) = 2 - 0.20",
                    "result": f"a = {a:.2f}",
                },
                {
                    "step": 3,
                    "title": "Compute Vector A & Condition Check",
                    "formula": "A = 2 * a * r1 - a",
                    "values": f"A = 2 * ({a:.2f}) * ({r1:.2f}) - {a:.2f} = {A:.2f}",
                    "result": f"p = {p:.2f} < 0.5 AND |A| = {abs(A):.2f} < 1.0 --> Shrinking Encircling Triggered!",
                },
                {
                    "step": 4,
                    "title": "Compute Coefficient C & Distance D",
                    "formula": "C = 2 * r2,   D = | C * X* - X1 |",
                    "values": f"C = 2 * {r2:.2f} = {C:.2f},   D = | {C:.2f} * ({X_star}) - {X1} | = |-6.40|",
                    "result": f"C = {C:.2f},   D = {D:.2f}",
                },
                {
                    "step": 5,
                    "title": "Update Position X1(1)",
                    "formula": "X1(1) = X* - (A * D)",
                    "values": f"X1(1) = {X_star} - ({A:.2f} * {D:.2f}) = {X_star} + {abs(A*D):.3f}",
                    "result": f"X1(1) = {X1_new:+.3f}   (f(X1) dropped from {f_X1:.1f} to {f_new:.2f}, 57.5% gain!)",
                },
            ],
            "final_position": round(X1_new, 3),
            "final_cost": round(f_new, 3),
        }

    elif req.problem_id == 2:
        p = req.p2_p if req.p2_p is not None else 0.85
        b = req.p2_b if req.p2_b is not None else 1.0
        l = req.p2_l if req.p2_l is not None else -0.50

        X_star = [1.0, 1.0]
        X1 = [3.0, 2.0]
        D_prime = [abs(X_star[0] - X1[0]), abs(X_star[1] - X1[1])]
        exp_factor = float(np.exp(b * l))
        cos_factor = float(np.cos(2.0 * np.pi * l))
        displacement = [
            D_prime[0] * exp_factor * cos_factor,
            D_prime[1] * exp_factor * cos_factor,
        ]
        X1_new = [displacement[0] + X_star[0], displacement[1] + X_star[1]]
        f_old = X1[0] ** 2 + X1[1] ** 2
        f_new = X1_new[0] ** 2 + X1_new[1] ** 2

        return {
            "problem_id": 2,
            "title": "Problem 2: 2D Logarithmic Spiral Attack",
            "given": {
                "objective": "Minimize f(x, y) = x^2 + y^2",
                "whales": {"X*": X_star, "X1": X1},
                "parameters": {"p": p, "b": b, "l": l},
            },
            "steps": [
                {
                    "step": 1,
                    "title": "Mechanism Decision",
                    "formula": "if p >= 0.5 --> Execute Logarithmic Spiral Attack",
                    "values": f"Given p = {p:.2f} >= 0.5",
                    "result": "Spiral Attack Triggered (|A| is completely ignored!)",
                },
                {
                    "step": 2,
                    "title": "Compute Distance Vector D'",
                    "formula": "D' = | X* - X1 |",
                    "values": f"D' = | ({X_star[0]} - {X1[0]}, {X_star[1]} - {X1[1]}) | = | (-2, -1) |",
                    "result": f"D' = ({D_prime[0]}, {D_prime[1]})",
                },
                {
                    "step": 3,
                    "title": "Evaluate Spiral Math Factors",
                    "formula": "e^(b*l),   cos(2*pi*l)",
                    "values": f"e^({b} * {l}) = {exp_factor:.4f},   cos(2*pi * {l}) = cos(-pi) = {cos_factor:.1f}",
                    "result": f"exp={exp_factor:.4f}, cos={cos_factor:.1f}",
                },
                {
                    "step": 4,
                    "title": "Compute Spiral Displacement Vector",
                    "formula": "Displacement = D' * e^(bl) * cos(2*pi*l)",
                    "values": f"Displacement = ({D_prime[0]}, {D_prime[1]}) * {exp_factor:.4f} * ({cos_factor:.1f})",
                    "result": f"Displacement = ({displacement[0]:.4f}, {displacement[1]:.4f})",
                },
                {
                    "step": 5,
                    "title": "Update Position X1(1)",
                    "formula": "X1(1) = Displacement + X*",
                    "values": f"X1(1) = ({displacement[0]:.4f} + {X_star[0]}, {displacement[1]:.4f} + {X_star[1]})",
                    "result": f"X1(1) = ({X1_new[0]:.4f}, {X1_new[1]:.4f})   (Cost drops from {f_old:.1f} to {f_new:.4f}, 98.5% drop!)",
                },
            ],
            "final_position": [round(X1_new[0], 4), round(X1_new[1], 4)],
            "final_cost": round(f_new, 4),
        }

    else:
        a = req.p3_a if req.p3_a is not None else 1.80
        r1 = req.p3_r1 if req.p3_r1 is not None else 0.90
        X1 = 10.0
        X_star = 2.0
        X_rand = 8.0
        r2 = 0.50
        p = 0.30

        A = 2.0 * a * r1 - a
        C = 2.0 * r2
        D = abs(C * X_rand - X1)
        X1_new = X_rand - (A * D)

        return {
            "problem_id": 3,
            "title": "Problem 3: Global Exploration Rule with Random Whale",
            "given": {
                "whales": {"X1": X1, "X*": X_star, "X_rand": X_rand},
                "parameters": {"a": a, "r1": r1, "r2": r2, "p": p},
            },
            "steps": [
                {
                    "step": 1,
                    "title": "Compute Vector A",
                    "formula": "A = 2 * a * r1 - a",
                    "values": f"A = 2 * ({a:.2f}) * ({r1:.2f}) - {a:.2f} = 3.24 - 1.80",
                    "result": f"A = {A:+.2f}",
                },
                {
                    "step": 2,
                    "title": "Evaluate Exploration Condition",
                    "formula": "if p < 0.5 AND |A| >= 1.0 --> Global Search with X_rand",
                    "values": f"p = {p:.2f} < 0.5 AND |A| = {abs(A):.2f} >= 1.0",
                    "result": "Whale MUST DIVERGE from leader X*=2.0 and follow random peer X_rand=8.0!",
                },
                {
                    "step": 3,
                    "title": "Compute Coefficient C & Distance D",
                    "formula": "C = 2 * r2,   D = | C * X_rand - X1 |",
                    "values": f"C = 2 * {r2:.2f} = {C:.2f},   D = | {C:.2f} * {X_rand} - {X1} | = |-2.0|",
                    "result": f"C = {C:.2f},   D = {D:.2f}",
                },
                {
                    "step": 4,
                    "title": "Global Search Position Update",
                    "formula": "X1(1) = X_rand - (A * D)",
                    "values": f"X1(1) = {X_rand} - ({A:.2f} * {D:.2f}) = {X_rand} - 2.88",
                    "result": f"X1(1) = {X1_new:+.2f}   (Whale moved to +5.12 exploring far away from leader!)",
                },
            ],
            "final_position": round(X1_new, 2),
        }
