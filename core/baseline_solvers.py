"""
================================================================================
BASELINE OPTIMIZATION ALGORITHMS FOR BENCHMARKING
================================================================================
Provides standard classical and metaheuristic baselines to compare against WOA:
    1. Random Search: Pure stochastic sampling across the search space.
    2. Grid Search: Systematic discretized Cartesian grid evaluation.
    3. Particle Swarm Optimization (PSO): Velocity & inertia-based swarm solver.
    4. Genetic Algorithm (GA): Selection, single-point crossover, and mutation.
================================================================================
"""

import time
from typing import Callable, List, Optional, Tuple, Union
import numpy as np


class RandomSearchOptimizer:
    """Standard uniform Random Search baseline."""

    def __init__(self, seed: Optional[int] = 42):
        self.rng = np.random.default_rng(seed)

    def optimize(
        self,
        objective_func: Callable[[np.ndarray], float],
        lb: Union[float, List[float], np.ndarray],
        ub: Union[float, List[float], np.ndarray],
        dim: int,
        max_evals: int = 1500,
    ) -> Tuple[np.ndarray, float, np.ndarray, float]:
        start = time.perf_counter()
        lb_arr = np.full(dim, lb) if np.isscalar(lb) else np.asarray(lb, dtype=np.float64)
        ub_arr = np.full(dim, ub) if np.isscalar(ub) else np.asarray(ub, dtype=np.float64)

        best_pos = None
        best_fit = float("inf")
        history = np.zeros(max_evals)

        for i in range(max_evals):
            sample = self.rng.uniform(lb_arr, ub_arr)
            score = objective_func(sample)
            if score < best_fit:
                best_fit = score
                best_pos = sample.copy()
            history[i] = best_fit

        duration = time.perf_counter() - start
        return best_pos, best_fit, history, duration


class GridSearchOptimizer:
    """Discretized Cartesian Grid Search baseline."""

    def __init__(self, points_per_dim: int = 4):
        self.points_per_dim = points_per_dim

    def optimize(
        self,
        objective_func: Callable[[np.ndarray], float],
        lb: Union[float, List[float], np.ndarray],
        ub: Union[float, List[float], np.ndarray],
        dim: int,
        max_evals: int = 1500,
    ) -> Tuple[np.ndarray, float, np.ndarray, float]:
        start = time.perf_counter()
        lb_arr = np.full(dim, lb) if np.isscalar(lb) else np.asarray(lb, dtype=np.float64)
        ub_arr = np.full(dim, ub) if np.isscalar(ub) else np.asarray(ub, dtype=np.float64)

        # Build grid ticks per dimension
        ticks = [np.linspace(lb_arr[d], ub_arr[d], self.points_per_dim) for d in range(dim)]
        mesh = np.meshgrid(*ticks)
        grid_points = np.stack([m.ravel() for m in mesh], axis=-1)

        # Cap at max_evals if grid is huge
        if len(grid_points) > max_evals:
            step = len(grid_points) // max_evals
            grid_points = grid_points[::step][:max_evals]

        best_pos = None
        best_fit = float("inf")
        history = np.zeros(len(grid_points))

        for i, pt in enumerate(grid_points):
            score = objective_func(pt)
            if score < best_fit:
                best_fit = score
                best_pos = pt.copy()
            history[i] = best_fit

        duration = time.perf_counter() - start
        return best_pos, best_fit, history, duration


class ParticleSwarmOptimizer:
    """Standard Particle Swarm Optimization (Kennedy & Eberhart 1995)."""

    def __init__(
        self,
        pop_size: int = 30,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        seed: Optional[int] = 42,
    ):
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.rng = np.random.default_rng(seed)

    def optimize(
        self,
        objective_func: Callable[[np.ndarray], float],
        lb: Union[float, List[float], np.ndarray],
        ub: Union[float, List[float], np.ndarray],
        dim: int,
        max_evals: int = 1500,
    ) -> Tuple[np.ndarray, float, np.ndarray, float]:
        start = time.perf_counter()
        lb_arr = np.full(dim, lb) if np.isscalar(lb) else np.asarray(lb, dtype=np.float64)
        ub_arr = np.full(dim, ub) if np.isscalar(ub) else np.asarray(ub, dtype=np.float64)

        max_iter = max_evals // self.pop_size
        positions = self.rng.uniform(lb_arr, ub_arr, size=(self.pop_size, dim))
        velocities = np.zeros_like(positions)

        pbest_positions = positions.copy()
        pbest_fitness = np.array([objective_func(p) for p in positions])

        gbest_idx = np.argmin(pbest_fitness)
        gbest_pos = pbest_positions[gbest_idx].copy()
        gbest_fit = pbest_fitness[gbest_idx]

        history = np.zeros(max_iter)

        for t in range(max_iter):
            r1 = self.rng.random(size=(self.pop_size, dim))
            r2 = self.rng.random(size=(self.pop_size, dim))

            velocities = (
                self.w * velocities
                + self.c1 * r1 * (pbest_positions - positions)
                + self.c2 * r2 * (gbest_pos - positions)
            )
            positions = np.clip(positions + velocities, lb_arr, ub_arr)

            for i in range(self.pop_size):
                fit = objective_func(positions[i])
                if fit < pbest_fitness[i]:
                    pbest_fitness[i] = fit
                    pbest_positions[i] = positions[i].copy()
                    if fit < gbest_fit:
                        gbest_fit = fit
                        gbest_pos = positions[i].copy()

            history[t] = gbest_fit

        duration = time.perf_counter() - start
        return gbest_pos, gbest_fit, history, duration


class GeneticAlgorithmOptimizer:
    """Continuous Real-coded Genetic Algorithm (GA) baseline."""

    def __init__(
        self,
        pop_size: int = 30,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.15,
        seed: Optional[int] = 42,
    ):
        self.pop_size = pop_size
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.rng = np.random.default_rng(seed)

    def optimize(
        self,
        objective_func: Callable[[np.ndarray], float],
        lb: Union[float, List[float], np.ndarray],
        ub: Union[float, List[float], np.ndarray],
        dim: int,
        max_evals: int = 1500,
    ) -> Tuple[np.ndarray, float, np.ndarray, float]:
        start = time.perf_counter()
        lb_arr = np.full(dim, lb) if np.isscalar(lb) else np.asarray(lb, dtype=np.float64)
        ub_arr = np.full(dim, ub) if np.isscalar(ub) else np.asarray(ub, dtype=np.float64)

        max_iter = max_evals // self.pop_size
        population = self.rng.uniform(lb_arr, ub_arr, size=(self.pop_size, dim))
        fitness = np.array([objective_func(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_pos = population[best_idx].copy()
        best_fit = fitness[best_idx]

        history = np.zeros(max_iter)

        for t in range(max_iter):
            # Tournament selection
            new_pop = []
            for _ in range(self.pop_size):
                i1, i2 = self.rng.integers(0, self.pop_size, size=2)
                winner = population[i1] if fitness[i1] < fitness[i2] else population[i2]
                new_pop.append(winner.copy())
            new_pop = np.array(new_pop)

            # Crossover & Mutation
            for i in range(0, self.pop_size - 1, 2):
                if self.rng.random() < self.cx_rate:
                    alpha = self.rng.random()
                    c1 = alpha * new_pop[i] + (1 - alpha) * new_pop[i + 1]
                    c2 = (1 - alpha) * new_pop[i] + alpha * new_pop[i + 1]
                    new_pop[i], new_pop[i + 1] = c1, c2

            for i in range(self.pop_size):
                if self.rng.random() < self.mut_rate:
                    mut_noise = self.rng.normal(0, 0.1 * (ub_arr - lb_arr), size=dim)
                    new_pop[i] += mut_noise

            population = np.clip(new_pop, lb_arr, ub_arr)
            fitness = np.array([objective_func(ind) for ind in population])

            cur_best_idx = np.argmin(fitness)
            if fitness[cur_best_idx] < best_fit:
                best_fit = fitness[cur_best_idx]
                best_pos = population[cur_best_idx].copy()

            history[t] = best_fit

        duration = time.perf_counter() - start
        return best_pos, best_fit, history, duration
