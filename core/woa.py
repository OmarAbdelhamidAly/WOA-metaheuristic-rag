"""
================================================================================
WHALE OPTIMIZATION ALGORITHM (WOA) - CORE NUMPY ENGINE
================================================================================
Reference:
    Mirjalili, S., & Lewis, A. (2016).
    "The Whale Optimization Algorithm."
    Advances in Engineering Software, Elsevier, 95, 51-67.
    DOI: 10.1016/j.advengsoft.2016.01.008

Key Characteristics:
    1. Mimics the bubble-net feeding behavior of Humpback Whales (Megaptera novaeangliae).
    2. Seamlessly balances Global Exploration (|A| >= 1) and Local Exploitation (|A| < 1).
    3. Employs a dual-hunting mechanism via random probability p (50/50 chance):
       - If p >= 0.5: 3D Logarithmic Spiral Attack around the prey.
       - If p < 0.5 and |A| < 1: Shrinking Encircling towards the best leader.
       - If p < 0.5 and |A| >= 1: Exploration divergence towards a random peer whale.
================================================================================
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class WOAConfig:
    """Hyperparameters and runtime settings for Whale Optimization Algorithm."""
    pop_size: int = 30           # Number of search agents (whales in the pod)
    max_iter: int = 50           # Maximum optimization iterations (T_max)
    b_spiral: float = 1.0        # Constant defining the shape of the logarithmic spiral
    a_decay: str = "linear"      # Decay mode for parameter 'a': 'linear' (default) or 'nonlinear'
    seed: Optional[int] = 42     # Random seed for reproducible experiments
    verbose: bool = True         # Whether to print iteration progress
    log_interval: int = 5        # Epoch interval for progress logging


@dataclass
class WOAResult:
    """Encapsulates the final optimization results and convergence telemetry."""
    best_position: np.ndarray
    best_fitness: float
    convergence_curve: np.ndarray
    mean_fitness_curve: np.ndarray
    diversity_curve: np.ndarray
    execution_time_sec: float
    total_evaluations: int
    trajectory: List[np.ndarray] = field(default_factory=list)


class WhaleOptimizer:
    """
    Pure NumPy implementation of the Whale Optimization Algorithm (WOA).
    Zero heavy dependencies; fully transparent, vectorized, and reproducible.
    """

    def __init__(self, config: Optional[WOAConfig] = None):
        self.config = config or WOAConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def _compute_a(self, t: int, max_iter: int) -> float:
        """
        Computes the decaying parameter 'a', which linearly decreases from 2 down to 0.
        Controls the range of vector A: when a > 1, |A| can be >= 1 (exploration).
        When a < 1, |A| is strictly < 1 (exploitation).
        """
        if self.config.a_decay == "nonlinear":
            # Nonlinear exponential decay: explores longer early on, converges sharply
            return 2.0 * np.cos((np.pi / 2.0) * (t / max_iter))
        # Standard linear decay (Mirjalili 2016 Eq. 2.3)
        return 2.0 - 2.0 * (t / max_iter)

    def optimize(
        self,
        objective_func: Callable[[np.ndarray], float],
        lb: Union[float, List[float], np.ndarray],
        ub: Union[float, List[float], np.ndarray],
        dim: int,
    ) -> WOAResult:
        """
        Executes the Whale Optimization Algorithm to minimize the objective function.

        Args:
            objective_func: Target function f(x) to MINIMIZE. Takes 1D array of shape (dim,).
            lb: Lower bound(s) for the decision variables.
            ub: Upper bound(s) for the decision variables.
            dim: Dimensionality of the search space (number of decision variables).

        Returns:
            WOAResult containing best position, best fitness, and telemetry curves.
        """
        start_time = time.perf_counter()

        # Convert bounds to numpy arrays
        lb_arr = np.full(dim, lb) if np.isscalar(lb) else np.asarray(lb, dtype=np.float64)
        ub_arr = np.full(dim, ub) if np.isscalar(ub) else np.asarray(ub, dtype=np.float64)
        assert len(lb_arr) == dim and len(ub_arr) == dim, "Bounds dimensionality mismatch"

        pop_size = self.config.pop_size
        max_iter = self.config.max_iter
        b = self.config.b_spiral

        # -------------------------------------------------------------
        # STEP 1: POPULATION INITIALIZATION
        # Uniform random distribution within hypercube [lb, ub]
        # -------------------------------------------------------------
        population = self.rng.uniform(lb_arr, ub_arr, size=(pop_size, dim))

        # Evaluate initial fitness of each whale
        fitness = np.zeros(pop_size, dtype=np.float64)
        for i in range(pop_size):
            fitness[i] = objective_func(population[i])

        # Identify initial best search agent (Leader X*)
        best_idx = np.argmin(fitness)
        best_fitness = float(fitness[best_idx])
        best_position = population[best_idx].copy()

        # Telemetry tracking
        convergence_curve = np.zeros(max_iter, dtype=np.float64)
        mean_fitness_curve = np.zeros(max_iter, dtype=np.float64)
        diversity_curve = np.zeros(max_iter, dtype=np.float64)
        trajectory: List[np.ndarray] = []

        total_evals = pop_size

        if self.config.verbose:
            print(f"{'='*75}")
            print(f" WHALE OPTIMIZATION ALGORITHM (WOA) - RUNTIME ENGINE")
            print(f"{'='*75}")
            print(f" Population Size: {pop_size:3d} Whales | Dimension: {dim:2d} | Max Iterations: {max_iter:3d}")
            print(f" Initial Best Fitness: {best_fitness:.6e}")
            print(f"{'-'*75}")
            print(f" {'Iter':>5} | {'Best Fitness':>16} | {'Mean Fitness':>16} | {'Swarm Diversity':>15} | {'a':>6}")
            print(f"{'-'*75}")

        # -------------------------------------------------------------
        # MAIN ITERATIVE EVOLUTION LOOP
        # -------------------------------------------------------------
        for t in range(max_iter):
            # Compute decay factor a: decreases from 2 to 0
            a = self._compute_a(t, max_iter)

            # Record swarm snapshot for trajectory tracking
            trajectory.append(population.copy())

            # Update each whale's position in the search space
            for i in range(pop_size):
                # Draw random numbers r1, r2 in [0, 1]
                r1 = self.rng.random(dim)
                r2 = self.rng.random(dim)

                # Compute coefficient vectors A and C (Eq. 2.3 & 2.4)
                A = 2.0 * a * r1 - a
                C = 2.0 * r2

                # Parameter l in [-1, 1] for logarithmic spiral
                l = self.rng.uniform(-1.0, 1.0)

                # Probability p in [0, 1] for dual hunting decision (Eq. 2.6)
                p = self.rng.random()

                if p < 0.5:
                    norm_A = np.linalg.norm(A) / np.sqrt(dim)
                    # Use component-wise or average norm threshold
                    # Mirjalili Eq. 2.7 & 2.8 branch on |A|
                    if np.mean(np.abs(A)) < 1.0:
                        # ------------------------------------------------------
                        # PHASE 2A: SHRINKING ENCIRCLING (EXPLOITATION)
                        # Whale moves closer towards the best leader X*
                        # D = | C * X*(t) - X(t) |
                        # X(t+1) = X*(t) - A * D
                        # ------------------------------------------------------
                        D = np.abs(C * best_position - population[i])
                        new_pos = best_position - A * D
                    else:
                        # ------------------------------------------------------
                        # PHASE 3: GLOBAL EXPLORATION (SEARCH FOR PREY)
                        # Diverge away from leader; follow a random whale peer
                        # D = | C * X_rand - X(t) |
                        # X(t+1) = X_rand - A * D
                        # ------------------------------------------------------
                        rand_whale_idx = self.rng.integers(0, pop_size)
                        X_rand = population[rand_whale_idx]
                        D = np.abs(C * X_rand - population[i])
                        new_pos = X_rand - A * D
                else:
                    # ----------------------------------------------------------
                    # PHASE 2B: LOGARITHMIC SPIRAL BUBBLE-NET ATTACK
                    # Whale swims along helix path towards leader X*
                    # D' = | X*(t) - X(t) |
                    # X(t+1) = D' * e^(b*l) * cos(2*pi*l) + X*(t)
                    # ----------------------------------------------------------
                    D_prime = np.abs(best_position - population[i])
                    spiral_factor = np.exp(b * l) * np.cos(2.0 * np.pi * l)
                    new_pos = D_prime * spiral_factor + best_position

                # -------------------------------------------------------------
                # BOUNDARY ENFORCEMENT: Clamp to [lb, ub]
                # -------------------------------------------------------------
                new_pos = np.clip(new_pos, lb_arr, ub_arr)
                population[i] = new_pos

                # Evaluate new fitness
                fit_val = objective_func(population[i])
                fitness[i] = fit_val
                total_evals += 1

                # Update best global solution if improved
                if fit_val < best_fitness:
                    best_fitness = float(fit_val)
                    best_position = population[i].copy()

            # Record telemetry for epoch t
            convergence_curve[t] = best_fitness
            mean_fitness_curve[t] = float(np.mean(fitness))
            # Swarm diversity: mean Euclidean distance from centroid
            centroid = np.mean(population, axis=0)
            diversity_curve[t] = float(np.mean(np.linalg.norm(population - centroid, axis=1)))

            # Logging
            if self.config.verbose and ((t + 1) % self.config.log_interval == 0 or t == max_iter - 1):
                print(
                    f" {t+1:5d} | {best_fitness:16.6e} | {mean_fitness_curve[t]:16.6e} | "
                    f"{diversity_curve[t]:15.4f} | {a:6.3f}"
                )

        exec_time = time.perf_counter() - start_time

        if self.config.verbose:
            print(f"{'-'*75}")
            print(f" [COMPLETE] Optimization Finished in {exec_time:.3f}s ({total_evals} evaluations)")
            print(f" Global Best Fitness: {best_fitness:.6e}")
            print(f" Global Best Solution: {np.round(best_position, 4)}")
            print(f"{'='*75}\n")

        return WOAResult(
            best_position=best_position,
            best_fitness=best_fitness,
            convergence_curve=convergence_curve,
            mean_fitness_curve=mean_fitness_curve,
            diversity_curve=diversity_curve,
            execution_time_sec=exec_time,
            total_evaluations=total_evals,
            trajectory=trajectory,
        )
