"""
================================================================================
STANDARD BENCHMARK OPTIMIZATION FUNCTIONS
================================================================================
Defines canonical unimodal and multimodal benchmark functions with known global
optima, commonly used in literature (Mirjalili 2016) to validate metaheuristics.
================================================================================
"""

from typing import Dict, Tuple
import numpy as np


def sphere(x: np.ndarray) -> float:
    """Unimodal Sphere function: f(x) = sum(x_i^2). Optimum f(0,...,0) = 0."""
    return float(np.sum(x**2))


def rosenbrock(x: np.ndarray) -> float:
    """Valley-shaped Rosenbrock function: f(1,...,1) = 0."""
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rastrigin(x: np.ndarray) -> float:
    """Highly multimodal Rastrigin function with many local minima: f(0,...,0) = 0."""
    return float(10.0 * len(x) + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    """Multimodal Ackley function with steep global basin: f(0,...,0) = 0."""
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2.0 * np.pi * x))
    return float(-20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20.0 + np.e)


def griewank(x: np.ndarray) -> float:
    """Multimodal Griewank function: f(0,...,0) = 0."""
    sum_term = np.sum(x**2) / 4000.0
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return float(sum_term - prod_term + 1.0)


# Dictionary metadata for easy programmatic access
BENCHMARK_FUNCTIONS: Dict[str, Dict] = {
    "sphere": {
        "func": sphere,
        "lb": -100.0,
        "ub": 100.0,
        "optimal_val": 0.0,
        "type": "Unimodal",
        "description": "Smooth quadratic bowl; tests pure exploitation convergence speed.",
    },
    "rosenbrock": {
        "func": rosenbrock,
        "lb": -30.0,
        "ub": 30.0,
        "optimal_val": 0.0,
        "type": "Unimodal / Valley",
        "description": "Narrow curved parabolic valley; tests ability to navigate deceptive gradients.",
    },
    "rastrigin": {
        "func": rastrigin,
        "lb": -5.12,
        "ub": 5.12,
        "optimal_val": 0.0,
        "type": "Multimodal",
        "description": "Contains thousands of local traps; tests global exploration & trap escape.",
    },
    "ackley": {
        "func": ackley,
        "lb": -32.768,
        "ub": 32.768,
        "optimal_val": 0.0,
        "type": "Multimodal",
        "description": "Nearly flat outer region with a central deep well; tests exploratory searching.",
    },
    "griewank": {
        "func": griewank,
        "lb": -600.0,
        "ub": 600.0,
        "optimal_val": 0.0,
        "type": "Multimodal",
        "description": "High-frequency oscillatory component overlaid on quadratic bowl.",
    },
}
