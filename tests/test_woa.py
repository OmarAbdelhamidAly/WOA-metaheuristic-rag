"""
================================================================================
UNIT TESTS: WOA CORE & RAG SEARCH SPACE
================================================================================
Verifies algorithm correctness, boundary clamping, numerical reproducibility,
and exact match against the 3 examination problems.
================================================================================
"""

import numpy as np
import pytest

from core.woa import WhaleOptimizer, WOAConfig
from core.test_functions import sphere, rastrigin
from rag_pipeline.search_space import RAGSearchSpace, RAGHyperparameters
from exam_problems.exam_solver import solve_problem_1, solve_problem_2, solve_problem_3


def test_woa_sphere_convergence():
    """Tests that WOA converges to near-zero on unimodal Sphere function."""
    config = WOAConfig(pop_size=25, max_iter=40, seed=42, verbose=False)
    opt = WhaleOptimizer(config)
    res = opt.optimize(sphere, lb=-10.0, ub=10.0, dim=3)
    assert res.best_fitness < 1e-3, f"Failed to converge on Sphere: {res.best_fitness}"


def test_woa_boundary_enforcement():
    """Tests that all whale positions remain strictly inside bounds [lb, ub]."""
    lb = np.array([-5.0, 10.0, 0.0])
    ub = np.array([5.0, 20.0, 1.0])
    config = WOAConfig(pop_size=20, max_iter=15, seed=123, verbose=False)
    opt = WhaleOptimizer(config)
    res = opt.optimize(sphere, lb=lb, ub=ub, dim=3)

    assert np.all(res.best_position >= lb)
    assert np.all(res.best_position <= ub)


def test_reproducibility():
    """Tests that the same random seed produces identical results."""
    cfg1 = WOAConfig(pop_size=15, max_iter=20, seed=999, verbose=False)
    cfg2 = WOAConfig(pop_size=15, max_iter=20, seed=999, verbose=False)

    opt1 = WhaleOptimizer(cfg1)
    opt2 = WhaleOptimizer(cfg2)

    res1 = opt1.optimize(rastrigin, lb=-5.12, ub=5.12, dim=2)
    res2 = opt2.optimize(rastrigin, lb=-5.12, ub=5.12, dim=2)

    assert np.allclose(res1.best_position, res2.best_position)
    assert np.isclose(res1.best_fitness, res2.best_fitness)


def test_rag_search_space_constraints():
    """Tests physical constraint enforcement in RAG hyperparameter decoding."""
    # Test case where raw overlap exceeds 50% of chunk size
    vec = np.array([256.0, 200.0, 15.0, 0.7, 0.4])
    hp = RAGSearchSpace.decode(vec)

    assert hp.chunk_size == 256
    # Overlap must be <= 0.4 * 256 = 102.4
    assert hp.chunk_overlap <= 103
    assert 1 <= hp.top_k <= 20
    assert 0.0 <= hp.hybrid_alpha <= 1.0
    assert 0.0 <= hp.rerank_threshold <= 0.90


def test_exam_problem_1():
    """Verifies numerical precision of Exam Problem 1."""
    sol = solve_problem_1()
    assert sol["X_star"] == -2.0
    assert np.isclose(sol["a"], 1.80)
    assert np.isclose(sol["A"], -0.72)
    assert np.isclose(sol["C"], 1.20)
    assert np.isclose(sol["D"], 6.40)
    assert np.isclose(sol["X1_new"], 2.608, atol=1e-3)
    assert np.isclose(sol["f_new"], 6.801664, atol=1e-2)


def test_exam_problem_2():
    """Verifies numerical precision of Exam Problem 2."""
    sol = solve_problem_2()
    assert np.allclose(sol["D_prime"], np.array([2.0, 1.0]))
    assert np.isclose(sol["exp_factor"], 0.60653, atol=1e-4)
    assert np.isclose(sol["cos_factor"], -1.0, atol=1e-4)
    assert np.isclose(sol["X1_new"][0], -0.2131, atol=1e-3)
    assert np.isclose(sol["X1_new"][1], 0.3935, atol=1e-3)
    assert np.isclose(sol["f_new"], 0.2002, atol=1e-3)


def test_exam_problem_3():
    """Verifies numerical precision of Exam Problem 3."""
    sol = solve_problem_3()
    assert np.isclose(sol["A"], 1.44)
    assert np.isclose(sol["C"], 1.00)
    assert np.isclose(sol["D"], 2.00)
    assert np.isclose(sol["X1_new"], 5.12, atol=1e-2)
