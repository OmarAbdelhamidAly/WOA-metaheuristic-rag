"""
================================================================================
WOA EXAM PROBLEM SOLVER - STEP-BY-STEP MATHEMATICAL PROOFS
================================================================================
Provides programmatic, step-by-step hand solutions for standard examination
questions on Whale Optimization Algorithm (WOA).

Problems Solved:
    Problem 1: 1D Shrinking Encircling Step (|A| < 1, p < 0.5)
    Problem 2: 2D Logarithmic Spiral Attack (p >= 0.5)
    Problem 3: Global Exploration with Random Whale (|A| >= 1, p < 0.5)
================================================================================
"""

import math
from typing import Dict
import numpy as np


def solve_problem_1() -> Dict:
    """
    Problem 1: 1D Shrinking Encircling Step
    Given:
        Objective: Minimize f(x) = x^2
        Whales: X1 = 4, X2 = -2, X3 = 5
        Iteration: t = 1, T_max = 10
        Random draws: r1 = 0.30, r2 = 0.60, p = 0.20
    """
    print(f"\n{'='*75}")
    print(f" EXAM PROBLEM 1: 1D SHRINKING ENCIRCLING STEP (DETAILED SOLUTION)")
    print(f"{'='*75}")

    # Given data
    X1, X2, X3 = 4.0, -2.0, 5.0
    f_X1, f_X2, f_X3 = X1**2, X2**2, X3**2
    t, T_max = 1, 10
    r1, r2, p = 0.30, 0.60, 0.20

    print(f"[Given Parameters]")
    print(f"  - Target function: f(x) = x^2 (Global optimum x* = 0)")
    print(f"  - Initial Population: X1 = {X1}, X2 = {X2}, X3 = {X3}")
    print(f"  - Iteration: t = {t}, T_max = {T_max}")
    print(f"  - Random Draws: r1 = {r1}, r2 = {r2}, p = {p}")

    # Step 1: Identify Best Solution
    print(f"\n[Step 1: Identify Leader Whale X*]")
    print(f"  - f(X1) = {X1}^2 = {f_X1:.1f}")
    print(f"  - f(X2) = ({X2})^2 = {f_X2:.1f}  <-- LOWEST COST! Best Leader X* = {X2}")
    print(f"  - f(X3) = {X3}^2 = {f_X3:.1f}")
    X_star = X2

    # Step 2: Compute parameter a
    a = 2.0 - 2.0 * (t / T_max)
    print(f"\n[Step 2: Compute Decaying Parameter a]")
    print(f"  - a = 2 - 2 * (t / T_max) = 2 - 2 * ({t}/{T_max}) = 2 - 0.20 = {a:.2f}")

    # Step 3: Compute Vector A and Condition Check
    A = 2.0 * a * r1 - a
    print(f"\n[Step 3: Compute Vector A & Operator Condition]")
    print(f"  - A = 2 * a * r1 - a = 2 * ({a:.2f}) * ({r1:.2f}) - {a:.2f} = {A:.2f}")
    print(f"  - Condition Check:")
    print(f"      p = {p:.2f} < 0.5  --> Encircling Branch")
    print(f"      |A| = |{A:.2f}| = {abs(A):.2f} < 1.0 --> SHRINKING ENCIRCLING TRIGGERED!")

    # Step 4: Vector C and Distance D
    C = 2.0 * r2
    D = abs(C * X_star - X1)
    print(f"\n[Step 4: Compute Coefficient C & Distance D]")
    print(f"  - C = 2 * r2 = 2 * {r2:.2f} = {C:.2f}")
    print(f"  - D = | C * X* - X1 | = | {C:.2f} * ({X_star}) - {X1} | = | -6.40 | = {D:.2f}")

    # Step 5: Position Update
    X1_new = X_star - (A * D)
    f_X1_new = X1_new**2
    improvement = ((f_X1 - f_X1_new) / f_X1) * 100.0

    print(f"\n[Step 5: Position Update X1(t+1)]")
    print(f"  - X1(1) = X* - (A * D)")
    print(f"          = {X_star} - ({A:.2f} * {D:.2f})")
    print(f"          = {X_star} - ({-A * D:.3f}) = {X_star} + {A * D * -1:.3f} = {X1_new:+.3f}")
    print(f"  - New Cost: f({X1_new:.3f}) = {X1_new:.3f}^2 = {f_X1_new:.2f}")
    print(f"  - Initial Cost: {f_X1:.2f} --> New Cost: {f_X1_new:.2f} (Improvement: {improvement:.1f}%)")
    print(f"{'='*75}\n")

    return {
        "X_star": X_star,
        "a": a,
        "A": A,
        "C": C,
        "D": D,
        "X1_new": X1_new,
        "f_new": f_X1_new,
        "improvement_pct": improvement,
    }


def solve_problem_2() -> Dict:
    """
    Problem 2: 2D Logarithmic Spiral Attack
    Given:
        Objective: Minimize f(x, y) = x^2 + y^2
        Whales: Best X* = (1, 1), Update Whale X1 = (3, 2)
        Parameters: p = 0.85, b = 1.0, l = -0.50
    """
    print(f"\n{'='*75}")
    print(f" EXAM PROBLEM 2: 2D LOGARITHMIC SPIRAL ATTACK (DETAILED SOLUTION)")
    print(f"{'='*75}")

    X_star = np.array([1.0, 1.0])
    X1 = np.array([3.0, 2.0])
    p, b, l = 0.85, 1.0, -0.50

    f_X1_old = float(np.sum(X1**2))

    print(f"[Given Parameters]")
    print(f"  - Target function: f(x, y) = x^2 + y^2")
    print(f"  - Leader Whale X* = ({X_star[0]}, {X_star[1]}) with f(X*) = {float(np.sum(X_star**2))}")
    print(f"  - Whale to Update X1 = ({X1[0]}, {X1[1]}) with f(X1) = {f_X1_old}")
    print(f"  - Parameters: p = {p}, b = {b}, l = {l}")

    # Step 1: Decision
    print(f"\n[Step 1: Mechanism Decision]")
    print(f"  - Given p = {p} >= 0.5 --> LOGARITHMIC SPIRAL ATTACK TRIGGERED!")

    # Step 2: Distance Vector D'
    D_prime = np.abs(X_star - X1)
    print(f"\n[Step 2: Distance Vector D']")
    print(f"  - D' = | X* - X1 | = | ({X_star[0]} - {X1[0]}, {X_star[1]} - {X1[1]}) |")
    print(f"       = | (-2.0, -1.0) | = ({D_prime[0]:.1f}, {D_prime[1]:.1f})")

    # Step 3: Spiral factors
    exp_factor = math.exp(b * l)
    cos_factor = math.cos(2.0 * math.pi * l)
    print(f"\n[Step 3: Evaluate Spiral Mathematical Factors]")
    print(f"  - e^(b * l) = e^({b} * {l}) = e^(-0.5) = {exp_factor:.4f}")
    print(f"  - cos(2 * pi * l) = cos(2 * pi * (-0.5)) = cos(-pi) = {cos_factor:.3f}")

    # Step 4: Spiral displacement
    displacement = D_prime * exp_factor * cos_factor
    print(f"\n[Step 4: Compute Spiral Displacement Vector]")
    print(f"  - Displacement = D' * e^(b*l) * cos(2*pi*l)")
    print(f"                 = ({D_prime[0]}, {D_prime[1]}) * {exp_factor:.4f} * ({cos_factor:.1f})")
    print(f"                 = ({displacement[0]:.4f}, {displacement[1]:.4f})")

    # Step 5: Position update
    X1_new = displacement + X_star
    f_X1_new = float(np.sum(X1_new**2))
    improvement = ((f_X1_old - f_X1_new) / f_X1_old) * 100.0

    print(f"\n[Step 5: Position Update X1(t+1)]")
    print(f"  - X1(1) = Displacement + X*")
    print(f"          = ({displacement[0]:.4f} + {X_star[0]}, {displacement[1]:.4f} + {X_star[1]})")
    print(f"          = ({X1_new[0]:.4f}, {X1_new[1]:.4f})")
    print(f"  - New Cost: f({X1_new[0]:.3f}, {X1_new[1]:.3f}) = {f_X1_new:.4f}")
    print(f"  - Initial Cost: {f_X1_old:.2f} --> New Cost: {f_X1_new:.4f} (Improvement: {improvement:.1f}%)")
    print(f"{'='*75}\n")

    return {
        "D_prime": D_prime,
        "exp_factor": exp_factor,
        "cos_factor": cos_factor,
        "displacement": displacement,
        "X1_new": X1_new,
        "f_new": f_X1_new,
        "improvement_pct": improvement,
    }


def solve_problem_3() -> Dict:
    """
    Problem 3: Global Exploration Rule with Random Whale
    Given:
        Whale position X1 = 10, Leader X* = 2, Random Whale X_rand = 8
        Parameters: a = 1.80, r1 = 0.90, r2 = 0.50, p = 0.30
    """
    print(f"\n{'='*75}")
    print(f" EXAM PROBLEM 3: GLOBAL EXPLORATION RULE (|A| >= 1) (DETAILED SOLUTION)")
    print(f"{'='*75}")

    X1 = 10.0
    X_star = 2.0
    X_rand = 8.0
    a, r1, r2, p = 1.80, 0.90, 0.50, 0.30

    print(f"[Given Parameters]")
    print(f"  - Whale Position X1 = {X1}")
    print(f"  - Leader Whale X* = {X_star}")
    print(f"  - Random Peer Whale X_rand = {X_rand}")
    print(f"  - Parameters: a = {a}, r1 = {r1}, r2 = {r2}, p = {p}")

    # Step 1: Compute Vector A
    A = 2.0 * a * r1 - a
    print(f"\n[Step 1: Compute Vector A]")
    print(f"  - A = 2 * a * r1 - a = 2 * ({a:.2f}) * ({r1:.2f}) - {a:.2f} = 3.24 - 1.80 = {A:+.2f}")

    # Step 2: Evaluate Exploration Condition
    print(f"\n[Step 2: Evaluate Exploration Condition]")
    print(f"  - p = {p:.2f} < 0.5  --> Encircling Branch")
    print(f"  - |A| = |{A:+.2f}| = {abs(A):.2f} >= 1.0 --> GLOBAL EXPLORATION TRIGGERED!")
    print(f"  - CRITICAL EXAM RULE: Because |A| >= 1, the whale MUST IGNORE leader X*={X_star}")
    print(f"    and follow random peer X_rand = {X_rand} to escape local minima!")

    # Step 3: Vector C & Distance D
    C = 2.0 * r2
    D = abs(C * X_rand - X1)
    print(f"\n[Step 3: Vector C & Distance D from X_rand]")
    print(f"  - C = 2 * r2 = 2 * {r2:.2f} = {C:.2f}")
    print(f"  - D = | C * X_rand - X1 | = | {C:.2f} * {X_rand} - {X1} | = | 8.00 - 10.00 | = {D:.2f}")

    # Step 4: Position Update
    X1_new = X_rand - (A * D)
    print(f"\n[Step 4: Global Search Position Update X1(t+1)]")
    print(f"  - X1(1) = X_rand - (A * D)")
    print(f"          = {X_rand} - ({A:.2f} * {D:.2f})")
    print(f"          = {X_rand} - {A * D:.2f} = {X1_new:+.2f}")
    print(f"  - Takeaway: Whale moved to {X1_new:+.2f}, safely exploring far away from X*={X_star}.")
    print(f"{'='*75}\n")

    return {
        "A": A,
        "C": C,
        "D": D,
        "X1_new": X1_new,
    }


def solve_all_exam_problems():
    """Runs and outputs solutions for all 3 exam problems."""
    solve_problem_1()
    solve_problem_2()
    solve_problem_3()


if __name__ == "__main__":
    solve_all_exam_problems()
