#!/usr/bin/env python3
"""
analyze_competing_risks.py — Gap 3: Competing-Risks Decomposition vs Naive Filtering

Demonstrates the methodological correction for handling planning failure:
Comparing biased (naively filtered, dropping plan failures) vs unbiased competing-risks
safety estimates:
  - Naive Filtered Model: P_naive(Y^H=1 | do(c)) = P(Y^H=1 | do(c), B^H=0)
  - Competing-Risks Model: P_total_fail(do(c)) = P(B^H=1 | do(c)) + P(Y^H=1 | do(c), B^H=0) * (1 - P(B^H=1 | do(c)))

Pure Python implementation using stdlib random and math.
"""

import math
import random


def main():
    print("=" * 75)
    print("Gap 3: Competing-Risks Decomposition vs Naive Filtering Analysis")
    print("=" * 75)
    
    random.seed(42)
    n_trials = 500
    
    # Configuration 1: Moderate Baseline
    c1_plan_fail_rate = 0.05
    c1_coll_given_plan_rate = 0.10
    
    c1_plan_fails = sum(1 for _ in range(n_trials) if random.random() < c1_plan_fail_rate)
    c1_navigated = n_trials - c1_plan_fails
    c1_collisions = sum(1 for _ in range(c1_navigated) if random.random() < c1_coll_given_plan_rate)
    
    # Configuration 2: Hyper-Restrictive Configuration (Carry Arm / High Inflation in Narrow Channel)
    c2_plan_fail_rate = 0.60
    c2_coll_given_plan_rate = 0.02
    
    c2_plan_fails = sum(1 for _ in range(n_trials) if random.random() < c2_plan_fail_rate)
    c2_navigated = n_trials - c2_plan_fails
    c2_collisions = sum(1 for _ in range(c2_navigated) if random.random() < c2_coll_given_plan_rate)
    
    # Computations for Config 1
    p1_plan_fail = c1_plan_fails / n_trials
    p1_coll_given_plan = c1_collisions / c1_navigated if c1_navigated > 0 else 0.0
    p1_naive_risk = p1_coll_given_plan
    p1_unbiased_total = p1_plan_fail + p1_coll_given_plan * (1.0 - p1_plan_fail)
    
    # Computations for Config 2
    p2_plan_fail = c2_plan_fails / n_trials
    p2_coll_given_plan = c2_collisions / c2_navigated if c2_navigated > 0 else 0.0
    p2_naive_risk = p2_coll_given_plan
    p2_unbiased_total = p2_plan_fail + p2_coll_given_plan * (1.0 - p2_plan_fail)
    
    print("\nMethodological Comparison Table:")
    print("-" * 75)
    print(f"{'Metric':<35} | {'Moderate Baseline':<16} | {'Hyper-Restrictive':<16}")
    print("-" * 75)
    print(f"{'Total Trials (N)':<35} | {n_trials:<16} | {n_trials:<16}")
    print(f"{'P(Plan Fail)':<35} | {p1_plan_fail:<16.3f} | {p2_plan_fail:<16.3f}")
    print(f"{'P(Collision | Plan Succeeded)':<35} | {p1_coll_given_plan:<16.3f} | {p2_coll_given_plan:<16.3f}")
    print(f"{'Naive Filtered Risk (BIASED)':<35} | {p1_naive_risk:<16.3f} | {p2_naive_risk:<16.3f}")
    print(f"{'Competing-Risks Risk (UNBIASED)':<35} | {p1_unbiased_total:<16.3f} | {p2_unbiased_total:<16.3f}")
    print(f"{'Distortion Error (Naive - Unbiased)':<35} | {p1_naive_risk - p1_unbiased_total:<16.3f} | {p2_naive_risk - p2_unbiased_total:<16.3f}")
    print("-" * 75)
    
    print("\nKey Takeaway:")
    print(f"  Under NAIVE FILTERING, Hyper-Restrictive appears SAFER ({p2_naive_risk:.3f} vs {p1_naive_risk:.3f})!")
    print(f"  Under COMPETING-RISKS, Hyper-Restrictive is correctly revealed as WORSE ({p2_unbiased_total:.3f} vs {p1_unbiased_total:.3f}).")
    print(f"  Dropping planning failures creates severe post-treatment collider bias ({p2_naive_risk - p2_unbiased_total:.3f} distortion).")
    print("=" * 75)


if __name__ == "__main__":
    main()
