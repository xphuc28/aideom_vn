from __future__ import annotations

from src.bai04_region_lp import H_MIN, REGION_MAX, REGION_MIN, solve_bai04_pulp


def test_bai04_total_budget_is_within_limit():
    result = solve_bai04_pulp(budget=50000, fairness=True, lambda_=0.7)
    allocation_matrix = result["allocation_matrix"]

    assert allocation_matrix.to_numpy().sum() <= 50000 + 1e-6


def test_bai04_each_region_satisfies_floor_and_cap():
    result = solve_bai04_pulp(budget=50000, fairness=True, lambda_=0.7)
    region_totals = result["allocation_matrix"].sum(axis=1)

    assert (region_totals >= REGION_MIN - 1e-6).all()
    assert (region_totals <= REGION_MAX + 1e-6).all()


def test_bai04_human_capital_minimum_is_satisfied():
    result = solve_bai04_pulp(budget=50000, fairness=True, lambda_=0.7)
    allocation_matrix = result["allocation_matrix"]

    assert allocation_matrix["H"].sum() >= H_MIN - 1e-6
