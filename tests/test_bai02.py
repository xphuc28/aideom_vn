from __future__ import annotations

from src.bai02_lp_budget import solve_bai02_scipy


def test_bai02_status_optimal_with_default_budget():
    result = solve_bai02_scipy(B=100, min_human=20)

    assert result["status"] == "optimal"


def test_bai02_total_allocation_is_within_budget():
    result = solve_bai02_scipy(B=100, min_human=20)
    allocation_df = result["allocation_df"]

    assert allocation_df["allocation"].sum() <= 100 + 1e-7


def test_bai02_min_constraints_are_satisfied():
    result = solve_bai02_scipy(B=100, min_human=20)
    allocation_df = result["allocation_df"].set_index("variable")

    assert allocation_df.loc["x1", "allocation"] >= 25 - 1e-7
    assert allocation_df.loc["x2", "allocation"] >= 15 - 1e-7
    assert allocation_df.loc["x3", "allocation"] >= 20 - 1e-7
    assert allocation_df.loc["x4", "allocation"] >= 10 - 1e-7
