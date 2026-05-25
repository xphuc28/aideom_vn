from __future__ import annotations

from src.bai09_labor_ai import solve_bai09


def test_bai09_netjob_is_nonnegative_if_optimal():
    result = solve_bai09()

    assert result["status"] == "optimal"
    assert (result["allocation_df"]["NetJob"] >= -1e-7).all()


def test_bai09_displaced_is_not_above_retraining_capacity():
    result = solve_bai09()
    df = result["allocation_df"]

    assert result["status"] == "optimal"
    assert (df["DisplacedJob"] <= df["RetrainingCapacity"] + 1e-7).all()
