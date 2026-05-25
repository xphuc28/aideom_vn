from __future__ import annotations

from src.bai05_mip_projects import solve_bai05


def test_bai05_selected_project_count_is_between_7_and_11_if_feasible():
    result = solve_bai05(total_budget=80000, early_budget=40000)

    assert result["feasibility"]
    assert 7 <= len(result["selected_df"]) <= 11


def test_bai05_p1_p2_mutual_exclusion_holds_without_force():
    result = solve_bai05(total_budget=80000, early_budget=40000, force_p1_p2=False)
    selected_ids = set(result["selected_df"]["project_id"])

    assert not {"P1", "P2"}.issubset(selected_ids)


def test_bai05_p14_is_selected():
    result = solve_bai05(total_budget=80000, early_budget=40000)
    selected_ids = set(result["selected_df"]["project_id"])

    assert "P14" in selected_ids
