from __future__ import annotations

from src.bai10_stochastic import solve_stochastic_pulp


def test_bai10_first_stage_budget_is_within_limit():
    result = solve_stochastic_pulp()
    first_stage_df = result["first_stage_df"]

    assert first_stage_df["allocation"].sum() <= 65000 + 1e-7


def test_bai10_second_stage_budget_by_scenario_is_within_limit():
    result = solve_stochastic_pulp()
    second_stage_df = result["second_stage_df"]
    scenario_totals = second_stage_df.groupby("scenario")["allocation"].sum()

    assert (scenario_totals <= 15000 + 1e-7).all()


def test_bai10_ai_recourse_is_limited_by_h_first_stage():
    result = solve_stochastic_pulp()
    first_stage_df = result["first_stage_df"].set_index("item")
    second_stage_df = result["second_stage_df"]
    x_h = first_stage_df.loc["H", "allocation"]
    ai_rows = second_stage_df[second_stage_df["item"] == "AI"]

    assert (ai_rows["allocation"] <= 0.5 * x_h + 1e-7).all()
