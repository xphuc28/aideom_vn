from __future__ import annotations

from src.scenario_engine import run_all_scenarios


def test_run_all_scenarios_returns_five_scenarios():
    result = run_all_scenarios()

    assert len(result) == 5
    assert set(result["scenario"]) == {"S1", "S2", "S3", "S4", "S5"}


def test_scenario_engine_core_kpis_are_not_null():
    result = run_all_scenarios()
    kpis = [
        "GDP_gain",
        "Digital_score",
        "AI_score",
        "Human_capital_score",
        "NetJob",
        "Inequality_risk",
        "Cyber_risk",
        "Emission_risk",
        "Overall_score",
    ]

    assert result[kpis].notna().all().all()


def test_overall_score_is_in_reasonable_scale():
    result = run_all_scenarios()

    assert (result["Overall_score"] >= 0).all()
    assert (result["Overall_score"] <= 100).all()
