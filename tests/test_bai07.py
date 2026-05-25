from __future__ import annotations

import numpy as np

from src.bai07_pareto import _candidate_vectors, _is_feasible, evaluate_solution, random_feasible_search, run_nsga2


def test_bai07_evaluate_solution_returns_four_objectives():
    x = np.zeros(24)
    result = evaluate_solution(x)

    assert set(result) == {"gdp_gain", "inequality", "emission", "net_cyber_risk"}


def test_bai07_random_fallback_creates_non_empty_pareto_df():
    result = random_feasible_search(n_samples=100, seed=123)

    assert not result["pareto_df"].empty


def test_bai07_invalid_vector_is_not_feasible_instead_of_raising():
    assert _is_feasible(np.zeros(1), budget=50000, fairness=True, lambda_=0.7) is False


def test_bai07_candidate_vectors_filters_partial_optimizer_output():
    candidates = _candidate_vectors([np.zeros(1), np.zeros(24)])

    assert len(candidates) == 1
    assert candidates[0].shape == (24,)


def test_bai07_run_nsga2_returns_result_without_redacted_streamlit_error():
    result = run_nsga2(pop_size=20, n_gen=10, seed=7)

    assert set(["pareto_df", "compromise_solution", "summary_df", "note", "method"]).issubset(result)
