from __future__ import annotations

import numpy as np

from src.bai07_pareto import evaluate_solution, random_feasible_search


def test_bai07_evaluate_solution_returns_four_objectives():
    x = np.zeros(24)
    result = evaluate_solution(x)

    assert set(result) == {"gdp_gain", "inequality", "emission", "net_cyber_risk"}


def test_bai07_random_fallback_creates_non_empty_pareto_df():
    result = random_feasible_search(n_samples=100, seed=123)

    assert not result["pareto_df"].empty
