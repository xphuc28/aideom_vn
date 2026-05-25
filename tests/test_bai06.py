from __future__ import annotations

from src.bai06_topsis import entropy_weights, topsis
from src.data_loader import load_regions


def test_bai06_scores_are_between_zero_and_one():
    results = topsis(load_regions())
    ranking = results["ranking_expert"]

    assert (ranking["topsis_score"] >= 0).all()
    assert (ranking["topsis_score"] <= 1).all()


def test_bai06_ranking_is_not_empty():
    results = topsis(load_regions())

    assert not results["ranking_expert"].empty
    assert not results["ranking_entropy"].empty


def test_bai06_entropy_weights_sum_to_one():
    results = topsis(load_regions())
    matrix = results["ranking_expert"][
        [
            "grdp_per_capita",
            "fdi_registered",
            "digital_index",
            "ai_readiness",
            "trained_labor",
            "rd_intensity",
            "internet_penetration",
            "gini",
        ]
    ]
    weights = entropy_weights(matrix)

    assert abs(weights.sum() - 1.0) < 1e-9
