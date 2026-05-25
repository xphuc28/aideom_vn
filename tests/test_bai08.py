from __future__ import annotations

from src.bai08_dynamic import simulate_policy


def test_bai08_simulate_policy_returns_10_years():
    result = simulate_policy([0.25, 0.25, 0.25, 0.25])

    assert len(result) == 10


def test_bai08_y_and_c_are_positive():
    result = simulate_policy([0.25, 0.25, 0.25, 0.25])

    assert (result["Y"] > 0).all()
    assert (result["C"] > 0).all()
