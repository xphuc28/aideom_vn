from __future__ import annotations

from src.bai01_cobb_douglas import DEFAULT_PARAMS, run_bai01
from src.data_loader import load_macro


def test_run_bai01_returns_required_keys():
    results = run_bai01(load_macro(), DEFAULT_PARAMS)

    assert set(results) == {
        "result_df",
        "growth_df",
        "contribution_df",
        "mape",
        "forecast_2030_df",
    }


def test_bai01_mape_is_non_negative_number():
    results = run_bai01(load_macro(), DEFAULT_PARAMS)

    assert isinstance(results["mape"], float)
    assert results["mape"] >= 0


def test_bai01_contribution_df_is_not_empty():
    results = run_bai01(load_macro(), DEFAULT_PARAMS)

    assert not results["contribution_df"].empty
