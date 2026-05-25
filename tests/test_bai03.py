from __future__ import annotations

from src.bai03_priority import compute_priority
from src.data_loader import load_sectors


def test_bai03_normalized_values_are_between_zero_and_one():
    results = compute_priority(load_sectors(), weights=None, risk_mode="inverted")
    normalized_df = results["normalized_df"]
    norm_columns = [column for column in normalized_df.columns if column.endswith("_norm")]

    assert norm_columns
    assert (normalized_df[norm_columns] >= 0).all().all()
    assert (normalized_df[norm_columns] <= 1).all().all()


def test_bai03_ranking_has_all_sectors():
    sectors = load_sectors()
    results = compute_priority(sectors, weights=None, risk_mode="inverted")
    ranking_df = results["ranking_df"]

    assert len(ranking_df) == len(sectors)


def test_bai03_priority_is_not_null():
    results = compute_priority(load_sectors(), weights=None, risk_mode="inverted")
    ranking_df = results["ranking_df"]

    assert ranking_df["priority_score"].notna().all()
