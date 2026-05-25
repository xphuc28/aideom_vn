from __future__ import annotations

from src.data_loader import data_summary, load_macro, load_regions, load_sectors


def test_load_macro_has_years():
    macro = load_macro()
    assert not macro.empty
    assert {"year", "GDP_growth_pct"}.issubset(macro.columns)


def test_load_sectors_has_sector_names():
    sectors = load_sectors()
    assert not sectors.empty
    assert {"sector_id", "sector_name_vi"}.issubset(sectors.columns)


def test_load_regions_has_region_names():
    regions = load_regions()
    assert not regions.empty
    assert {"region_id", "region_name_vi"}.issubset(regions.columns)


def test_data_summary_counts_rows():
    summary = data_summary()
    assert summary["macro_rows"] > 0
    assert summary["sector_rows"] > 0
    assert summary["region_rows"] > 0
