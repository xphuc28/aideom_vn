"""Data loading utilities for AIDEOM-VN.

The dashboard pages should use these functions instead of reading CSV files
directly. Keeping data access in one module makes local runs, tests, and
Streamlit Cloud deployment easier to maintain.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MACRO_FILE = "vietnam_macro_2020_2025.csv"
SECTORS_FILE = "vietnam_sectors_2024.csv"
REGIONS_FILE = "vietnam_regions_2024.csv"

DATA_FILES = {
    "macro": MACRO_FILE,
    "sectors": SECTORS_FILE,
    "regions": REGIONS_FILE,
}


def _read_csv(filename: str) -> pd.DataFrame:
    """Read a project CSV file from the data directory."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    return pd.read_csv(path)


def load_macro() -> pd.DataFrame:
    """Load Vietnam macro indicators for 2020-2025."""
    df = _read_csv(MACRO_FILE)
    if "year" in df.columns:
        df["year"] = df["year"].astype(int)
    return df


def load_sectors() -> pd.DataFrame:
    """Load 2024 sector-level economic and digital readiness indicators."""
    df = _read_csv(SECTORS_FILE)
    if "sector_id" in df.columns:
        df["sector_id"] = df["sector_id"].astype(int)
    return df


def load_regions() -> pd.DataFrame:
    """Load 2024 region-level economic, social, and digital indicators."""
    df = _read_csv(REGIONS_FILE)
    if "region_id" in df.columns:
        df["region_id"] = df["region_id"].astype(int)
    return df


def data_status() -> dict[str, dict[str, object]]:
    """Return existence, shape, and columns for required CSV files."""
    status: dict[str, dict[str, object]] = {}

    for dataset_name, filename in DATA_FILES.items():
        path = DATA_DIR / filename
        item: dict[str, object] = {
            "filename": filename,
            "relative_path": f"data/{filename}",
            "exists": path.exists(),
            "shape": None,
            "columns": [],
            "error": None,
        }

        if path.exists():
            try:
                df = pd.read_csv(path)
                item["shape"] = df.shape
                item["columns"] = list(df.columns)
            except Exception as exc:
                item["error"] = str(exc)

        status[dataset_name] = item

    return status


def data_summary() -> dict[str, int]:
    """Return a compact summary of available rows for the home page."""
    return {
        "macro_rows": len(load_macro()),
        "sector_rows": len(load_sectors()),
        "region_rows": len(load_regions()),
    }
