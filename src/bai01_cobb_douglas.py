"""Bai 1: Extended Cobb-Douglas model with AI and digitalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import unicodedata

import numpy as np
import pandas as pd


MODULE_TITLE = "Bài 1 - Cobb-Douglas mở rộng với AI và số hóa"

DEFAULT_PARAMS = {
    "alpha": 0.33,
    "beta": 0.42,
    "gamma": 0.10,
    "delta": 0.08,
    "theta": 0.07,
}

EPSILON = 1e-9


@dataclass(frozen=True)
class ColumnMap:
    """Resolved input columns used by the model."""

    year: str | None
    output: str
    capital: str | None
    labor: str | None
    digital: str | None
    ai: str | None
    human_capital: str | None
    fdi: str | None
    productivity: str | None
    population: str | None


def flexible_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Return the first matching column from a flexible list of candidates.

    Matching is case-insensitive and also ignores spaces, underscores, hyphens,
    percent symbols, and accents where possible. This lets the model tolerate
    CSV files with slightly different naming conventions.
    """

    def normalize(value: str) -> str:
        replacements = {
            "%": "pct",
            " ": "",
            "_": "",
            "-": "",
            ".": "",
            "/": "",
            "(": "",
            ")": "",
        }
        normalized = str(value).strip().lower().replace("đ", "d")
        normalized = unicodedata.normalize("NFD", normalized)
        normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        return normalized

    exact_lookup = {str(column).strip().lower(): column for column in df.columns}
    normalized_lookup = {normalize(column): column for column in df.columns}

    for candidate in candidates:
        candidate_text = str(candidate).strip().lower()
        if candidate_text in exact_lookup:
            return exact_lookup[candidate_text]

        normalized_candidate = normalize(candidate)
        if normalized_candidate in normalized_lookup:
            return normalized_lookup[normalized_candidate]

        if len(normalized_candidate) >= 3:
            for normalized_column, original_column in normalized_lookup.items():
                if normalized_candidate in normalized_column:
                    return original_column

    return None


def module_status() -> str:
    """Return a short implementation status for the Streamlit page."""
    return "Đã triển khai mô hình Cobb-Douglas mở rộng, phân rã tăng trưởng và mô phỏng 2030."


def _positive_numeric(series: pd.Series, fallback: float = 1.0) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().any():
        values = values.interpolate(limit_direction="both").fillna(values.mean())
    else:
        values = pd.Series(fallback, index=series.index, dtype=float)
    return values.astype(float).clip(lower=EPSILON)


def _scaled_proxy(series: pd.Series, lower: float, upper: float) -> pd.Series:
    values = _positive_numeric(series)
    span = values.max() - values.min()
    if span <= EPSILON:
        return pd.Series((lower + upper) / 2, index=series.index, dtype=float)
    return lower + (values - values.min()) / span * (upper - lower)


def _resolve_columns(df: pd.DataFrame) -> ColumnMap:
    year = flexible_column(df, ["year", "nam", "năm", "time", "period"])
    output = flexible_column(
        df,
        [
            "Y",
            "GDP",
            "GDP_trillion_VND",
            "GDP_billion_USD",
            "gdp_current",
            "output",
            "san_luong",
        ],
    )
    if output is None:
        raise ValueError("Không tìm thấy cột GDP/Y để làm biến phụ thuộc Y_t.")

    capital = flexible_column(
        df,
        [
            "K",
            "capital",
            "capital_stock",
            "capital_stock_trillion_VND",
            "gross_capital",
            "von",
            "vốn",
        ],
    )
    labor = flexible_column(
        df,
        ["L", "labor", "labor_million", "employment_million", "workforce_million", "lao_dong"],
    )
    digital = flexible_column(
        df,
        [
            "D",
            "digital",
            "digital_economy_share_GDP_pct",
            "digital_share_pct",
            "digital_index",
            "digital_index_0_100",
        ],
    )
    ai = flexible_column(
        df,
        [
            "AI",
            "ai",
            "ai_firms_thousand",
            "ai_enterprises_thousand",
            "ai_adoption",
            "ai_readiness_0_100",
        ],
    )
    human_capital = flexible_column(
        df,
        [
            "H",
            "human_capital",
            "human_capital_pct",
            "trained_labor_pct",
            "skilled_labor_pct",
            "education_index",
        ],
    )
    fdi = flexible_column(df, ["FDI_disbursed_billion_USD", "fdi", "investment", "I"])
    productivity = flexible_column(
        df,
        ["labor_productivity_million_VND", "labor_productivity", "productivity"],
    )
    population = flexible_column(df, ["population_million", "population", "pop_million"])

    return ColumnMap(
        year=year,
        output=output,
        capital=capital,
        labor=labor,
        digital=digital,
        ai=ai,
        human_capital=human_capital,
        fdi=fdi,
        productivity=productivity,
        population=population,
    )


def _prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = _resolve_columns(df)
    model = pd.DataFrame(index=df.index)

    if columns.year:
        model["year"] = pd.to_numeric(df[columns.year], errors="coerce").astype("Int64")
    else:
        model["year"] = np.arange(1, len(df) + 1)

    model["Y_actual"] = _positive_numeric(df[columns.output])

    if columns.capital:
        model["K"] = _positive_numeric(df[columns.capital])
    else:
        if columns.fdi:
            investment = _positive_numeric(df[columns.fdi])
            base_capital = model["Y_actual"].iloc[0] * 2.8
            capital_values = [base_capital]
            for value in investment.iloc[1:]:
                capital_values.append(capital_values[-1] * 0.94 + value * 25.0)
            model["K"] = capital_values
        else:
            model["K"] = model["Y_actual"] * 2.8

    if columns.labor:
        model["L"] = _positive_numeric(df[columns.labor])
    elif columns.population:
        model["L"] = _positive_numeric(df[columns.population]) * 0.49
    elif columns.productivity:
        model["L"] = model["Y_actual"] / _positive_numeric(df[columns.productivity])
    else:
        model["L"] = pd.Series(np.linspace(45.0, 52.0, len(df)), index=df.index)

    if columns.digital:
        model["D"] = _positive_numeric(df[columns.digital])
    else:
        model["D"] = pd.Series(np.linspace(10.0, 20.0, len(df)), index=df.index)

    if columns.ai:
        model["AI"] = _positive_numeric(df[columns.ai])
    else:
        model["AI"] = _scaled_proxy(model["D"], 25.0, 65.0)

    if columns.human_capital:
        model["H"] = _positive_numeric(df[columns.human_capital])
    elif columns.productivity:
        model["H"] = _scaled_proxy(df[columns.productivity], 25.0, 35.0)
    else:
        model["H"] = pd.Series(np.linspace(25.0, 35.0, len(df)), index=df.index)

    return model.sort_values("year").reset_index(drop=True)


def _coefficient_params(params: dict | None) -> dict[str, float]:
    merged = DEFAULT_PARAMS.copy()
    if params:
        merged.update({key: float(value) for key, value in params.items() if key in merged})
    return merged


def _production(df: pd.DataFrame, params: dict[str, float], tfp: pd.Series | float) -> pd.Series:
    return (
        tfp
        * np.power(df["K"], params["alpha"])
        * np.power(df["L"], params["beta"])
        * np.power(df["D"], params["gamma"])
        * np.power(df["AI"], params["delta"])
        * np.power(df["H"], params["theta"])
    )


def _growth_decomposition(model: pd.DataFrame, params: dict[str, float]) -> pd.DataFrame:
    log_values = np.log(model[["Y_actual", "TFP_A", "K", "L", "D", "AI", "H"]])
    growth = log_values.diff().iloc[1:] * 100.0
    years = model["year"].iloc[1:].astype(int).to_numpy()

    growth_df = pd.DataFrame(
        {
            "year": years,
            "GDP_log_growth_pct": growth["Y_actual"].to_numpy(),
            "TFP_contribution_pct_points": growth["TFP_A"].to_numpy(),
            "K_contribution_pct_points": params["alpha"] * growth["K"].to_numpy(),
            "L_contribution_pct_points": params["beta"] * growth["L"].to_numpy(),
            "D_contribution_pct_points": params["gamma"] * growth["D"].to_numpy(),
            "AI_contribution_pct_points": params["delta"] * growth["AI"].to_numpy(),
            "H_contribution_pct_points": params["theta"] * growth["H"].to_numpy(),
        }
    )
    component_columns = [
        "TFP_contribution_pct_points",
        "K_contribution_pct_points",
        "L_contribution_pct_points",
        "D_contribution_pct_points",
        "AI_contribution_pct_points",
        "H_contribution_pct_points",
    ]
    growth_df["model_log_growth_pct"] = growth_df[component_columns].sum(axis=1)
    return growth_df


def _average_contribution(growth_df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "K": "K_contribution_pct_points",
        "L": "L_contribution_pct_points",
        "D": "D_contribution_pct_points",
        "AI": "AI_contribution_pct_points",
        "H": "H_contribution_pct_points",
        "TFP": "TFP_contribution_pct_points",
    }

    rows = []
    for factor, column in mapping.items():
        average = float(growth_df[column].mean()) if not growth_df.empty else 0.0
        rows.append({"factor": factor, "avg_contribution_pct_points": average})

    contribution_df = pd.DataFrame(rows)
    total_abs = contribution_df["avg_contribution_pct_points"].abs().sum()
    if total_abs > EPSILON:
        contribution_df["share_abs_pct"] = (
            contribution_df["avg_contribution_pct_points"].abs() / total_abs * 100.0
        )
    else:
        contribution_df["share_abs_pct"] = 0.0
    return contribution_df


def _forecast_to_2030(model: pd.DataFrame, params: dict[str, float]) -> pd.DataFrame:
    base = model.iloc[-1]
    base_year = int(base["year"])
    target_year = 2030
    years = np.arange(base_year, target_year + 1)
    horizon = max(target_year - base_year, 1)
    rows = []

    for year in years:
        step = int(year - base_year)
        progress = min(max(step / horizon, 0.0), 1.0)
        values = {
            "year": int(year),
            "K": float(base["K"]) * (1.06**step),
            "L": float(base["L"]) * (1.06**step),
            "D": float(base["D"]) + (30.0 - float(base["D"])) * progress,
            "AI": float(base["AI"]) + (100.0 - float(base["AI"])) * progress,
            "H": float(base["H"]) + (35.0 - float(base["H"])) * progress,
            "TFP_A": float(base["TFP_A"]) * (1.012**step),
        }
        values["Y_forecast"] = float(
            values["TFP_A"]
            * values["K"] ** params["alpha"]
            * values["L"] ** params["beta"]
            * values["D"] ** params["gamma"]
            * values["AI"] ** params["delta"]
            * values["H"] ** params["theta"]
        )
        rows.append(values)

    return pd.DataFrame(rows)


def run_bai01(df: pd.DataFrame, params: dict | None = None) -> dict[str, object]:
    """Run the extended Cobb-Douglas model and return all result tables."""
    if df.empty:
        raise ValueError("Dữ liệu đầu vào rỗng.")

    coefficients = _coefficient_params(params)
    model = _prepare_model_frame(df)
    production_without_tfp = _production(model, coefficients, 1.0)
    model["TFP_A"] = model["Y_actual"] / production_without_tfp

    tfp_geomean = float(np.exp(np.log(model["TFP_A"]).mean()))
    model["Y_hat"] = _production(model, coefficients, tfp_geomean)
    model["error_pct"] = ((model["Y_hat"] - model["Y_actual"]).abs() / model["Y_actual"]) * 100.0

    result_df = model[["year", "Y_actual", "TFP_A", "Y_hat", "error_pct"]].copy()
    growth_df = _growth_decomposition(model, coefficients)
    contribution_df = _average_contribution(growth_df)
    forecast_2030_df = _forecast_to_2030(model, coefficients)
    mape = float(result_df["error_pct"].mean())

    return {
        "result_df": result_df,
        "growth_df": growth_df,
        "contribution_df": contribution_df,
        "mape": mape,
        "forecast_2030_df": forecast_2030_df,
    }
