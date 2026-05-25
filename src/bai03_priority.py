"""Bai 3: Priority ranking for 10 Vietnamese economic sectors."""

from __future__ import annotations

from typing import Iterable
import unicodedata

import numpy as np
import pandas as pd


MODULE_TITLE = "Bài 3 - Priority 10 ngành Việt Nam"

DEFAULT_WEIGHTS = {
    "growth": 0.18,
    "productivity": 0.16,
    "spillover": 0.16,
    "export": 0.14,
    "labor": 0.10,
    "ai_readiness": 0.16,
    "automation_risk": 0.10,
}

GROWTH_ORIENTED_WEIGHTS = {
    "growth": 0.25,
    "productivity": 0.20,
    "spillover": 0.15,
    "export": 0.18,
    "labor": 0.06,
    "ai_readiness": 0.12,
    "automation_risk": 0.04,
}

INCLUSIVE_ORIENTED_WEIGHTS = {
    "growth": 0.12,
    "productivity": 0.10,
    "spillover": 0.13,
    "export": 0.08,
    "labor": 0.25,
    "ai_readiness": 0.12,
    "automation_risk": 0.20,
}

CRITERIA = [
    "growth",
    "productivity",
    "spillover",
    "export",
    "labor",
    "ai_readiness",
    "automation_risk",
]


def module_status() -> str:
    """Return a short implementation status for the Streamlit page."""
    return "Đã triển khai xếp hạng ưu tiên ngành, nhạy cảm trọng số AI và so sánh chính sách."


def _normalize_name(value: str) -> str:
    text = str(value).strip().lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    for old, new in {
        "%": "pct",
        " ": "",
        "_": "",
        "-": "",
        ".": "",
        "/": "",
        "(": "",
        ")": "",
    }.items():
        text = text.replace(old, new)
    return text


def _flexible_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    exact_lookup = {str(column).strip().lower(): column for column in df.columns}
    normalized_lookup = {_normalize_name(column): column for column in df.columns}

    for candidate in candidates:
        exact_candidate = str(candidate).strip().lower()
        if exact_candidate in exact_lookup:
            return exact_lookup[exact_candidate]

        normalized_candidate = _normalize_name(candidate)
        if normalized_candidate in normalized_lookup:
            return normalized_lookup[normalized_candidate]

        if len(normalized_candidate) >= 3:
            for normalized_column, original_column in normalized_lookup.items():
                if normalized_candidate in normalized_column:
                    return original_column
    return None


def normalize_minmax(values: pd.Series | np.ndarray | list[float], invert: bool = False) -> pd.Series:
    """Normalize numeric values to [0, 1] with optional cost-to-benefit inversion."""
    series = pd.Series(values, dtype="float64")
    if series.notna().any():
        series = series.interpolate(limit_direction="both").fillna(series.mean())
    else:
        series = pd.Series(0.0, index=series.index, dtype="float64")

    min_value = float(series.min())
    max_value = float(series.max())
    if np.isclose(max_value, min_value):
        normalized = pd.Series(0.5, index=series.index, dtype="float64")
    else:
        normalized = (series - min_value) / (max_value - min_value)

    if invert:
        normalized = 1.0 - normalized
    return normalized.clip(0.0, 1.0)


def _normalized_weights(weights: dict[str, float] | None) -> dict[str, float]:
    merged = DEFAULT_WEIGHTS.copy()
    if weights:
        for key, value in weights.items():
            if key in merged:
                merged[key] = max(float(value), 0.0)

    total = sum(merged.values())
    if total <= 0:
        return DEFAULT_WEIGHTS.copy()
    return {key: value / total for key, value in merged.items()}


def _risk_flags(risk_mode: str | dict | None) -> tuple[bool, bool]:
    if risk_mode is None:
        return False, True

    if isinstance(risk_mode, str):
        text = risk_mode.strip().lower()
        if text in {"cost", "risk_as_cost", "subtract"}:
            return True, False
        if text in {"inverted", "risk_inverted", "benefit"}:
            return False, True

    if isinstance(risk_mode, dict):
        risk_as_cost = bool(risk_mode.get("risk_as_cost", False))
        risk_inverted = bool(risk_mode.get("risk_inverted", not risk_as_cost))
        if risk_as_cost and risk_inverted:
            risk_as_cost = False
        return risk_as_cost, risk_inverted

    return False, True


def _sector_names(df: pd.DataFrame) -> pd.Series:
    name_column = _flexible_column(df, ["sector_name_vi", "sector_name", "nganh", "sector"])
    if name_column:
        return df[name_column].astype(str)
    return pd.Series([f"Ngành {index + 1}" for index in range(len(df))], index=df.index)


def _raw_criteria(df: pd.DataFrame) -> pd.DataFrame:
    growth_col = _flexible_column(df, ["growth_rate_2024_pct", "growth_rate", "growth", "tang_truong"])
    gdp_share_col = _flexible_column(df, ["gdp_share_2024_pct", "gdp_share", "gdp_pct"])
    productivity_col = _flexible_column(
        df,
        ["productivity", "labor_productivity", "labor_productivity_million_VND", "nang_suat"],
    )
    spillover_col = _flexible_column(df, ["spillover_coef_0_1", "spillover", "lan_toa"])
    export_col = _flexible_column(df, ["export_billion_USD", "exports_billion_USD", "export", "exports"])
    labor_col = _flexible_column(df, ["labor_million", "labor", "employment", "lao_dong"])
    ai_col = _flexible_column(df, ["ai_readiness_0_100", "ai_readiness", "ai"])
    risk_col = _flexible_column(df, ["automation_risk_pct", "automation_risk", "risk"])

    missing = [
        label
        for label, column in {
            "growth": growth_col,
            "spillover": spillover_col,
            "export": export_col,
            "labor": labor_col,
            "ai_readiness": ai_col,
            "automation_risk": risk_col,
        }.items()
        if column is None
    ]
    if missing:
        raise ValueError(f"Thiếu cột tiêu chí bắt buộc: {', '.join(missing)}")

    raw = pd.DataFrame(index=df.index)
    raw["sector_name"] = _sector_names(df)
    raw["growth"] = pd.to_numeric(df[growth_col], errors="coerce")

    if productivity_col:
        raw["productivity"] = pd.to_numeric(df[productivity_col], errors="coerce")
    elif gdp_share_col and labor_col:
        gdp_share = pd.to_numeric(df[gdp_share_col], errors="coerce")
        labor = pd.to_numeric(df[labor_col], errors="coerce").replace(0, np.nan)
        raw["productivity"] = gdp_share / labor
    elif gdp_share_col:
        raw["productivity"] = pd.to_numeric(df[gdp_share_col], errors="coerce")
    else:
        raw["productivity"] = raw["growth"]

    raw["spillover"] = pd.to_numeric(df[spillover_col], errors="coerce")
    raw["export"] = pd.to_numeric(df[export_col], errors="coerce")
    raw["labor"] = pd.to_numeric(df[labor_col], errors="coerce")
    raw["ai_readiness"] = pd.to_numeric(df[ai_col], errors="coerce")
    raw["automation_risk"] = pd.to_numeric(df[risk_col], errors="coerce")
    return raw


def compute_priority(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
    risk_mode: str | dict | None = "inverted",
) -> dict[str, pd.DataFrame]:
    """Compute sector priority ranking from normalized multi-criteria scores."""
    if df.empty:
        raise ValueError("Dữ liệu ngành rỗng.")

    normalized_weights = _normalized_weights(weights)
    risk_as_cost, risk_inverted = _risk_flags(risk_mode)
    raw = _raw_criteria(df)

    normalized_df = raw[["sector_name"]].copy()
    for criterion in CRITERIA:
        invert = criterion == "automation_risk" and risk_inverted
        normalized_df[f"{criterion}_norm"] = normalize_minmax(raw[criterion], invert=invert)

    good_criteria = [criterion for criterion in CRITERIA if criterion != "automation_risk"]
    normalized_df["good_part"] = sum(
        normalized_weights[criterion] * normalized_df[f"{criterion}_norm"]
        for criterion in good_criteria
    )

    risk_weight = normalized_weights["automation_risk"]
    if risk_as_cost:
        normalized_df["risk_component"] = -risk_weight * normalized_df["automation_risk_norm"]
    else:
        normalized_df["risk_component"] = risk_weight * normalized_df["automation_risk_norm"]

    normalized_df["priority_score"] = normalized_df["good_part"] + normalized_df["risk_component"]

    ranking_df = normalized_df[["sector_name", "priority_score"]].copy()
    for criterion in CRITERIA:
        ranking_df[criterion] = raw[criterion]
        ranking_df[f"{criterion}_norm"] = normalized_df[f"{criterion}_norm"]
    ranking_df["rank"] = ranking_df["priority_score"].rank(ascending=False, method="first").astype(int)
    ranking_df = ranking_df.sort_values(["rank", "sector_name"]).reset_index(drop=True)

    return {
        "ranking_df": ranking_df,
        "normalized_df": normalized_df,
    }


def sensitivity_ai_weight(
    df: pd.DataFrame,
    ai_weights=np.arange(0.05, 0.45, 0.05),
) -> pd.DataFrame:
    """Track the top-ranked sector as AI-readiness weight changes."""
    rows = []
    base_non_ai_total = sum(value for key, value in DEFAULT_WEIGHTS.items() if key != "ai_readiness")

    for ai_weight in ai_weights:
        remaining = max(1.0 - float(ai_weight), 0.0)
        weights = {}
        for key, value in DEFAULT_WEIGHTS.items():
            if key == "ai_readiness":
                weights[key] = float(ai_weight)
            else:
                weights[key] = value / base_non_ai_total * remaining

        ranking_df = compute_priority(df, weights, risk_mode="inverted")["ranking_df"]
        for _, row in ranking_df.iterrows():
            rows.append(
                {
                    "ai_weight": round(float(ai_weight), 3),
                    "sector_name": row["sector_name"],
                    "rank": int(row["rank"]),
                    "priority_score": float(row["priority_score"]),
                    "is_top_rank": int(row["rank"] == 1),
                }
            )

    return pd.DataFrame(rows)


def compare_policy_weights(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compare top sectors under growth-oriented and inclusive-oriented weights."""
    growth_ranking = compute_priority(
        df,
        GROWTH_ORIENTED_WEIGHTS,
        risk_mode={"risk_inverted": True},
    )["ranking_df"]
    inclusive_ranking = compute_priority(
        df,
        INCLUSIVE_ORIENTED_WEIGHTS,
        risk_mode={"risk_inverted": True},
    )["ranking_df"]

    return {
        "growth_oriented": growth_ranking,
        "inclusive_oriented": inclusive_ranking,
        "top3_comparison": pd.DataFrame(
            {
                "rank": [1, 2, 3],
                "growth_oriented": growth_ranking.head(3)["sector_name"].to_list(),
                "inclusive_oriented": inclusive_ranking.head(3)["sector_name"].to_list(),
            }
        ),
    }
