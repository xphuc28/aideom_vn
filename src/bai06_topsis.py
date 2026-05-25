"""Bai 6: TOPSIS ranking for six Vietnamese regions as AI investment hubs."""

from __future__ import annotations

from typing import Iterable
import unicodedata

import numpy as np
import pandas as pd


MODULE_TITLE = "Bài 6 - TOPSIS xếp hạng 6 vùng đầu tư AI"

CRITERIA = [
    "grdp_per_capita",
    "fdi_registered",
    "digital_index",
    "ai_readiness",
    "trained_labor",
    "rd_intensity",
    "internet_penetration",
    "gini",
]

DEFAULT_WEIGHTS = {
    "grdp_per_capita": 0.14,
    "fdi_registered": 0.13,
    "digital_index": 0.16,
    "ai_readiness": 0.18,
    "trained_labor": 0.13,
    "rd_intensity": 0.11,
    "internet_penetration": 0.10,
    "gini": 0.05,
}

IS_BENEFIT = {
    "grdp_per_capita": True,
    "fdi_registered": True,
    "digital_index": True,
    "ai_readiness": True,
    "trained_labor": True,
    "rd_intensity": True,
    "internet_penetration": True,
    "gini": False,
}

CRITERION_CANDIDATES = {
    "grdp_per_capita": ["grdp_per_capita_million_VND", "grdp_per_capita", "income_per_capita"],
    "fdi_registered": ["fdi_registered_billion_USD", "fdi_registered", "fdi"],
    "digital_index": ["digital_index_0_100", "digital_index", "digital"],
    "ai_readiness": ["ai_readiness_0_100", "ai_readiness", "ai"],
    "trained_labor": ["trained_labor_pct", "trained_labor", "skilled_labor_pct"],
    "rd_intensity": ["rd_intensity_pct", "rd_intensity", "r&d"],
    "internet_penetration": ["internet_penetration_pct", "internet_penetration", "internet"],
    "gini": ["gini_coef", "gini", "inequality"],
}


def module_status() -> str:
    """Return a short implementation status for the Streamlit page."""
    return "Đã triển khai TOPSIS expert/entropy, sensitivity trọng số AI và ranking 6 vùng."


def _normalize_name(value: str) -> str:
    text = str(value).strip().lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    for old, new in {
        "%": "pct",
        "&": "and",
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


def _region_names(df: pd.DataFrame) -> pd.Series:
    column = _flexible_column(df, ["region_name_vi", "region_name", "region", "vung"])
    if column:
        return df[column].astype(str)
    return pd.Series([f"Vùng {index + 1}" for index in range(len(df))], index=df.index)


def _criteria_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.DataFrame(index=df.index)
    missing = []

    for criterion in CRITERIA:
        column = _flexible_column(df, CRITERION_CANDIDATES[criterion])
        if column is None:
            missing.append(criterion)
            continue
        data[criterion] = pd.to_numeric(df[column], errors="coerce")

    if missing:
        raise ValueError(f"Thiếu cột tiêu chí TOPSIS: {', '.join(missing)}")

    data = data.interpolate(limit_direction="both").fillna(data.mean(numeric_only=True))
    return data.astype(float), _region_names(df)


def vector_normalize(X) -> np.ndarray:
    """Vector-normalize a matrix column-wise."""
    array = np.asarray(X, dtype=float)
    denominator = np.sqrt((array**2).sum(axis=0))
    denominator = np.where(denominator == 0, 1.0, denominator)
    return array / denominator


def _weights_array(weights: dict[str, float] | Iterable[float] | None) -> np.ndarray:
    if weights is None:
        raw = np.array([DEFAULT_WEIGHTS[criterion] for criterion in CRITERIA], dtype=float)
    elif isinstance(weights, dict):
        raw = np.array([float(weights.get(criterion, DEFAULT_WEIGHTS[criterion])) for criterion in CRITERIA])
    else:
        raw = np.array(list(weights), dtype=float)
        if len(raw) != len(CRITERIA):
            raise ValueError("weights phải có đúng 8 phần tử.")

    raw = np.maximum(raw, 0.0)
    total = raw.sum()
    if total <= 0:
        return np.ones(len(CRITERIA)) / len(CRITERIA)
    return raw / total


def _benefit_array(is_benefit: dict[str, bool] | Iterable[bool] | None) -> np.ndarray:
    if is_benefit is None:
        return np.array([IS_BENEFIT[criterion] for criterion in CRITERIA], dtype=bool)
    if isinstance(is_benefit, dict):
        return np.array([bool(is_benefit.get(criterion, IS_BENEFIT[criterion])) for criterion in CRITERIA])
    values = np.array(list(is_benefit), dtype=bool)
    if len(values) != len(CRITERIA):
        raise ValueError("is_benefit phải có đúng 8 phần tử.")
    return values


def _rank_from_matrix(
    X: pd.DataFrame,
    region_names: pd.Series,
    weights: dict[str, float] | Iterable[float] | None,
    is_benefit: dict[str, bool] | Iterable[bool] | None,
) -> pd.DataFrame:
    normalized = vector_normalize(X.to_numpy())
    w = _weights_array(weights)
    benefit = _benefit_array(is_benefit)
    weighted = normalized * w

    ideal_best = np.where(benefit, weighted.max(axis=0), weighted.min(axis=0))
    ideal_worst = np.where(benefit, weighted.min(axis=0), weighted.max(axis=0))
    d_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    score = d_worst / (d_best + d_worst + 1e-12)

    ranking = pd.DataFrame(
        {
            "region_name": region_names.to_numpy(),
            "topsis_score": score,
            "distance_best": d_best,
            "distance_worst": d_worst,
        }
    )
    for criterion in CRITERIA:
        ranking[criterion] = X[criterion].to_numpy()
    ranking["rank"] = ranking["topsis_score"].rank(ascending=False, method="first").astype(int)
    return ranking.sort_values(["rank", "region_name"]).reset_index(drop=True)


def entropy_weights(X) -> np.ndarray:
    """Compute entropy weights for a non-negative decision matrix."""
    array = np.asarray(X, dtype=float)
    shifted = array - np.nanmin(array, axis=0)
    column_sums = shifted.sum(axis=0)
    p = np.divide(shifted, column_sums, out=np.zeros_like(shifted), where=column_sums != 0)
    p_safe = np.where(p <= 0, 1.0, p)
    k = 1.0 / np.log(array.shape[0]) if array.shape[0] > 1 else 1.0
    entropy = -k * (p * np.log(p_safe)).sum(axis=0)
    diversity = 1.0 - entropy
    if diversity.sum() <= 1e-12:
        return np.ones(array.shape[1]) / array.shape[1]
    return diversity / diversity.sum()


def topsis(
    df: pd.DataFrame,
    weights: dict[str, float] | Iterable[float] | None = None,
    is_benefit: dict[str, bool] | Iterable[bool] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run TOPSIS with expert weights and entropy weights."""
    X, region_names = _criteria_matrix(df)
    expert_ranking = _rank_from_matrix(X, region_names, weights, is_benefit)
    entropy_w = entropy_weights(X.to_numpy())
    entropy_ranking = _rank_from_matrix(X, region_names, entropy_w, is_benefit)

    weights_df = pd.DataFrame(
        {
            "criterion": CRITERIA,
            "expert_weight": _weights_array(weights),
            "entropy_weight": entropy_w,
            "is_benefit": _benefit_array(is_benefit),
        }
    )

    return {
        "ranking_expert": expert_ranking,
        "ranking_entropy": entropy_ranking,
        "weights_df": weights_df,
        "normalized_matrix": pd.DataFrame(vector_normalize(X.to_numpy()), columns=CRITERIA),
    }


def sensitivity_ai_weight(
    df: pd.DataFrame,
    ai_weight_grid,
) -> pd.DataFrame:
    """Track TOPSIS top rank as the AI-readiness expert weight changes."""
    rows = []
    base = DEFAULT_WEIGHTS.copy()
    non_ai_total = sum(value for key, value in base.items() if key != "ai_readiness")

    for ai_weight in ai_weight_grid:
        remaining = max(1.0 - float(ai_weight), 0.0)
        weights = {}
        for criterion, value in base.items():
            if criterion == "ai_readiness":
                weights[criterion] = float(ai_weight)
            else:
                weights[criterion] = value / non_ai_total * remaining

        ranking = topsis(df, weights, IS_BENEFIT)["ranking_expert"]
        for _, row in ranking.iterrows():
            rows.append(
                {
                    "ai_weight": round(float(ai_weight), 3),
                    "region_name": row["region_name"],
                    "rank": int(row["rank"]),
                    "topsis_score": float(row["topsis_score"]),
                    "is_top_rank": int(row["rank"] == 1),
                }
            )

    return pd.DataFrame(rows)
