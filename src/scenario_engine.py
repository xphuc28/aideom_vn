"""Scenario aggregation utilities for Bai 12 AIDEOM-VN."""

from __future__ import annotations

import numpy as np
import pandas as pd


SCENARIOS = {
    "S1": {
        "name": "Truyền thống",
        "description": "Duy trì ưu tiên vốn vật chất và hạ tầng truyền thống.",
        "shares": {"K": 0.70, "D": 0.10, "AI": 0.10, "H": 0.10},
    },
    "S2": {
        "name": "Số hóa nhanh",
        "description": "Tăng tốc hạ tầng số, dịch vụ số và nền tảng dữ liệu.",
        "shares": {"K": 0.25, "D": 0.45, "AI": 0.15, "H": 0.15},
    },
    "S3": {
        "name": "AI dẫn dắt",
        "description": "Dồn trọng tâm vào năng lực AI, dữ liệu và tự động hóa.",
        "shares": {"K": 0.20, "D": 0.20, "AI": 0.45, "H": 0.15},
    },
    "S4": {
        "name": "Bao trùm số",
        "description": "Ưu tiên vốn nhân lực, đào tạo lại và tiếp cận số rộng.",
        "shares": {"K": 0.30, "D": 0.20, "AI": 0.10, "H": 0.40},
    },
    "S5": {
        "name": "Tối ưu cân bằng",
        "description": "Weighted compromise fallback giữa tăng trưởng, số hóa, AI, việc làm và rủi ro.",
        "shares": {"K": 0.30, "D": 0.28, "AI": 0.22, "H": 0.20},
    },
}

ITEM_LABELS = {
    "K": "Vốn/hạ tầng",
    "D": "Số hóa",
    "AI": "AI và dữ liệu",
    "H": "Vốn nhân lực",
}

BENEFIT_KPIS = ["GDP_gain", "Digital_score", "AI_score", "Human_capital_score", "NetJob"]
COST_KPIS = ["Inequality_risk", "Cyber_risk", "Emission_risk"]


def list_scenarios() -> list[dict[str, str]]:
    """Return scenario metadata as dictionaries for display."""
    return [
        {
            "code": code,
            "name": spec["name"],
            "description": spec["description"],
        }
        for code, spec in SCENARIOS.items()
    ]


def _normalize_shares(shares: dict[str, float]) -> dict[str, float]:
    total = sum(float(shares[item]) for item in ["K", "D", "AI", "H"])
    if total <= 0:
        raise ValueError("Tổng allocation shares phải dương.")
    return {item: float(shares[item]) / total for item in ["K", "D", "AI", "H"]}


def run_scenario(name: str, budget: float = 50000) -> dict[str, object]:
    """Run one policy scenario and return KPI approximations."""
    if name not in SCENARIOS:
        raise ValueError(f"Scenario không hợp lệ: {name}")

    spec = SCENARIOS[name]
    shares = _normalize_shares(spec["shares"])
    budget = float(budget)
    allocation = {item: shares[item] * budget for item in shares}

    k, d, ai, h = shares["K"], shares["D"], shares["AI"], shares["H"]
    balance_bonus = 1.0 - float(np.std([k, d, ai, h]))
    s5_bonus = 0.04 if name == "S5" else 0.0

    gdp_gain = budget * (0.88 * k + 1.10 * d + 1.34 * ai + 0.96 * h) * (1.0 + 0.08 * balance_bonus + s5_bonus)
    digital_score = 100.0 * (0.25 + 1.35 * d + 0.35 * ai + 0.15 * h)
    ai_score = 100.0 * (0.18 + 1.45 * ai + 0.25 * d + 0.15 * h)
    human_capital_score = 100.0 * (0.20 + 1.25 * h + 0.18 * d)
    net_job = budget * (0.42 * h + 0.26 * d + 0.30 * ai - 0.18 * ai**1.25 + 0.08 * k)
    inequality_risk = 100.0 * max(0.05, 0.55 - 0.70 * h - 0.16 * d - (0.12 if name == "S5" else 0.0))
    cyber_risk = 100.0 * max(0.04, 0.14 + 0.86 * ai + 0.22 * d - 0.48 * h)
    emission_risk = 100.0 * max(0.05, 0.10 + 0.62 * k + 0.34 * ai - 0.20 * d - 0.16 * h)

    return {
        "scenario": name,
        "scenario_name": spec["name"],
        "description": spec["description"],
        "budget": budget,
        "K_share": k,
        "D_share": d,
        "AI_share": ai,
        "H_share": h,
        "K_allocation": allocation["K"],
        "D_allocation": allocation["D"],
        "AI_allocation": allocation["AI"],
        "H_allocation": allocation["H"],
        "GDP_gain": gdp_gain,
        "Digital_score": min(digital_score, 100.0),
        "AI_score": min(ai_score, 100.0),
        "Human_capital_score": min(human_capital_score, 100.0),
        "NetJob": net_job,
        "Inequality_risk": min(inequality_risk, 100.0),
        "Cyber_risk": min(cyber_risk, 100.0),
        "Emission_risk": min(emission_risk, 100.0),
    }


def compute_overall_score(kpi_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize KPIs and compute an overall 0-100 score."""
    if kpi_df.empty:
        return kpi_df.copy()

    scored = kpi_df.copy()
    component_weights = {
        "GDP_gain": 0.24,
        "Digital_score": 0.13,
        "AI_score": 0.13,
        "Human_capital_score": 0.13,
        "NetJob": 0.15,
        "Inequality_risk": 0.10,
        "Cyber_risk": 0.06,
        "Emission_risk": 0.06,
    }

    total_score = np.zeros(len(scored))
    for column, weight in component_weights.items():
        values = scored[column].astype(float)
        min_value = values.min()
        max_value = values.max()
        if np.isclose(min_value, max_value):
            normalized = pd.Series(0.5, index=scored.index)
        else:
            normalized = (values - min_value) / (max_value - min_value)
        if column in COST_KPIS:
            normalized = 1.0 - normalized
        scored[f"{column}_norm"] = normalized
        total_score += weight * normalized.to_numpy()

    scored["Overall_score"] = np.clip(total_score * 100.0, 0.0, 100.0)
    return scored


def run_all_scenarios(budget: float = 50000) -> pd.DataFrame:
    """Run all five AIDEOM-VN scenarios."""
    rows = [run_scenario(code, budget=budget) for code in SCENARIOS]
    return compute_overall_score(pd.DataFrame(rows))


def allocation_long(kpi_df: pd.DataFrame) -> pd.DataFrame:
    """Return long allocation table for charts."""
    rows = []
    for _, row in kpi_df.iterrows():
        for item in ["K", "D", "AI", "H"]:
            rows.append(
                {
                    "scenario": row["scenario"],
                    "scenario_name": row["scenario_name"],
                    "item": item,
                    "item_label": ITEM_LABELS[item],
                    "share": row[f"{item}_share"],
                    "allocation": row[f"{item}_allocation"],
                }
            )
    return pd.DataFrame(rows)


def recommendation_text(kpi_df: pd.DataFrame) -> list[str]:
    """Generate concise policy recommendations from scenario scores."""
    if kpi_df.empty:
        return ["Chưa có dữ liệu kịch bản để khuyến nghị."]

    best = kpi_df.sort_values("Overall_score", ascending=False).iloc[0]
    ai_best = kpi_df.sort_values("AI_score", ascending=False).iloc[0]
    job_best = kpi_df.sort_values("NetJob", ascending=False).iloc[0]
    low_risk = kpi_df.assign(total_risk=kpi_df[COST_KPIS].sum(axis=1)).sort_values("total_risk").iloc[0]

    return [
        f"Kịch bản tổng hợp tốt nhất là {best['scenario']} - {best['scenario_name']} với Overall_score {best['Overall_score']:.1f}/100.",
        f"Nếu ưu tiên năng lực AI thuần túy, kịch bản nổi bật là {ai_best['scenario']} - {ai_best['scenario_name']}.",
        f"Nếu ưu tiên việc làm ròng và đào tạo lại, kịch bản mạnh nhất là {job_best['scenario']} - {job_best['scenario_name']}.",
        f"Nếu ưu tiên giảm rủi ro tổng hợp, nên xem xét {low_risk['scenario']} - {low_risk['scenario_name']}.",
        "Dashboard là công cụ hỗ trợ ra quyết định định lượng; không thay thế thảo luận chính trị, xã hội và phân bổ nguồn lực thực tế.",
    ]
