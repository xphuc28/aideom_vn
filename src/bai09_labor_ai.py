"""Bai 9: AI impacts on Vietnam's labor market."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog


MODULE_TITLE = "Bài 9 - Tác động AI tới thị trường lao động Việt Nam"

LABOR_SECTORS = [
    {
        "sector": "Nông-Lâm-Thủy sản",
        "labor_million": 13.20,
        "risk_pct": 18,
        "a1": 0.92,
        "a2": 0.22,
        "b1": 0.74,
        "c1": 0.62,
        "d1": 1.10,
    },
    {
        "sector": "Công nghiệp chế biến chế tạo",
        "labor_million": 11.50,
        "risk_pct": 42,
        "a1": 1.18,
        "a2": 0.36,
        "b1": 0.82,
        "c1": 0.88,
        "d1": 1.24,
    },
    {
        "sector": "Xây dựng",
        "labor_million": 4.80,
        "risk_pct": 25,
        "a1": 0.84,
        "a2": 0.20,
        "b1": 0.68,
        "c1": 0.58,
        "d1": 1.02,
    },
    {
        "sector": "Bán buôn-bán lẻ",
        "labor_million": 7.80,
        "risk_pct": 38,
        "a1": 1.04,
        "a2": 0.28,
        "b1": 0.78,
        "c1": 0.82,
        "d1": 1.16,
    },
    {
        "sector": "Tài chính-Ngân hàng-Bảo hiểm",
        "labor_million": 0.55,
        "risk_pct": 52,
        "a1": 1.35,
        "a2": 0.42,
        "b1": 0.90,
        "c1": 0.95,
        "d1": 1.28,
    },
    {
        "sector": "Logistics-Vận tải-Kho bãi",
        "labor_million": 1.95,
        "risk_pct": 35,
        "a1": 1.08,
        "a2": 0.30,
        "b1": 0.80,
        "c1": 0.76,
        "d1": 1.14,
    },
    {
        "sector": "Thông tin-Truyền thông-CNTT",
        "labor_million": 0.62,
        "risk_pct": 28,
        "a1": 1.55,
        "a2": 0.48,
        "b1": 0.96,
        "c1": 0.54,
        "d1": 1.32,
    },
    {
        "sector": "Giáo dục-Đào tạo",
        "labor_million": 2.15,
        "risk_pct": 22,
        "a1": 0.96,
        "a2": 0.34,
        "b1": 0.88,
        "c1": 0.46,
        "d1": 1.25,
    },
]


def module_status() -> str:
    """Return a short implementation status for the Streamlit page."""
    return "Đã triển khai LP phân bổ AI/đào tạo lại theo 8 ngành lao động."


def labor_dataframe() -> pd.DataFrame:
    """Return the hard-coded 8-sector labor dataset."""
    df = pd.DataFrame(LABOR_SECTORS)
    df["risk"] = df["risk_pct"] / 100.0
    return df


def _empty_result(status: str, note: str, risk_multiplier: float) -> dict[str, object]:
    df = labor_dataframe()
    result = df.copy()
    for column in ["x_AI", "x_H", "NewJob", "UpgradeJob", "DisplacedJob", "RetrainingCapacity", "NetJob"]:
        result[column] = 0.0
    result["total_allocation"] = 0.0
    return {
        "allocation_df": result,
        "objective": None,
        "status": status,
        "note": note,
        "risk_multiplier": risk_multiplier,
    }


def _build_result(df: pd.DataFrame, x: np.ndarray, objective: float, status: str, note: str, risk_multiplier: float) -> dict[str, object]:
    n = len(df)
    result = df.copy()
    result["x_AI"] = x[:n]
    result["x_H"] = x[n:]
    risk = result["risk"] * risk_multiplier
    result["NewJob"] = result["a1"] * result["x_AI"]
    result["UpgradeJob"] = result["b1"] * result["x_H"]
    result["DisplacedJob"] = result["c1"] * risk * result["x_AI"]
    result["RetrainingCapacity"] = result["d1"] * result["x_H"]
    result["NetJob"] = result["NewJob"] + result["UpgradeJob"] - result["DisplacedJob"]
    result["total_allocation"] = result["x_AI"] + result["x_H"]
    return {
        "allocation_df": result,
        "objective": float(objective),
        "status": status,
        "note": note,
        "risk_multiplier": risk_multiplier,
    }


def solve_bai09(
    budget: float = 30000,
    max_displaced_share: float | None = None,
    risk_multiplier: float = 1.0,
) -> dict[str, object]:
    """Solve the AI-labor allocation LP with SciPy HiGHS."""
    df = labor_dataframe()
    n = len(df)
    budget = float(budget)
    risk_multiplier = float(risk_multiplier)
    if budget < 0:
        raise ValueError("budget không được âm.")
    if risk_multiplier < 0:
        raise ValueError("risk_multiplier không được âm.")

    risk = df["risk"].to_numpy() * risk_multiplier
    ai_net_coef = df["a1"].to_numpy() - df["c1"].to_numpy() * risk
    h_net_coef = df["b1"].to_numpy()
    c = -np.concatenate([ai_net_coef, h_net_coef])

    a_ub = []
    b_ub = []

    budget_row = np.ones(2 * n)
    a_ub.append(budget_row)
    b_ub.append(budget)

    for idx in range(n):
        # NetJob_i >= 0 -> -(a1-c1*risk)xAI - b1*xH <= 0
        row = np.zeros(2 * n)
        row[idx] = -ai_net_coef[idx]
        row[n + idx] = -h_net_coef[idx]
        a_ub.append(row)
        b_ub.append(0.0)

        # DisplacedJob_i <= RetrainingCapacity_i
        row = np.zeros(2 * n)
        row[idx] = df.loc[idx, "c1"] * risk[idx]
        row[n + idx] = -df.loc[idx, "d1"]
        a_ub.append(row)
        b_ub.append(0.0)

        if max_displaced_share is not None:
            # labor_million is converted to thousand workers by multiplying by 1000.
            row = np.zeros(2 * n)
            row[idx] = df.loc[idx, "c1"] * risk[idx]
            a_ub.append(row)
            b_ub.append(float(max_displaced_share) * df.loc[idx, "labor_million"] * 1000.0)

    result = linprog(
        c,
        A_ub=np.array(a_ub, dtype=float),
        b_ub=np.array(b_ub, dtype=float),
        bounds=[(0.0, None) for _ in range(2 * n)],
        method="highs",
    )

    if not result.success:
        return _empty_result(result.message, "SciPy HiGHS không tìm được nghiệm tối ưu.", risk_multiplier)

    objective = -float(result.fun)
    return _build_result(
        df=df,
        x=np.maximum(result.x, 0.0),
        objective=objective,
        status="optimal",
        note="Solved with SciPy HiGHS.",
        risk_multiplier=risk_multiplier,
    )


def manufacturing_training_threshold(x_ai_value: float) -> pd.DataFrame:
    """Return minimum manufacturing training investment for a fixed AI allocation."""
    df = labor_dataframe()
    manufacturing = df[df["sector"] == "Công nghiệp chế biến chế tạo"].iloc[0]
    x_ai = float(x_ai_value)
    risk = manufacturing["risk"]
    displaced = manufacturing["c1"] * risk * x_ai
    x_h_for_retraining = displaced / manufacturing["d1"] if manufacturing["d1"] > 0 else np.inf
    net_without_h = manufacturing["a1"] * x_ai - displaced
    x_h_for_net = max(0.0, -net_without_h / manufacturing["b1"])
    threshold = max(x_h_for_retraining, x_h_for_net)
    return pd.DataFrame(
        [
            {
                "sector": manufacturing["sector"],
                "x_AI": x_ai,
                "displaced_job": displaced,
                "x_H_for_retraining": x_h_for_retraining,
                "x_H_for_nonnegative_net": x_h_for_net,
                "minimum_x_H_required": threshold,
            }
        ]
    )


def stress_test_risk(
    risk_multiplier: float,
    budget: float = 30000,
    max_displaced_share: float | None = None,
) -> dict[str, object]:
    """Solve the model under higher or lower automation risk."""
    return solve_bai09(
        budget=budget,
        max_displaced_share=max_displaced_share,
        risk_multiplier=risk_multiplier,
    )
