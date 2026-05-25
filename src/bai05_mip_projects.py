"""Bai 5: MIP project portfolio selection for digital transformation."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd


MODULE_TITLE = "Bài 5 - MIP lựa chọn 15 dự án chuyển đổi số"

PROJECTS = [
    {
        "project_id": "P1",
        "project_name": "Trung tâm dữ liệu quốc gia",
        "sector": "Hạ tầng số",
        "cost": 16000,
        "benefit": 27000,
        "cost_year_1_2": 9000,
        "cost_year_3_5": 7000,
        "success_probability": 0.86,
    },
    {
        "project_id": "P2",
        "project_name": "Nền tảng cloud chính phủ",
        "sector": "Hạ tầng số",
        "cost": 14000,
        "benefit": 23500,
        "cost_year_1_2": 8500,
        "cost_year_3_5": 5500,
        "success_probability": 0.88,
    },
    {
        "project_id": "P3",
        "project_name": "Băng rộng nông thôn",
        "sector": "Bao trùm số",
        "cost": 9000,
        "benefit": 15000,
        "cost_year_1_2": 5500,
        "cost_year_3_5": 3500,
        "success_probability": 0.90,
    },
    {
        "project_id": "P4",
        "project_name": "Định danh số liên thông",
        "sector": "Dịch vụ công",
        "cost": 6500,
        "benefit": 13000,
        "cost_year_1_2": 3500,
        "cost_year_3_5": 3000,
        "success_probability": 0.92,
    },
    {
        "project_id": "P5",
        "project_name": "Cổng dịch vụ công thông minh",
        "sector": "Dịch vụ công",
        "cost": 7000,
        "benefit": 12500,
        "cost_year_1_2": 4000,
        "cost_year_3_5": 3000,
        "success_probability": 0.91,
    },
    {
        "project_id": "P6",
        "project_name": "Voucher chuyển đổi số SME",
        "sector": "Doanh nghiệp số",
        "cost": 6000,
        "benefit": 10800,
        "cost_year_1_2": 3200,
        "cost_year_3_5": 2800,
        "success_probability": 0.84,
    },
    {
        "project_id": "P7",
        "project_name": "SOC an ninh mạng quốc gia",
        "sector": "An toàn số",
        "cost": 8500,
        "benefit": 16000,
        "cost_year_1_2": 5000,
        "cost_year_3_5": 3500,
        "success_probability": 0.87,
    },
    {
        "project_id": "P8",
        "project_name": "AI chẩn đoán y tế",
        "sector": "Y tế AI",
        "cost": 9000,
        "benefit": 19000,
        "cost_year_1_2": 4200,
        "cost_year_3_5": 4800,
        "success_probability": 0.72,
    },
    {
        "project_id": "P9",
        "project_name": "Hành lang logistics thông minh",
        "sector": "Logistics",
        "cost": 9500,
        "benefit": 17000,
        "cost_year_1_2": 5000,
        "cost_year_3_5": 4500,
        "success_probability": 0.82,
    },
    {
        "project_id": "P10",
        "project_name": "Chuỗi cung ứng nông nghiệp số",
        "sector": "Nông nghiệp số",
        "cost": 7500,
        "benefit": 14500,
        "cost_year_1_2": 4200,
        "cost_year_3_5": 3300,
        "success_probability": 0.85,
    },
    {
        "project_id": "P11",
        "project_name": "Sàn dữ liệu mở cho doanh nghiệp",
        "sector": "Dữ liệu",
        "cost": 8000,
        "benefit": 14200,
        "cost_year_1_2": 4500,
        "cost_year_3_5": 3500,
        "success_probability": 0.80,
    },
    {
        "project_id": "P12",
        "project_name": "Kho dữ liệu AI quốc gia",
        "sector": "Dữ liệu AI",
        "cost": 7000,
        "benefit": 12000,
        "cost_year_1_2": 3800,
        "cost_year_3_5": 3200,
        "success_probability": 0.83,
    },
    {
        "project_id": "P13",
        "project_name": "AI giáo dục cá nhân hóa",
        "sector": "Giáo dục AI",
        "cost": 6500,
        "benefit": 14000,
        "cost_year_1_2": 2500,
        "cost_year_3_5": 4000,
        "success_probability": 0.76,
    },
    {
        "project_id": "P14",
        "project_name": "Đào tạo kỹ năng số quốc gia",
        "sector": "Nhân lực số",
        "cost": 5000,
        "benefit": 9500,
        "cost_year_1_2": 2500,
        "cost_year_3_5": 2500,
        "success_probability": 0.94,
    },
    {
        "project_id": "P15",
        "project_name": "Trợ lý số cho chính quyền địa phương",
        "sector": "GovTech",
        "cost": 6000,
        "benefit": 11200,
        "cost_year_1_2": 3500,
        "cost_year_3_5": 2500,
        "success_probability": 0.86,
    },
]


def module_status() -> str:
    """Return a short implementation status for the Streamlit page."""
    return "Đã triển khai MIP chọn danh mục 15 dự án bằng PuLP optional và exhaustive-search fallback."


def project_dataframe() -> pd.DataFrame:
    """Return the hard-coded 15-project dataset."""
    df = pd.DataFrame(PROJECTS)
    df["benefit_cost_ratio"] = df["benefit"] / df["cost"]
    df["expected_benefit"] = df["benefit"] * df["success_probability"]
    return df


def _evaluate_solution(projects: pd.DataFrame, y: np.ndarray, expected_value: bool) -> dict[str, float]:
    benefit_col = "expected_benefit" if expected_value else "benefit"
    return {
        "count": int(y.sum()),
        "total_cost": float(projects["cost"].to_numpy() @ y),
        "early_cost": float(projects["cost_year_1_2"].to_numpy() @ y),
        "objective": float(projects[benefit_col].to_numpy() @ y),
        "benefit": float(projects["benefit"].to_numpy() @ y),
        "expected_benefit": float(projects["expected_benefit"].to_numpy() @ y),
    }


def _is_feasible(projects: pd.DataFrame, y: np.ndarray, total_budget: float, early_budget: float, force_p1_p2: bool) -> bool:
    stats = _evaluate_solution(projects, y, expected_value=False)

    if stats["total_cost"] > total_budget + 1e-9:
        return False
    if stats["early_cost"] > early_budget + 1e-9:
        return False
    if y[0] + y[1] > 1:
        return False
    if force_p1_p2 and not (y[0] == 1 and y[1] == 1):
        return False
    if y[7] > y[11]:
        return False
    if y[12] > y[11]:
        return False
    if y[3] + y[4] < 1:
        return False
    if y[13] < 1:
        return False
    if stats["count"] < 7 or stats["count"] > 11:
        return False
    return True


def _solve_exhaustive(
    total_budget: float,
    early_budget: float,
    force_p1_p2: bool,
    expected_value: bool,
) -> dict[str, object]:
    projects = project_dataframe()
    best_y = None
    best_key = None

    for bits in product([0, 1], repeat=len(projects)):
        y = np.array(bits, dtype=int)
        if not _is_feasible(projects, y, total_budget, early_budget, force_p1_p2):
            continue

        stats = _evaluate_solution(projects, y, expected_value)
        key = (stats["objective"], -stats["total_cost"], stats["count"])
        if best_key is None or key > best_key:
            best_key = key
            best_y = y

    if best_y is None:
        return _result_dict(projects, np.zeros(len(projects), dtype=int), None, False, "Không có danh mục khả thi.")

    stats = _evaluate_solution(projects, best_y, expected_value)
    return _result_dict(projects, best_y, stats["objective"], True, "Solved by exhaustive search fallback.")


def _solve_pulp(
    total_budget: float,
    early_budget: float,
    force_p1_p2: bool,
    expected_value: bool,
) -> dict[str, object] | None:
    try:
        import pulp
    except ImportError:
        return None

    projects = project_dataframe()
    benefit_col = "expected_benefit" if expected_value else "benefit"
    model = pulp.LpProblem("bai05_mip_projects", pulp.LpMaximize)
    y = {
        project_id: pulp.LpVariable(project_id, cat="Binary")
        for project_id in projects["project_id"]
    }

    model += pulp.lpSum(row[benefit_col] * y[row["project_id"]] for _, row in projects.iterrows())
    model += pulp.lpSum(row["cost"] * y[row["project_id"]] for _, row in projects.iterrows()) <= total_budget
    model += pulp.lpSum(row["cost_year_1_2"] * y[row["project_id"]] for _, row in projects.iterrows()) <= early_budget
    model += y["P1"] + y["P2"] <= 1
    if force_p1_p2:
        model += y["P1"] == 1
        model += y["P2"] == 1
    model += y["P8"] <= y["P12"]
    model += y["P13"] <= y["P12"]
    model += y["P4"] + y["P5"] >= 1
    model += y["P14"] >= 1
    model += pulp.lpSum(y[project_id] for project_id in projects["project_id"]) >= 7
    model += pulp.lpSum(y[project_id] for project_id in projects["project_id"]) <= 11

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus.get(model.status, str(model.status)).lower()
    if status != "optimal":
        return _result_dict(projects, np.zeros(len(projects), dtype=int), None, False, f"PuLP status: {status}.")

    selected = np.array([int(round(y[project_id].value())) for project_id in projects["project_id"]], dtype=int)
    return _result_dict(projects, selected, float(pulp.value(model.objective)), True, "Solved with PuLP/CBC.")


def _result_dict(
    projects: pd.DataFrame,
    y: np.ndarray,
    objective: float | None,
    feasible: bool,
    note: str,
) -> dict[str, object]:
    all_projects_df = projects.copy()
    all_projects_df["selected"] = y.astype(int)
    all_projects_df["selected_label"] = np.where(all_projects_df["selected"] == 1, "Chọn", "Không chọn")
    selected_df = all_projects_df[all_projects_df["selected"] == 1].reset_index(drop=True)
    total_cost = float(selected_df["cost"].sum()) if feasible else 0.0
    total_benefit = float(selected_df["benefit"].sum()) if feasible else 0.0
    benefit_cost_ratio = total_benefit / total_cost if total_cost > 0 else 0.0

    return {
        "selected_df": selected_df,
        "all_projects_df": all_projects_df,
        "objective": objective,
        "total_cost": total_cost,
        "benefit_cost_ratio": benefit_cost_ratio,
        "feasibility": feasible,
        "note": note,
    }


def solve_bai05(
    total_budget: float = 80000,
    early_budget: float = 40000,
    force_p1_p2: bool = False,
    expected_value: bool = False,
) -> dict[str, object]:
    """Solve the 15-project binary MIP portfolio model."""
    total_budget = float(total_budget)
    early_budget = float(early_budget)
    if total_budget < 0 or early_budget < 0:
        raise ValueError("Ngân sách không được âm.")

    pulp_result = _solve_pulp(total_budget, early_budget, force_p1_p2, expected_value)
    if pulp_result is not None:
        return pulp_result
    return _solve_exhaustive(total_budget, early_budget, force_p1_p2, expected_value)


def solve_budget_scenario(total_budget: float = 100000) -> dict[str, object]:
    """Solve a larger-budget scenario while keeping early budget proportional."""
    return solve_bai05(total_budget=total_budget, early_budget=total_budget * 0.5)


def solve_with_risk_probabilities() -> dict[str, object]:
    """Solve using expected benefits adjusted by project success probabilities."""
    return solve_bai05(total_budget=80000, early_budget=40000, expected_value=True)
