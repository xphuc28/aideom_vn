"""Bai 4: Linear programming for digital budget allocation by region-item."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linprog


MODULE_TITLE = "Bài 4 - LP phân bổ ngân sách số theo ngành-vùng"

REGIONS = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
REGION_NAMES = {
    "NMM": "Trung du miền núi phía Bắc",
    "RRD": "Đồng bằng sông Hồng",
    "NCC": "Bắc Trung Bộ và duyên hải miền Trung",
    "CH": "Tây Nguyên",
    "SE": "Đông Nam Bộ",
    "MD": "Đồng bằng sông Cửu Long",
}
ITEMS = ["I", "D", "AI", "H"]
ITEM_NAMES = {
    "I": "Hạ tầng số",
    "D": "Chuyển đổi số",
    "AI": "AI và dữ liệu",
    "H": "Nhân lực số",
}

BETA = {
    "NMM": {"I": 1.15, "D": 0.85, "AI": 0.55, "H": 1.30},
    "RRD": {"I": 0.95, "D": 1.25, "AI": 1.40, "H": 1.05},
    "NCC": {"I": 1.05, "D": 0.95, "AI": 0.85, "H": 1.15},
    "CH": {"I": 1.20, "D": 0.75, "AI": 0.45, "H": 1.35},
    "SE": {"I": 0.90, "D": 1.30, "AI": 1.55, "H": 1.00},
    "MD": {"I": 1.10, "D": 0.85, "AI": 0.65, "H": 1.25},
}

D0 = {"NMM": 38, "RRD": 78, "NCC": 55, "CH": 32, "SE": 82, "MD": 48}
DEFAULT_BUDGET = 50000.0
REGION_MIN = 5000.0
REGION_MAX = 12000.0
H_MIN = 12000.0
GAMMA = 0.002
TOLERANCE = 1e-6


def module_status() -> str:
    """Return a short implementation status for the Streamlit page."""
    return "Đã triển khai LP ngành-vùng với ràng buộc công bằng và fallback SciPy khi thiếu PuLP/CVXPY."


def _x_index(region_index: int, item_index: int) -> int:
    return region_index * len(ITEMS) + item_index


def _m_index() -> int:
    return len(REGIONS) * len(ITEMS)


def _beta_vector(include_m: bool = False) -> np.ndarray:
    values = [BETA[region][item] for region in REGIONS for item in ITEMS]
    if include_m:
        values.append(0.0)
    return np.array(values, dtype=float)


def _matrix_from_values(values: Iterable[float]) -> pd.DataFrame:
    array = np.array(list(values), dtype=float).reshape((len(REGIONS), len(ITEMS)))
    return pd.DataFrame(array, index=REGIONS, columns=ITEMS)


def _long_frame(allocation_matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        for item in ITEMS:
            allocation = float(allocation_matrix.loc[region, item])
            beta = float(BETA[region][item])
            rows.append(
                {
                    "region": region,
                    "region_name": REGION_NAMES[region],
                    "item": item,
                    "item_name": ITEM_NAMES[item],
                    "allocation": allocation,
                    "beta": beta,
                    "value_contribution": allocation * beta,
                    "digital_after": D0[region] + GAMMA * allocation_matrix.loc[region, "D"],
                }
            )
    return pd.DataFrame(rows)


def _result_dict(
    allocation_matrix: pd.DataFrame,
    objective: float | None,
    status: str,
    note: str,
    solver: str,
    fairness: bool,
    lambda_: float,
    fairness_cost: float | None = None,
    effective_lambda: float | None = None,
) -> dict[str, object]:
    long_df = _long_frame(allocation_matrix)
    region_totals = (
        long_df.groupby(["region", "region_name"], as_index=False)["allocation"].sum()
        .rename(columns={"allocation": "region_total"})
        .sort_values("region")
        .reset_index(drop=True)
    )
    region_totals["digital_after"] = region_totals["region"].map(
        lambda region: D0[region] + GAMMA * allocation_matrix.loc[region, "D"]
    )
    item_totals = (
        long_df.groupby(["item", "item_name"], as_index=False)["allocation"].sum()
        .rename(columns={"allocation": "item_total"})
        .sort_values("item")
        .reset_index(drop=True)
    )
    return {
        "allocation_matrix": allocation_matrix,
        "long_df": long_df,
        "objective": objective,
        "region_totals": region_totals,
        "item_totals": item_totals,
        "fairness_cost": fairness_cost,
        "status": status,
        "note": note,
        "solver": solver,
        "fairness": fairness,
        "lambda": lambda_,
        "effective_lambda": lambda_ if effective_lambda is None else effective_lambda,
    }


def _empty_result(status: str, note: str, solver: str, fairness: bool, lambda_: float) -> dict[str, object]:
    return _result_dict(
        allocation_matrix=pd.DataFrame(0.0, index=REGIONS, columns=ITEMS),
        objective=None,
        status=status,
        note=note,
        solver=solver,
        fairness=fairness,
        lambda_=lambda_,
    )


def _max_feasible_lambda() -> float:
    """Return a conservative lambda cap implied by D0, gamma, and region max."""
    max_m = max(D0.values())
    return min((D0[region] + GAMMA * REGION_MAX) / max_m for region in REGIONS)


def _build_scipy_problem(budget: float, fairness: bool, lambda_: float):
    n_x = len(REGIONS) * len(ITEMS)
    n_variables = n_x + (1 if fairness else 0)
    c = np.zeros(n_variables)
    c[:n_x] = -_beta_vector(include_m=False)

    a_ub = []
    b_ub = []

    total_budget = np.zeros(n_variables)
    total_budget[:n_x] = 1.0
    a_ub.append(total_budget)
    b_ub.append(float(budget))

    for r_idx, _region in enumerate(REGIONS):
        region_total = np.zeros(n_variables)
        for j_idx, _item in enumerate(ITEMS):
            region_total[_x_index(r_idx, j_idx)] = 1.0
        a_ub.append(region_total)
        b_ub.append(REGION_MAX)
        a_ub.append(-region_total)
        b_ub.append(-REGION_MIN)

    h_total = np.zeros(n_variables)
    h_idx = ITEMS.index("H")
    for r_idx, _region in enumerate(REGIONS):
        h_total[_x_index(r_idx, h_idx)] = -1.0
    a_ub.append(h_total)
    b_ub.append(-H_MIN)

    if fairness:
        d_idx = ITEMS.index("D")
        m_idx = _m_index()
        for r_idx, region in enumerate(REGIONS):
            upper = np.zeros(n_variables)
            upper[_x_index(r_idx, d_idx)] = GAMMA
            upper[m_idx] = -1.0
            a_ub.append(upper)
            b_ub.append(-D0[region])

            lower = np.zeros(n_variables)
            lower[_x_index(r_idx, d_idx)] = -GAMMA
            lower[m_idx] = float(lambda_)
            a_ub.append(lower)
            b_ub.append(D0[region])

    bounds = [(0.0, None) for _ in range(n_variables)]
    if fairness:
        bounds[_m_index()] = (max(D0.values()), None)

    return c, np.array(a_ub, dtype=float), np.array(b_ub, dtype=float), bounds


def _solve_scipy_core(budget: float, fairness: bool, lambda_: float, solver_label: str) -> dict[str, object]:
    effective_lambda = float(lambda_)
    if fairness:
        lambda_cap = _max_feasible_lambda()
        if effective_lambda > lambda_cap:
            effective_lambda = max(lambda_cap - 1e-5, 0.0)

    c, a_ub, b_ub, bounds = _build_scipy_problem(budget, fairness, effective_lambda)
    result = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not result.success:
        return _empty_result(
            status=result.message,
            note="SciPy HiGHS không tìm được nghiệm khả thi/tối ưu.",
            solver=solver_label,
            fairness=fairness,
            lambda_=lambda_,
        )

    allocation_matrix = _matrix_from_values(result.x[: len(REGIONS) * len(ITEMS)])
    objective = float(_beta_vector(include_m=False) @ result.x[: len(REGIONS) * len(ITEMS)])
    note = "Solved with SciPy HiGHS."
    if fairness and effective_lambda < float(lambda_):
        note += f" Lambda yêu cầu {lambda_:.3f} không khả thi; đã dùng effective_lambda={effective_lambda:.3f}."
    return _result_dict(
        allocation_matrix=allocation_matrix,
        objective=objective,
        status="optimal",
        note=note,
        solver=solver_label,
        fairness=fairness,
        lambda_=lambda_,
        effective_lambda=effective_lambda,
    )


def solve_bai04_pulp(
    budget: float = DEFAULT_BUDGET,
    fairness: bool = True,
    lambda_: float = 0.7,
) -> dict[str, object]:
    """Solve Bai 4 with PuLP if installed, otherwise use SciPy fallback."""
    try:
        import pulp
    except ImportError:
        result = _solve_scipy_core(budget, fairness, lambda_, "SciPy fallback for PuLP")
        result["note"] = "PuLP chưa được cài; đã dùng SciPy HiGHS fallback để giải LP."
        if result.get("effective_lambda") != result.get("lambda"):
            result["note"] += f" Lambda hiệu dụng: {result['effective_lambda']:.3f}."
        return result

    effective_lambda = float(lambda_)
    if fairness:
        lambda_cap = _max_feasible_lambda()
        if effective_lambda > lambda_cap:
            effective_lambda = max(lambda_cap - 1e-5, 0.0)

    model = pulp.LpProblem("bai04_region_item_lp", pulp.LpMaximize)
    x = {
        (region, item): pulp.LpVariable(f"x_{region}_{item}", lowBound=0)
        for region in REGIONS
        for item in ITEMS
    }
    m_var = pulp.LpVariable("M", lowBound=max(D0.values())) if fairness else None

    model += pulp.lpSum(BETA[region][item] * x[(region, item)] for region in REGIONS for item in ITEMS)
    model += pulp.lpSum(x[(region, item)] for region in REGIONS for item in ITEMS) <= float(budget)

    for region in REGIONS:
        model += pulp.lpSum(x[(region, item)] for item in ITEMS) >= REGION_MIN
        model += pulp.lpSum(x[(region, item)] for item in ITEMS) <= REGION_MAX

    model += pulp.lpSum(x[(region, "H")] for region in REGIONS) >= H_MIN

    if fairness:
        for region in REGIONS:
            model += D0[region] + GAMMA * x[(region, "D")] <= m_var
            model += D0[region] + GAMMA * x[(region, "D")] >= effective_lambda * m_var

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus.get(model.status, str(model.status)).lower()
    if status != "optimal":
        return _empty_result(
            status=status,
            note="PuLP/CBC không tìm được nghiệm tối ưu.",
            solver="PuLP/CBC",
            fairness=fairness,
            lambda_=lambda_,
        )

    allocation_matrix = pd.DataFrame(index=REGIONS, columns=ITEMS, dtype=float)
    for region in REGIONS:
        for item in ITEMS:
            allocation_matrix.loc[region, item] = float(x[(region, item)].value())

    note = "Solved with PuLP/CBC."
    if fairness and effective_lambda < float(lambda_):
        note += f" Lambda yêu cầu {lambda_:.3f} không khả thi; đã dùng effective_lambda={effective_lambda:.3f}."

    return _result_dict(
        allocation_matrix=allocation_matrix,
        objective=float(pulp.value(model.objective)),
        status="optimal",
        note=note,
        solver="PuLP/CBC",
        fairness=fairness,
        lambda_=lambda_,
        effective_lambda=effective_lambda,
    )


def solve_bai04_cvxpy(
    budget: float = DEFAULT_BUDGET,
    fairness: bool = True,
    lambda_: float = 0.7,
) -> dict[str, object]:
    """Solve Bai 4 with CVXPY if installed; otherwise use SciPy fallback."""
    try:
        import cvxpy as cp
    except ImportError:
        result = _solve_scipy_core(budget, fairness, lambda_, "SciPy fallback for CVXPY")
        result["note"] = "CVXPY chưa được cài; đã dùng SciPy HiGHS fallback để giải LP."
        if result.get("effective_lambda") != result.get("lambda"):
            result["note"] += f" Lambda hiệu dụng: {result['effective_lambda']:.3f}."
        return result

    effective_lambda = float(lambda_)
    if fairness:
        lambda_cap = _max_feasible_lambda()
        if effective_lambda > lambda_cap:
            effective_lambda = max(lambda_cap - 1e-5, 0.0)

    x = cp.Variable((len(REGIONS), len(ITEMS)), nonneg=True)
    constraints = [cp.sum(x) <= float(budget)]
    for r_idx, _region in enumerate(REGIONS):
        constraints += [cp.sum(x[r_idx, :]) >= REGION_MIN, cp.sum(x[r_idx, :]) <= REGION_MAX]
    constraints.append(cp.sum(x[:, ITEMS.index("H")]) >= H_MIN)

    if fairness:
        m_var = cp.Variable(nonneg=True)
        constraints.append(m_var >= max(D0.values()))
        d_idx = ITEMS.index("D")
        for r_idx, region in enumerate(REGIONS):
            digital_after = D0[region] + GAMMA * x[r_idx, d_idx]
            constraints += [digital_after <= m_var, digital_after >= effective_lambda * m_var]

    beta = np.array([[BETA[region][item] for item in ITEMS] for region in REGIONS])
    problem = cp.Problem(cp.Maximize(cp.sum(cp.multiply(beta, x))), constraints)
    problem.solve(solver=cp.CLARABEL if "CLARABEL" in cp.installed_solvers() else None)

    if problem.status not in {"optimal", "optimal_inaccurate"}:
        return _empty_result(
            status=str(problem.status),
            note="CVXPY không tìm được nghiệm tối ưu.",
            solver="CVXPY",
            fairness=fairness,
            lambda_=lambda_,
        )

    note = "Solved with CVXPY."
    if fairness and effective_lambda < float(lambda_):
        note += f" Lambda yêu cầu {lambda_:.3f} không khả thi; đã dùng effective_lambda={effective_lambda:.3f}."

    return _result_dict(
        allocation_matrix=pd.DataFrame(np.maximum(x.value, 0.0), index=REGIONS, columns=ITEMS),
        objective=float(problem.value),
        status="optimal",
        note=note,
        solver="CVXPY",
        fairness=fairness,
        lambda_=lambda_,
        effective_lambda=effective_lambda,
    )


def compare_fairness(
    budget: float = DEFAULT_BUDGET,
    lambda_: float = 0.7,
) -> dict[str, object]:
    """Compare solutions with and without fairness constraints."""
    no_fairness = solve_bai04_pulp(budget=budget, fairness=False, lambda_=lambda_)
    fairness_result = solve_bai04_pulp(budget=budget, fairness=True, lambda_=lambda_)

    if no_fairness["objective"] is not None and fairness_result["objective"] is not None:
        fairness_cost = float(no_fairness["objective"] - fairness_result["objective"])
    else:
        fairness_cost = None

    fairness_result["fairness_cost"] = fairness_cost
    comparison_df = pd.DataFrame(
        [
            {
                "scenario": "Không fairness",
                "objective": no_fairness["objective"],
                "status": no_fairness["status"],
            },
            {
                "scenario": "Có fairness",
                "objective": fairness_result["objective"],
                "status": fairness_result["status"],
            },
        ]
    )
    return {
        "no_fairness": no_fairness,
        "fairness": fairness_result,
        "fairness_cost": fairness_cost,
        "comparison_df": comparison_df,
    }
