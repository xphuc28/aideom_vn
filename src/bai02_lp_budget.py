"""Bai 2: Linear programming for digital budget allocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linprog


MODULE_TITLE = "Bài 2 - LP phân bổ ngân sách số"

DECISION_NAMES = {
    "x1": "Hạ tầng số",
    "x2": "AI và dữ liệu",
    "x3": "Nhân lực số",
    "x4": "R&D công nghệ",
}

OBJECTIVE_COEFFICIENTS = np.array([0.85, 1.20, 0.95, 1.35], dtype=float)
DEFAULT_BUDGETS = [100, 120, 140]
TOLERANCE = 1e-7


@dataclass(frozen=True)
class LPResult:
    """Structured LP result used internally before conversion to dictionaries."""

    allocation_df: pd.DataFrame
    objective: float | None
    status: str
    binding_constraints: list[str]
    note: str
    shadow_prices: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "allocation_df": self.allocation_df,
            "objective": self.objective,
            "status": self.status,
            "binding_constraints": self.binding_constraints,
            "note": self.note,
            "shadow_prices": self.shadow_prices,
        }


def module_status() -> str:
    """Return a short implementation status for the Streamlit page."""
    return "Đã triển khai mô hình LP bằng SciPy và PuLP optional với phân tích độ nhạy ngân sách."


def _minimums(min_human: float) -> np.ndarray:
    return np.array([25.0, 15.0, float(min_human), 10.0], dtype=float)


def _empty_allocation() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variable": list(DECISION_NAMES),
            "category": list(DECISION_NAMES.values()),
            "allocation": [0.0] * 4,
            "coefficient": OBJECTIVE_COEFFICIENTS,
            "value_contribution": [0.0] * 4,
        }
    )


def _allocation_frame(values: Iterable[float]) -> pd.DataFrame:
    allocation = np.array(list(values), dtype=float)
    return pd.DataFrame(
        {
            "variable": list(DECISION_NAMES),
            "category": list(DECISION_NAMES.values()),
            "allocation": allocation,
            "coefficient": OBJECTIVE_COEFFICIENTS,
            "value_contribution": allocation * OBJECTIVE_COEFFICIENTS,
        }
    )


def _constraint_slacks(x: np.ndarray, B: float, min_human: float) -> dict[str, float]:
    mins = _minimums(min_human)
    total = float(x.sum())
    return {
        "budget_total": float(B - total),
        "min_x1_hatang": float(x[0] - mins[0]),
        "min_x2_ai_data": float(x[1] - mins[1]),
        "min_x3_nhan_luc": float(x[2] - mins[2]),
        "min_x4_rd": float(x[3] - mins[3]),
        "ai_rd_share": float((x[1] + x[3]) - 0.35 * total),
    }


def _binding_constraints(x: np.ndarray, B: float, min_human: float) -> list[str]:
    slacks = _constraint_slacks(x, B, min_human)
    return [name for name, slack in slacks.items() if abs(slack) <= TOLERANCE]


def _validate_inputs(B: float, min_human: float) -> tuple[float, float]:
    budget = float(B)
    human = float(min_human)
    if budget < 0:
        raise ValueError("B phải không âm.")
    if human < 0:
        raise ValueError("min_human phải không âm.")
    return budget, human


def solve_bai02_scipy(B: float = 100, min_human: float = 20) -> dict[str, object]:
    """Solve Bai 2 with scipy.optimize.linprog."""
    budget, human = _validate_inputs(B, min_human)
    mins = _minimums(human)

    c = -OBJECTIVE_COEFFICIENTS
    a_ub = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [0.35, -0.65, 0.35, -0.65],
        ]
    )
    b_ub = np.array([budget, 0.0])
    bounds = [(mins[index], None) for index in range(4)]

    result = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not result.success:
        return LPResult(
            allocation_df=_empty_allocation(),
            objective=None,
            status=result.message,
            binding_constraints=[],
            note="SciPy không tìm được nghiệm tối ưu. Kiểm tra ngân sách so với mức tối thiểu.",
            shadow_prices={},
        ).as_dict()

    x = np.maximum(result.x, 0.0)
    objective = float(OBJECTIVE_COEFFICIENTS @ x)
    shadow_prices: dict[str, float] = {}
    if hasattr(result, "ineqlin") and hasattr(result.ineqlin, "marginals"):
        shadow_prices["budget_total"] = float(-result.ineqlin.marginals[0])
        shadow_prices["ai_rd_share"] = float(-result.ineqlin.marginals[1])
    if hasattr(result, "lower") and hasattr(result.lower, "marginals"):
        for index, variable in enumerate(DECISION_NAMES):
            shadow_prices[f"lower_{variable}"] = float(-result.lower.marginals[index])

    return LPResult(
        allocation_df=_allocation_frame(x),
        objective=objective,
        status="optimal",
        binding_constraints=_binding_constraints(x, budget, human),
        note="SciPy HiGHS solved successfully.",
        shadow_prices=shadow_prices,
    ).as_dict()


def solve_bai02_pulp(B: float = 100, min_human: float = 20) -> dict[str, object]:
    """Solve Bai 2 with PuLP if available; otherwise return a graceful note."""
    budget, human = _validate_inputs(B, min_human)
    try:
        import pulp
    except ImportError:
        return LPResult(
            allocation_df=_empty_allocation(),
            objective=None,
            status="unavailable",
            binding_constraints=[],
            note="PuLP chưa được cài trong môi trường hiện tại. Cài `pip install -r requirements.txt` để dùng PuLP/CBC.",
            shadow_prices={},
        ).as_dict()

    mins = _minimums(human)
    model = pulp.LpProblem("bai02_lp_ngan_sach_so", pulp.LpMaximize)
    variables = {
        variable: pulp.LpVariable(variable, lowBound=float(mins[index]))
        for index, variable in enumerate(DECISION_NAMES)
    }

    x1, x2, x3, x4 = (variables[f"x{index}"] for index in range(1, 5))
    model += (
        OBJECTIVE_COEFFICIENTS[0] * x1
        + OBJECTIVE_COEFFICIENTS[1] * x2
        + OBJECTIVE_COEFFICIENTS[2] * x3
        + OBJECTIVE_COEFFICIENTS[3] * x4
    )
    model += x1 + x2 + x3 + x4 <= budget, "budget_total"
    model += x2 + x4 >= 0.35 * (x1 + x2 + x3 + x4), "ai_rd_share"

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)
    status = pulp.LpStatus.get(model.status, str(model.status)).lower()
    if status != "optimal":
        return LPResult(
            allocation_df=_empty_allocation(),
            objective=None,
            status=status,
            binding_constraints=[],
            note="PuLP/CBC không tìm được nghiệm tối ưu. Kiểm tra ngân sách so với mức tối thiểu.",
            shadow_prices={},
        ).as_dict()

    x = np.array([float(variables[f"x{index}"].value()) for index in range(1, 5)])
    shadow_prices = {}
    dual_notes = []
    for name, constraint in model.constraints.items():
        pi = getattr(constraint, "pi", None)
        if pi is None:
            dual_notes.append(name)
        else:
            shadow_prices[name] = float(pi)

    note = "PuLP/CBC solved successfully."
    if dual_notes:
        note += " CBC không trả dual values cho một số ràng buộc; phần shadow price chỉ mang tính tham khảo."

    return LPResult(
        allocation_df=_allocation_frame(x),
        objective=float(pulp.value(model.objective)),
        status="optimal",
        binding_constraints=_binding_constraints(x, budget, human),
        note=note,
        shadow_prices=shadow_prices,
    ).as_dict()


def sensitivity_budget(budgets: Iterable[float] = DEFAULT_BUDGETS) -> pd.DataFrame:
    """Solve the LP across multiple budgets for sensitivity analysis."""
    rows = []
    for budget in budgets:
        result = solve_bai02_scipy(B=float(budget), min_human=20)
        allocation_df = result["allocation_df"]
        row = {
            "budget": float(budget),
            "objective": result["objective"],
            "status": result["status"],
            "binding_constraints": ", ".join(result["binding_constraints"]),
            "budget_shadow_price": result["shadow_prices"].get("budget_total"),
        }
        for _, allocation_row in allocation_df.iterrows():
            row[str(allocation_row["variable"])] = float(allocation_row["allocation"])
        rows.append(row)
    return pd.DataFrame(rows)


def scenario_human_priority(B: float = 100, min_human: float = 30) -> dict[str, object]:
    """Run a scenario with a higher minimum allocation for digital human capital."""
    return solve_bai02_scipy(B=B, min_human=min_human)
