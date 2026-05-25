"""Bai 10: Two-stage stochastic programming model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog


MODULE_TITLE = "Bài 10 - Quy hoạch ngẫu nhiên hai giai đoạn"

ITEMS = ["I", "D", "AI", "H"]
ITEM_NAMES = {
    "I": "Hạ tầng",
    "D": "Số hóa",
    "AI": "AI và dữ liệu",
    "H": "Nhân lực",
}
SCENARIOS = ["s1", "s2", "s3", "s4"]
SCENARIO_PROBS = {"s1": 0.30, "s2": 0.45, "s3": 0.20, "s4": 0.05}
SCENARIO_LABELS = {
    "s1": "Tăng tốc thuận lợi",
    "s2": "Cơ sở",
    "s3": "Chậm lại",
    "s4": "Cú sốc bất lợi",
}

BETA_FIRST = {"I": 1.00, "D": 1.10, "AI": 1.25, "H": 0.95}
BETA_SCENARIO = {
    "s1": {"I": 1.25, "D": 1.35, "AI": 1.55, "H": 1.05},
    "s2": {"I": 1.00, "D": 1.10, "AI": 1.25, "H": 0.95},
    "s3": {"I": 0.75, "D": 0.85, "AI": 0.90, "H": 1.00},
    "s4": {"I": 0.40, "D": 0.50, "AI": 0.55, "H": 1.10},
}

FIRST_STAGE_BUDGET = 65000.0
SECOND_STAGE_BUDGET = 15000.0
TOLERANCE = 1e-8


def module_status() -> str:
    """Return a short implementation status for the Streamlit page."""
    return "Đã triển khai quy hoạch ngẫu nhiên hai giai đoạn với PuLP optional và SciPy fallback."


def scenario_table() -> pd.DataFrame:
    """Return the hard-coded scenario probability and beta table."""
    rows = []
    for scenario in SCENARIOS:
        row = {
            "scenario": scenario,
            "label": SCENARIO_LABELS[scenario],
            "probability": SCENARIO_PROBS[scenario],
        }
        for item in ITEMS:
            row[f"beta_{item}"] = BETA_SCENARIO[scenario][item]
        rows.append(row)
    return pd.DataFrame(rows)


def _x_index(item: str) -> int:
    return ITEMS.index(item)


def _y_index(scenario: str, item: str, scenarios: list[str]) -> int:
    return len(ITEMS) + scenarios.index(scenario) * len(ITEMS) + ITEMS.index(item)


def _first_stage_df(x: np.ndarray) -> pd.DataFrame:
    rows = []
    for item in ITEMS:
        allocation = float(x[_x_index(item)])
        beta = BETA_FIRST[item]
        rows.append(
            {
                "item": item,
                "item_name": ITEM_NAMES[item],
                "allocation": allocation,
                "beta_first": beta,
                "value_contribution": allocation * beta,
            }
        )
    return pd.DataFrame(rows)


def _second_stage_df(y: np.ndarray, scenarios: list[str], probs: dict[str, float]) -> pd.DataFrame:
    rows = []
    y_matrix = y.reshape((len(scenarios), len(ITEMS)))
    for s_idx, scenario in enumerate(scenarios):
        for item in ITEMS:
            allocation = float(y_matrix[s_idx, ITEMS.index(item)])
            beta = BETA_SCENARIO[scenario][item]
            probability = probs[scenario]
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "probability": probability,
                    "item": item,
                    "item_name": ITEM_NAMES[item],
                    "allocation": allocation,
                    "beta_s": beta,
                    "expected_value_contribution": probability * beta * allocation,
                }
            )
    return pd.DataFrame(rows)


def _result_dict(
    x: np.ndarray,
    y: np.ndarray,
    scenarios: list[str],
    probs: dict[str, float],
    objective: float | None,
    status: str,
    note: str,
    include_empty_metrics: bool = True,
) -> dict[str, object]:
    first_stage_df = _first_stage_df(x)
    second_stage_df = _second_stage_df(y, scenarios, probs)
    return {
        "first_stage_df": first_stage_df,
        "second_stage_df": second_stage_df,
        "objective": None if objective is None else float(objective),
        "status": status,
        "note": note,
        "vss_evpi_df": pd.DataFrame() if include_empty_metrics else None,
    }


def _solve_scipy(
    scenarios: list[str] | None = None,
    probs: dict[str, float] | None = None,
    fixed_x: dict[str, float] | None = None,
    budget: float = FIRST_STAGE_BUDGET,
) -> dict[str, object]:
    scenarios = list(SCENARIOS if scenarios is None else scenarios)
    probs = SCENARIO_PROBS.copy() if probs is None else probs.copy()

    n_vars = len(ITEMS) + len(scenarios) * len(ITEMS)
    c = np.zeros(n_vars)
    for item in ITEMS:
        c[_x_index(item)] = -BETA_FIRST[item]
    for scenario in scenarios:
        for item in ITEMS:
            c[_y_index(scenario, item, scenarios)] = -probs[scenario] * BETA_SCENARIO[scenario][item]

    a_ub = []
    b_ub = []
    row = np.zeros(n_vars)
    for item in ITEMS:
        row[_x_index(item)] = 1.0
    a_ub.append(row)
    b_ub.append(float(budget))

    for scenario in scenarios:
        row = np.zeros(n_vars)
        for item in ITEMS:
            row[_y_index(scenario, item, scenarios)] = 1.0
        a_ub.append(row)
        b_ub.append(SECOND_STAGE_BUDGET)

        row = np.zeros(n_vars)
        row[_y_index(scenario, "AI", scenarios)] = 1.0
        row[_x_index("H")] = -0.5
        a_ub.append(row)
        b_ub.append(0.0)

    bounds = [(0.0, None) for _ in range(n_vars)]
    if fixed_x is not None:
        for item in ITEMS:
            value = float(fixed_x[item])
            bounds[_x_index(item)] = (value, value)

    result = linprog(
        c,
        A_ub=np.array(a_ub, dtype=float),
        b_ub=np.array(b_ub, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return _result_dict(
            x=np.zeros(len(ITEMS)),
            y=np.zeros(len(scenarios) * len(ITEMS)),
            scenarios=scenarios,
            probs=probs,
            objective=None,
            status=result.message,
            note="SciPy HiGHS không tìm được nghiệm tối ưu.",
        )

    x = np.maximum(result.x[: len(ITEMS)], 0.0)
    y = np.maximum(result.x[len(ITEMS) :], 0.0)
    return _result_dict(
        x=x,
        y=y,
        scenarios=scenarios,
        probs=probs,
        objective=-float(result.fun),
        status="optimal",
        note="Solved with SciPy HiGHS.",
    )


def _solve_pulp_or_none() -> dict[str, object] | None:
    try:
        import pulp
    except ImportError:
        return None

    model = pulp.LpProblem("bai10_two_stage_stochastic", pulp.LpMaximize)
    x = {item: pulp.LpVariable(f"x_{item}", lowBound=0) for item in ITEMS}
    y = {
        (scenario, item): pulp.LpVariable(f"y_{scenario}_{item}", lowBound=0)
        for scenario in SCENARIOS
        for item in ITEMS
    }

    model += pulp.lpSum(BETA_FIRST[item] * x[item] for item in ITEMS) + pulp.lpSum(
        SCENARIO_PROBS[scenario] * BETA_SCENARIO[scenario][item] * y[(scenario, item)]
        for scenario in SCENARIOS
        for item in ITEMS
    )
    model += pulp.lpSum(x[item] for item in ITEMS) <= FIRST_STAGE_BUDGET
    for scenario in SCENARIOS:
        model += pulp.lpSum(y[(scenario, item)] for item in ITEMS) <= SECOND_STAGE_BUDGET
        model += y[(scenario, "AI")] <= 0.5 * x["H"]

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus.get(model.status, str(model.status)).lower()
    if status != "optimal":
        return _result_dict(
            x=np.zeros(len(ITEMS)),
            y=np.zeros(len(SCENARIOS) * len(ITEMS)),
            scenarios=SCENARIOS,
            probs=SCENARIO_PROBS,
            objective=None,
            status=status,
            note="PuLP/CBC không tìm được nghiệm tối ưu.",
        )

    x_array = np.array([float(x[item].value()) for item in ITEMS])
    y_array = np.array([float(y[(scenario, item)].value()) for scenario in SCENARIOS for item in ITEMS])
    return _result_dict(
        x=x_array,
        y=y_array,
        scenarios=SCENARIOS,
        probs=SCENARIO_PROBS,
        objective=float(pulp.value(model.objective)),
        status="optimal",
        note="Solved with PuLP/CBC.",
    )


def solve_stochastic_pulp() -> dict[str, object]:
    """Solve the two-stage stochastic program with PuLP, falling back to SciPy."""
    result = _solve_pulp_or_none()
    if result is None:
        result = _solve_scipy()
        result["note"] = "PuLP chưa được cài; đã dùng SciPy HiGHS fallback."
    result["vss_evpi_df"] = compute_vss_evpi()
    return result


def solve_deterministic_scenario(s: str) -> dict[str, object]:
    """Solve a deterministic wait-and-see problem for one scenario."""
    if s not in SCENARIOS:
        raise ValueError(f"Scenario không hợp lệ: {s}")
    return _solve_scipy(scenarios=[s], probs={s: 1.0})


def solve_expected_value() -> dict[str, object]:
    """Solve the expected-value deterministic approximation."""
    expected_beta = {
        item: sum(SCENARIO_PROBS[scenario] * BETA_SCENARIO[scenario][item] for scenario in SCENARIOS)
        for item in ITEMS
    }
    original = {scenario: BETA_SCENARIO[scenario].copy() for scenario in SCENARIOS}
    try:
        BETA_SCENARIO["s2"] = expected_beta
        result = _solve_scipy(scenarios=["s2"], probs={"s2": 1.0})
        result["second_stage_df"]["scenario"] = "expected"
        result["second_stage_df"]["scenario_label"] = "Expected value"
        result["note"] = "Solved expected-value deterministic approximation."
        return result
    finally:
        for scenario in SCENARIOS:
            BETA_SCENARIO[scenario] = original[scenario]


def _evaluate_fixed_first_stage(fixed_x: dict[str, float]) -> dict[str, object]:
    return _solve_scipy(scenarios=SCENARIOS, probs=SCENARIO_PROBS, fixed_x=fixed_x)


def compute_vss_evpi() -> pd.DataFrame:
    """Compute VSS and EVPI when all component LPs are feasible."""
    stochastic = _solve_scipy()
    expected = solve_expected_value()

    if stochastic["objective"] is None or expected["objective"] is None:
        return pd.DataFrame()

    fixed_x = dict(
        zip(
            expected["first_stage_df"]["item"],
            expected["first_stage_df"]["allocation"],
        )
    )
    eev = _evaluate_fixed_first_stage(fixed_x)
    deterministic_results = [solve_deterministic_scenario(scenario) for scenario in SCENARIOS]
    if eev["objective"] is None or any(result["objective"] is None for result in deterministic_results):
        return pd.DataFrame()

    rp = float(stochastic["objective"])
    eev_value = float(eev["objective"])
    ws = float(
        sum(SCENARIO_PROBS[scenario] * deterministic_results[index]["objective"] for index, scenario in enumerate(SCENARIOS))
    )
    return pd.DataFrame(
        [
            {"metric": "RP_stochastic", "value": rp, "description": "Optimal stochastic recourse value"},
            {"metric": "EEV", "value": eev_value, "description": "Expected result of expected-value first-stage"},
            {"metric": "WS", "value": ws, "description": "Wait-and-see value with perfect information"},
            {"metric": "VSS", "value": rp - eev_value, "description": "Value of the stochastic solution"},
            {"metric": "EVPI", "value": ws - rp, "description": "Expected value of perfect information"},
        ]
    )


def robust_minimax_regret() -> dict[str, object]:
    """Optional robust fallback: minimize maximum regret across scenarios."""
    best_by_scenario = {scenario: solve_deterministic_scenario(scenario)["objective"] for scenario in SCENARIOS}
    if any(value is None for value in best_by_scenario.values()):
        return _solve_scipy()

    n_vars = len(ITEMS) + len(SCENARIOS) * len(ITEMS) + 1
    regret_idx = n_vars - 1
    c = np.zeros(n_vars)
    c[regret_idx] = 1.0

    a_ub = []
    b_ub = []
    row = np.zeros(n_vars)
    for item in ITEMS:
        row[_x_index(item)] = 1.0
    a_ub.append(row)
    b_ub.append(FIRST_STAGE_BUDGET)

    for scenario in SCENARIOS:
        row = np.zeros(n_vars)
        for item in ITEMS:
            row[_y_index(scenario, item, SCENARIOS)] = 1.0
        a_ub.append(row)
        b_ub.append(SECOND_STAGE_BUDGET)

        row = np.zeros(n_vars)
        row[_y_index(scenario, "AI", SCENARIOS)] = 1.0
        row[_x_index("H")] = -0.5
        a_ub.append(row)
        b_ub.append(0.0)

        # best_s - value_s <= R -> -value_s - R <= -best_s
        row = np.zeros(n_vars)
        for item in ITEMS:
            row[_x_index(item)] = -BETA_FIRST[item]
            row[_y_index(scenario, item, SCENARIOS)] = -BETA_SCENARIO[scenario][item]
        row[regret_idx] = -1.0
        a_ub.append(row)
        b_ub.append(-float(best_by_scenario[scenario]))

    result = linprog(
        c,
        A_ub=np.array(a_ub, dtype=float),
        b_ub=np.array(b_ub, dtype=float),
        bounds=[(0.0, None) for _ in range(n_vars)],
        method="highs",
    )
    if not result.success:
        return _solve_scipy()

    x = result.x[: len(ITEMS)]
    y = result.x[len(ITEMS) : len(ITEMS) + len(SCENARIOS) * len(ITEMS)]
    output = _result_dict(
        x=x,
        y=y,
        scenarios=SCENARIOS,
        probs=SCENARIO_PROBS,
        objective=float(result.x[regret_idx]),
        status="optimal",
        note="Solved minimax regret fallback; objective is maximum regret.",
    )
    output["max_regret"] = float(result.x[regret_idx])
    return output
