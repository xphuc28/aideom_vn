"""Bai 8: Dynamic optimization for 2026-2035."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


MODULE_TITLE = "Bài 8 - Tối ưu động 2026-2035"

YEARS = list(range(2026, 2036))
CONTROL_NAMES = ["share_K", "share_D", "share_AI", "share_H"]

COBB_DOUGLAS = {
    "alpha": 0.33,
    "beta": 0.42,
    "gamma": 0.10,
    "delta": 0.08,
    "theta": 0.07,
}

DEFAULT_PARAMS = {
    "delta_K": 0.05,
    "delta_D": 0.12,
    "delta_AI": 0.15,
    "theta_H": 0.8,
    "mu": 0.02,
    "phi1": 0.003,
    "phi2": 0.002,
    "phi3": 0.004,
    "rho": 0.97,
    "investment_rate": 0.28,
}

INITIAL_STATE = {
    "K": 27500.0,
    "L": 53.9,
    "D": 20.3,
    "AI": 86.0,
    "H": 30.0,
    "A": 1.0,
}

EPSILON = 1e-9


def module_status() -> str:
    """Return a short implementation status for the Streamlit page."""
    return "Đã triển khai mô phỏng và tối ưu động SLSQP cho giai đoạn 2026-2035."


def _production(state: dict[str, float]) -> float:
    return float(
        state["A"]
        * state["K"] ** COBB_DOUGLAS["alpha"]
        * state["L"] ** COBB_DOUGLAS["beta"]
        * state["D"] ** COBB_DOUGLAS["gamma"]
        * state["AI"] ** COBB_DOUGLAS["delta"]
        * state["H"] ** COBB_DOUGLAS["theta"]
    )


def _policy_matrix(policy_shares, T: int) -> np.ndarray:
    values = np.asarray(policy_shares, dtype=float)
    if values.ndim == 1:
        if values.size != 4:
            raise ValueError("policy_shares dạng vector phải có 4 phần tử cho K,D,AI,H.")
        values = np.tile(values, (T, 1))
    if values.shape != (T, 4):
        raise ValueError(f"policy_shares phải có shape {(T, 4)} hoặc vector 4 phần tử.")

    values = np.clip(values, 0.0, None)
    row_sums = values.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= EPSILON, 1.0, row_sums)
    return values / row_sums


def _shock_multiplier(year: int, shock) -> float:
    if shock is None:
        return 1.0
    if isinstance(shock, str) and shock == "shock_2028":
        return 0.92 if year == 2028 else 1.0
    if isinstance(shock, dict):
        return float(shock.get(year, 1.0))
    return 1.0


def simulate_policy(
    policy_shares,
    shock=None,
    T: int = 10,
    rho: float = 0.97,
    investment_rate: float = 0.28,
    initial_state: dict[str, float] | None = None,
    params: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Simulate a 10-year policy path for K, D, AI, H, A, Y, and C."""
    model_params = DEFAULT_PARAMS.copy()
    if params:
        model_params.update(params)
    model_params["rho"] = float(rho)
    model_params["investment_rate"] = float(investment_rate)

    state = INITIAL_STATE.copy()
    if initial_state:
        state.update({key: float(value) for key, value in initial_state.items() if key in state})

    shares = _policy_matrix(policy_shares, T)
    rows = []

    for t in range(T):
        year = YEARS[t] if t < len(YEARS) else YEARS[0] + t
        y_pre_shock = _production(state)
        y = y_pre_shock * _shock_multiplier(year, shock)
        investment_budget = model_params["investment_rate"] * y
        consumption = max((1.0 - model_params["investment_rate"]) * y, EPSILON)
        s_k, s_d, s_ai, s_h = shares[t]

        rows.append(
            {
                "year": year,
                "K": state["K"],
                "L": state["L"],
                "D": state["D"],
                "AI": state["AI"],
                "H": state["H"],
                "A": state["A"],
                "Y": max(y, EPSILON),
                "Y_pre_shock": y_pre_shock,
                "C": consumption,
                "investment_budget": investment_budget,
                "share_K": s_k,
                "share_D": s_d,
                "share_AI": s_ai,
                "share_H": s_h,
                "discounted_log_C": (model_params["rho"] ** t) * np.log(consumption),
            }
        )

        next_k = (1.0 - model_params["delta_K"]) * state["K"] + s_k * investment_budget
        next_d = (1.0 - model_params["delta_D"]) * state["D"] + model_params["phi1"] * s_d * investment_budget
        next_ai = (1.0 - model_params["delta_AI"]) * state["AI"] + model_params["phi2"] * s_ai * investment_budget
        next_h = state["H"] + model_params["theta_H"] * model_params["phi3"] * s_h * investment_budget + model_params["mu"] * (40.0 - state["H"])
        digital_ai_spillover = 0.0002 * (next_d - state["D"]) + 0.00015 * (next_ai - state["AI"]) + 0.0001 * (next_h - state["H"])
        next_a = state["A"] * (1.0 + model_params["mu"] + max(digital_ai_spillover, -0.01))

        state = {
            "K": max(next_k, EPSILON),
            "L": state["L"] * 1.006,
            "D": max(next_d, EPSILON),
            "AI": max(next_ai, EPSILON),
            "H": max(next_h, EPSILON),
            "A": max(next_a, EPSILON),
        }

    return pd.DataFrame(rows)


def _welfare_from_vector(vector: np.ndarray, T: int, rho: float, investment_rate: float, shock) -> float:
    shares = _policy_matrix(vector.reshape((T, 4)), T)
    trajectory = simulate_policy(shares, shock=shock, T=T, rho=rho, investment_rate=investment_rate)
    return float(trajectory["discounted_log_C"].sum())


def optimize_dynamic(
    T: int = 10,
    rho: float = 0.97,
    bounds: tuple[float, float] = (0.02, 0.90),
    investment_rate: float = 0.28,
    shock=None,
    maxiter: int = 250,
) -> dict[str, object]:
    """Optimize yearly allocation shares with scipy.optimize.minimize SLSQP."""
    lower, upper = bounds
    x0 = np.tile(np.array([0.25, 0.25, 0.25, 0.25]), T)
    variable_bounds = [(lower, upper) for _ in range(4 * T)]
    constraints = []
    for t in range(T):
        constraints.append(
            {
                "type": "eq",
                "fun": lambda x, t=t: np.sum(x[t * 4 : (t + 1) * 4]) - 1.0,
            }
        )

    def objective(x):
        return -_welfare_from_vector(x, T=T, rho=rho, investment_rate=investment_rate, shock=shock)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=variable_bounds,
        constraints=constraints,
        options={"maxiter": int(maxiter), "ftol": 1e-8, "disp": False},
    )

    if result.success:
        shares = _policy_matrix(result.x.reshape((T, 4)), T)
        status = "optimal"
        note = "SLSQP solved successfully."
    else:
        shares = _policy_matrix(x0.reshape((T, 4)), T)
        status = "fallback_equal"
        note = f"SLSQP không hội tụ ({result.message}); dùng equal investment fallback."

    trajectory_df = simulate_policy(shares, shock=shock, T=T, rho=rho, investment_rate=investment_rate)
    return {
        "trajectory_df": trajectory_df,
        "policy_df": trajectory_df[["year", *CONTROL_NAMES]].copy(),
        "welfare": float(trajectory_df["discounted_log_C"].sum()),
        "status": status,
        "note": note,
        "optimizer_message": str(result.message),
    }


def _front_load_policy(T: int) -> np.ndarray:
    policy = np.zeros((T, 4))
    for t in range(T):
        if t < min(4, T):
            policy[t] = [0.18, 0.30, 0.34, 0.18]
        else:
            policy[t] = [0.34, 0.18, 0.16, 0.32]
    return policy


def shock_2028() -> dict[int, float]:
    """Return a shock that reduces 2028 output by 8%."""
    return {2028: 0.92}


def compare_strategies(
    rho: float = 0.97,
    investment_rate: float = 0.28,
    shock=None,
    T: int = 10,
) -> dict[str, pd.DataFrame]:
    """Compare equal investment, front-load, and optimized strategies."""
    equal_policy = np.tile(np.array([0.25, 0.25, 0.25, 0.25]), (T, 1))
    front_policy = _front_load_policy(T)
    optimized = optimize_dynamic(T=T, rho=rho, investment_rate=investment_rate, shock=shock)

    strategy_runs = {
        "equal investment": {
            "trajectory_df": simulate_policy(equal_policy, shock=shock, T=T, rho=rho, investment_rate=investment_rate),
            "status": "simulated",
            "note": "Equal shares across K,D,AI,H.",
        },
        "front-load": {
            "trajectory_df": simulate_policy(front_policy, shock=shock, T=T, rho=rho, investment_rate=investment_rate),
            "status": "simulated",
            "note": "Front-load D/AI, then rebalance toward K/H.",
        },
        "optimized": optimized,
    }

    trajectories = []
    summary_rows = []
    for strategy, result in strategy_runs.items():
        trajectory = result["trajectory_df"].copy()
        trajectory["strategy"] = strategy
        trajectories.append(trajectory)
        summary_rows.append(
            {
                "strategy": strategy,
                "status": result["status"],
                "welfare": float(trajectory["discounted_log_C"].sum()),
                "Y_2035": float(trajectory["Y"].iloc[-1]),
                "C_2035": float(trajectory["C"].iloc[-1]),
                "K_2035": float(trajectory["K"].iloc[-1]),
                "D_2035": float(trajectory["D"].iloc[-1]),
                "AI_2035": float(trajectory["AI"].iloc[-1]),
                "H_2035": float(trajectory["H"].iloc[-1]),
                "note": result["note"],
            }
        )

    return {
        "trajectory_df": pd.concat(trajectories, ignore_index=True),
        "summary_df": pd.DataFrame(summary_rows).sort_values("welfare", ascending=False).reset_index(drop=True),
        "optimized_result": optimized,
    }
