"""Bai 7: Multi-objective Pareto optimization with NSGA-II fallback support."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.bai04_region_lp import (
    BETA,
    D0,
    GAMMA,
    H_MIN,
    ITEMS,
    ITEM_NAMES,
    REGIONS,
    REGION_MAX,
    REGION_MIN,
    REGION_NAMES,
)


MODULE_TITLE = "Bài 7 - Tối ưu đa mục tiêu Pareto NSGA-II"

DEFAULT_BUDGET = 50000.0
DEFAULT_LAMBDA = 0.7

EMISSION_COEF = {
    "NMM": 0.72,
    "RRD": 0.62,
    "NCC": 0.68,
    "CH": 0.78,
    "SE": 0.58,
    "MD": 0.70,
}

CYBER_RHO = {
    "NMM": 0.42,
    "RRD": 0.34,
    "NCC": 0.38,
    "CH": 0.46,
    "SE": 0.32,
    "MD": 0.40,
}

HUMAN_SIGMA = {
    "NMM": 0.24,
    "RRD": 0.30,
    "NCC": 0.26,
    "CH": 0.23,
    "SE": 0.31,
    "MD": 0.25,
}


def module_status() -> str:
    """Return a short implementation status for the Streamlit page."""
    return "Đã triển khai tối ưu Pareto 4 mục tiêu với NSGA-II optional và random feasible fallback."


def _max_feasible_lambda() -> float:
    max_m = max(D0.values())
    return min((D0[region] + GAMMA * REGION_MAX) / max_m for region in REGIONS)


def _effective_lambda(lambda_: float) -> float:
    return min(float(lambda_), max(_max_feasible_lambda() - 1e-5, 0.0))


def _as_matrix(x) -> np.ndarray:
    matrix = np.asarray(x, dtype=float)
    if matrix.size != len(REGIONS) * len(ITEMS):
        raise ValueError("x phải có 24 biến = 6 vùng x 4 hạng mục.")
    return matrix.reshape((len(REGIONS), len(ITEMS)))


def _candidate_vectors(raw_solutions) -> list[np.ndarray]:
    """Normalize optimizer output into valid 24-variable candidate vectors.

    Some pymoo failure modes on hosted environments return ``None`` or a
    partial/object-shaped ``result.X`` instead of a full ``(n, 24)`` array.
    Streamlit Cloud then surfaced a redacted ``ValueError`` when those partial
    values reached ``_as_matrix``. This helper filters such candidates and lets
    ``run_nsga2`` fall back gracefully to random feasible search.
    """
    expected_size = len(REGIONS) * len(ITEMS)
    if raw_solutions is None:
        return []

    try:
        array = np.asarray(raw_solutions, dtype=float)
    except (TypeError, ValueError):
        vectors: list[np.ndarray] = []
        try:
            iterator = list(raw_solutions)
        except TypeError:
            return []
        for item in iterator:
            try:
                vector = np.asarray(item, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
            if vector.size == expected_size:
                vectors.append(vector)
        return vectors

    if array.ndim == 1:
        return [array.reshape(-1)] if array.size == expected_size else []
    if array.ndim >= 2:
        return [row.reshape(-1) for row in array if row.size == expected_size]
    return []


def _bounded_random_allocation(total: float, caps: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    """Allocate a total across bounded buckets using a sequential random repair."""
    caps = np.asarray(caps, dtype=float)
    if total < -1e-9 or total > caps.sum() + 1e-9:
        return None

    allocation = np.zeros_like(caps)
    remaining = float(total)
    order = rng.permutation(len(caps))
    for position, idx in enumerate(order):
        later = order[position + 1 :]
        min_keep = max(0.0, remaining - caps[later].sum())
        max_take = min(caps[idx], remaining)
        if position == len(order) - 1:
            take = remaining
        elif max_take <= min_keep:
            take = min_keep
        else:
            take = rng.uniform(min_keep, max_take)
        allocation[idx] = take
        remaining -= take
    return np.minimum(np.maximum(allocation, 0.0), caps)


def _sample_feasible(
    rng: np.random.Generator,
    budget: float,
    fairness: bool,
    lambda_: float,
) -> np.ndarray | None:
    d_alloc = np.zeros(len(REGIONS))
    if fairness:
        effective_lambda = _effective_lambda(lambda_)
        m_ref = max(D0.values())
        for idx, region in enumerate(REGIONS):
            d_alloc[idx] = max(0.0, (effective_lambda * m_ref - D0[region]) / GAMMA)
            d_cap = max(0.0, (m_ref - D0[region]) / GAMMA)
            d_alloc[idx] = min(d_alloc[idx], d_cap, REGION_MAX)

    lower_totals = np.maximum(REGION_MIN, d_alloc)
    if lower_totals.sum() > budget:
        return None

    extra_caps = REGION_MAX - lower_totals
    extra_budget = min(budget - lower_totals.sum(), extra_caps.sum())
    total_extra = rng.uniform(0.0, extra_budget)
    extra = _bounded_random_allocation(total_extra, extra_caps, rng)
    if extra is None:
        return None
    region_totals = lower_totals + extra

    h_caps = np.maximum(region_totals - d_alloc, 0.0)
    if h_caps.sum() < H_MIN:
        return None
    h_alloc = _bounded_random_allocation(H_MIN, h_caps, rng)
    if h_alloc is None:
        return None

    matrix = np.zeros((len(REGIONS), len(ITEMS)))
    d_idx = ITEMS.index("D")
    h_idx = ITEMS.index("H")
    i_idx = ITEMS.index("I")
    ai_idx = ITEMS.index("AI")
    matrix[:, d_idx] = d_alloc
    matrix[:, h_idx] = h_alloc

    remaining = np.maximum(region_totals - matrix[:, d_idx] - matrix[:, h_idx], 0.0)
    for r_idx, amount in enumerate(remaining):
        shares = rng.dirichlet([1.2, 1.4, 0.8])
        matrix[r_idx, i_idx] += amount * shares[0]
        matrix[r_idx, ai_idx] += amount * shares[1]
        matrix[r_idx, h_idx] += amount * shares[2]

    if matrix.sum() > budget + 1e-6:
        return None
    return matrix.reshape(-1)


def evaluate_solution(x) -> dict[str, float]:
    """Evaluate one 24-variable allocation solution on four objectives."""
    matrix = _as_matrix(x)
    i_idx = ITEMS.index("I")
    ai_idx = ITEMS.index("AI")
    h_idx = ITEMS.index("H")

    gdp_gain = 0.0
    emission = 0.0
    net_cyber_risk = 0.0
    for r_idx, region in enumerate(REGIONS):
        for j_idx, item in enumerate(ITEMS):
            gdp_gain += BETA[region][item] * matrix[r_idx, j_idx]
        emission += EMISSION_COEF[region] * (matrix[r_idx, i_idx] + matrix[r_idx, ai_idx])
        net_cyber_risk += CYBER_RHO[region] * matrix[r_idx, ai_idx] - HUMAN_SIGMA[region] * matrix[r_idx, h_idx]

    region_totals = matrix.sum(axis=1)
    inequality = float(np.mean(np.abs(region_totals - region_totals.mean())))
    return {
        "gdp_gain": float(gdp_gain),
        "inequality": inequality,
        "emission": float(emission),
        "net_cyber_risk": float(net_cyber_risk),
    }


def _is_feasible(x, budget: float, fairness: bool, lambda_: float) -> bool:
    try:
        matrix = _as_matrix(x)
    except (TypeError, ValueError):
        return False
    if np.any(matrix < -1e-8):
        return False
    if matrix.sum() > budget + 1e-6:
        return False
    region_totals = matrix.sum(axis=1)
    if np.any(region_totals < REGION_MIN - 1e-6) or np.any(region_totals > REGION_MAX + 1e-6):
        return False
    if matrix[:, ITEMS.index("H")].sum() < H_MIN - 1e-6:
        return False
    if fairness:
        digital_after = np.array([D0[region] + GAMMA * matrix[idx, ITEMS.index("D")] for idx, region in enumerate(REGIONS)])
        m_value = digital_after.max()
        if np.any(digital_after < _effective_lambda(lambda_) * m_value - 1e-6):
            return False
    return True


def _pareto_filter(df: pd.DataFrame) -> pd.DataFrame:
    objectives = df[["gdp_gain", "inequality", "emission", "net_cyber_risk"]].to_numpy()
    minimize = np.column_stack([-objectives[:, 0], objectives[:, 1], objectives[:, 2], objectives[:, 3]])
    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if not keep[i]:
            continue
        dominated = np.all(minimize <= minimize[i], axis=1) & np.any(minimize < minimize[i], axis=1)
        if dominated.any():
            keep[i] = False
    return df.loc[keep].sort_values("gdp_gain", ascending=False).reset_index(drop=True)


def _rows_from_solutions(solutions: list[np.ndarray], method: str) -> pd.DataFrame:
    rows = []
    for sol_idx, x in enumerate(solutions):
        metrics = evaluate_solution(x)
        matrix = _as_matrix(x)
        row = {
            "solution_id": f"{method}_{sol_idx + 1}",
            "method": method,
            "total_budget": float(matrix.sum()),
            **metrics,
        }
        for r_idx, region in enumerate(REGIONS):
            for j_idx, item in enumerate(ITEMS):
                row[f"{region}_{item}"] = float(matrix[r_idx, j_idx])
        rows.append(row)
    return pd.DataFrame(rows)


def choose_compromise_topsis(
    pareto_df: pd.DataFrame,
    weights=(0.40, 0.25, 0.20, 0.15),
) -> pd.DataFrame:
    """Choose one compromise Pareto solution using TOPSIS over four objectives."""
    if pareto_df.empty:
        return pd.DataFrame()

    objective_cols = ["gdp_gain", "inequality", "emission", "net_cyber_risk"]
    data = pareto_df[objective_cols].astype(float).copy()
    benefit = np.array([True, False, False, False])
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    normalized = pd.DataFrame(index=data.index)
    for col in objective_cols:
        min_value = data[col].min()
        max_value = data[col].max()
        if np.isclose(max_value, min_value):
            normalized[col] = 0.5
        else:
            normalized[col] = (data[col] - min_value) / (max_value - min_value)

    weighted = normalized.to_numpy() * w
    ideal_best = np.where(benefit, weighted.max(axis=0), weighted.min(axis=0))
    ideal_worst = np.where(benefit, weighted.min(axis=0), weighted.max(axis=0))
    d_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    score = d_worst / (d_best + d_worst + 1e-12)

    chosen = pareto_df.iloc[[int(np.argmax(score))]].copy()
    chosen["compromise_score"] = float(score.max())
    return chosen.reset_index(drop=True)


def _summary_from_pareto(pareto_df: pd.DataFrame, compromise_solution: pd.DataFrame) -> pd.DataFrame:
    if pareto_df.empty:
        return pd.DataFrame()
    max_growth = pareto_df.sort_values("gdp_gain", ascending=False).head(1).copy()
    max_growth["scenario"] = "Tăng trưởng cao nhất"
    compromise = compromise_solution.copy()
    compromise["scenario"] = "Nghiệm thỏa hiệp"
    return pd.concat([max_growth, compromise], ignore_index=True, sort=False)[
        ["scenario", "solution_id", "gdp_gain", "inequality", "emission", "net_cyber_risk", "total_budget"]
    ]


def random_feasible_search(
    n_samples: int = 2000,
    seed: int = 42,
    budget: float = DEFAULT_BUDGET,
    fairness: bool = True,
    lambda_: float = DEFAULT_LAMBDA,
) -> dict[str, object]:
    """Generate feasible random allocations and return an approximate Pareto set."""
    rng = np.random.default_rng(seed)
    solutions = []
    attempts = max(n_samples * 5, 1000)
    for _ in range(attempts):
        x = _sample_feasible(rng, budget, fairness, lambda_)
        if x is None or not _is_feasible(x, budget, fairness, lambda_):
            continue
        solutions.append(x)
        if len(solutions) >= n_samples:
            break

    if not solutions:
        return {
            "pareto_df": pd.DataFrame(),
            "compromise_solution": pd.DataFrame(),
            "summary_df": pd.DataFrame(),
            "note": "Không tạo được nghiệm khả thi bằng random search.",
            "method": "random",
        }

    all_df = _rows_from_solutions(solutions, "random")
    pareto_df = _pareto_filter(all_df)
    compromise = choose_compromise_topsis(pareto_df)
    summary_df = _summary_from_pareto(pareto_df, compromise)
    return {
        "pareto_df": pareto_df,
        "compromise_solution": compromise,
        "summary_df": summary_df,
        "note": f"Random feasible search: {len(solutions)} feasible samples, {len(pareto_df)} Pareto candidates.",
        "method": "random",
    }


def run_nsga2(
    pop_size: int = 100,
    n_gen: int = 200,
    seed: int = 42,
    budget: float = DEFAULT_BUDGET,
    fairness: bool = True,
    lambda_: float = DEFAULT_LAMBDA,
) -> dict[str, object]:
    """Run NSGA-II with pymoo when available; otherwise use random feasible search."""
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.optimize import minimize
    except Exception:
        fallback_samples = max(int(pop_size) * max(int(n_gen), 1), 500)
        result = random_feasible_search(
            n_samples=min(fallback_samples, 5000),
            seed=seed,
            budget=budget,
            fairness=fairness,
            lambda_=lambda_,
        )
        result["note"] = "pymoo chưa khả dụng; " + result["note"]
        return result

    class AideomParetoProblem(ElementwiseProblem):
        def __init__(self):
            n_constraints = 1 + 2 * len(REGIONS) + 1 + (len(REGIONS) if fairness else 0)
            super().__init__(
                n_var=len(REGIONS) * len(ITEMS),
                n_obj=4,
                n_ieq_constr=n_constraints,
                xl=0.0,
                xu=REGION_MAX,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            metrics = evaluate_solution(x)
            matrix = _as_matrix(x)
            region_totals = matrix.sum(axis=1)
            constraints = [matrix.sum() - budget]
            constraints.extend(REGION_MIN - region_totals)
            constraints.extend(region_totals - REGION_MAX)
            constraints.append(H_MIN - matrix[:, ITEMS.index("H")].sum())
            if fairness:
                digital_after = np.array(
                    [D0[region] + GAMMA * matrix[idx, ITEMS.index("D")] for idx, region in enumerate(REGIONS)]
                )
                m_value = digital_after.max()
                constraints.extend(_effective_lambda(lambda_) * m_value - digital_after)

            out["F"] = [
                -metrics["gdp_gain"],
                metrics["inequality"],
                metrics["emission"],
                metrics["net_cyber_risk"],
            ]
            out["G"] = constraints

    algorithm = NSGA2(pop_size=int(pop_size))
    result = minimize(
        AideomParetoProblem(),
        algorithm,
        ("n_gen", int(n_gen)),
        seed=int(seed),
        verbose=False,
    )

    feasible_solutions = []
    for x in _candidate_vectors(getattr(result, "X", None)):
        if _is_feasible(x, budget, fairness, lambda_):
            feasible_solutions.append(np.asarray(x, dtype=float).reshape(-1))

    if not feasible_solutions:
        fallback_result = random_feasible_search(
            n_samples=max(int(pop_size), 500),
            seed=seed,
            budget=budget,
            fairness=fairness,
            lambda_=lambda_,
        )
        fallback_result["note"] = "NSGA-II không trả nghiệm khả thi hợp lệ; " + fallback_result["note"]
        return fallback_result

    all_df = _rows_from_solutions(feasible_solutions, "nsga2")
    pareto_df = _pareto_filter(all_df)
    compromise = choose_compromise_topsis(pareto_df)
    summary_df = _summary_from_pareto(pareto_df, compromise)
    return {
        "pareto_df": pareto_df,
        "compromise_solution": compromise,
        "summary_df": summary_df,
        "note": f"NSGA-II completed: {len(feasible_solutions)} feasible solutions, {len(pareto_df)} Pareto candidates.",
        "method": "nsga2",
    }
