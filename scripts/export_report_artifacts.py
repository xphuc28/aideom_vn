"""Run AIDEOM-VN models and export report artifacts.

This script does not change model logic. It imports the existing computation
modules, runs representative/default scenarios, and writes CSV, Markdown, and
PNG artifacts for the final report into reports/figures/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "models"))
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.bai01_cobb_douglas import DEFAULT_PARAMS as BAI01_PARAMS
from src.bai01_cobb_douglas import run_bai01
from src.bai02_lp_budget import scenario_human_priority, sensitivity_budget, solve_bai02_scipy
from src.bai03_priority import DEFAULT_WEIGHTS as BAI03_WEIGHTS
from src.bai03_priority import compare_policy_weights, compute_priority, sensitivity_ai_weight as bai03_sensitivity
from src.bai04_region_lp import compare_fairness, solve_bai04_pulp
from src.bai05_mip_projects import project_dataframe, solve_bai05, solve_budget_scenario, solve_with_risk_probabilities
from src.bai06_topsis import DEFAULT_WEIGHTS as BAI06_WEIGHTS
from src.bai06_topsis import IS_BENEFIT, sensitivity_ai_weight as bai06_sensitivity, topsis
from src.bai07_pareto import run_nsga2
from src.bai08_dynamic import compare_strategies, optimize_dynamic
from src.bai09_labor_ai import labor_dataframe, manufacturing_training_threshold, solve_bai09, stress_test_risk
from src.bai10_stochastic import (
    compute_vss_evpi,
    robust_minimax_regret,
    scenario_table,
    solve_deterministic_scenario,
    solve_expected_value,
    solve_stochastic_pulp,
)
from src.bai11_q_learning import ACTION_ALLOCATIONS, ACTION_NAMES, SAMPLE_STATES, state_to_tuple, train_q_learning
from src.data_loader import load_macro, load_regions, load_sectors
from src.scenario_engine import allocation_long, recommendation_text, run_all_scenarios


OUT_DIR = ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, name: str) -> None:
    """Save a dataframe as CSV and Markdown in reports/figures."""
    csv_path = OUT_DIR / f"{name}.csv"
    md_path = OUT_DIR / f"{name}.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")


def save_series_table(rows: list[dict[str, object]], name: str) -> pd.DataFrame:
    """Save a list of dictionaries as a compact table."""
    df = pd.DataFrame(rows)
    save_table(df, name)
    return df


def clean_figure(ax, title: str) -> None:
    """Apply a simple report-friendly Matplotlib style."""
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_bar(df: pd.DataFrame, x: str, y: str, title: str, filename: str, horizontal: bool = False) -> None:
    """Save a bar chart from a dataframe."""
    fig, ax = plt.subplots(figsize=(10, 6))
    if horizontal:
        ax.barh(df[x].astype(str), df[y].astype(float), color="#7C3AED")
        ax.set_xlabel(y)
        ax.set_ylabel(x)
    else:
        ax.bar(df[x].astype(str), df[y].astype(float), color="#7C3AED")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.tick_params(axis="x", rotation=30)
    clean_figure(ax, title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180)
    plt.close(fig)


def save_line(df: pd.DataFrame, x: str, ys: list[str], title: str, filename: str) -> None:
    """Save a multi-line chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for y in ys:
        ax.plot(df[x], df[y], marker="o", linewidth=2, label=y)
    ax.set_xlabel(x)
    ax.legend()
    clean_figure(ax, title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180)
    plt.close(fig)


def save_heatmap(matrix: pd.DataFrame, title: str, filename: str) -> None:
    """Save a simple heatmap from a numeric dataframe."""
    fig, ax = plt.subplots(figsize=(10, 6))
    values = matrix.astype(float).to_numpy()
    image = ax.imshow(values, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180)
    plt.close(fig)


def run_all() -> None:
    """Run all main AIDEOM-VN modules and export report artifacts."""
    macro = load_macro()
    sectors = load_sectors()
    regions = load_regions()
    save_table(macro, "00_input_macro")
    save_table(sectors, "00_input_sectors")
    save_table(regions, "00_input_regions")

    summary_rows: list[dict[str, object]] = []

    # Bai 1
    bai01 = run_bai01(macro, BAI01_PARAMS)
    save_table(bai01["result_df"], "01_bai01_cobb_douglas_fit")
    save_table(bai01["growth_df"], "01_bai01_growth_decomposition")
    save_table(bai01["contribution_df"], "01_bai01_average_contribution")
    save_table(bai01["forecast_2030_df"], "01_bai01_forecast_2030")
    save_line(bai01["result_df"], "year", ["Y_actual", "Y_hat"], "Bài 1 - GDP thực tế và mô phỏng", "01_bai01_gdp_fit.png")
    save_line(bai01["forecast_2030_df"], "year", ["Y_forecast"], "Bài 1 - Forecast GDP đến 2030", "01_bai01_forecast_2030.png")
    summary_rows.append({"module": "Bài 1", "main_result": f"MAPE={bai01['mape']:.4f}", "status": "completed"})

    # Bai 2
    bai02 = solve_bai02_scipy(B=100, min_human=20)
    bai02_sens = sensitivity_budget([100, 120, 140])
    bai02_human = scenario_human_priority(B=100, min_human=30)
    save_table(bai02["allocation_df"], "02_bai02_allocation")
    save_table(bai02_sens, "02_bai02_budget_sensitivity")
    save_table(bai02_human["allocation_df"], "02_bai02_human_priority")
    save_bar(bai02["allocation_df"], "category", "allocation", "Bài 2 - Phân bổ ngân sách tối ưu", "02_bai02_allocation.png")
    save_line(bai02_sens, "budget", ["objective"], "Bài 2 - Độ nhạy ngân sách", "02_bai02_budget_sensitivity.png")
    summary_rows.append({"module": "Bài 2", "main_result": f"objective={bai02['objective']}", "status": bai02["status"]})

    # Bai 3
    bai03 = compute_priority(sectors, BAI03_WEIGHTS, risk_mode="inverted")
    bai03_sens = bai03_sensitivity(sectors)
    bai03_compare = compare_policy_weights(sectors)["top3_comparison"]
    save_table(bai03["normalized_df"], "03_bai03_normalized_matrix")
    save_table(bai03["ranking_df"], "03_bai03_priority_ranking")
    save_table(bai03_sens, "03_bai03_ai_sensitivity")
    save_table(bai03_compare, "03_bai03_policy_top3_comparison")
    save_bar(
        bai03["ranking_df"].sort_values("priority_score"),
        "sector_name",
        "priority_score",
        "Bài 3 - Priority score theo ngành",
        "03_bai03_priority_ranking.png",
        horizontal=True,
    )
    summary_rows.append({"module": "Bài 3", "main_result": str(bai03["ranking_df"].iloc[0]["sector_name"]), "status": "completed"})

    # Bai 4
    bai04 = solve_bai04_pulp(budget=50000, fairness=True, lambda_=0.7)
    bai04_compare = compare_fairness(budget=50000, lambda_=0.7)
    save_table(bai04["allocation_matrix"].reset_index(names="region"), "04_bai04_allocation_matrix")
    save_table(bai04["long_df"], "04_bai04_long_allocation")
    save_table(bai04["region_totals"], "04_bai04_region_totals")
    save_table(bai04["item_totals"], "04_bai04_item_totals")
    save_table(bai04_compare["comparison_df"], "04_bai04_fairness_comparison")
    save_heatmap(bai04["allocation_matrix"], "Bài 4 - Ma trận phân bổ vùng-hạng mục", "04_bai04_allocation_heatmap.png")
    save_bar(bai04["region_totals"], "region_name", "region_total", "Bài 4 - Tổng ngân sách theo vùng", "04_bai04_region_totals.png", horizontal=True)
    summary_rows.append({"module": "Bài 4", "main_result": f"objective={bai04['objective']}", "status": bai04["status"]})

    # Bai 5
    projects = project_dataframe()
    bai05 = solve_bai05(total_budget=80000, early_budget=40000)
    bai05_100 = solve_budget_scenario(total_budget=100000)
    bai05_risk = solve_with_risk_probabilities()
    save_table(projects, "05_bai05_project_dataset")
    save_table(bai05["selected_df"], "05_bai05_selected_projects")
    save_table(bai05["all_projects_df"], "05_bai05_all_projects_solution")
    save_series_table(
        [
            {"scenario": "budget_80000", "objective": bai05["objective"], "total_cost": bai05["total_cost"], "selected_count": len(bai05["selected_df"])},
            {"scenario": "budget_100000", "objective": bai05_100["objective"], "total_cost": bai05_100["total_cost"], "selected_count": len(bai05_100["selected_df"])},
            {"scenario": "expected_value", "objective": bai05_risk["objective"], "total_cost": bai05_risk["total_cost"], "selected_count": len(bai05_risk["selected_df"])},
        ],
        "05_bai05_budget_comparison",
    )
    save_bar(bai05["all_projects_df"].sort_values("benefit_cost_ratio"), "project_id", "benefit_cost_ratio", "Bài 5 - Benefit/cost theo dự án", "05_bai05_benefit_cost.png", horizontal=True)
    summary_rows.append({"module": "Bài 5", "main_result": f"selected={len(bai05['selected_df'])}", "status": "feasible" if bai05["feasibility"] else "infeasible"})

    # Bai 6
    bai06 = topsis(regions, BAI06_WEIGHTS, IS_BENEFIT)
    bai06_sens = bai06_sensitivity(regions, np.arange(0.05, 0.45, 0.05))
    save_table(bai06["ranking_expert"], "06_bai06_topsis_expert")
    save_table(bai06["ranking_entropy"], "06_bai06_topsis_entropy")
    save_table(bai06["weights_df"], "06_bai06_weights")
    save_table(bai06_sens, "06_bai06_ai_sensitivity")
    save_bar(
        bai06["ranking_expert"].sort_values("topsis_score"),
        "region_name",
        "topsis_score",
        "Bài 6 - TOPSIS score expert",
        "06_bai06_topsis_score.png",
        horizontal=True,
    )
    summary_rows.append({"module": "Bài 6", "main_result": str(bai06["ranking_expert"].iloc[0]["region_name"]), "status": "completed"})

    # Bai 7
    bai07 = run_nsga2(pop_size=80, n_gen=80, seed=42, fairness=True, lambda_=0.7)
    save_table(bai07["pareto_df"], "07_bai07_pareto_set")
    save_table(bai07["compromise_solution"], "07_bai07_compromise_solution")
    save_table(bai07["summary_df"], "07_bai07_growth_vs_compromise")
    if not bai07["pareto_df"].empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        pareto = bai07["pareto_df"]
        scatter = ax.scatter(pareto["gdp_gain"], pareto["emission"], c=pareto["net_cyber_risk"], cmap="viridis", s=45)
        ax.set_xlabel("gdp_gain")
        ax.set_ylabel("emission")
        ax.set_title("Bài 7 - Pareto: GDP gain và emission", fontsize=12, fontweight="bold")
        fig.colorbar(scatter, ax=ax, label="net_cyber_risk")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "07_bai07_pareto_scatter.png", dpi=180)
        plt.close(fig)
    summary_rows.append({"module": "Bài 7", "main_result": f"pareto={len(bai07['pareto_df'])}", "status": bai07["method"]})

    # Bai 8
    bai08 = optimize_dynamic(T=10, rho=0.97, investment_rate=0.28, maxiter=180)
    bai08_compare = compare_strategies(rho=0.97, investment_rate=0.28, T=10)
    save_table(bai08["trajectory_df"], "08_bai08_optimized_trajectory")
    save_table(bai08["policy_df"], "08_bai08_optimized_policy")
    save_table(bai08_compare["summary_df"], "08_bai08_strategy_summary")
    save_line(bai08["trajectory_df"], "year", ["K", "D", "AI", "H", "Y", "C"], "Bài 8 - Quỹ đạo trạng thái tối ưu", "08_bai08_trajectory.png")
    save_bar(bai08_compare["summary_df"], "strategy", "welfare", "Bài 8 - Welfare theo chiến lược", "08_bai08_strategy_welfare.png")
    summary_rows.append({"module": "Bài 8", "main_result": f"welfare={bai08['welfare']:.4f}", "status": bai08["status"]})

    # Bai 9
    labor = labor_dataframe()
    bai09 = solve_bai09(budget=30000)
    bai09_threshold = manufacturing_training_threshold(5000)
    bai09_stress = stress_test_risk(1.0)
    save_table(labor, "09_bai09_labor_dataset")
    save_table(bai09["allocation_df"], "09_bai09_allocation_jobs")
    save_table(bai09_threshold, "09_bai09_manufacturing_training_threshold")
    save_table(bai09_stress["allocation_df"], "09_bai09_risk_stress_test")
    save_bar(bai09["allocation_df"].sort_values("NetJob"), "sector", "NetJob", "Bài 9 - NetJob theo ngành", "09_bai09_netjob.png", horizontal=True)
    summary_rows.append({"module": "Bài 9", "main_result": f"objective={bai09['objective']}", "status": bai09["status"]})

    # Bai 10
    bai10 = solve_stochastic_pulp()
    deterministic_rows = []
    for scenario in ["s1", "s2", "s3", "s4"]:
        result = solve_deterministic_scenario(scenario)
        deterministic_rows.append({"scenario": scenario, "objective": result["objective"], "status": result["status"]})
    deterministic_df = pd.DataFrame(deterministic_rows)
    expected = solve_expected_value()
    vss_evpi = compute_vss_evpi()
    robust = robust_minimax_regret()
    save_table(scenario_table(), "10_bai10_scenario_table")
    save_table(bai10["first_stage_df"], "10_bai10_first_stage")
    save_table(bai10["second_stage_df"], "10_bai10_second_stage")
    save_table(deterministic_df, "10_bai10_deterministic_comparison")
    save_table(vss_evpi, "10_bai10_vss_evpi")
    save_table(expected["first_stage_df"], "10_bai10_expected_value_first_stage")
    save_table(robust["first_stage_df"], "10_bai10_robust_first_stage")
    save_bar(bai10["first_stage_df"], "item_name", "allocation", "Bài 10 - First-stage allocation", "10_bai10_first_stage.png")
    save_bar(deterministic_df, "scenario", "objective", "Bài 10 - Deterministic objective theo kịch bản", "10_bai10_deterministic.png")
    summary_rows.append({"module": "Bài 10", "main_result": f"objective={bai10['objective']}", "status": bai10["status"]})

    # Bai 11
    bai11 = train_q_learning(n_episodes=3000, alpha=0.1, gamma=0.95, seed=42)
    rewards_df = pd.DataFrame(
        {
            "episode": np.arange(1, len(bai11["rewards"]) + 1),
            "reward": bai11["rewards"],
            "smoothed_reward": bai11["smoothed_rewards"],
        }
    )
    sample_rows = []
    q = bai11["Q"]
    for label, state_labels in SAMPLE_STATES.items():
        state = state_to_tuple(state_labels)
        action = int(q[state].argmax())
        sample_rows.append(
            {
                "sample_state": label,
                "state_labels": " | ".join(state_labels),
                "recommended_action": action,
                "action_name": ACTION_NAMES[action],
                "allocation": ACTION_ALLOCATIONS[action],
                "q_value": float(q[state + (action,)]),
            }
        )
    actions_df = pd.DataFrame(
        [{"action": action, "action_name": ACTION_NAMES[action], "allocation": ACTION_ALLOCATIONS[action]} for action in ACTION_NAMES]
    )
    save_table(actions_df, "11_bai11_action_mapping")
    save_table(rewards_df, "11_bai11_learning_curve")
    save_table(pd.DataFrame(sample_rows), "11_bai11_sample_policy")
    save_table(bai11["policy_df"], "11_bai11_full_policy")
    save_table(bai11["comparison_df"], "11_bai11_policy_comparison")
    save_line(rewards_df, "episode", ["reward", "smoothed_reward"], "Bài 11 - Learning curve", "11_bai11_learning_curve.png")
    save_bar(bai11["comparison_df"].sort_values("mean_reward"), "policy_type", "mean_reward", "Bài 11 - Mean reward theo policy", "11_bai11_policy_comparison.png", horizontal=True)
    summary_rows.append({"module": "Bài 11", "main_result": f"last100={np.mean(bai11['rewards'][-100:]):.4f}", "status": "completed"})

    # Bai 12
    bai12 = run_all_scenarios(budget=50000)
    bai12_alloc = allocation_long(bai12)
    save_table(bai12, "12_bai12_aideom_kpi")
    save_table(bai12_alloc, "12_bai12_allocation_long")
    save_series_table([{"recommendation": text} for text in recommendation_text(bai12)], "12_bai12_policy_recommendations")
    save_bar(bai12.sort_values("Overall_score"), "scenario_name", "Overall_score", "Bài 12 - Overall_score theo kịch bản", "12_bai12_overall_score.png", horizontal=True)
    alloc_matrix = bai12.set_index("scenario")[["K_share", "D_share", "AI_share", "H_share"]] * 100.0
    save_heatmap(alloc_matrix, "Bài 12 - Heatmap phân bổ kịch bản", "12_bai12_allocation_heatmap.png")
    risk_long = bai12.melt(
        id_vars=["scenario"],
        value_vars=["Inequality_risk", "Cyber_risk", "Emission_risk"],
        var_name="risk_type",
        value_name="risk_value",
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    for risk_type, group in risk_long.groupby("risk_type"):
        ax.plot(group["scenario"], group["risk_value"], marker="o", linewidth=2, label=risk_type)
    ax.legend()
    ax.set_ylabel("risk_value")
    clean_figure(ax, "Bài 12 - Rủi ro theo kịch bản")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "12_bai12_risk_chart.png", dpi=180)
    plt.close(fig)
    summary_rows.append({"module": "Bài 12", "main_result": str(bai12.sort_values("Overall_score", ascending=False).iloc[0]["scenario_name"]), "status": "completed"})

    summary_df = pd.DataFrame(summary_rows)
    save_table(summary_df, "00_model_run_summary")

    manifest = OUT_DIR / "00_artifact_manifest.md"
    artifact_files = sorted(path.name for path in OUT_DIR.iterdir() if path.is_file() and not path.name.startswith("."))
    manifest.write_text(
        "# AIDEOM-VN report artifact manifest\n\n"
        "Các artifact này được sinh từ `scripts/export_report_artifacts.py` bằng logic hiện có trong `src/`.\n\n"
        "## Files\n\n"
        + "\n".join(f"- `{name}`" for name in artifact_files)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    run_all()
    print(f"Exported report artifacts to {OUT_DIR}")
