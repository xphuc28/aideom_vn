from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bai05_mip_projects import (
    MODULE_TITLE,
    module_status,
    project_dataframe,
    solve_bai05,
    solve_budget_scenario,
    solve_with_risk_probabilities,
)
from src.ui import apply_dashboard_style, policy_box, render_page_badges, render_sidebar
from src.visualization import download_dataframe_button, render_kpi_cards


st.set_page_config(page_title=MODULE_TITLE, page_icon="📦", layout="wide")
apply_dashboard_style()
render_sidebar(MODULE_TITLE, "Khó", "Mixed Integer Programming")


@st.cache_data(show_spinner=False)
def get_projects():
    return project_dataframe()


@st.cache_data(show_spinner=False)
def solve_cached(total_budget: float, early_budget: float, force_p1_p2: bool, expected_value: bool):
    return solve_bai05(
        total_budget=total_budget,
        early_budget=early_budget,
        force_p1_p2=force_p1_p2,
        expected_value=expected_value,
    )


@st.cache_data(show_spinner=False)
def scenario_cached(total_budget: float):
    return solve_budget_scenario(total_budget=total_budget)


@st.cache_data(show_spinner=False)
def risk_cached():
    return solve_with_risk_probabilities()


def policy_interpretation(result: dict[str, object], scenario_100k: dict[str, object]) -> list[str]:
    if not result["feasibility"]:
        return [
            "Danh mục hiện tại không khả thi. Cần tăng ngân sách, tăng early budget hoặc bỏ yêu cầu force P1 và P2.",
            "Đặc biệt, P1 và P2 không thể cùng được chọn vì ràng buộc loại trừ `y1 + y2 <= 1`.",
        ]

    selected = result["selected_df"]
    top_project = selected.sort_values("benefit_cost_ratio", ascending=False).iloc[0]
    extra_projects = set(scenario_100k["selected_df"]["project_id"]) - set(selected["project_id"])

    return [
        f"Danh mục tối ưu chọn {len(selected)} dự án với tổng chi phí {result['total_cost']:,.0f} tỷ VND.",
        f"Dự án có benefit/cost cao nhất trong danh mục là {top_project['project_id']} - {top_project['project_name']}.",
        "P14 được chọn bắt buộc, phản ánh ưu tiên nền tảng về kỹ năng số quốc gia.",
        "Khi tăng ngân sách lên 100.000 tỷ VND, các dự án bổ sung tiềm năng là: "
        + (", ".join(sorted(extra_projects)) if extra_projects else "không có dự án mới")
        + ".",
    ]


projects = get_projects()

st.title(MODULE_TITLE)
render_page_badges("Khó", "Mixed Integer Programming")
st.info(module_status())

st.header("🧭 1. Bối cảnh")
st.write(
    "Bài 5 lựa chọn danh mục dự án chuyển đổi số trong điều kiện ngân sách tổng, "
    "ngân sách giai đoạn đầu, ràng buộc phụ thuộc dự án và các yêu cầu chính sách. "
    "Đây là bài toán MIP nhị phân với biến `y_i = 1` nếu dự án được chọn."
)

st.header("🧮 2. Mô hình toán học")
st.latex(r"\max \sum_i B_i y_i")
st.latex(
    r"""
    \begin{aligned}
    \sum_i C_i y_i &\le total\_budget\\
    \sum_i C1_i y_i &\le early\_budget\\
    y_1+y_2 &\le 1,\quad y_8 \le y_{12},\quad y_{13} \le y_{12}\\
    y_4+y_5 &\ge 1,\quad y_{14} \ge 1\\
    7 \le \sum_i y_i &\le 11,\quad y_i \in \{0,1\}
    \end{aligned}
    """
)

st.header("🧾 3. Dữ liệu/tham số")
param_cols = st.columns(4)
total_budget = param_cols[0].number_input("Total budget, tỷ VND", min_value=30000.0, max_value=150000.0, value=80000.0, step=5000.0)
early_budget = param_cols[1].number_input("Early budget năm 1-2, tỷ VND", min_value=10000.0, max_value=90000.0, value=40000.0, step=2500.0)
force_p1_p2 = param_cols[2].checkbox("Force P1 and P2", value=False, help="Bật để kiểm tra xung đột với ràng buộc y1+y2<=1.")
expected_value = param_cols[3].checkbox("Expected benefit with risk probabilities", value=False)

st.subheader("Dữ liệu 15 dự án")
st.dataframe(projects, use_container_width=True)
download_dataframe_button(projects, "bai_5_projects.csv")

st.header("📊 4. Kết quả")
result = solve_cached(total_budget, early_budget, force_p1_p2, expected_value)
scenario_80k = solve_bai05(total_budget=80000, early_budget=40000)
scenario_100k = scenario_cached(100000)
risk_result = risk_cached()

render_kpi_cards(
    {
        "Feasible": "Có" if result["feasibility"] else "Không",
        "Objective": "N/A" if result["objective"] is None else f"{result['objective']:,.0f}",
        "Total cost": f"{result['total_cost']:,.0f}",
        "B/C ratio": f"{result['benefit_cost_ratio']:.2f}",
    }
)
st.caption(result["note"])

if result["feasibility"]:
    selected_df = result["selected_df"]
    all_projects_df = result["all_projects_df"]

    st.subheader("Danh sách dự án được chọn")
    st.dataframe(selected_df, use_container_width=True)
    download_dataframe_button(selected_df, "bai_5_selected_projects.csv")

    fig_ratio = px.bar(
        all_projects_df.sort_values("benefit_cost_ratio", ascending=True),
        x="benefit_cost_ratio",
        y="project_id",
        color="selected_label",
        orientation="h",
        hover_data=["project_name", "cost", "benefit", "success_probability"],
        title="Benefit/cost theo dự án",
        labels={"benefit_cost_ratio": "Benefit/cost", "project_id": "Dự án", "selected_label": "Trạng thái"},
    )
    st.plotly_chart(fig_ratio, use_container_width=True)

    comparison_df = pd.DataFrame(
        [
            {
                "scenario": "Budget 80k",
                "objective": scenario_80k["objective"],
                "total_cost": scenario_80k["total_cost"],
                "selected_count": len(scenario_80k["selected_df"]),
            },
            {
                "scenario": "Budget 100k",
                "objective": scenario_100k["objective"],
                "total_cost": scenario_100k["total_cost"],
                "selected_count": len(scenario_100k["selected_df"]),
            },
            {
                "scenario": "Expected benefit",
                "objective": risk_result["objective"],
                "total_cost": risk_result["total_cost"],
                "selected_count": len(risk_result["selected_df"]),
            },
        ]
    )
    st.subheader("So sánh ngân sách 80k vs 100k")
    st.dataframe(comparison_df, use_container_width=True)
    download_dataframe_button(comparison_df, "bai_5_budget_comparison.csv")

    fig_compare = px.bar(
        comparison_df,
        x="scenario",
        y="objective",
        color="scenario",
        title="So sánh objective theo kịch bản",
        labels={"scenario": "Kịch bản", "objective": "Objective"},
    )
    fig_compare.update_layout(showlegend=False)
    st.plotly_chart(fig_compare, use_container_width=True)

    with st.expander("Toàn bộ dự án kèm trạng thái chọn"):
        st.dataframe(all_projects_df, use_container_width=True)
        download_dataframe_button(all_projects_df, "bai_5_all_projects_solution.csv")
else:
    st.error("Không tìm được danh mục khả thi với các tham số hiện tại.")

st.header("🏛️ 5. Diễn giải chính sách")
policy_box(policy_interpretation(result, scenario_100k), kind="success")
