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
from src.assignment_ui import render_assignment_answers
from src.ui import (
    apply_dashboard_style,
    policy_box,
    render_page_badges,
    render_sidebar,
)
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


def assignment_answers(result, scenario_100k, forced_result, risk_result, projects):
    """Answer Bài 5 and disclose ID differences from the supplied PDF."""
    if result["feasibility"]:
        selected_ids = set(result["selected_df"]["project_id"])
        selected_text = ", ".join(result["selected_df"]["project_id"])
        count = len(result["selected_df"])
    else:
        selected_ids = set()
        selected_text = "không có"
        count = 0
    selected_100k = set(scenario_100k["selected_df"]["project_id"])
    added = sorted(selected_100k - selected_ids)
    removed = sorted(selected_ids - selected_100k)
    open_data_row = projects[projects["project_name"].str.contains("dữ liệu mở", case=False)].iloc[0]
    mandatory_row = projects.loc[projects["project_id"] == "P14"].iloc[0]

    programming = [
        {
            "code": "Câu 5.4.1",
            "question": "Giải MIP và báo cáo dự án chọn, tổng chi phí, lợi ích và B/C.",
            "answer": (
                f"Feasible={result['feasibility']}; chọn {count} dự án: {selected_text}. "
                f"Tổng chi phí={result['total_cost']:,.0f}, objective={result['objective']}, "
                f"benefit/cost={result['benefit_cost_ratio']:.3f}."
            ),
            "evidence": "Bảng Danh sách dự án được chọn và KPI đầu trang.",
        },
        {
            "code": "Câu 5.4.2",
            "question": "Nới ngân sách lên 100.000 tỷ.",
            "answer": (
                f"Kịch bản 100k có objective={scenario_100k['objective']}, chi phí={scenario_100k['total_cost']:,.0f}, "
                f"chọn {len(scenario_100k['selected_df'])} dự án. Dự án thêm: {', '.join(added) or 'không có'}; "
                f"dự án bị loại: {', '.join(removed) or 'không có'}."
            ),
            "evidence": "Bảng So sánh ngân sách 80k vs 100k.",
        },
        {
            "code": "Câu 5.4.3",
            "question": "Bắt buộc chọn đồng thời P1 và P2.",
            "answer": (
                f"Kịch bản force P1 và P2 có feasibility={forced_result['feasibility']}. "
                "Do mô hình vẫn giữ y1+y2<=1, yêu cầu chọn cả hai tạo mâu thuẫn logic và làm bài toán không khả thi."
            ),
            "evidence": "Ràng buộc loại trừ y1+y2<=1.",
        },
        {
            "code": "Câu 5.4.4",
            "question": "Tối đa hóa lợi ích kỳ vọng theo xác suất thành công.",
            "answer": (
                f"Nghiệm expected-value có objective={risk_result['objective']}, "
                f"tổng chi phí={risk_result['total_cost']:,.0f}, chọn {len(risk_result['selected_df'])} dự án."
            ),
            "evidence": "Kịch bản Expected benefit trong bảng so sánh.",
        },
    ]
    policy = [
        {
            "code": "Câu 5.5a",
            "question": "Vì sao dự án Open Data có thể bị loại dù B/C cao?",
            "answer": (
                f"Trong source hiện tại, dự án dữ liệu mở là {open_data_row['project_id']} - {open_data_row['project_name']}, "
                f"B/C={open_data_row['benefit_cost_ratio']:.3f} và trạng thái chọn="
                f"{'có' if open_data_row['project_id'] in selected_ids else 'không'}. Nếu bị loại, nguyên nhân là chi phí cơ hội "
                "trong ngân sách tổng/early budget và tương tác với các ràng buộc danh mục, không phải chỉ vì B/C riêng lẻ."
            ),
            "evidence": "project_dataframe và selected_df.",
        },
        {
            "code": "Câu 5.5b",
            "question": "Dự án bắt buộc P14 có làm giảm Z* và có hợp lý không?",
            "answer": (
                f"Source hiện tại gán P14 là “{mandatory_row['project_name']}”, không phải an ninh mạng như câu chữ trong PDF; "
                "SOC an ninh mạng là P7. Vì chưa có counterfactual bỏ y14>=1, dashboard chưa định lượng được mức giảm Z*. "
                "Về chính sách, một dự án nền tảng bắt buộc có thể hợp lý nhưng chi phí cơ hội phải được báo cáo bằng một lần chạy đối chứng."
            ),
            "status": "Phát hiện lệch mã dự án giữa PDF và source; không suy diễn số liệu.",
        },
        {
            "code": "Câu 5.5c",
            "question": "Mô hình hóa lợi ích cộng hưởng giữa hai dự án.",
            "answer": (
                "Thêm biến nhị phân z_ij biểu diễn hai dự án cùng được chọn, với z<=y_i, z<=y_j, "
                "z>=y_i+y_j-1; sau đó cộng synergy_ij*z_ij vào hàm mục tiêu. Đây là mở rộng MIP chuẩn."
            ),
            "evidence": "Giải pháp mô hình hóa, chưa làm thay đổi nghiệm hiện tại.",
        },
    ]
    return programming, policy


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
forced_result = solve_cached(total_budget, early_budget, True, expected_value)

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
programming_answers, discussion_answers = assignment_answers(
    result, scenario_100k, forced_result, risk_result, projects
)
render_assignment_answers(
    programming_answers,
    discussion_answers,
    note=(
        "Lưu ý đối chiếu đề: tên/mã một số dự án trong source hiện tại không trùng hoàn toàn "
        "với bảng dự án trong PDF. Phần trả lời dùng đúng project_dataframe đang chạy và ghi rõ chỗ lệch."
    ),
)
