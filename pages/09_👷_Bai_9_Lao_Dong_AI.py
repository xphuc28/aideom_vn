from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bai09_labor_ai import (
    MODULE_TITLE,
    labor_dataframe,
    manufacturing_training_threshold,
    module_status,
    solve_bai09,
    stress_test_risk,
)
from src.ui import apply_dashboard_style, policy_box, render_page_badges, render_sidebar
from src.visualization import download_dataframe_button, render_kpi_cards


st.set_page_config(page_title=MODULE_TITLE, page_icon="👷", layout="wide")
apply_dashboard_style()
render_sidebar(MODULE_TITLE, "Trung bình", "LP lao động - AI")


@st.cache_data(show_spinner=False)
def get_labor_data():
    return labor_dataframe()


@st.cache_data(show_spinner=False)
def solve_cached(budget: float, risk_multiplier: float, cap_displaced: bool):
    return solve_bai09(
        budget=budget,
        risk_multiplier=risk_multiplier,
        max_displaced_share=0.05 if cap_displaced else None,
    )


def policy_interpretation(result: dict[str, object]) -> list[str]:
    if result["status"] != "optimal":
        return ["Mô hình chưa có nghiệm tối ưu; cần tăng ngân sách hoặc nới ràng buộc displaced."]

    df = result["allocation_df"]
    retraining_focus = df.sort_values("x_H", ascending=False).iloc[0]
    displaced_focus = df.sort_values("DisplacedJob", ascending=False).iloc[0]
    net_focus = df.sort_values("NetJob", ascending=False).iloc[0]

    return [
        f"Ngành cần đào tạo lại nhiều nhất là {retraining_focus['sector']} với x_H={retraining_focus['x_H']:,.1f}.",
        f"Ngành có displaced jobs lớn nhất là {displaced_focus['sector']} với {displaced_focus['DisplacedJob']:,.1f}.",
        f"Đóng góp NetJob cao nhất đến từ {net_focus['sector']} với {net_focus['NetJob']:,.1f}.",
        "Chính sách nên ghép đầu tư AI với ngân sách đào tạo lại bắt buộc, vì ràng buộc displaced <= retraining capacity là hàng rào an toàn xã hội.",
    ]


labor_df = get_labor_data()

st.title(MODULE_TITLE)
render_page_badges("Trung bình", "LP lao động - AI")
st.info(module_status())

st.header("🧭 1. Bối cảnh")
st.write(
    "Bài 9 đánh giá tác động của AI tới thị trường lao động Việt Nam theo 8 ngành. "
    "Mô hình lựa chọn phân bổ ngân sách vào ứng dụng AI và đào tạo kỹ năng để tối đa hóa việc làm ròng, "
    "đồng thời kiểm soát lao động bị thay thế và năng lực đào tạo lại."
)

st.header("🧮 2. Mô hình toán học")
st.latex(r"NewJob_i=a1_i xAI_i,\quad UpgradeJob_i=b1_i xH_i,\quad DisplacedJob_i=c1_i risk_i xAI_i")
st.latex(r"\max \sum_i NetJob_i = \sum_i (NewJob_i+UpgradeJob_i-DisplacedJob_i)")
st.latex(
    r"""
    \begin{aligned}
    \sum_i(xAI_i+xH_i)&\le 30000\\
    NetJob_i&\ge 0\\
    DisplacedJob_i&\le RetrainingCapacity_i=d1_i xH_i\\
    xAI_i,xH_i&\ge 0
    \end{aligned}
    """
)

st.header("🧾 3. Dữ liệu/tham số")
param_cols = st.columns(3)
budget = param_cols[0].number_input("Budget", min_value=5000.0, max_value=100000.0, value=30000.0, step=2500.0)
risk_multiplier = param_cols[1].slider("Risk multiplier", 0.50, 2.00, 1.00, 0.05)
cap_displaced = param_cols[2].toggle("Displaced <= 5% labor", value=False)

st.subheader("Dữ liệu 8 ngành")
st.dataframe(labor_df, use_container_width=True)
download_dataframe_button(labor_df, "bai_9_labor_dataset.csv")

threshold_df = manufacturing_training_threshold(5000)
with st.expander("Ngưỡng đào tạo lại ngành chế biến chế tạo khi x_AI=5000"):
    st.dataframe(threshold_df, use_container_width=True)
    download_dataframe_button(threshold_df, "bai_9_manufacturing_threshold.csv")

st.header("📊 4. Kết quả")
result = solve_cached(budget, risk_multiplier, cap_displaced)
allocation_df = result["allocation_df"]

render_kpi_cards(
    {
        "Status": result["status"],
        "NetJob objective": "N/A" if result["objective"] is None else f"{result['objective']:,.1f}",
        "Total allocation": f"{allocation_df['total_allocation'].sum():,.1f}",
        "Risk multiplier": f"{risk_multiplier:.2f}",
    }
)
st.caption(result["note"])

allocation_cols = ["sector", "x_AI", "x_H", "total_allocation"]
job_cols = ["sector", "NewJob", "UpgradeJob", "DisplacedJob", "RetrainingCapacity", "NetJob"]

tab_alloc, tab_jobs, tab_stress = st.tabs(["Phân bổ", "Việc làm", "Stress test"])
with tab_alloc:
    st.dataframe(allocation_df[allocation_cols], use_container_width=True)
    download_dataframe_button(allocation_df[allocation_cols], "bai_9_allocation.csv")
with tab_jobs:
    st.dataframe(allocation_df[job_cols], use_container_width=True)
    download_dataframe_button(allocation_df[job_cols], "bai_9_jobs_result.csv")
with tab_stress:
    stress_rows = []
    for multiplier in [0.8, 1.0, 1.2, 1.5]:
        stress = stress_test_risk(
            multiplier,
            budget=budget,
            max_displaced_share=0.05 if cap_displaced else None,
        )
        stress_rows.append(
            {
                "risk_multiplier": multiplier,
                "status": stress["status"],
                "objective": stress["objective"],
                "total_displaced": stress["allocation_df"]["DisplacedJob"].sum(),
                "total_retraining_capacity": stress["allocation_df"]["RetrainingCapacity"].sum(),
            }
        )
    stress_df = pd.DataFrame(stress_rows)
    st.dataframe(stress_df, use_container_width=True)
    download_dataframe_button(stress_df, "bai_9_risk_stress_test.csv")

chart_cols = st.columns(2)
with chart_cols[0]:
    fig_net = px.bar(
        allocation_df.sort_values("NetJob", ascending=True),
        x="NetJob",
        y="sector",
        orientation="h",
        title="NetJob theo ngành",
        labels={"NetJob": "Việc làm ròng", "sector": "Ngành"},
    )
    st.plotly_chart(fig_net, use_container_width=True)

with chart_cols[1]:
    fig_h = px.bar(
        allocation_df.sort_values("x_H", ascending=True),
        x="x_H",
        y="sector",
        orientation="h",
        title="Ngân sách đào tạo lại x_H theo ngành",
        labels={"x_H": "x_H", "sector": "Ngành"},
    )
    st.plotly_chart(fig_h, use_container_width=True)

st.header("🏛️ 5. Diễn giải chính sách")
policy_box(policy_interpretation(result), kind="success")
