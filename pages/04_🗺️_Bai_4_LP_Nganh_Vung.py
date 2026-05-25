from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bai04_region_lp import (
    BETA,
    D0,
    GAMMA,
    H_MIN,
    ITEMS,
    ITEM_NAMES,
    MODULE_TITLE,
    REGIONS,
    REGION_MAX,
    REGION_MIN,
    REGION_NAMES,
    compare_fairness,
    module_status,
    solve_bai04_cvxpy,
    solve_bai04_pulp,
)
from src.data_loader import load_regions
from src.ui import apply_dashboard_style, policy_box, render_page_badges, render_sidebar
from src.visualization import download_dataframe_button, render_kpi_cards


st.set_page_config(page_title=MODULE_TITLE, page_icon="🗺️", layout="wide")
apply_dashboard_style()
render_sidebar(MODULE_TITLE, "Khó", "LP ngành-vùng")


@st.cache_data(show_spinner=False)
def get_data():
    return load_regions()


@st.cache_data(show_spinner=False)
def solve_cached(budget: float, fairness: bool, lambda_: float, solver: str):
    if solver == "CVXPY":
        return solve_bai04_cvxpy(budget=budget, fairness=fairness, lambda_=lambda_)
    return solve_bai04_pulp(budget=budget, fairness=fairness, lambda_=lambda_)


@st.cache_data(show_spinner=False)
def compare_cached(budget: float, lambda_: float):
    return compare_fairness(budget=budget, lambda_=lambda_)


def beta_table() -> pd.DataFrame:
    return pd.DataFrame(BETA).T.loc[REGIONS, ITEMS]


def policy_interpretation(result: dict[str, object], comparison: dict[str, object]) -> list[str]:
    region_totals = result["region_totals"].sort_values("region_total", ascending=False)
    item_totals = result["item_totals"].sort_values("item_total", ascending=False)
    top_region = region_totals.iloc[0]
    top_item = item_totals.iloc[0]
    fairness_cost = comparison.get("fairness_cost")
    cost_text = "chưa tính được" if fairness_cost is None else f"{fairness_cost:,.1f}"

    return [
        f"Vùng nhận ngân sách lớn nhất là {top_region['region_name']} với {top_region['region_total']:,.1f} tỷ VND.",
        f"Hạng mục được ưu tiên nhất là {top_item['item_name']} với {top_item['item_total']:,.1f} tỷ VND.",
        f"Chi phí công bằng Z_no_fairness - Z_fairness là {cost_text}; đây là phần mục tiêu phải đánh đổi để thu hẹp khoảng cách số.",
        "Nếu lambda hiệu dụng nhỏ hơn lambda nhập vào, bộ ràng buộc gốc quá chặt so với trần 12.000 tỷ/vùng và D0 hiện tại.",
    ]


regions_df = get_data()

st.title(MODULE_TITLE)
render_page_badges("Khó", "LP ngành-vùng")
st.info(module_status())

st.header("🧭 1. Bối cảnh")
st.write(
    "Bài 4 phân bổ ngân sách số theo 6 vùng kinh tế và 4 hạng mục đầu tư: hạ tầng số, "
    "chuyển đổi số, AI và dữ liệu, nhân lực số. Mô hình cân bằng giữa hiệu quả biên "
    "theo vùng-hạng mục và ràng buộc công bằng về chỉ số số hóa."
)

st.header("🧮 2. Mô hình toán học")
st.latex(r"\max \sum_{r \in R}\sum_{j \in J}\beta_{rj}x_{rj}")
st.latex(
    r"""
    \begin{aligned}
    \sum_j x_{rj} &\ge 5000,\quad \sum_j x_{rj} \le 12000\\
    \sum_r x_{rH} &\ge 12000,\quad \sum_{r,j}x_{rj} \le Budget\\
    D0_r + \gamma x_{rD} &\le M\\
    D0_r + \gamma x_{rD} &\ge \lambda M
    \end{aligned}
    """
)
st.write(f"`gamma = {GAMMA}`, sàn vùng = `{REGION_MIN:,.0f}`, trần vùng = `{REGION_MAX:,.0f}`, tổng H tối thiểu = `{H_MIN:,.0f}` tỷ VND.")

st.header("🧾 3. Dữ liệu/tham số")
param_cols = st.columns(3)
budget = param_cols[0].number_input("Budget tổng, tỷ VND", min_value=30000.0, max_value=72000.0, value=50000.0, step=1000.0)
lambda_ = param_cols[1].slider("Lambda công bằng", 0.10, 0.90, 0.70, 0.01)
fairness = param_cols[2].toggle("Bật fairness", value=True)
solver = st.radio("Solver", ["PuLP", "CVXPY"], horizontal=True, help="Nếu thư viện chưa cài, hệ thống tự dùng SciPy fallback.")

tab_beta, tab_d0, tab_regions = st.tabs(["Beta", "D0", "Dữ liệu vùng"])
with tab_beta:
    beta_df = beta_table()
    st.dataframe(beta_df, use_container_width=True)
    download_dataframe_button(beta_df.reset_index(names="region"), "bai_4_beta.csv")
with tab_d0:
    d0_df = pd.DataFrame({"region": REGIONS, "region_name": [REGION_NAMES[r] for r in REGIONS], "D0": [D0[r] for r in REGIONS]})
    st.dataframe(d0_df, use_container_width=True)
    download_dataframe_button(d0_df, "bai_4_d0.csv")
with tab_regions:
    st.dataframe(regions_df, use_container_width=True)
    download_dataframe_button(regions_df, "bai_4_regions_context.csv")

st.header("📊 4. Kết quả")
result = solve_cached(budget, fairness, lambda_, solver)
comparison = compare_cached(budget, lambda_)

objective = result["objective"]
render_kpi_cards(
    {
        "Status": result["status"],
        "Z tối ưu": "N/A" if objective is None else f"{objective:,.1f}",
        "Fairness cost": "N/A" if comparison["fairness_cost"] is None else f"{comparison['fairness_cost']:,.1f}",
        "Lambda hiệu dụng": f"{result.get('effective_lambda', lambda_):.3f}",
    }
)
st.caption(result["note"])

allocation_matrix = result["allocation_matrix"]
long_df = result["long_df"]
region_totals = result["region_totals"]
item_totals = result["item_totals"]

st.subheader("Ma trận phân bổ 6x4")
st.dataframe(allocation_matrix, use_container_width=True)
download_dataframe_button(allocation_matrix.reset_index(names="region"), "bai_4_allocation_matrix.csv")

chart_cols = st.columns(2)
with chart_cols[0]:
    fig_heatmap = px.imshow(
        allocation_matrix,
        text_auto=".0f",
        aspect="auto",
        color_continuous_scale="YlGnBu",
        title="Heatmap phân bổ vùng-hạng mục",
        labels={"x": "Hạng mục", "y": "Vùng", "color": "Tỷ VND"},
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

with chart_cols[1]:
    fig_region = px.bar(
        region_totals,
        x="region",
        y="region_total",
        color="region",
        title="Tổng phân bổ theo vùng",
        labels={"region": "Vùng", "region_total": "Tỷ VND"},
    )
    fig_region.update_layout(showlegend=False)
    st.plotly_chart(fig_region, use_container_width=True)

chart_cols = st.columns(2)
with chart_cols[0]:
    fig_item = px.bar(
        item_totals,
        x="item",
        y="item_total",
        color="item",
        title="Tổng phân bổ theo hạng mục",
        labels={"item": "Hạng mục", "item_total": "Tỷ VND"},
    )
    fig_item.update_layout(showlegend=False)
    st.plotly_chart(fig_item, use_container_width=True)

with chart_cols[1]:
    comparison_df = comparison["comparison_df"]
    fig_compare = px.bar(
        comparison_df,
        x="scenario",
        y="objective",
        color="scenario",
        title="So sánh có/không fairness",
        labels={"scenario": "Kịch bản", "objective": "Z"},
    )
    fig_compare.update_layout(showlegend=False)
    st.plotly_chart(fig_compare, use_container_width=True)

tab_long, tab_region_total, tab_item_total, tab_compare = st.tabs(
    ["Long table", "Tổng theo vùng", "Tổng theo hạng mục", "Fairness comparison"]
)
with tab_long:
    st.dataframe(long_df, use_container_width=True)
    download_dataframe_button(long_df, "bai_4_long_allocation.csv")
with tab_region_total:
    st.dataframe(region_totals, use_container_width=True)
    download_dataframe_button(region_totals, "bai_4_region_totals.csv")
with tab_item_total:
    st.dataframe(item_totals, use_container_width=True)
    download_dataframe_button(item_totals, "bai_4_item_totals.csv")
with tab_compare:
    st.dataframe(comparison_df, use_container_width=True)
    download_dataframe_button(comparison_df, "bai_4_fairness_comparison.csv")

st.header("🏛️ 5. Diễn giải chính sách")
policy_box(policy_interpretation(result, comparison), kind="success")
