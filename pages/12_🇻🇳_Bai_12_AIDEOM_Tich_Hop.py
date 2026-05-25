from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_macro, load_regions, load_sectors
from src.scenario_engine import (
    COST_KPIS,
    SCENARIOS,
    allocation_long,
    recommendation_text,
    run_all_scenarios,
)
from src.ui import apply_dashboard_style, policy_box, render_page_badges, render_sidebar
from src.visualization import download_dataframe_button, render_kpi_cards


st.set_page_config(page_title="Bài 12 - AIDEOM-VN tích hợp", page_icon="🇻🇳", layout="wide")
apply_dashboard_style()
render_sidebar("Bài 12 - Dashboard tích hợp", "Trung bình", "Scenario engine")


@st.cache_data(show_spinner=False)
def get_data(budget: float):
    return load_macro(), load_sectors(), load_regions(), run_all_scenarios(budget=budget)


def radar_chart(kpi_df):
    radar_cols = [
        "GDP_gain_norm",
        "Digital_score_norm",
        "AI_score_norm",
        "Human_capital_score_norm",
        "NetJob_norm",
        "Inequality_risk_norm",
        "Cyber_risk_norm",
        "Emission_risk_norm",
    ]
    labels = ["GDP", "Digital", "AI", "Human", "NetJob", "Equity", "Cyber", "Emission"]
    fig = go.Figure()
    for _, row in kpi_df.iterrows():
        values = [row[col] for col in radar_cols]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                fill="toself",
                name=f"{row['scenario']} {row['scenario_name']}",
            )
        )
    fig.update_layout(
        title="Radar normalized KPI theo kịch bản",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


st.title("Bài 12 - Dashboard tích hợp AIDEOM-VN")
render_page_badges("Trung bình", "Scenario engine", "Tổng hợp 5 kịch bản")
st.caption("Tổng hợp kết quả Bài 1-11 thành 5 kịch bản chính sách")

budget = st.sidebar.number_input("Ngân sách tổng hợp", min_value=10000.0, max_value=200000.0, value=50000.0, step=5000.0)
macro, sectors, regions, kpi_df = get_data(budget)
alloc_df = allocation_long(kpi_df)
best = kpi_df.sort_values("Overall_score", ascending=False).iloc[0]

tabs = st.tabs(["📌 Tổng quan", "🧩 5 kịch bản", "🧮 Phân bổ", "👷 Lao động & AI", "🛡️ Rủi ro", "🏛️ Khuyến nghị"])

with tabs[0]:
    st.subheader("Tổng quan")
    st.write(
        "Dashboard tích hợp này dùng approximation minh bạch từ allocation shares để tổng hợp các chiều "
        "tăng trưởng, số hóa, AI, vốn nhân lực, việc làm và rủi ro. Các mô hình chi tiết nằm ở Bài 1-11."
    )
    render_kpi_cards(
        {
            "Kịch bản tốt nhất": f"{best['scenario']} - {best['scenario_name']}",
            "Overall score": f"{best['Overall_score']:.1f}/100",
            "Budget": f"{budget:,.0f}",
            "Số kịch bản": len(kpi_df),
        }
    )

    chart_cols = st.columns(2)
    with chart_cols[0]:
        fig_overall = px.bar(
            kpi_df.sort_values("Overall_score", ascending=True),
            x="Overall_score",
            y="scenario_name",
            color="scenario",
            orientation="h",
            title="Overall_score theo kịch bản",
        )
        st.plotly_chart(fig_overall, use_container_width=True)
    with chart_cols[1]:
        st.plotly_chart(radar_chart(kpi_df), use_container_width=True)

    st.dataframe(kpi_df, use_container_width=True)
    download_dataframe_button(kpi_df, "bai_12_aideom_kpi.csv")

with tabs[1]:
    st.subheader("5 kịch bản chính sách")
    scenario_meta = [
        {
            "scenario": code,
            "scenario_name": spec["name"],
            "description": spec["description"],
            **{f"{item}_share": spec["shares"][item] for item in ["K", "D", "AI", "H"]},
        }
        for code, spec in SCENARIOS.items()
    ]
    st.dataframe(scenario_meta, use_container_width=True)
    grouped_cols = ["GDP_gain", "Digital_score", "AI_score", "Human_capital_score", "NetJob"]
    grouped = kpi_df.melt(id_vars=["scenario", "scenario_name"], value_vars=grouped_cols, var_name="KPI", value_name="value")
    fig_grouped = px.bar(
        grouped,
        x="scenario",
        y="value",
        color="KPI",
        barmode="group",
        title="So sánh KPI lợi ích theo kịch bản",
    )
    st.plotly_chart(fig_grouped, use_container_width=True)

with tabs[2]:
    st.subheader("Phân bổ ngân sách")
    st.dataframe(alloc_df, use_container_width=True)
    download_dataframe_button(alloc_df, "bai_12_allocation_long.csv")
    heatmap = alloc_df.pivot(index="scenario", columns="item_label", values="share")
    fig_heatmap = px.imshow(
        heatmap,
        text_auto=".0%",
        aspect="auto",
        color_continuous_scale="YlGnBu",
        title="Heatmap allocation scenario x item",
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    fig_alloc = px.bar(
        alloc_df,
        x="scenario",
        y="allocation",
        color="item_label",
        title="Phân bổ ngân sách theo kịch bản",
    )
    st.plotly_chart(fig_alloc, use_container_width=True)

with tabs[3]:
    st.subheader("Lao động & AI")
    labor_cols = ["scenario", "scenario_name", "AI_score", "Human_capital_score", "NetJob", "AI_share", "H_share"]
    st.dataframe(kpi_df[labor_cols], use_container_width=True)
    fig_labor = px.scatter(
        kpi_df,
        x="AI_score",
        y="NetJob",
        size="Human_capital_score",
        color="scenario",
        hover_name="scenario_name",
        title="AI score, NetJob và vốn nhân lực",
    )
    st.plotly_chart(fig_labor, use_container_width=True)

with tabs[4]:
    st.subheader("Rủi ro")
    risk_df = kpi_df.melt(
        id_vars=["scenario", "scenario_name"],
        value_vars=COST_KPIS,
        var_name="risk_type",
        value_name="risk_value",
    )
    st.dataframe(risk_df, use_container_width=True)
    fig_risk = px.bar(
        risk_df,
        x="scenario",
        y="risk_value",
        color="risk_type",
        barmode="group",
        title="Inequality, cyber và emission risk",
    )
    st.plotly_chart(fig_risk, use_container_width=True)

with tabs[5]:
    st.subheader("Khuyến nghị chính sách tự động")
    policy_box(recommendation_text(kpi_df), title="Khuyến nghị chính sách", kind="success")

    st.info(
        "AIDEOM-VN là công cụ hỗ trợ ra quyết định định lượng. Kết quả không thay thế quyết định "
        "chính trị-xã hội, tham vấn chuyên gia, phân tích ngân sách thực tế và đánh giá tác động phân phối."
    )

    with st.expander("Dữ liệu nền đã nạp"):
        st.write("Macro")
        st.dataframe(macro, use_container_width=True)
        st.write("Sectors")
        st.dataframe(sectors, use_container_width=True)
        st.write("Regions")
        st.dataframe(regions, use_container_width=True)
