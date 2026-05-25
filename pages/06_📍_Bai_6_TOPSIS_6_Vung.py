from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bai06_topsis import (
    CRITERIA,
    DEFAULT_WEIGHTS,
    IS_BENEFIT,
    MODULE_TITLE,
    module_status,
    sensitivity_ai_weight,
    topsis,
)
from src.data_loader import load_regions
from src.ui import apply_dashboard_style, policy_box, render_page_badges, render_sidebar
from src.visualization import download_dataframe_button, render_kpi_cards


st.set_page_config(page_title=MODULE_TITLE, page_icon="📍", layout="wide")
apply_dashboard_style()
render_sidebar(MODULE_TITLE, "Trung bình", "TOPSIS")


@st.cache_data(show_spinner=False)
def get_data():
    return load_regions()


@st.cache_data(show_spinner=False)
def run_topsis_cached(data, weights):
    return topsis(data, weights, IS_BENEFIT)


@st.cache_data(show_spinner=False)
def run_sensitivity_cached(data):
    return sensitivity_ai_weight(data, np.arange(0.05, 0.45, 0.05))


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return DEFAULT_WEIGHTS.copy()
    return {key: value / total for key, value in weights.items()}


def radar_top3(ranking_df: pd.DataFrame) -> go.Figure:
    top3 = ranking_df.head(3)
    radar_criteria = [
        "grdp_per_capita",
        "fdi_registered",
        "digital_index",
        "ai_readiness",
        "trained_labor",
        "rd_intensity",
        "internet_penetration",
    ]
    values = ranking_df[radar_criteria].astype(float)
    normalized = (values - values.min()) / (values.max() - values.min()).replace(0, 1)

    fig = go.Figure()
    for _, row in top3.iterrows():
        region_values = normalized.loc[ranking_df["region_name"] == row["region_name"], radar_criteria].iloc[0].to_list()
        fig.add_trace(
            go.Scatterpolar(
                r=region_values + [region_values[0]],
                theta=radar_criteria + [radar_criteria[0]],
                fill="toself",
                name=row["region_name"],
            )
        )
    fig.update_layout(
        title="Radar top 3 vùng theo TOPSIS expert",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def policy_interpretation(expert_ranking: pd.DataFrame, entropy_ranking: pd.DataFrame) -> list[str]:
    top3 = expert_ranking.head(3)["region_name"].to_list()
    entropy_top3 = entropy_ranking.head(3)["region_name"].to_list()
    overlap = set(top3).intersection(entropy_top3)

    return [
        "Ba vùng trung tâm AI đề xuất theo expert weights là: " + ", ".join(top3) + ".",
        f"TOPSIS expert và entropy có {len(overlap)} vùng trùng nhau trong top-3, cho thấy mức ổn định của khuyến nghị.",
        "Vùng top-1 nên đóng vai trò hạt nhân AI; vùng top-2/top-3 phù hợp làm cực tăng trưởng bổ trợ và lan tỏa.",
        "Các vùng điểm thấp không nên bị bỏ lại; chính sách phù hợp hơn là đầu tư nền tảng số, kỹ năng số và kết nối trước khi mở rộng AI chuyên sâu.",
    ]


regions = get_data()

st.title(MODULE_TITLE)
render_page_badges("Trung bình", "TOPSIS")
st.info(module_status())

st.header("🧭 1. Bối cảnh")
st.write(
    "Bài 6 dùng TOPSIS để xếp hạng 6 vùng kinh tế theo năng lực trở thành trung tâm đầu tư AI. "
    "Mô hình kết hợp quy mô kinh tế, FDI, chỉ số số hóa, sẵn sàng AI, lao động đào tạo, R&D, "
    "internet và bất bình đẳng thu nhập."
)

st.header("🧮 2. Mô hình toán học")
st.latex(r"C_i = \frac{D_i^-}{D_i^+ + D_i^-}")
st.write(
    "Bảy tiêu chí đầu là benefit criteria; `gini_coef` là cost criterion. "
    "Điểm càng gần 1 thì vùng càng gần nghiệm lý tưởng dương."
)

st.header("🧾 3. Dữ liệu/tham số")
st.subheader("Dữ liệu vùng")
st.dataframe(regions, use_container_width=True)
download_dataframe_button(regions, "bai_6_regions_input.csv")

st.subheader("Trọng số expert")
weight_cols = st.columns(4)
raw_weights = {
    "grdp_per_capita": weight_cols[0].slider("GRDP/người", 0.0, 1.0, DEFAULT_WEIGHTS["grdp_per_capita"], 0.01),
    "fdi_registered": weight_cols[1].slider("FDI", 0.0, 1.0, DEFAULT_WEIGHTS["fdi_registered"], 0.01),
    "digital_index": weight_cols[2].slider("Digital index", 0.0, 1.0, DEFAULT_WEIGHTS["digital_index"], 0.01),
    "ai_readiness": weight_cols[3].slider("AI readiness", 0.0, 1.0, DEFAULT_WEIGHTS["ai_readiness"], 0.01),
    "trained_labor": weight_cols[0].slider("Lao động đào tạo", 0.0, 1.0, DEFAULT_WEIGHTS["trained_labor"], 0.01),
    "rd_intensity": weight_cols[1].slider("R&D intensity", 0.0, 1.0, DEFAULT_WEIGHTS["rd_intensity"], 0.01),
    "internet_penetration": weight_cols[2].slider("Internet", 0.0, 1.0, DEFAULT_WEIGHTS["internet_penetration"], 0.01),
    "gini": weight_cols[3].slider("Gini cost", 0.0, 1.0, DEFAULT_WEIGHTS["gini"], 0.01),
}
weights = normalize_weights(raw_weights)
normalize_clicked = st.button("Normalize weights")
if normalize_clicked:
    st.success("Đã normalize weights để tổng bằng 1 trong mô hình tính toán.")
st.caption(f"Tổng trọng số nhập: {sum(raw_weights.values()):.2f}; tổng sau normalize: {sum(weights.values()):.2f}.")

st.header("📊 4. Kết quả")
results = run_topsis_cached(regions, weights)
expert_ranking = results["ranking_expert"]
entropy_ranking = results["ranking_entropy"]
weights_df = results["weights_df"]
sensitivity_df = run_sensitivity_cached(regions)

render_kpi_cards(
    {
        "Top expert": expert_ranking.iloc[0]["region_name"],
        "Score top": f"{expert_ranking.iloc[0]['topsis_score']:.3f}",
        "Top entropy": entropy_ranking.iloc[0]["region_name"],
    }
)

tab_expert, tab_entropy, tab_weights, tab_sensitivity = st.tabs(
    ["Ranking expert", "Ranking entropy", "Weights", "Sensitivity AI"]
)
with tab_expert:
    st.dataframe(expert_ranking, use_container_width=True)
    download_dataframe_button(expert_ranking, "bai_6_ranking_expert.csv")
with tab_entropy:
    st.dataframe(entropy_ranking, use_container_width=True)
    download_dataframe_button(entropy_ranking, "bai_6_ranking_entropy.csv")
with tab_weights:
    st.dataframe(weights_df, use_container_width=True)
    download_dataframe_button(weights_df, "bai_6_weights.csv")
with tab_sensitivity:
    st.dataframe(sensitivity_df, use_container_width=True)
    download_dataframe_button(sensitivity_df, "bai_6_ai_weight_sensitivity.csv")

comparison_df = expert_ranking[["region_name", "rank", "topsis_score"]].merge(
    entropy_ranking[["region_name", "rank", "topsis_score"]],
    on="region_name",
    suffixes=("_expert", "_entropy"),
)
st.subheader("So sánh ranking expert vs entropy")
st.dataframe(comparison_df, use_container_width=True)
download_dataframe_button(comparison_df, "bai_6_ranking_comparison.csv")

chart_cols = st.columns(2)
with chart_cols[0]:
    fig_score = px.bar(
        expert_ranking.sort_values("topsis_score", ascending=True),
        x="topsis_score",
        y="region_name",
        orientation="h",
        title="TOPSIS score theo vùng",
        labels={"topsis_score": "TOPSIS score", "region_name": "Vùng"},
    )
    st.plotly_chart(fig_score, use_container_width=True)

with chart_cols[1]:
    heatmap_data = (
        sensitivity_df.pivot(index="region_name", columns="ai_weight", values="is_top_rank")
        .fillna(0)
        .astype(float)
    )
    fig_sens = px.imshow(
        heatmap_data,
        aspect="auto",
        color_continuous_scale=["#F2F4F7", "#2563EB"],
        title="Top-1 khi thay đổi trọng số AI readiness",
        labels={"x": "Trọng số AI", "y": "Vùng", "color": "Top-1"},
    )
    st.plotly_chart(fig_sens, use_container_width=True)

st.plotly_chart(radar_top3(expert_ranking), use_container_width=True)

st.header("🏛️ 5. Diễn giải chính sách")
policy_box(policy_interpretation(expert_ranking, entropy_ranking), kind="success")
