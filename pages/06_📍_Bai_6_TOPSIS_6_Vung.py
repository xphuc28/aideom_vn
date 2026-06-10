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
from src.ui import (
    apply_dashboard_style,
    policy_box,
    render_assignment_answers,
    render_page_badges,
    render_sidebar,
)
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


def assignment_answers(expert_ranking, entropy_ranking, sensitivity_df, comparison_df):
    """Answer the TOPSIS ranking and policy questions in Bài 6."""
    top3 = expert_ranking.head(3)["region_name"].tolist()
    entropy_top3 = entropy_ranking.head(3)["region_name"].tolist()
    comparison = comparison_df.copy()
    comparison["rank_change"] = (comparison["rank_expert"] - comparison["rank_entropy"]).abs()
    largest_change = comparison.sort_values("rank_change", ascending=False).iloc[0]
    top3_sets = (
        sensitivity_df[sensitivity_df["rank"] <= 3]
        .groupby("ai_weight")["region_name"]
        .apply(lambda values: tuple(values))
    )
    stable_top3 = len(set(top3_sets.tolist())) == 1

    programming = [
        {
            "code": "Câu 6.4.1",
            "question": "Tính TOPSIS với trọng số chuyên gia.",
            "answer": (
                f"Vùng dẫn đầu là {expert_ranking.iloc[0]['region_name']} với C*="
                f"{expert_ranking.iloc[0]['topsis_score']:.4f}. Top-3: {', '.join(top3)}."
            ),
            "evidence": "Tab Ranking expert.",
        },
        {
            "code": "Câu 6.4.2",
            "question": "Tính trọng số Entropy và so sánh thứ hạng.",
            "answer": (
                f"Top-3 Entropy: {', '.join(entropy_top3)}. Vùng thay đổi hạng nhiều nhất là "
                f"{largest_change['region_name']}, chênh {int(largest_change['rank_change'])} bậc."
            ),
            "evidence": "Tab Ranking entropy, Weights và bảng so sánh ranking.",
        },
        {
            "code": "Câu 6.4.3",
            "question": "Độ nhạy khi thay đổi trọng số AI từ 0,10 đến 0,40.",
            "answer": (
                f"Top-3 {'ổn định' if stable_top3 else 'có thay đổi'} trên lưới trọng số đang chạy. "
                f"Có {len(top3_sets)} mức trọng số AI được kiểm tra."
            ),
            "evidence": "Tab Sensitivity AI và heatmap top-1.",
        },
        {
            "code": "Câu 6.4.4",
            "question": "Mở rộng so sánh thêm AHP.",
            "answer": (
                "Phiên bản hiện tại chưa triển khai ma trận so sánh cặp AHP; chỉ có TOPSIS expert và Entropy. "
                "Không có kết quả AHP để kết luận phương pháp nào tốt hơn."
            ),
            "status": "Phần mở rộng AHP chưa triển khai.",
        },
    ]
    policy = [
        {
            "code": "Câu 6.5a",
            "question": "Vùng dẫn đầu có nên đặt trung tâm AI đầu tiên không?",
            "answer": (
                f"{top3[0]} dẫn đầu theo TOPSIS. Đây là ứng viên định lượng mạnh nhất, nhưng quyết định trung tâm AI "
                "còn cần đất đai, điện, trung tâm dữ liệu, liên kết đại học và yêu cầu an ninh."
            ),
            "evidence": "Điểm TOPSIS expert cao nhất.",
        },
        {
            "code": "Câu 6.5b",
            "question": "Vùng nào thay đổi hạng nhiều nhất khi dùng Entropy?",
            "answer": (
                f"{largest_change['region_name']} thay đổi {int(largest_change['rank_change'])} bậc. "
                "Nguyên nhân là Entropy ưu tiên tiêu chí có độ phân tán dữ liệu lớn, khác với trọng số chuyên gia."
            ),
            "evidence": "rank_expert và rank_entropy.",
        },
        {
            "code": "Câu 6.5c",
            "question": "Tương quan giữa AI readiness và internet ảnh hưởng thế nào?",
            "answer": (
                "Nếu hai tiêu chí tương quan cao, TOPSIS có thể đếm hai lần cùng một lợi thế và làm vùng dẫn đầu mạnh hơn quá mức. "
                "Nên kiểm tra ma trận tương quan, giảm/gộp trọng số hoặc dùng PCA/robustness analysis."
            ),
            "evidence": "Giới hạn phương pháp TOPSIS tuyến tính.",
        },
        {
            "code": "Câu 6.5d",
            "question": "Chọn ba vùng cho ba trung tâm AI.",
            "answer": (
                f"Theo kết quả thuần TOPSIS: {', '.join(top3)}. Trước quyết định cuối cần thêm tiêu chí địa-chính trị, "
                "an ninh năng lượng, liên kết vùng và mục tiêu thu hẹp khoảng cách."
            ),
            "evidence": "Top-3 expert ranking.",
        },
    ]
    return programming, policy


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
programming_answers, discussion_answers = assignment_answers(
    expert_ranking, entropy_ranking, sensitivity_df, comparison_df
)
render_assignment_answers(programming_answers, discussion_answers)
