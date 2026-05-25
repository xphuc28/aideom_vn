from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bai03_priority import (
    DEFAULT_WEIGHTS,
    MODULE_TITLE,
    compare_policy_weights,
    compute_priority,
    module_status,
    sensitivity_ai_weight,
)
from src.data_loader import load_sectors
from src.ui import apply_dashboard_style, policy_box, render_page_badges, render_sidebar
from src.visualization import download_dataframe_button, render_kpi_cards


st.set_page_config(page_title=MODULE_TITLE, page_icon="🏭", layout="wide")
apply_dashboard_style()
render_sidebar(MODULE_TITLE, "Dễ - Trung bình", "MCDA Priority Score")


@st.cache_data(show_spinner=False)
def get_data():
    return load_sectors()


def normalized_slider_weights(raw_weights: dict[str, float]) -> dict[str, float]:
    total = sum(raw_weights.values())
    if total <= 0:
        return DEFAULT_WEIGHTS.copy()
    return {key: value / total for key, value in raw_weights.items()}


def policy_interpretation(ranking_df, comparison_df) -> list[str]:
    top = ranking_df.iloc[0]
    top3 = ranking_df.head(3)["sector_name"].to_list()
    overlap = set(comparison_df["growth_oriented"]).intersection(comparison_df["inclusive_oriented"])

    return [
        f"Ngành ưu tiên cao nhất theo trọng số hiện tại là {top['sector_name']} "
        f"với điểm {top['priority_score']:.3f}.",
        "Nhóm top-3 hiện tại gồm: " + ", ".join(top3) + ".",
        f"Hai cách đặt trọng số chính sách có {len(overlap)} ngành trùng nhau trong top-3; "
        "mức trùng càng cao thì khuyến nghị càng ổn định giữa mục tiêu tăng trưởng và bao trùm.",
        "Nếu tăng trọng số AI readiness làm thay đổi ngành đứng đầu, chính sách cần tách riêng "
        "gói thúc đẩy AI cho ngành dẫn dắt và gói giảm rủi ro tự động hóa cho ngành nhiều lao động.",
    ]


sectors = get_data()

st.title(MODULE_TITLE)
render_page_badges("Dễ - Trung bình", "MCDA Priority Score")
st.info(module_status())

st.header("🧭 1. Bối cảnh")
st.write(
    "Bài 3 xếp hạng 10 ngành kinh tế Việt Nam theo bộ tiêu chí đa chiều: tăng trưởng, "
    "năng suất, lan tỏa, xuất khẩu, quy mô lao động, mức sẵn sàng AI và rủi ro tự động hóa. "
    "Kết quả giúp xác định ngành nên ưu tiên trong chiến lược chuyển đổi số và AI."
)

st.header("🧮 2. Mô hình toán học")
st.latex(r"Priority_i = \sum_{j \in Good} w_j z_{ij} + w_r RiskGood_i")
st.write(
    "Các tiêu chí được chuẩn hóa min-max về `[0,1]`. Với rủi ro tự động hóa, mặc định "
    "dùng `risk_inverted=True`, tức rủi ro thấp được xem là điểm tốt để tránh trừ hai lần."
)

st.header("🧾 3. Dữ liệu/tham số")
st.subheader("Dữ liệu ngành")
st.dataframe(sectors, use_container_width=True)
download_dataframe_button(sectors, "bai_3_input_sectors.csv")

st.subheader("Trọng số tiêu chí")
weight_cols = st.columns(4)
raw_weights = {
    "growth": weight_cols[0].slider("Tăng trưởng", 0.0, 1.0, DEFAULT_WEIGHTS["growth"], 0.01),
    "productivity": weight_cols[1].slider("Năng suất", 0.0, 1.0, DEFAULT_WEIGHTS["productivity"], 0.01),
    "spillover": weight_cols[2].slider("Lan tỏa", 0.0, 1.0, DEFAULT_WEIGHTS["spillover"], 0.01),
    "export": weight_cols[3].slider("Xuất khẩu", 0.0, 1.0, DEFAULT_WEIGHTS["export"], 0.01),
    "labor": weight_cols[0].slider("Lao động", 0.0, 1.0, DEFAULT_WEIGHTS["labor"], 0.01),
    "ai_readiness": weight_cols[1].slider("AI readiness", 0.0, 1.0, DEFAULT_WEIGHTS["ai_readiness"], 0.01),
    "automation_risk": weight_cols[2].slider("Rủi ro tự động hóa", 0.0, 1.0, DEFAULT_WEIGHTS["automation_risk"], 0.01),
}
weights = normalized_slider_weights(raw_weights)
st.caption(
    f"Tổng trọng số nhập vào: {sum(raw_weights.values()):.2f}. "
    "Mô hình tự chuẩn hóa tổng trọng số về 1 trước khi tính điểm."
)

risk_choice = st.radio(
    "Cách xử lý rủi ro tự động hóa",
    ["risk_inverted", "risk_as_cost"],
    horizontal=True,
    help="risk_inverted là mặc định: rủi ro thấp thành điểm tốt. risk_as_cost trừ trực tiếp điểm rủi ro.",
)
risk_mode = {"risk_inverted": risk_choice == "risk_inverted", "risk_as_cost": risk_choice == "risk_as_cost"}

st.header("📊 4. Kết quả")
try:
    results = compute_priority(sectors, weights, risk_mode=risk_mode)
    ranking_df = results["ranking_df"]
    normalized_df = results["normalized_df"]
    sensitivity_df = sensitivity_ai_weight(sectors)
    policy_compare = compare_policy_weights(sectors)
    top3_comparison = policy_compare["top3_comparison"]

    render_kpi_cards(
        {
            "Số ngành": len(ranking_df),
            "Top priority": ranking_df.iloc[0]["sector_name"],
            "Điểm top": f"{ranking_df.iloc[0]['priority_score']:.3f}",
        }
    )

    st.subheader("Ma trận chuẩn hóa")
    st.dataframe(normalized_df, use_container_width=True)
    download_dataframe_button(normalized_df, "bai_3_normalized_matrix.csv")

    st.subheader("Ranking 10 ngành")
    st.dataframe(ranking_df, use_container_width=True)
    download_dataframe_button(ranking_df, "bai_3_priority_ranking.csv")

    chart_cols = st.columns(2)
    with chart_cols[0]:
        fig_priority = px.bar(
            ranking_df.sort_values("priority_score", ascending=True),
            x="priority_score",
            y="sector_name",
            orientation="h",
            title="Priority score theo ngành",
            labels={"priority_score": "Priority score", "sector_name": "Ngành"},
        )
        st.plotly_chart(fig_priority, use_container_width=True)

    with chart_cols[1]:
        heatmap_data = (
            sensitivity_df.pivot(index="sector_name", columns="ai_weight", values="is_top_rank")
            .fillna(0)
            .astype(float)
        )
        fig_heatmap = px.imshow(
            heatmap_data,
            aspect="auto",
            color_continuous_scale=["#F2F4F7", "#0F766E"],
            title="Heatmap ngành top-1 khi thay đổi trọng số AI readiness",
            labels={"x": "Trọng số AI readiness", "y": "Ngành", "color": "Top-1"},
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    st.subheader("Sensitivity theo trọng số AI readiness")
    st.dataframe(sensitivity_df, use_container_width=True)
    download_dataframe_button(sensitivity_df, "bai_3_ai_weight_sensitivity.csv")

    st.subheader("So sánh top-3 theo định hướng chính sách")
    st.dataframe(top3_comparison, use_container_width=True)
    download_dataframe_button(top3_comparison, "bai_3_policy_top3_comparison.csv")

    st.header("🏛️ 5. Diễn giải chính sách")
    policy_box(policy_interpretation(ranking_df, top3_comparison), kind="success")

except Exception as exc:
    st.error(f"Không chạy được Bài 3: {exc}")
