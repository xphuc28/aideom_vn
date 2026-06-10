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
from src.ui import (
    apply_dashboard_style,
    policy_box,
    render_assignment_answers,
    render_page_badges,
    render_sidebar,
)
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


def assignment_answers(result, threshold_df, capped_result, cap_displaced):
    """Answer labor-allocation and social-safeguard questions."""
    df = result["allocation_df"]
    threshold = threshold_df.iloc[0]
    finance = df[df["sector"].str.contains("Tài chính", case=False)].iloc[0]
    agriculture = df[df["sector"].str.contains("Nông", case=False)].iloc[0]
    top_training = df.sort_values("x_H", ascending=False).iloc[0]
    total_net = float(df["NetJob"].sum())
    constraints_ok = bool(
        (df["NetJob"] >= -1e-6).all()
        and (df["DisplacedJob"] <= df["RetrainingCapacity"] + 1e-6).all()
    )

    programming = [
        {
            "code": "Câu 9.4.1",
            "question": "Giải phân bổ x_AI, x_H và tính NetJob từng ngành.",
            "answer": (
                f"Status={result['status']}; tổng NetJob={total_net:,.2f}; tổng phân bổ="
                f"{df['total_allocation'].sum():,.2f}. Các ràng buộc NetJob>=0 và Displaced<=RetrainCap "
                f"{'đều thỏa' if constraints_ok else 'chưa thỏa'}."
            ),
            "evidence": "Tabs Phân bổ và Việc làm.",
        },
        {
            "code": "Câu 9.4.2",
            "question": "Ngưỡng đào tạo ngành chế biến chế tạo khi đầu tư AI tối đa.",
            "answer": (
                f"Với x_AI={threshold['x_AI']:,.2f}, x_H tối thiểu theo ràng buộc đào tạo lại là "
                f"{threshold['minimum_x_H_required']:,.2f}; displaced jobs={threshold['displaced_job']:,.2f}."
            ),
            "evidence": "Bảng Ngưỡng đào tạo lại ngành chế biến chế tạo.",
        },
        {
            "code": "Câu 9.4.3",
            "question": "Mô phỏng nhóm dễ tổn thương và vẽ Sankey dịch chuyển lao động.",
            "answer": (
                "Dashboard đã tính NewJob, UpgradeJob, DisplacedJob và NetJob cho các ngành 1, 3, 4, "
                "nhưng chưa dựng Sankey/swimming lane riêng cho luồng lao động."
            ),
            "status": "Thiếu trực quan Sankey theo đúng phần mở rộng của đề.",
        },
        {
            "code": "Câu 9.4.4",
            "question": "Thêm ràng buộc không ngành nào mất quá 5% lao động.",
            "answer": (
                f"Kịch bản có cap 5% cho status={capped_result['status']} và objective={capped_result['objective']}. "
                f"Công tắc hiện tại đang {'bật' if cap_displaced else 'tắt'}."
            ),
            "evidence": "Kết quả solve_bai09 với max_displaced_share=0.05.",
        },
    ]
    policy = [
        {
            "code": "Câu 9.5a",
            "question": "Ngành nào cần đào tạo lại nhiều nhất?",
            "answer": (
                f"{top_training['sector']} có x_H cao nhất, bằng {top_training['x_H']:,.2f}. "
                "Đây là kết quả tối ưu theo hệ số mô hình; cần đối chiếu dữ liệu kỹ năng thực tế trước khi kết luận chính sách."
            ),
            "evidence": "Sắp xếp allocation_df theo x_H.",
        },
        {
            "code": "Câu 9.5b",
            "question": "Chiến lược cho Tài chính–Ngân hàng khi rủi ro thay thế cao.",
            "answer": (
                f"Nghiệm hiện tại phân bổ x_AI={finance['x_AI']:,.2f}, x_H={finance['x_H']:,.2f}, "
                f"NetJob={finance['NetJob']:,.2f}. Mô hình yêu cầu mọi tự động hóa phải đi cùng năng lực đào tạo lại đủ lớn."
            ),
            "evidence": "Dòng Tài chính-Ngân hàng trong allocation_df.",
        },
        {
            "code": "Câu 9.5c",
            "question": "Có nên đầu tư AI vào Nông-Lâm-Thủy sản?",
            "answer": (
                f"Nghiệm hiện tại chọn x_AI={agriculture['x_AI']:,.2f}, x_H={agriculture['x_H']:,.2f} cho ngành này. "
                "Nếu bằng 0, đó là kết quả của hiệu suất biên tương đối; không đồng nghĩa ngành không cần chính sách số bao trùm."
            ),
            "evidence": "Dòng Nông-Lâm-Thủy sản trong allocation_df.",
        },
        {
            "code": "Câu 9.5d",
            "question": "Tự động hóa không vượt năng lực đào tạo lại được biểu diễn thế nào?",
            "answer": (
                "Ràng buộc trực tiếp là DisplacedJob_i <= RetrainingCapacity_i. Có thể bổ sung cap 5% lao động, "
                "sàn x_H theo ngành dễ tổn thương và ngân sách hỗ trợ thu nhập chuyển tiếp."
            ),
            "evidence": "Hệ ràng buộc trong mô hình và tùy chọn max_displaced_share.",
        },
    ]
    return programming, policy


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

threshold_df = manufacturing_training_threshold(budget)
with st.expander(f"Ngưỡng đào tạo lại ngành chế biến chế tạo khi x_AI={budget:,.0f}"):
    st.dataframe(threshold_df, use_container_width=True)
    download_dataframe_button(threshold_df, "bai_9_manufacturing_threshold.csv")

st.header("📊 4. Kết quả")
result = solve_cached(budget, risk_multiplier, cap_displaced)
capped_result = solve_cached(budget, risk_multiplier, True)
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
programming_answers, discussion_answers = assignment_answers(
    result, threshold_df, capped_result, cap_displaced
)
render_assignment_answers(programming_answers, discussion_answers)
