from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bai08_dynamic import (
    MODULE_TITLE,
    compare_strategies,
    module_status,
    optimize_dynamic,
    shock_2028,
)
from src.assignment_ui import render_assignment_answers
from src.data_loader import load_macro
from src.ui import (
    apply_dashboard_style,
    policy_box,
    render_page_badges,
    render_sidebar,
)
from src.visualization import download_dataframe_button, render_kpi_cards


st.set_page_config(page_title=MODULE_TITLE, page_icon="⏳", layout="wide")
apply_dashboard_style()
render_sidebar(MODULE_TITLE, "Rất khó", "Tối ưu động SLSQP")


@st.cache_data(show_spinner=False)
def get_data():
    return load_macro()


@st.cache_data(show_spinner=True)
def optimize_cached(rho: float, investment_rate: float, use_shock: bool):
    shock = shock_2028() if use_shock else None
    return optimize_dynamic(T=10, rho=rho, investment_rate=investment_rate, shock=shock, maxiter=180)


@st.cache_data(show_spinner=True)
def compare_cached(rho: float, investment_rate: float, use_shock: bool):
    shock = shock_2028() if use_shock else None
    return compare_strategies(rho=rho, investment_rate=investment_rate, shock=shock, T=10)


def policy_interpretation(summary_df, optimized_df) -> list[str]:
    best = summary_df.iloc[0]
    final = optimized_df.iloc[-1]
    first_policy = optimized_df[["share_K", "share_D", "share_AI", "share_H"]].mean().sort_values(ascending=False)
    top_control = first_policy.index[0].replace("share_", "")

    return [
        f"Chiến lược có welfare cao nhất là {best['strategy']} với Y_2035 khoảng {best['Y_2035']:.2f}.",
        f"Nghiệm tối ưu phân bổ trung bình nhiều nhất vào {top_control}, phản ánh nút thắt động lớn nhất trong hàm sản xuất.",
        f"Đến 2035, quỹ đạo tối ưu đạt K={final['K']:.1f}, D={final['D']:.2f}, AI={final['AI']:.2f}, H={final['H']:.2f}.",
        "Nếu bật shock 2028, chính sách nên giữ nhịp đầu tư ổn định thay vì phản ứng cực đoan một năm, vì mục tiêu là welfare chiết khấu dài hạn.",
    ]


def assignment_answers(optimized, trajectory_df, policy_df, summary_df, use_shock, rho):
    """Answer the dynamic-optimization requirements for the active run."""
    optimized_summary = summary_df[summary_df["strategy"] == "optimized"].iloc[0]
    equal_summary = summary_df[summary_df["strategy"] == "equal investment"].iloc[0]
    front_summary = summary_df[summary_df["strategy"] == "front-load"].iloc[0]
    early = policy_df.head(3)[["share_K", "share_D", "share_AI", "share_H"]].mean()
    late = policy_df.tail(3)[["share_K", "share_D", "share_AI", "share_H"]].mean()
    ai_h_ratio = (policy_df["share_AI"] / policy_df["share_H"].replace(0, float("nan"))).dropna()
    shock_answer = (
        "Shock 2028 đang bật. Quỹ đạo hiển thị trực tiếp Y, C và tỷ trọng đầu tư sau cú sốc; "
        f"Y_2028={trajectory_df.loc[trajectory_df['year'] == 2028, 'Y'].iloc[0]:.2f}, "
        f"Y_2029={trajectory_df.loc[trajectory_df['year'] == 2029, 'Y'].iloc[0]:.2f}."
        if use_shock
        else "Shock 2028 chưa bật trong lần chạy này. Bật công tắc và chạy lại để câu trả lời dùng quỹ đạo giảm Y năm 2028."
    )

    programming = [
        {
            "code": "Câu 8.3.1",
            "question": "Giải bài toán phi tuyến bằng SLSQP hoặc CVXPY.",
            "answer": (
                f"Dashboard dùng SLSQP; status={optimized['status']}, welfare={optimized['welfare']:.4f}. "
                f"Thông báo solver: {optimized['note']}"
            ),
            "evidence": "KPI Status/Welfare và note của optimize_dynamic.",
        },
        {
            "code": "Câu 8.3.2",
            "question": "Vẽ quỹ đạo K, D, AI, H, Y, C giai đoạn 2026-2035.",
            "answer": (
                f"Quỹ đạo gồm {len(trajectory_df)} năm. Năm 2035: K={trajectory_df['K'].iloc[-1]:,.2f}, "
                f"D={trajectory_df['D'].iloc[-1]:.2f}, AI={trajectory_df['AI'].iloc[-1]:.2f}, "
                f"H={trajectory_df['H'].iloc[-1]:.2f}, Y={trajectory_df['Y'].iloc[-1]:.2f}, "
                f"C={trajectory_df['C'].iloc[-1]:.2f}."
            ),
            "evidence": "Biểu đồ quỹ đạo và tab Trajectory.",
        },
        {
            "code": "Câu 8.3.3",
            "question": "Phân tích cú sốc 2028 làm Y giảm 8%.",
            "answer": shock_answer,
            "evidence": "Công tắc Shock 2028 và trajectory_df.",
            "status": "Cần bật shock để có kết quả phản thực." if not use_shock else "Đã trả lời",
        },
        {
            "code": "Câu 8.3.4",
            "question": "So sánh equal investment, front-load và optimized.",
            "answer": (
                f"Welfare: optimized={optimized_summary['welfare']:.4f}, "
                f"equal={equal_summary['welfare']:.4f}, front-load={front_summary['welfare']:.4f}. "
                f"Chiến lược cao nhất là {summary_df.iloc[0]['strategy']}."
            ),
            "evidence": "Tab Strategy summary.",
        },
    ]
    policy = [
        {
            "code": "Câu 8.4a",
            "question": "Quỹ đạo tối ưu front-loaded hay back-loaded?",
            "answer": (
                f"Ba năm đầu phân bổ bình quân K/D/AI/H = {early.to_dict()}, ba năm cuối = {late.to_dict()}. "
                "So sánh hai vector cho biết yếu tố nào được dồn sớm hay muộn; trong nghiệm hiện tại D được ưu tiên mạnh ở đầu kỳ."
            ),
            "evidence": "policy_df, trung bình 3 năm đầu và 3 năm cuối.",
        },
        {
            "code": "Câu 8.4b",
            "question": "Tỷ lệ đầu tư AI/H có ổn định không?",
            "answer": (
                f"Tỷ lệ share_AI/share_H dao động từ {ai_h_ratio.min():.2f} đến {ai_h_ratio.max():.2f}. "
                "Nếu tỷ lệ biến động, mô hình không ủng hộ một tỷ lệ cố định; đào tạo và AI cần được điều chỉnh đồng thời theo trạng thái."
            ),
            "evidence": "Các cột share_AI và share_H trong policy_df.",
        },
        {
            "code": "Câu 8.4c",
            "question": "Nếu rho giảm từ 0,97 xuống 0,90 thì kết quả thay đổi thế nào?",
            "answer": (
                f"Lần chạy hiện tại dùng rho={rho:.3f}. Dashboard cho phép đặt rho=0,90 và chạy lại, "
                "nhưng không tự chạy song song hai nghiệm nên chưa đưa ra chênh lệch số khi chưa thực hiện lần đối chứng. "
                "Về cơ chế, rho thấp làm giảm trọng số lợi ích xa trong tương lai."
            ),
            "status": "Cần chạy lại với rho=0,90 để định lượng chênh lệch.",
        },
    ]
    return programming, policy


macro = get_data()

st.title(MODULE_TITLE)
render_page_badges("Rất khó", "Tối ưu động SLSQP")
st.info(module_status())

st.header("🧭 1. Bối cảnh")
st.write(
    "Bài 8 mô phỏng quyết định đầu tư liên thời gian 2026-2035. Nhà hoạch định phân bổ "
    "ngân sách đầu tư hằng năm vào vốn vật chất K, số hóa D, AI và vốn nhân lực H, trong khi "
    "tối đa hóa phúc lợi tiêu dùng chiết khấu."
)

st.header("🧮 2. Mô hình toán học")
st.latex(r"\max \sum_{t=2026}^{2035}\rho^{t-2026}\log(C_t)")
st.latex(
    r"""
    \begin{aligned}
    Y_t &= A_t K_t^\alpha L_t^\beta D_t^\gamma AI_t^\delta H_t^\theta\\
    I_t &= investment\_rate \cdot Y_t,\quad C_t=(1-investment\_rate)Y_t\\
    K_{t+1} &= (1-\delta_K)K_t+s_{K,t}I_t\\
    D_{t+1} &= (1-\delta_D)D_t+\phi_1s_{D,t}I_t\\
    AI_{t+1} &= (1-\delta_{AI})AI_t+\phi_2s_{AI,t}I_t\\
    H_{t+1} &= H_t+\theta_H\phi_3s_{H,t}I_t+\mu(40-H_t)
    \end{aligned}
    """
)
st.write("Mỗi năm các tỷ trọng `s_K, s_D, s_AI, s_H` cộng lại bằng 1. SLSQP được dùng làm solver mặc định.")

st.header("🧾 3. Dữ liệu/tham số")
param_cols = st.columns(3)
rho = param_cols[0].slider("rho - hệ số chiết khấu", 0.85, 0.995, 0.97, 0.005)
investment_rate = param_cols[1].slider("investment_rate", 0.10, 0.45, 0.28, 0.01)
use_shock = param_cols[2].toggle("Shock 2028: Y giảm 8%", value=False)

with st.expander("Dữ liệu macro tham khảo"):
    st.dataframe(macro, use_container_width=True)
    download_dataframe_button(macro, "bai_8_macro_context.csv")

st.header("📊 4. Kết quả")
if st.button("Chạy tối ưu động 2026-2035", type="primary"):
    optimized = optimize_cached(rho, investment_rate, use_shock)
    comparison = compare_cached(rho, investment_rate, use_shock)
    trajectory_df = optimized["trajectory_df"]
    policy_df = optimized["policy_df"]
    compare_trajectory_df = comparison["trajectory_df"]
    summary_df = comparison["summary_df"]

    render_kpi_cards(
        {
            "Status": optimized["status"],
            "Welfare": f"{optimized['welfare']:.3f}",
            "Y 2035": f"{trajectory_df['Y'].iloc[-1]:.2f}",
            "C 2035": f"{trajectory_df['C'].iloc[-1]:.2f}",
        }
    )
    st.caption(optimized["note"])

    st.subheader("Quỹ đạo tối ưu K, D, AI, H, Y, C")
    state_long = trajectory_df.melt(
        id_vars="year",
        value_vars=["K", "D", "AI", "H", "Y", "C"],
        var_name="variable",
        value_name="value",
    )
    fig_state = px.line(
        state_long,
        x="year",
        y="value",
        color="variable",
        markers=True,
        title="Quỹ đạo trạng thái và kết quả tối ưu",
    )
    st.plotly_chart(fig_state, use_container_width=True)

    st.subheader("Policy shares tối ưu")
    policy_long = policy_df.melt(id_vars="year", value_vars=["share_K", "share_D", "share_AI", "share_H"])
    fig_policy = px.area(
        policy_long,
        x="year",
        y="value",
        color="variable",
        title="Tỷ trọng phân bổ đầu tư theo năm",
        labels={"value": "Share", "variable": "Control"},
    )
    st.plotly_chart(fig_policy, use_container_width=True)

    st.subheader("So sánh equal vs front-load vs optimized")
    compare_y = compare_trajectory_df.melt(
        id_vars=["year", "strategy"],
        value_vars=["Y", "C"],
        var_name="metric",
        value_name="value",
    )
    fig_compare = px.line(
        compare_y,
        x="year",
        y="value",
        color="strategy",
        line_dash="metric",
        markers=True,
        title="So sánh Y và C theo chiến lược",
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    tab_traj, tab_policy, tab_summary = st.tabs(["Trajectory", "Policy", "Strategy summary"])
    with tab_traj:
        st.dataframe(trajectory_df, use_container_width=True)
        download_dataframe_button(trajectory_df, "bai_8_optimized_trajectory.csv")
    with tab_policy:
        st.dataframe(policy_df, use_container_width=True)
        download_dataframe_button(policy_df, "bai_8_optimized_policy.csv")
    with tab_summary:
        st.dataframe(summary_df, use_container_width=True)
        download_dataframe_button(summary_df, "bai_8_strategy_comparison.csv")

    st.header("🏛️ 5. Diễn giải chính sách")
    policy_box(policy_interpretation(summary_df, trajectory_df), kind="success")
    programming_answers, discussion_answers = assignment_answers(
        optimized, trajectory_df, policy_df, summary_df, use_shock, rho
    )
    render_assignment_answers(programming_answers, discussion_answers)
else:
    st.info("Nhấn **Chạy tối ưu động 2026-2035** để tối ưu policy shares và so sánh ba chiến lược.")
    st.header("🏛️ 5. Diễn giải chính sách")
    policy_box("Diễn giải chính sách sẽ xuất hiện sau khi chạy mô hình.")
