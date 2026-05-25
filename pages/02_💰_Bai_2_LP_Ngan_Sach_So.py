from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bai02_lp_budget import (
    DECISION_NAMES,
    MODULE_TITLE,
    module_status,
    scenario_human_priority,
    sensitivity_budget,
    solve_bai02_pulp,
    solve_bai02_scipy,
)
from src.data_loader import load_sectors
from src.ui import apply_dashboard_style, policy_box, render_page_badges, render_sidebar
from src.visualization import download_dataframe_button, render_kpi_cards


st.set_page_config(page_title=MODULE_TITLE, page_icon="💰", layout="wide")
apply_dashboard_style()
render_sidebar(MODULE_TITLE, "Trung bình", "Linear Programming")


@st.cache_data(show_spinner=False)
def get_data():
    return load_sectors()


@st.cache_data(show_spinner=False)
def run_scipy_cached(B: float, min_human: float):
    return solve_bai02_scipy(B=B, min_human=min_human)


@st.cache_data(show_spinner=False)
def run_pulp_cached(B: float, min_human: float):
    return solve_bai02_pulp(B=B, min_human=min_human)


@st.cache_data(show_spinner=False)
def run_sensitivity_cached(max_budget: float):
    budgets = [100, 120, 140, float(max_budget)]
    budgets = sorted(set(budget for budget in budgets if budget >= 70))
    return sensitivity_budget(budgets)


def explain_shadow_price(result: dict[str, object]) -> str:
    shadow_prices = result.get("shadow_prices", {})
    price = shadow_prices.get("budget_total")
    if result.get("status") != "optimal":
        return "Chưa có shadow price vì mô hình chưa có nghiệm tối ưu."
    if price is None:
        return "Solver không trả dual value cho ngân sách; xem binding constraints để nhận diện nút thắt."
    return (
        f"Shadow price ngân sách xấp xỉ {price:.3f}: nếu tăng thêm 1 nghìn tỷ VND "
        "và cấu trúc ràng buộc không đổi, mục tiêu Z kỳ vọng tăng khoảng mức này."
    )


def policy_answers(result: dict[str, object]) -> list[str]:
    if result.get("status") != "optimal":
        return ["Ngân sách hiện tại không tạo được nghiệm tối ưu; cần tăng B hoặc giảm mức tối thiểu."]

    allocation_df = result["allocation_df"].sort_values("allocation", ascending=False)
    top = allocation_df.iloc[0]
    binding = result.get("binding_constraints", [])
    binding_text = ", ".join(binding) if binding else "không có ràng buộc sát nút rõ ràng"
    total = allocation_df["allocation"].sum()
    ai_rd = allocation_df.loc[allocation_df["variable"].isin(["x2", "x4"]), "allocation"].sum()

    return [
        f"Hạng mục nhận nhiều ngân sách nhất là {top['category']} với {top['allocation']:.2f} nghìn tỷ VND.",
        f"Tỷ trọng AI + R&D đạt {ai_rd / total * 100:.1f}% tổng phân bổ, so với ngưỡng tối thiểu 35%.",
        f"Các ràng buộc binding: {binding_text}. Đây là nơi chính sách nên xem xét nới hoặc giữ chặt.",
        "Nếu mục tiêu là tăng tốc công nghệ lõi, nên theo dõi x2 và x4; nếu mục tiêu là bao trùm, "
        "so sánh thêm kịch bản tăng tối thiểu nhân lực số.",
    ]


sectors = get_data()

st.title(MODULE_TITLE)
render_page_badges("Trung bình", "Linear Programming")
st.info(module_status())

st.header("🧭 1. Bối cảnh")
st.write(
    "Bài 2 mô hình hóa bài toán phân bổ ngân sách chuyển đổi số giữa bốn nhóm đầu tư: "
    "hạ tầng số, AI và dữ liệu, nhân lực số, và R&D công nghệ. Mục tiêu là tối đa hóa "
    "tác động tổng hợp trong khi vẫn bảo đảm các mức đầu tư tối thiểu."
)

st.header("🧮 2. Mô hình toán học")
st.latex(r"\max Z = 0.85x_1 + 1.20x_2 + 0.95x_3 + 1.35x_4")
st.latex(
    r"""
    \begin{aligned}
    x_1+x_2+x_3+x_4 &\le B\\
    x_1 &\ge 25,\quad x_2 \ge 15,\quad x_3 \ge min\_human,\quad x_4 \ge 10\\
    x_2+x_4 &\ge 0.35(x_1+x_2+x_3+x_4)\\
    x_i &\ge 0
    \end{aligned}
    """
)
st.write("Đơn vị của các biến quyết định là nghìn tỷ VND.")

st.header("🧾 3. Dữ liệu/tham số")
param_cols = st.columns(2)
B = param_cols[0].number_input("Tổng ngân sách B", min_value=0.0, max_value=500.0, value=100.0, step=5.0)
min_human = param_cols[1].number_input(
    "Tối thiểu nhân lực số x3",
    min_value=0.0,
    max_value=200.0,
    value=20.0,
    step=5.0,
)

st.subheader("Bảng hạng mục quyết định")
decision_df = pd.DataFrame(
    {
        "variable": list(DECISION_NAMES),
        "category": list(DECISION_NAMES.values()),
        "unit": ["nghìn tỷ VND"] * 4,
        "objective_coef": [0.85, 1.20, 0.95, 1.35],
        "minimum": [25, 15, min_human, 10],
    }
)
st.dataframe(decision_df, use_container_width=True)

with st.expander("Dữ liệu ngành tham khảo từ vietnam_sectors_2024.csv"):
    st.dataframe(sectors, use_container_width=True)
    download_dataframe_button(sectors, "bai_2_sector_context.csv")

st.header("📊 4. Kết quả")
button_cols = st.columns(2)
run_scipy = button_cols[0].button("Chạy SciPy", type="primary")
run_pulp = button_cols[1].button("Chạy PuLP")

result = None
solver_name = None
if run_scipy:
    result = run_scipy_cached(B, min_human)
    solver_name = "SciPy HiGHS"
elif run_pulp:
    result = run_pulp_cached(B, min_human)
    solver_name = "PuLP/CBC"
else:
    st.info("Nhấn **Chạy SciPy** hoặc **Chạy PuLP** để giải mô hình LP.")

if result is not None:
    allocation_df = result["allocation_df"]
    objective = result["objective"]
    objective_text = "N/A" if objective is None else f"{objective:.2f}"
    render_kpi_cards(
        {
            "Solver": solver_name,
            "Status": result["status"],
            "Z tối ưu": objective_text,
        }
    )
    st.caption(result.get("note", ""))

    st.subheader("Bảng phân bổ tối ưu")
    st.dataframe(allocation_df, use_container_width=True)
    download_dataframe_button(allocation_df, "bai_2_allocation.csv")

    fig_alloc = px.bar(
        allocation_df,
        x="category",
        y="allocation",
        color="category",
        title="Phân bổ ngân sách tối ưu",
        labels={"category": "Hạng mục", "allocation": "Nghìn tỷ VND"},
    )
    fig_alloc.update_layout(showlegend=False, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig_alloc, use_container_width=True)

    st.subheader("Binding constraints và shadow price")
    st.write("Binding constraints:", result.get("binding_constraints", []))
    st.write(explain_shadow_price(result))
    if result.get("shadow_prices"):
        shadow_df = pd.DataFrame(
            {
            "constraint": list(result["shadow_prices"].keys()),
            "shadow_price": list(result["shadow_prices"].values()),
            }
        )
        st.dataframe(shadow_df, use_container_width=True)

st.subheader("Phân tích độ nhạy ngân sách")
sensitivity_df = run_sensitivity_cached(B)
st.dataframe(sensitivity_df, use_container_width=True)
download_dataframe_button(sensitivity_df, "bai_2_budget_sensitivity.csv")
fig_sens = px.line(
    sensitivity_df,
    x="budget",
    y="objective",
    markers=True,
    title="Z*(B) theo tổng ngân sách",
    labels={"budget": "B - nghìn tỷ VND", "objective": "Z tối ưu"},
)
st.plotly_chart(fig_sens, use_container_width=True)

st.subheader("Kịch bản ưu tiên nhân lực số")
human_result = scenario_human_priority(B=B, min_human=max(min_human, 30.0))
st.write(
    f"Khi yêu cầu nhân lực số tối thiểu là {max(min_human, 30.0):.0f}, "
    f"status = `{human_result['status']}`, Z = `{human_result['objective']}`."
)
st.dataframe(human_result["allocation_df"], use_container_width=True)

st.header("🏛️ 5. Diễn giải chính sách")
if result is None:
    policy_box("Sau khi chạy solver, trang sẽ trả lời câu hỏi ngân sách nên ưu tiên vào đâu và nút thắt nằm ở ràng buộc nào.")
else:
    policy_box(policy_answers(result), kind="success")
