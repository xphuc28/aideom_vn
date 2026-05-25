from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bai04_region_lp import ITEMS, ITEM_NAMES, REGIONS, REGION_NAMES
from src.bai07_pareto import MODULE_TITLE, module_status, run_nsga2
from src.data_loader import load_regions
from src.ui import apply_dashboard_style, policy_box, render_page_badges, render_sidebar
from src.visualization import download_dataframe_button, render_kpi_cards


st.set_page_config(page_title=MODULE_TITLE, page_icon="🧬", layout="wide")
apply_dashboard_style()
render_sidebar(MODULE_TITLE, "Rất khó", "Pareto / NSGA-II")


@st.cache_data(show_spinner=False)
def get_data():
    return load_regions()


@st.cache_data(show_spinner=True)
def run_pareto_cached(pop_size: int, n_gen: int, seed: int, fairness: bool, lambda_: float):
    return run_nsga2(
        pop_size=pop_size,
        n_gen=n_gen,
        seed=seed,
        fairness=fairness,
        lambda_=lambda_,
    )


def allocation_from_solution(row: pd.Series) -> pd.DataFrame:
    matrix = []
    for region in REGIONS:
        matrix.append([float(row.get(f"{region}_{item}", 0.0)) for item in ITEMS])
    return pd.DataFrame(matrix, index=[REGION_NAMES[r] for r in REGIONS], columns=[ITEM_NAMES[i] for i in ITEMS])


def policy_interpretation(pareto_df: pd.DataFrame, summary_df: pd.DataFrame) -> list[str]:
    if pareto_df.empty or summary_df.empty:
        return ["Chưa có tập Pareto khả dụng; cần tăng số mẫu hoặc nới ràng buộc."]

    max_growth = summary_df[summary_df["scenario"] == "Tăng trưởng cao nhất"].iloc[0]
    compromise = summary_df[summary_df["scenario"] == "Nghiệm thỏa hiệp"].iloc[0]
    gdp_loss = max_growth["gdp_gain"] - compromise["gdp_gain"]
    emission_gain = max_growth["emission"] - compromise["emission"]
    risk_gain = max_growth["net_cyber_risk"] - compromise["net_cyber_risk"]

    return [
        f"Tập Pareto có {len(pareto_df)} nghiệm không bị trội, thể hiện nhiều cách đánh đổi chính sách.",
        f"Nghiệm thỏa hiệp hy sinh khoảng {gdp_loss:,.1f} điểm GDP gain so với nghiệm tăng trưởng cao nhất.",
        f"Đổi lại, nghiệm thỏa hiệp giảm emission khoảng {emission_gain:,.1f} và giảm net cyber risk khoảng {risk_gain:,.1f}.",
        "Nếu mục tiêu là bứt tốc, chọn nghiệm tăng trưởng cao nhất; nếu cần cân bằng phát triển, phát thải, rủi ro và công bằng vùng, chọn nghiệm thỏa hiệp.",
    ]


regions = get_data()

st.title(MODULE_TITLE)
render_page_badges("Rất khó", "Pareto / NSGA-II")
st.info(module_status())

st.header("🧭 1. Bối cảnh")
st.write(
    "Bài 7 mở rộng bài toán phân bổ vùng-hạng mục thành tối ưu đa mục tiêu. "
    "Không chỉ tối đa hóa tăng trưởng, mô hình còn xem xét bất bình đẳng phân bổ vùng, "
    "phát thải và rủi ro an ninh mạng ròng khi đầu tư AI."
)

st.header("🧮 2. Mô hình toán học")
st.latex(r"\max f_1(x)=GDPGain,\quad \min f_2(x)=MAD,\quad \min f_3(x)=Emission,\quad \min f_4(x)=CyberRisk")
st.write(
    "Trong `pymoo`, mục tiêu tăng trưởng được chuyển thành minimize `-GDP_gain`. "
    "Các ràng buộc giống Bài 4: tổng ngân sách, sàn/trần vùng, tổng H và fairness tùy chọn."
)

st.header("🧾 3. Dữ liệu/tham số")
param_cols = st.columns(5)
pop_size = int(param_cols[0].number_input("pop_size", min_value=20, max_value=300, value=80, step=10))
n_gen = int(param_cols[1].number_input("n_gen", min_value=10, max_value=300, value=80, step=10))
seed = int(param_cols[2].number_input("seed", min_value=1, max_value=9999, value=42, step=1))
fairness = param_cols[3].toggle("Fairness", value=True)
lambda_ = param_cols[4].slider("lambda", 0.10, 0.90, 0.70, 0.01)

with st.expander("Dữ liệu vùng tham khảo"):
    st.dataframe(regions, use_container_width=True)
    download_dataframe_button(regions, "bai_7_regions_context.csv")

st.header("📊 4. Kết quả")
if st.button("Chạy tối ưu Pareto Bài 7", type="primary"):
    result = run_pareto_cached(pop_size, n_gen, seed, fairness, lambda_)
    pareto_df = result["pareto_df"]
    compromise = result["compromise_solution"]
    summary_df = result["summary_df"]

    st.caption(result["note"])
    if pareto_df.empty:
        st.error("Không tạo được Pareto set. Hãy tăng pop_size/n_gen hoặc tắt fairness.")
    else:
        render_kpi_cards(
            {
                "Method": result["method"],
                "Pareto candidates": len(pareto_df),
                "Best GDP gain": f"{pareto_df['gdp_gain'].max():,.1f}",
                "Compromise GDP": f"{compromise['gdp_gain'].iloc[0]:,.1f}",
            }
        )

        fig_3d = px.scatter_3d(
            pareto_df,
            x="gdp_gain",
            y="inequality",
            z="emission",
            color="net_cyber_risk",
            hover_name="solution_id",
            title="Pareto 3D: GDP gain - inequality - emission",
            labels={
                "gdp_gain": "GDP gain",
                "inequality": "Inequality MAD",
                "emission": "Emission",
                "net_cyber_risk": "Net cyber risk",
            },
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        fig_parallel = px.parallel_coordinates(
            pareto_df,
            dimensions=["gdp_gain", "inequality", "emission", "net_cyber_risk", "total_budget"],
            color="gdp_gain",
            title="Parallel coordinates cho 4 objective",
            color_continuous_scale=px.colors.sequential.Viridis,
        )
        st.plotly_chart(fig_parallel, use_container_width=True)

        st.subheader("Nghiệm thỏa hiệp TOPSIS")
        st.dataframe(compromise, use_container_width=True)
        download_dataframe_button(compromise, "bai_7_compromise_solution.csv")

        st.subheader("Ma trận phân bổ của nghiệm thỏa hiệp")
        compromise_matrix = allocation_from_solution(compromise.iloc[0])
        st.dataframe(compromise_matrix, use_container_width=True)
        download_dataframe_button(compromise_matrix.reset_index(names="region"), "bai_7_compromise_allocation.csv")

        st.subheader("So sánh nghiệm tăng trưởng cao nhất với nghiệm thỏa hiệp")
        st.dataframe(summary_df, use_container_width=True)
        download_dataframe_button(summary_df, "bai_7_growth_vs_compromise.csv")

        st.subheader("Pareto dataframe")
        st.dataframe(pareto_df, use_container_width=True)
        download_dataframe_button(pareto_df, "bai_7_pareto_set.csv")

        st.header("🏛️ 5. Diễn giải chính sách")
        policy_box(policy_interpretation(pareto_df, summary_df), kind="success")
else:
    st.info("Nhấn **Chạy tối ưu Pareto Bài 7** để sinh tập Pareto bằng NSGA-II hoặc random feasible fallback.")
    st.header("🏛️ 5. Diễn giải chính sách")
    policy_box("Diễn giải Pareto trade-off sẽ xuất hiện sau khi chạy mô hình.")
