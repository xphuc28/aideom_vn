from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bai01_cobb_douglas import DEFAULT_PARAMS, MODULE_TITLE, module_status, run_bai01
from src.assignment_ui import render_assignment_answers
from src.data_loader import load_macro
from src.ui import (
    apply_dashboard_style,
    policy_box,
    render_page_badges,
    render_sidebar,
)
from src.visualization import download_dataframe_button, render_kpi_cards


st.set_page_config(page_title=MODULE_TITLE, page_icon="📈", layout="wide")
apply_dashboard_style()
render_sidebar(MODULE_TITLE, "Trung bình", "Cobb-Douglas mở rộng")


@st.cache_data(show_spinner=False)
def get_data():
    return load_macro()


@st.cache_data(show_spinner=False)
def run_model(data, params):
    return run_bai01(data, params)


def policy_interpretation(results: dict[str, object]) -> list[str]:
    result_df = results["result_df"]
    contribution_df = results["contribution_df"]
    forecast_df = results["forecast_2030_df"]
    mape = float(results["mape"])
    tfp_avg = float(result_df["TFP_A"].mean())

    leading_row = contribution_df.reindex(
        contribution_df["avg_contribution_pct_points"].abs().sort_values(ascending=False).index
    ).iloc[0]
    base_y = float(forecast_df["Y_forecast"].iloc[0])
    target_y = float(forecast_df["Y_forecast"].iloc[-1])
    forecast_growth = (target_y / base_y - 1.0) * 100.0 if base_y else 0.0

    fit_text = (
        "Mô hình bám dữ liệu tốt, phù hợp dùng làm khung mô phỏng chính sách."
        if mape < 5
        else "Sai số còn đáng kể, nên xem kết quả như mô phỏng định hướng thay vì dự báo điểm."
    )
    tfp_text = (
        "TFP bình quân dương cho thấy phần tăng trưởng ngoài các yếu tố K, L, D, AI, H vẫn quan trọng."
        if tfp_avg > 0
        else "TFP bình quân thấp, cần kiểm tra lại thang đo dữ liệu và giả định hệ số."
    )

    return [
        fit_text,
        tfp_text,
        f"Yếu tố đóng góp bình quân nổi bật nhất là {leading_row['factor']} "
        f"với khoảng {leading_row['avg_contribution_pct_points']:.2f} điểm phần trăm log mỗi năm.",
        f"Kịch bản đến 2030 cho GDP mô phỏng tăng khoảng {forecast_growth:.1f}% so với năm gốc, "
        "với giả định kinh tế số đạt 30%, AI đạt 100 nghìn doanh nghiệp và H đạt 35%.",
    ]


def assignment_answers(results: dict[str, object]):
    """Answer the programming and policy questions stated in Bài 1."""
    result_df = results["result_df"]
    contribution_df = results["contribution_df"]
    forecast_df = results["forecast_2030_df"]
    mape = float(results["mape"])

    first_tfp = float(result_df["TFP_A"].iloc[0])
    last_tfp = float(result_df["TFP_A"].iloc[-1])
    tfp_change = (last_tfp / first_tfp - 1.0) * 100.0
    max_error = result_df.loc[result_df["error_pct"].abs().idxmax()]
    contribution_text = ", ".join(
        f"{row.factor}={row.avg_contribution_pct_points:.2f} điểm %"
        for row in contribution_df.itertuples()
    )
    new_factors = contribution_df[contribution_df["factor"].isin(["D", "AI", "H"])]
    leading_new = new_factors.loc[new_factors["avg_contribution_pct_points"].idxmax()]
    target_2030 = forecast_df.loc[forecast_df["year"] == 2030].iloc[0]

    programming = [
        {
            "code": "Câu 1.4.1",
            "question": "Ước lượng TFP A_t từng năm và bình luận xu hướng.",
            "answer": (
                f"TFP tăng từ {first_tfp:.4f} năm {int(result_df['year'].iloc[0])} "
                f"lên {last_tfp:.4f} năm {int(result_df['year'].iloc[-1])}, "
                f"tương đương tăng {tfp_change:.1f}% trong cả giai đoạn. "
                "Dù có năm giảm nhẹ, xu hướng cuối kỳ là đi lên."
            ),
            "evidence": "Bảng Kết quả fit, cột TFP_A và biểu đồ TFP theo năm.",
        },
        {
            "code": "Câu 1.4.2",
            "question": "Tính GDP dự báo từ A trung bình và báo cáo MAPE.",
            "answer": (
                f"MAPE của mô hình là {mape:.2f}%. Sai số tuyệt đối lớn nhất xuất hiện "
                f"năm {int(max_error['year'])}, bằng {abs(float(max_error['error_pct'])):.2f}%. "
                "Vì vậy mô hình phù hợp cho mô phỏng định hướng, nhưng không nên coi là dự báo điểm chính thức."
            ),
            "evidence": "Bảng Kết quả fit, các cột Y_actual, Y_hat, error_pct.",
        },
        {
            "code": "Câu 1.4.3",
            "question": "Phân rã đóng góp tăng trưởng của K, L, D, AI, H và TFP.",
            "answer": f"Đóng góp bình quân theo output hiện tại: {contribution_text}.",
            "evidence": "Bảng Đóng góp và biểu đồ Đóng góp tăng trưởng bình quân.",
        },
        {
            "code": "Câu 1.4.4",
            "question": "Mô phỏng kịch bản D=30%, AI=100 nghìn DN, H=35% đến 2030.",
            "answer": (
                f"GDP mô phỏng năm 2030 là {float(target_2030['Y_forecast']):,.1f}. "
                f"Tại năm đích, TFP={float(target_2030['TFP_A']):.4f}, "
                f"K={float(target_2030['K']):,.1f}, L={float(target_2030['L']):,.1f}."
            ),
            "evidence": "Bảng Forecast 2030, dòng năm 2030.",
        },
    ]
    policy = [
        {
            "code": "Câu 1.5a",
            "question": "TFP tăng hay giảm và nói gì về chất lượng tăng trưởng?",
            "answer": (
                f"TFP cuối kỳ cao hơn đầu kỳ {tfp_change:.1f}%, nên mô hình ghi nhận chất lượng tăng trưởng "
                "được cải thiện về cuối giai đoạn. Tuy nhiên TFP là phần dư của mô hình và còn phụ thuộc cách đo K, L, D, AI, H."
            ),
            "evidence": "Chuỗi TFP_A 2020-2025.",
        },
        {
            "code": "Câu 1.5b",
            "question": "Trong D, AI, H, yếu tố nào đóng góp nhiều nhất?",
            "answer": (
                f"{leading_new['factor']} có đóng góp bình quân lớn nhất trong ba yếu tố mới, "
                f"bằng {leading_new['avg_contribution_pct_points']:.2f} điểm % log/năm theo bộ dữ liệu và hệ số hiện tại."
            ),
            "evidence": "Lọc ba dòng D, AI, H trong contribution_df.",
        },
        {
            "code": "Câu 1.5c",
            "question": "Mục tiêu kinh tế số 30% GDP năm 2030 có khả thi không?",
            "answer": (
                "Mô hình cho thấy kịch bản D=30% tạo được một quỹ đạo GDP dương, nhưng không tự chứng minh tính khả thi "
                "thể chế hay ngân sách vì D=30% là giả định đầu vào. Cần bổ sung ràng buộc vốn đầu tư, nhân lực, hạ tầng "
                "dữ liệu, an ninh mạng và tốc độ hấp thụ của doanh nghiệp."
            ),
            "evidence": "D=30% được đặt ngoại sinh trong forecast_2030_df.",
        },
    ]
    return programming, policy


macro = get_data()

st.title(MODULE_TITLE)
render_page_badges("Trung bình", "Cobb-Douglas mở rộng")
st.info(module_status())

st.header("🧭 1. Bối cảnh")
st.write(
    "Bài 1 mở rộng hàm sản xuất Cobb-Douglas truyền thống bằng cách thêm ba yếu tố "
    "đại diện cho chuyển đổi số, mức ứng dụng AI và vốn nhân lực. Mục tiêu là ước "
    "lượng TFP, so sánh GDP thực tế với GDP mô phỏng, phân rã tăng trưởng và tạo "
    "kịch bản đến năm 2030."
)

st.header("🧮 2. Mô hình toán học")
st.latex(r"Y_t = A_t K_t^\alpha L_t^\beta D_t^\gamma AI_t^\delta H_t^\theta")
st.write(
    "Trong đó `Y` là GDP, `K` là vốn, `L` là lao động, `D` là mức số hóa, "
    "`AI` là mức ứng dụng AI, `H` là vốn nhân lực và `A` là năng suất nhân tố tổng hợp."
)

st.header("🧾 3. Dữ liệu/tham số")
st.subheader("Tham số hệ số co giãn")
input_cols = st.columns(5)
alpha = input_cols[0].slider("alpha - vốn K", 0.0, 1.0, DEFAULT_PARAMS["alpha"], 0.01)
beta = input_cols[1].slider("beta - lao động L", 0.0, 1.0, DEFAULT_PARAMS["beta"], 0.01)
gamma = input_cols[2].slider("gamma - số hóa D", 0.0, 1.0, DEFAULT_PARAMS["gamma"], 0.01)
delta = input_cols[3].slider("delta - AI", 0.0, 1.0, DEFAULT_PARAMS["delta"], 0.01)
theta = input_cols[4].slider("theta - vốn nhân lực H", 0.0, 1.0, DEFAULT_PARAMS["theta"], 0.01)

params = {
    "alpha": alpha,
    "beta": beta,
    "gamma": gamma,
    "delta": delta,
    "theta": theta,
}
coef_sum = sum(params.values())
st.write(f"**Tổng hệ số:** `{coef_sum:.2f}`")
if abs(coef_sum - 1.0) > 0.01:
    st.warning("Tổng hệ số khác 1. Mô hình khi đó không còn giả định hiệu suất không đổi theo quy mô.")

st.subheader("Dữ liệu macro")
st.dataframe(macro, use_container_width=True)
download_dataframe_button(macro, "bai_1_macro_input.csv")

st.header("📊 4. Kết quả")
bai01_results = None
if st.button("Chạy Bài 1", type="primary"):
    try:
        bai01_results = run_model(macro, params)
        result_df = bai01_results["result_df"]
        growth_df = bai01_results["growth_df"]
        contribution_df = bai01_results["contribution_df"]
        forecast_df = bai01_results["forecast_2030_df"]

        render_kpi_cards(
            {
                "MAPE": f"{float(bai01_results['mape']):.2f}%",
                "TFP trung bình": f"{result_df['TFP_A'].mean():.4f}",
                "GDP forecast 2030": f"{forecast_df['Y_forecast'].iloc[-1]:,.1f}",
            }
        )

        chart_cols = st.columns(2)
        with chart_cols[0]:
            fig_tfp = px.line(
                result_df,
                x="year",
                y="TFP_A",
                markers=True,
                title="TFP theo năm",
                labels={"year": "Năm", "TFP_A": "TFP A_t"},
            )
            st.plotly_chart(fig_tfp, use_container_width=True)

        with chart_cols[1]:
            fig_fit = go.Figure()
            fig_fit.add_trace(
                go.Scatter(
                    x=result_df["year"],
                    y=result_df["Y_actual"],
                    mode="lines+markers",
                    name="GDP thực tế",
                )
            )
            fig_fit.add_trace(
                go.Scatter(
                    x=result_df["year"],
                    y=result_df["Y_hat"],
                    mode="lines+markers",
                    name="GDP dự báo",
                )
            )
            fig_fit.update_layout(
                title="GDP thực tế vs dự báo",
                xaxis_title="Năm",
                yaxis_title="GDP",
                margin=dict(l=20, r=20, t=60, b=20),
            )
            st.plotly_chart(fig_fit, use_container_width=True)

        chart_cols = st.columns(2)
        with chart_cols[0]:
            fig_contrib = px.bar(
                contribution_df,
                x="factor",
                y="avg_contribution_pct_points",
                title="Đóng góp tăng trưởng bình quân",
                labels={
                    "factor": "Yếu tố",
                    "avg_contribution_pct_points": "Điểm % log/năm",
                },
            )
            st.plotly_chart(fig_contrib, use_container_width=True)

        with chart_cols[1]:
            fig_forecast = go.Figure()
            fig_forecast.add_trace(
                go.Scatter(
                    x=result_df["year"],
                    y=result_df["Y_actual"],
                    mode="lines+markers",
                    name="GDP thực tế",
                )
            )
            fig_forecast.add_trace(
                go.Scatter(
                    x=forecast_df["year"],
                    y=forecast_df["Y_forecast"],
                    mode="lines+markers",
                    name="GDP forecast đến 2030",
                )
            )
            fig_forecast.update_layout(
                title="GDP forecast đến 2030",
                xaxis_title="Năm",
                yaxis_title="GDP",
                margin=dict(l=20, r=20, t=60, b=20),
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

        st.subheader("Bảng kết quả")
        tab_result, tab_growth, tab_contrib, tab_forecast = st.tabs(
            ["Kết quả fit", "Phân rã tăng trưởng", "Đóng góp", "Forecast 2030"]
        )
        with tab_result:
            st.dataframe(result_df, use_container_width=True)
            download_dataframe_button(result_df, "bai_1_result.csv")
        with tab_growth:
            st.dataframe(growth_df, use_container_width=True)
            download_dataframe_button(growth_df, "bai_1_growth_decomposition.csv")
        with tab_contrib:
            st.dataframe(contribution_df, use_container_width=True)
            download_dataframe_button(contribution_df, "bai_1_contribution.csv")
        with tab_forecast:
            st.dataframe(forecast_df, use_container_width=True)
            download_dataframe_button(forecast_df, "bai_1_forecast_2030.csv")

    except Exception as exc:
        st.error(f"Không chạy được Bài 1: {exc}")
else:
    st.info("Nhấn **Chạy Bài 1** để ước lượng TFP, phân rã tăng trưởng và tạo forecast 2030.")

st.header("🏛️ 5. Diễn giải chính sách")
if bai01_results is None:
    policy_box("Diễn giải tự động sẽ xuất hiện sau khi chạy mô hình.")
else:
    policy_box(policy_interpretation(bai01_results), kind="success")
    programming_answers, discussion_answers = assignment_answers(bai01_results)
    render_assignment_answers(programming_answers, discussion_answers)
