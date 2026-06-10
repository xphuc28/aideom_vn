from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bai10_stochastic import (
    ITEMS,
    MODULE_TITLE,
    compute_vss_evpi,
    module_status,
    robust_minimax_regret,
    scenario_table,
    solve_deterministic_scenario,
    solve_expected_value,
    solve_stochastic_pulp,
)
from src.data_loader import load_macro
from src.ui import (
    apply_dashboard_style,
    policy_box,
    render_assignment_answers,
    render_page_badges,
    render_sidebar,
)
from src.visualization import download_dataframe_button, render_kpi_cards


st.set_page_config(page_title=MODULE_TITLE, page_icon="🎲", layout="wide")
apply_dashboard_style()
render_sidebar(MODULE_TITLE, "Khó", "Stochastic Programming")


@st.cache_data(show_spinner=False)
def get_data():
    return load_macro(), scenario_table()


@st.cache_data(show_spinner=True)
def run_sp_cached():
    return solve_stochastic_pulp()


@st.cache_data(show_spinner=True)
def deterministic_cached():
    rows = []
    details = {}
    for scenario in ["s1", "s2", "s3", "s4"]:
        result = solve_deterministic_scenario(scenario)
        details[scenario] = result
        rows.append(
            {
                "scenario": scenario,
                "objective": result["objective"],
                "status": result["status"],
                "first_stage_budget": result["first_stage_df"]["allocation"].sum(),
                "second_stage_budget": result["second_stage_df"]["allocation"].sum(),
            }
        )
    return pd.DataFrame(rows), details


@st.cache_data(show_spinner=True)
def expected_cached():
    return solve_expected_value()


@st.cache_data(show_spinner=True)
def robust_cached():
    return robust_minimax_regret()


def policy_interpretation(result: dict[str, object], deterministic_df: pd.DataFrame, vss_evpi_df: pd.DataFrame) -> list[str]:
    first = result["first_stage_df"].sort_values("allocation", ascending=False).iloc[0]
    second = (
        result["second_stage_df"]
        .groupby("item_name", as_index=False)["allocation"]
        .sum()
        .sort_values("allocation", ascending=False)
        .iloc[0]
    )
    best_det = deterministic_df.sort_values("objective", ascending=False).iloc[0]
    vss = 0.0
    evpi = 0.0
    if not vss_evpi_df.empty:
        metric_map = dict(zip(vss_evpi_df["metric"], vss_evpi_df["value"]))
        vss = metric_map.get("VSS", 0.0)
        evpi = metric_map.get("EVPI", 0.0)

    return [
        f"First-stage ưu tiên lớn nhất là {first['item_name']} với {first['allocation']:,.0f}.",
        f"Recourse second-stage dùng nhiều nhất cho {second['item_name']} với tổng {second['allocation']:,.0f} qua các kịch bản.",
        f"Kịch bản deterministic có objective cao nhất là {best_det['scenario']} với {best_det['objective']:,.1f}.",
        f"VSS={vss:,.2f} và EVPI={evpi:,.2f}; nếu hai giá trị thấp, quyết định first-stage khá ổn định trước bất định trong bộ hệ số này.",
    ]


def assignment_answers(result, deterministic_df, expected_result, vss_evpi_df, robust_result):
    """Answer stochastic-programming requirements from computed scenarios."""
    first = result["first_stage_df"]
    expected_first = expected_result["first_stage_df"]
    first_map = dict(zip(first["item"], first["allocation"]))
    expected_map = dict(zip(expected_first["item"], expected_first["allocation"]))
    metrics = dict(zip(vss_evpi_df["metric"], vss_evpi_df["value"]))
    robust_first = robust_result["first_stage_df"]
    robust_map = dict(zip(robust_first["item"], robust_first["allocation"]))

    programming = [
        {
            "code": "Câu 10.5.1",
            "question": "Mô hình hóa first-stage/second-stage và báo cáo quyết định tối ưu.",
            "answer": (
                f"Phiên bản deploy dùng PuLP/CBC hoặc SciPy fallback thay cho Pyomo/GLPK để bảo đảm chạy được. "
                f"Status={result['status']}, objective={result['objective']}. First-stage: "
                + ", ".join(f"{item}={value:,.0f}" for item, value in first_map.items())
                + "."
            ),
            "evidence": "Bảng First-stage và Second-stage recourse.",
        },
        {
            "code": "Câu 10.5.2",
            "question": "So sánh deterministic, expected-value và stochastic SP.",
            "answer": (
                f"Expected-value first-stage: {expected_map}. SP first-stage: {first_map}. "
                f"Objective deterministic theo bốn kịch bản nằm từ {deterministic_df['objective'].min():,.1f} "
                f"đến {deterministic_df['objective'].max():,.1f}."
            ),
            "evidence": "Bảng deterministic từng kịch bản và expected-value result.",
        },
        {
            "code": "Câu 10.5.3",
            "question": "Tính VSS và EVPI.",
            "answer": (
                f"VSS={metrics.get('VSS', float('nan')):,.2f}; EVPI={metrics.get('EVPI', float('nan')):,.2f}. "
                "VSS đo lợi ích của quyết định có xét bất định; EVPI đo giá trị tối đa của thông tin hoàn hảo."
            ),
            "evidence": "Bảng Expected value, VSS và EVPI.",
        },
        {
            "code": "Câu 10.5.4",
            "question": "Robust minimax regret và so sánh với SP.",
            "answer": (
                f"Robust first-stage: {robust_map}. Giá trị max regret/objective được báo cáo là "
                f"{robust_result.get('max_regret', robust_result.get('objective'))}."
            ),
            "evidence": robust_result.get("note", ""),
        },
    ]
    policy = [
        {
            "code": "Câu 10.6a",
            "question": "SP đầu tư H nhiều hay ít hơn lời giải xác định/kỳ vọng?",
            "answer": (
                f"SP phân bổ H={first_map.get('H', 0):,.0f}; expected-value phân bổ H={expected_map.get('H', 0):,.0f}. "
                "Trong bộ hệ số hiện tại, chênh lệch này cho biết nhân lực có được dùng như lớp đệm trước bất định hay không."
            ),
            "evidence": "So sánh cột allocation của first-stage_df.",
        },
        {
            "code": "Câu 10.6b",
            "question": "VSS dương nói lên điều gì?",
            "answer": (
                f"VSS hiện bằng {metrics.get('VSS', 0):,.2f}. Nếu bằng 0, bộ kịch bản hiện tại chưa tạo lợi ích đo được "
                "từ việc đổi first-stage so với expected-value; điều đó không có nghĩa tư duy xác suất vô giá trị trong mọi dữ liệu."
            ),
            "evidence": "Metric VSS.",
        },
        {
            "code": "Câu 10.6c",
            "question": "Có đang dưới đầu tư nhân lực số như một hàng hóa bảo hiểm?",
            "answer": (
                "Mô hình hiện chỉ cho bằng chứng nội bộ qua mức H và recourse trong kịch bản xấu; không đủ dữ liệu để kết luận "
                "Việt Nam thực tế đang dưới đầu tư. Có thể dùng sensitivity beta_H và xác suất shock để kiểm tra giả thuyết này."
            ),
            "evidence": "Giới hạn phạm vi mô hình kịch bản.",
        },
    ]
    return programming, policy


macro, scenarios = get_data()

st.title(MODULE_TITLE)
render_page_badges("Khó", "Stochastic Programming")
st.info(module_status())

st.header("🧭 1. Bối cảnh")
st.write(
    "Bài 10 mô hình hóa quyết định hai giai đoạn trong điều kiện bất định. "
    "Giai đoạn 1 chọn phân bổ nền tảng trước khi biết kịch bản; giai đoạn 2 điều chỉnh recourse "
    "theo từng kịch bản tăng trưởng/chuyển đổi số."
)

st.header("🧮 2. Mô hình toán học")
st.latex(
    r"\max \sum_{j \in J}\beta_jx_j + \sum_{s \in S}p_s\sum_{j \in J}\beta_{sj}y_{sj}"
)
st.latex(
    r"""
    \begin{aligned}
    \sum_j x_j &\le 65000\\
    \sum_j y_{sj} &\le 15000,\quad \forall s\\
    y_{s,AI} &\le 0.5x_H,\quad \forall s\\
    x_j,y_{sj} &\ge 0
    \end{aligned}
    """
)

st.header("🧾 3. Dữ liệu/tham số")
st.subheader("Scenario table")
st.dataframe(scenarios, use_container_width=True)
download_dataframe_button(scenarios, "bai_10_scenario_table.csv")

with st.expander("Dữ liệu macro tham khảo"):
    st.dataframe(macro, use_container_width=True)
    download_dataframe_button(macro, "bai_10_macro_context.csv")

st.header("📊 4. Kết quả")
if st.button("Chạy quy hoạch ngẫu nhiên hai giai đoạn", type="primary"):
    result = run_sp_cached()
    deterministic_df, deterministic_details = deterministic_cached()
    expected_result = expected_cached()
    vss_evpi_df = compute_vss_evpi()
    robust_result = robust_cached()

    render_kpi_cards(
        {
            "Status": result["status"],
            "Objective SP": "N/A" if result["objective"] is None else f"{result['objective']:,.1f}",
            "First budget": f"{result['first_stage_df']['allocation'].sum():,.0f}",
            "Solver note": "PuLP" if "PuLP" in result["note"] else "Fallback",
        }
    )
    st.caption(result["note"])

    st.subheader("First-stage allocation")
    first_stage_df = result["first_stage_df"]
    st.dataframe(first_stage_df, use_container_width=True)
    download_dataframe_button(first_stage_df, "bai_10_first_stage.csv")

    fig_first = px.bar(
        first_stage_df,
        x="item_name",
        y="allocation",
        color="item",
        title="First-stage allocation x_j",
        labels={"item_name": "Hạng mục", "allocation": "Ngân sách"},
    )
    st.plotly_chart(fig_first, use_container_width=True)

    st.subheader("Second-stage recourse theo scenario")
    second_stage_df = result["second_stage_df"]
    st.dataframe(second_stage_df, use_container_width=True)
    download_dataframe_button(second_stage_df, "bai_10_second_stage.csv")

    fig_second = px.bar(
        second_stage_df,
        x="scenario",
        y="allocation",
        color="item_name",
        title="Second-stage y_sj theo kịch bản",
        labels={"scenario": "Scenario", "allocation": "Ngân sách", "item_name": "Hạng mục"},
    )
    st.plotly_chart(fig_second, use_container_width=True)

    st.subheader("So sánh deterministic từng kịch bản")
    st.dataframe(deterministic_df, use_container_width=True)
    download_dataframe_button(deterministic_df, "bai_10_deterministic_comparison.csv")

    fig_det = px.bar(
        deterministic_df,
        x="scenario",
        y="objective",
        color="scenario",
        title="Objective deterministic theo từng kịch bản",
    )
    fig_det.update_layout(showlegend=False)
    st.plotly_chart(fig_det, use_container_width=True)

    st.subheader("Expected value, VSS và EVPI")
    expected_first = expected_result["first_stage_df"].copy()
    expected_first["model"] = "expected_value"
    st.write(f"Expected-value objective: `{expected_result['objective']}`")
    st.dataframe(vss_evpi_df, use_container_width=True)
    download_dataframe_button(vss_evpi_df, "bai_10_vss_evpi.csv")

    with st.expander("Robust minimax regret optional fallback"):
        st.write(robust_result["note"])
        st.write("Max regret/objective:", robust_result.get("max_regret", robust_result["objective"]))
        st.dataframe(robust_result["first_stage_df"], use_container_width=True)

    st.header("🏛️ 5. Diễn giải chính sách")
    policy_box(policy_interpretation(result, deterministic_df, vss_evpi_df), kind="success")
    programming_answers, discussion_answers = assignment_answers(
        result, deterministic_df, expected_result, vss_evpi_df, robust_result
    )
    render_assignment_answers(
        programming_answers,
        discussion_answers,
        note="Đề gợi ý Pyomo; bản deploy dùng PuLP mặc định để không phụ thuộc GLPK bên ngoài.",
    )
else:
    st.info("Nhấn **Chạy quy hoạch ngẫu nhiên hai giai đoạn** để giải stochastic program, deterministic scenarios và VSS/EVPI.")
    st.header("🏛️ 5. Diễn giải chính sách")
    policy_box("Diễn giải chính sách sẽ xuất hiện sau khi chạy mô hình.")
