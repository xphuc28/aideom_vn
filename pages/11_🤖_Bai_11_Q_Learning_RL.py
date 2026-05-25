from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bai11_q_learning import (
    ACTION_ALLOCATIONS,
    ACTION_NAMES,
    MODULE_TITLE,
    SAMPLE_STATES,
    module_status,
    state_to_tuple,
    train_q_learning,
)
from src.data_loader import load_macro
from src.ui import apply_dashboard_style, policy_box, render_page_badges, render_sidebar
from src.visualization import download_dataframe_button, render_kpi_cards


st.set_page_config(page_title=MODULE_TITLE, page_icon="🤖", layout="wide")
apply_dashboard_style()
render_sidebar(MODULE_TITLE, "Khó", "Q-learning MDP")


@st.cache_data(show_spinner=False)
def get_data():
    return load_macro()


@st.cache_resource(show_spinner=True)
def train_cached(n_episodes: int, alpha: float, gamma: float, seed: int):
    return train_q_learning(n_episodes=n_episodes, alpha=alpha, gamma=gamma, seed=seed)


def sample_policy_table(Q) -> pd.DataFrame:
    rows = []
    for label, state_labels in SAMPLE_STATES.items():
        state = state_to_tuple(state_labels)
        action = int(Q[state].argmax())
        rows.append(
            {
                "sample_state": label,
                "GDP_growth": state_labels[0],
                "Digital_index": state_labels[1],
                "AI_capacity": state_labels[2],
                "Unemployment_risk": state_labels[3],
                "recommended_action": action,
                "action_name": ACTION_NAMES[action],
                "allocation": ACTION_ALLOCATIONS[action],
                "q_value": float(Q[state + (action,)]),
            }
        )
    return pd.DataFrame(rows)


def policy_interpretation(sample_df: pd.DataFrame, comparison_df: pd.DataFrame) -> list[str]:
    best_policy = comparison_df.sort_values("mean_reward", ascending=False).iloc[0]
    crisis_action = sample_df[sample_df["sample_state"] == "GDP low, D low, AI low, U high"].iloc[0]
    advanced_action = sample_df[sample_df["sample_state"] == "GDP high, D high, AI high, U low"].iloc[0]

    return [
        f"Policy có mean reward cao nhất trong mô phỏng là {best_policy['policy_type']} với reward trung bình {best_policy['mean_reward']:.3f}.",
        f"Khi GDP thấp, số hóa thấp, AI thấp và thất nghiệp cao, agent khuyến nghị {crisis_action['action_name']}.",
        f"Khi nền kinh tế đã mạnh và AI cao, agent khuyến nghị {advanced_action['action_name']}, phản ánh khả năng thích nghi theo trạng thái.",
        "Q-learning không đưa ra một công thức chính sách cố định; nó chọn hành động theo trạng thái vĩ mô, năng lực số, năng lực AI và rủi ro thất nghiệp.",
    ]


macro = get_data()

st.title(MODULE_TITLE)
render_page_badges("Khó", "Q-learning MDP")
st.info(module_status())

st.header("🧭 1. Bối cảnh")
st.write(
    "Bài 11 mô phỏng chính sách kinh tế thích nghi bằng Q-learning. Nhà hoạch định quan sát "
    "trạng thái gồm tăng trưởng GDP, chỉ số số hóa, năng lực AI và rủi ro thất nghiệp, sau đó "
    "chọn một trong năm gói chính sách phân bổ nguồn lực."
)

st.header("🧮 2. Mô hình toán học")
st.latex(r"Q(s,a) \leftarrow Q(s,a)+\alpha\left[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right]")
st.latex(r"R=0.40\Delta GDP-0.25\Delta Unemployment-0.20CyberRisk-0.15Emission")
st.write("State space có `3^4 = 81` trạng thái và 5 hành động chính sách.")

st.header("🧾 3. Dữ liệu/tham số")
param_cols = st.columns(4)
episodes = int(param_cols[0].number_input("Episodes", min_value=100, max_value=10000, value=3000, step=100))
alpha = param_cols[1].slider("alpha", 0.01, 0.50, 0.10, 0.01)
gamma = param_cols[2].slider("gamma", 0.50, 0.99, 0.95, 0.01)
seed = int(param_cols[3].number_input("seed", min_value=1, max_value=99999, value=42, step=1))

actions_df = pd.DataFrame(
    [
        {
            "action": action,
            "action_name": ACTION_NAMES[action],
            "traditional": allocation[0],
            "digital": allocation[1],
            "ai": allocation[2],
            "inclusive": allocation[3],
        }
        for action, allocation in ACTION_ALLOCATIONS.items()
    ]
)
st.subheader("Action mapping")
st.dataframe(actions_df, use_container_width=True)
download_dataframe_button(actions_df, "bai_11_action_mapping.csv")

with st.expander("Dữ liệu macro tham khảo"):
    st.dataframe(macro, use_container_width=True)
    download_dataframe_button(macro, "bai_11_macro_context.csv")

st.header("📊 4. Kết quả")
if st.button("Huấn luyện Q-learning", type="primary"):
    result = train_cached(episodes, alpha, gamma, seed)
    rewards = result["rewards"]
    smoothed = result["smoothed_rewards"]
    Q = result["Q"]
    policy_df = result["policy_df"]
    comparison_df = result["comparison_df"]
    sample_df = sample_policy_table(Q)
    mean_last_100 = sum(rewards[-100:]) / min(100, len(rewards))

    render_kpi_cards(
        {
            "Q shape": str(Q.shape),
            "Mean reward cuối 100 ep": f"{mean_last_100:.3f}",
            "Best baseline": comparison_df.sort_values("mean_reward", ascending=False).iloc[0]["policy_type"],
        }
    )

    curve_df = pd.DataFrame(
        {
            "episode": list(range(1, len(rewards) + 1)),
            "reward": rewards,
            "smoothed_reward": smoothed,
        }
    )
    curve_long = curve_df.melt(id_vars="episode", value_vars=["reward", "smoothed_reward"], var_name="series", value_name="value")
    fig_curve = px.line(
        curve_long,
        x="episode",
        y="value",
        color="series",
        title="Learning curve per-episode và smoothed",
    )
    st.plotly_chart(fig_curve, use_container_width=True)

    st.subheader("Policy cho 5 trạng thái mẫu")
    st.dataframe(sample_df, use_container_width=True)
    download_dataframe_button(sample_df, "bai_11_sample_policy.csv")

    st.subheader("So sánh learned policy với baseline")
    st.dataframe(comparison_df, use_container_width=True)
    download_dataframe_button(comparison_df, "bai_11_policy_comparison.csv")
    fig_compare = px.bar(
        comparison_df.sort_values("mean_reward", ascending=True),
        x="mean_reward",
        y="policy_type",
        orientation="h",
        error_x="std_reward",
        title="Mean reward theo policy",
        labels={"mean_reward": "Mean reward", "policy_type": "Policy"},
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    with st.expander("Full learned policy 81 states"):
        st.dataframe(policy_df, use_container_width=True)
        download_dataframe_button(policy_df, "bai_11_full_policy_81_states.csv")

    st.header("🏛️ 5. Diễn giải chính sách")
    policy_box(policy_interpretation(sample_df, comparison_df), kind="success")
else:
    st.info("Nhấn **Huấn luyện Q-learning** để huấn luyện và xem policy thích nghi.")
    st.header("🏛️ 5. Diễn giải chính sách")
    policy_box("Diễn giải chính sách sẽ xuất hiện sau khi huấn luyện.")
