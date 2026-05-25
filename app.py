"""Streamlit home page for the AIDEOM-VN dashboard."""

from __future__ import annotations

import streamlit as st

from src.data_loader import data_status, load_macro, load_regions, load_sectors
from src.ui import apply_dashboard_style, render_page_badges, render_sidebar, section_title
from src.visualization import download_dataframe_button, render_kpi_cards


st.set_page_config(
    page_title="AIDEOM-VN",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_dashboard_style()
render_sidebar("Trang chủ", "Nền tảng", "Data audit")


@st.cache_data(show_spinner=False)
def get_data_status():
    """Load required CSV status for the dashboard landing page."""
    return data_status()


@st.cache_data(show_spinner=False)
def load_dataset(dataset_name: str):
    """Load one required dataset by logical name."""
    loaders = {
        "macro": load_macro,
        "sectors": load_sectors,
        "regions": load_regions,
    }
    return loaders[dataset_name]()


def main() -> None:
    status = get_data_status()

    st.title("AIDEOM-VN")
    render_page_badges("Nền tảng", "Dashboard Streamlit", "Kiểm tra dữ liệu")
    st.caption("Dashboard đồ án môn Các mô hình ra quyết định")

    st.markdown(
        """
        AIDEOM-VN là dashboard Streamlit phục vụ đồ án mô hình ra quyết định
        cho bài toán phát triển kinh tế số, AI và phân bổ nguồn lực tại Việt Nam.
        Trang chủ hiện kiểm tra nền dữ liệu đầu vào trước khi triển khai các
        mô hình tối ưu ở từng page.
        """
    )

    present_count = sum(1 for item in status.values() if item["exists"])
    render_kpi_cards(
        {
            "CSV bắt buộc": len(status),
            "CSV đã có": present_count,
            "CSV còn thiếu": len(status) - present_count,
        }
    )

    section_title("🗂️", "Kiểm tra 3 file CSV trong data/")
    st.write(
        "Project cần đủ ba file: `vietnam_macro_2020_2025.csv`, "
        "`vietnam_sectors_2024.csv`, `vietnam_regions_2024.csv`."
    )

    labels = {
        "macro": "Vĩ mô 2020-2025",
        "sectors": "Ngành kinh tế 2024",
        "regions": "Vùng kinh tế 2024",
    }

    for dataset_name, item in status.items():
        st.markdown(f"### {labels[dataset_name]}")

        if not item["exists"]:
            st.error(
                f"Thiếu `{item['filename']}`. Hãy copy file này vào thư mục "
                f"`{item['relative_path']}` rồi chạy lại `streamlit run app.py`."
            )
            continue

        if item["error"]:
            st.error(f"Không đọc được `{item['filename']}`: {item['error']}")
            continue

        df = load_dataset(dataset_name)
        shape = item["shape"]
        st.success(f"Đã tìm thấy `{item['relative_path']}`")
        st.write(f"**Shape:** `{shape[0]} rows x {shape[1]} columns`")
        st.write("**Columns:**")
        st.code(", ".join(item["columns"]), language="text")
        st.write("**Head:**")
        st.dataframe(df.head(), use_container_width=True)
        download_dataframe_button(df, item["filename"])


if __name__ == "__main__":
    main()
