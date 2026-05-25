"""Reusable Plotly visualization helpers."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui import apply_dashboard_style


def render_kpi_cards(metrics: dict) -> None:
    """Render a row of Streamlit KPI cards from a label-value dictionary."""
    if not metrics:
        return

    apply_dashboard_style()
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics.items()):
        column.metric(label, value)


def download_dataframe_button(df: pd.DataFrame, filename: str) -> None:
    """Render a CSV download button for a dataframe."""
    st.download_button(
        label=f"Tải xuống CSV: {filename}",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
    )


def macro_growth_line(macro: pd.DataFrame) -> go.Figure:
    """Create a GDP growth line chart for the home page."""
    fig = px.line(
        macro,
        x="year",
        y="GDP_growth_pct",
        markers=True,
        title="Tăng trưởng GDP Việt Nam 2020-2025",
        labels={"year": "Năm", "GDP_growth_pct": "Tăng trưởng GDP (%)"},
    )
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig


def sector_digital_bar(sectors: pd.DataFrame) -> go.Figure:
    """Create a sector digital index bar chart for the home page."""
    fig = px.bar(
        sectors.sort_values("digital_index_0_100", ascending=True),
        x="digital_index_0_100",
        y="sector_name_vi",
        orientation="h",
        title="Chỉ số số hóa theo ngành năm 2024",
        labels={"digital_index_0_100": "Chỉ số số hóa", "sector_name_vi": "Ngành"},
    )
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=520)
    return fig
