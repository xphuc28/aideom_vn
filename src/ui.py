"""Shared Streamlit UI helpers for the AIDEOM-VN dashboard.

The project pages are model-heavy, so visual consistency is centralized here:
dark dashboard styling, sidebar identity, difficulty badges, section headings,
and policy interpretation boxes. These helpers deliberately avoid changing any
model logic; they only shape how existing results are presented.
"""

from __future__ import annotations

from collections.abc import Iterable

import plotly.io as pio
import streamlit as st


def apply_dashboard_style() -> None:
    """Inject a restrained dark dashboard skin used across all Streamlit pages."""
    pio.templates.default = "plotly_dark"
    st.markdown(
        """
        <style>
        :root {
            --aideom-bg: #060711;
            --aideom-panel: #10121e;
            --aideom-panel-soft: #151827;
            --aideom-border: #2a2e40;
            --aideom-text: #f8fafc;
            --aideom-muted: #9ca3af;
            --aideom-accent: #a78bfa;
            --aideom-accent-2: #22d3ee;
            --aideom-good: #34d399;
            --aideom-warn: #fbbf24;
        }

        .stApp {
            background: radial-gradient(circle at top right, rgba(167, 139, 250, 0.12), transparent 30%),
                        linear-gradient(180deg, #060711 0%, #0a0c16 100%);
            color: var(--aideom-text);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #080914 0%, #0d0f1a 100%);
            border-right: 1px solid var(--aideom-border);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: var(--aideom-text);
        }

        .block-container {
            padding-top: 2.1rem;
            padding-bottom: 4rem;
            max-width: 1480px;
        }

        h1 {
            font-size: clamp(2.4rem, 4vw, 4.2rem) !important;
            letter-spacing: 0 !important;
            line-height: 1.03 !important;
            margin-bottom: 0.35rem !important;
        }

        h2, h3 {
            letter-spacing: 0 !important;
        }

        div[data-testid="stMetric"],
        div[data-testid="stDataFrame"],
        div[data-testid="stPlotlyChart"],
        div[data-testid="stExpander"] {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            background: rgba(16, 18, 30, 0.84);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.24);
        }

        div[data-testid="stMetric"] {
            padding: 1rem 1.05rem;
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--aideom-muted) !important;
            font-size: 0.86rem;
        }

        div[data-testid="stMetricValue"] {
            color: var(--aideom-text) !important;
        }

        div[data-testid="stPlotlyChart"] {
            padding: 0.4rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
            background: rgba(16, 18, 30, 0.56);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 999px;
            padding: 0.35rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            color: #cbd5e1;
            padding: 0.55rem 0.9rem;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(167, 139, 250, 0.18);
            color: #ffffff;
        }

        div.stButton > button,
        div.stDownloadButton > button {
            border: 1px solid rgba(167, 139, 250, 0.52);
            border-radius: 999px;
            background: linear-gradient(135deg, #a78bfa 0%, #7c3aed 100%);
            color: #f8fafc;
            font-weight: 700;
            min-height: 2.7rem;
        }

        div.stDownloadButton > button {
            background: rgba(16, 18, 30, 0.9);
        }

        div[data-testid="stAlert"] {
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.22);
        }

        .aideom-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin: 0.4rem 0 1.1rem 0;
        }

        .aideom-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border: 1px solid rgba(167, 139, 250, 0.35);
            background: rgba(167, 139, 250, 0.13);
            color: #ede9fe;
            border-radius: 999px;
            padding: 0.35rem 0.7rem;
            font-size: 0.82rem;
            font-weight: 700;
        }

        .aideom-section {
            margin: 1.25rem 0 0.7rem 0;
            color: #f8fafc;
            font-size: 1.15rem;
            font-weight: 800;
        }

        .aideom-sidebar-title {
            font-size: 1.35rem;
            font-weight: 900;
            margin-bottom: 0.2rem;
        }

        .aideom-sidebar-card {
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 16px;
            background: rgba(16, 18, 30, 0.72);
            padding: 0.85rem;
            margin: 0.8rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(module_name: str, difficulty: str, model_type: str) -> None:
    """Render a consistent sidebar identity block for a module page."""
    with st.sidebar:
        st.markdown('<div class="aideom-sidebar-title">AIDEOM-VN</div>', unsafe_allow_html=True)
        st.caption("Dashboard mô hình ra quyết định")
        st.markdown(
            f"""
            <div class="aideom-sidebar-card">
              <div style="font-weight:800; margin-bottom:0.35rem;">{module_name}</div>
              <div style="color:#9ca3af; font-size:0.9rem;">Độ khó: <b>{difficulty}</b></div>
              <div style="color:#9ca3af; font-size:0.9rem;">Mô hình: <b>{model_type}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("**Điều hướng**")
        st.caption("Chọn page ở menu Streamlit để chuyển giữa 12 bài.")


def render_page_badges(difficulty: str, model_type: str, status: str = "Sẵn sàng chạy") -> None:
    """Render compact badges below each page title."""
    st.markdown(
        f"""
        <div class="aideom-badges">
          <span class="aideom-badge">Độ khó: {difficulty}</span>
          <span class="aideom-badge">Mô hình: {model_type}</span>
          <span class="aideom-badge">Trạng thái: {status}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(icon: str, title: str) -> None:
    """Render a section heading with a stable icon prefix."""
    st.markdown(f'<div class="aideom-section">{icon} {title}</div>', unsafe_allow_html=True)


def policy_box(
    lines: Iterable[str] | str,
    title: str = "Diễn giải chính sách",
    kind: str = "info",
) -> None:
    """Render policy interpretation in a Streamlit info/success container."""
    if isinstance(lines, str):
        body = lines
    else:
        bullet_lines = [str(line).strip() for line in lines if str(line).strip()]
        body = "\n".join(f"- {line}" for line in bullet_lines)

    message = f"**{title}**\n\n{body}" if body else f"**{title}**"
    if kind == "success":
        st.success(message)
    else:
        st.info(message)
