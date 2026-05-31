"""Generate an academic PDF report for AIDEOM-VN.

The report includes Vietnamese academic prose, dashboard illustrations, and
result tables/figures generated from the same Bai 12 scenario engine used by
the Streamlit application.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "models"))
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.backends.backend_pdf import PdfPages

from src.scenario_engine import COST_KPIS, allocation_long, recommendation_text, run_all_scenarios


REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PDF_PATH = REPORTS_DIR / "aideom_vn_research_report.pdf"
BUDGET = 50000.0

PRIMARY = "#A78BFA"
SECONDARY = "#22D3EE"
GOOD = "#34D399"
WARN = "#FBBF24"
DANGER = "#F87171"
TEXT = "#111827"
MUTED = "#4B5563"
GRID = "#D1D5DB"


def wrap(text: str, width: int = 96) -> str:
    """Wrap Vietnamese prose for Matplotlib text blocks."""
    return "\n".join(textwrap.wrap(text.strip(), width=width, replace_whitespace=False))


def new_page(figsize: tuple[float, float] = (8.27, 11.69)):
    """Create a portrait A4 page."""
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("white")
    return fig


def page_number(fig, page: int) -> None:
    """Draw report footer and page number."""
    fig.text(0.08, 0.035, "AIDEOM-VN | Báo cáo nghiên cứu", fontsize=8, color=MUTED)
    fig.text(0.92, 0.035, str(page), fontsize=8, color=MUTED, ha="right")


def header(fig, title: str, subtitle: str | None = None) -> None:
    """Draw a consistent report header."""
    fig.text(0.08, 0.94, title, fontsize=17, weight="bold", color=TEXT)
    if subtitle:
        fig.text(0.08, 0.915, subtitle, fontsize=9.3, color=MUTED)
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.895, 0.895], transform=fig.transFigure, color=PRIMARY, lw=1.4))


def text_blocks(fig, blocks: list[tuple[str, str]], start_y: float = 0.86, width: int = 95) -> None:
    """Draw multiple titled prose blocks."""
    y = start_y
    for title, body in blocks:
        fig.text(0.08, y, title, fontsize=12.2, weight="bold", color=TEXT)
        y -= 0.028
        body_wrapped = wrap(body, width)
        fig.text(0.08, y, body_wrapped, fontsize=9.4, color=TEXT, va="top", linespacing=1.35)
        y -= 0.030 * (body_wrapped.count("\n") + 1) + 0.035


def bullet_block(fig, title: str, bullets: list[str], start_y: float = 0.86, width: int = 90) -> None:
    """Draw a titled bullet list."""
    fig.text(0.08, start_y, title, fontsize=12.2, weight="bold", color=TEXT)
    y = start_y - 0.035
    for bullet in bullets:
        line = wrap(bullet, width)
        fig.text(0.10, y, "- " + line.replace("\n", "\n  "), fontsize=9.3, color=TEXT, va="top", linespacing=1.32)
        y -= 0.027 * (line.count("\n") + 1) + 0.018


def rounded_box(ax, xy, w, h, face, edge="#2A2E40", radius=0.02, alpha=1.0):
    """Draw a rounded rectangle in dashboard illustrations."""
    box = patches.FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=patches.BoxStyle("Round", pad=0.012, rounding_size=radius),
        facecolor=face,
        edgecolor=edge,
        linewidth=1,
        alpha=alpha,
    )
    ax.add_patch(box)
    return box


def save_pdf_page(pdf: PdfPages, fig, page: int, png_name: str | None = None) -> int:
    """Save one PDF page and optionally export it as PNG."""
    page_number(fig, page)
    if png_name:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / png_name, dpi=180, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return page + 1


def module_summary_table() -> pd.DataFrame:
    """Return the module summary table."""
    rows = [
        ("1", "Cobb-Douglas AI", "TFP, phân rã tăng trưởng, forecast 2030", "Hàm sản xuất mở rộng"),
        ("2", "LP ngân sách số", "Phân bổ K, D, AI/R&D, H", "Linear Programming"),
        ("3", "Priority ngành", "Xếp hạng 10 ngành ưu tiên", "MCDA/min-max"),
        ("4", "LP ngành-vùng", "Phân bổ vùng-hạng mục có công bằng", "LP fairness"),
        ("5", "MIP dự án", "Chọn danh mục 15 dự án", "Mixed Integer Programming"),
        ("6", "TOPSIS vùng", "Xếp hạng 6 vùng đầu tư AI", "TOPSIS + entropy"),
        ("7", "Pareto", "Đánh đổi GDP, công bằng, phát thải, rủi ro", "NSGA-II/fallback"),
        ("8", "Tối ưu động", "Quỹ đạo đầu tư 2026-2035", "SLSQP"),
        ("9", "Lao động AI", "Việc làm ròng và đào tạo lại", "LP lao động"),
        ("10", "Stochastic SP", "Hai giai đoạn, VSS/EVPI", "Stochastic programming"),
        ("11", "Q-learning", "Chính sách thích nghi", "Reinforcement learning"),
        ("12", "AIDEOM tích hợp", "So sánh 5 kịch bản", "Scenario engine"),
    ]
    return pd.DataFrame(rows, columns=["Bài", "Mô-đun", "Đầu ra chính", "Phương pháp"])


def result_tables(kpi_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build four core result tables from Bai 12."""
    table_1 = kpi_df[["scenario", "scenario_name", "Overall_score", "GDP_gain", "Digital_score", "AI_score", "NetJob"]]
    table_1 = table_1.rename(
        columns={
            "scenario": "Mã",
            "scenario_name": "Kịch bản",
            "Overall_score": "Overall",
            "GDP_gain": "GDP gain",
            "Digital_score": "Digital",
            "AI_score": "AI",
        }
    )
    table_2 = kpi_df[
        ["scenario", "K_share", "D_share", "AI_share", "H_share", "K_allocation", "D_allocation", "AI_allocation", "H_allocation"]
    ].rename(
        columns={
            "scenario": "Mã",
            "K_share": "K %",
            "D_share": "D %",
            "AI_share": "AI %",
            "H_share": "H %",
            "K_allocation": "K",
            "D_allocation": "D",
            "AI_allocation": "AI",
            "H_allocation": "H",
        }
    )
    for col in ["K %", "D %", "AI %", "H %"]:
        table_2[col] *= 100.0
    table_3 = kpi_df[["scenario", "scenario_name", "Inequality_risk", "Cyber_risk", "Emission_risk"]].rename(
        columns={
            "scenario": "Mã",
            "scenario_name": "Kịch bản",
            "Inequality_risk": "Bất bình đẳng",
            "Cyber_risk": "Rủi ro mạng",
            "Emission_risk": "Phát thải",
        }
    )
    table_3["Tổng rủi ro"] = kpi_df[COST_KPIS].sum(axis=1)
    table_4 = kpi_df[
        ["scenario", "scenario_name", "GDP_gain_norm", "Digital_score_norm", "AI_score_norm", "Human_capital_score_norm", "NetJob_norm", "Overall_score"]
    ].rename(
        columns={
            "scenario": "Mã",
            "scenario_name": "Kịch bản",
            "GDP_gain_norm": "GDP norm",
            "Digital_score_norm": "Digital norm",
            "AI_score_norm": "AI norm",
            "Human_capital_score_norm": "H norm",
            "NetJob_norm": "NetJob norm",
            "Overall_score": "Overall",
        }
    )
    return {"kpi": table_1, "allocation": table_2, "risk": table_3, "normalized": table_4}


def table_page(pdf: PdfPages, page: int, title: str, subtitle: str, df: pd.DataFrame, note: str | None = None) -> int:
    """Render a dataframe as a table page."""
    fig = new_page()
    header(fig, title, subtitle)
    ax = fig.add_axes([0.055, 0.14, 0.89, 0.70])
    ax.axis("off")
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda value: f"{value:,.2f}")
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns, cellLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.7)
    table.scale(1.0, 1.45)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#E5E7EB")
        if row == 0:
            cell.set_facecolor("#EDE9FE")
            cell.set_text_props(weight="bold", color=TEXT)
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#F9FAFB")
    if note:
        fig.text(0.08, 0.095, wrap(note, 105), fontsize=8.5, color=MUTED)
    return save_pdf_page(pdf, fig, page)


def webapp_home_page(pdf: PdfPages, page: int) -> int:
    """Draw a dashboard-style image describing the home page."""
    fig = new_page()
    header(fig, "Hình 1. Giao diện trang chủ webapp", "Minh họa dark theme, sidebar, badge, KPI cards và kiểm tra dữ liệu")
    ax = fig.add_axes([0.07, 0.12, 0.86, 0.72])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rounded_box(ax, (0, 0), 1, 1, "#060711", edge="#060711", radius=0.02)
    rounded_box(ax, (0.02, 0.04), 0.22, 0.92, "#0D0F1A", radius=0.025)
    ax.text(0.05, 0.90, "AIDEOM-VN", color="white", fontsize=13, weight="bold")
    ax.text(0.05, 0.865, "Dashboard mô hình ra quyết định", color="#9CA3AF", fontsize=7.5)
    for idx, label in enumerate(["Trang chủ", "Bài 1-4", "Bài 5-8", "Bài 9-12", "Báo cáo"]):
        y = 0.78 - idx * 0.075
        rounded_box(ax, (0.045, y), 0.17, 0.045, "#1B1D2B" if idx == 0 else "#0D0F1A", radius=0.012)
        ax.text(0.065, y + 0.014, label, color="white" if idx == 0 else "#CBD5E1", fontsize=7.5)
    ax.text(0.29, 0.88, "AIDEOM-VN", color="white", fontsize=22, weight="bold")
    ax.text(0.29, 0.835, "Dashboard đồ án môn Các mô hình ra quyết định", color="#CBD5E1", fontsize=8.5)
    for idx, label in enumerate(["Độ khó: Nền tảng", "Mô hình: Dashboard", "Trạng thái: Data OK"]):
        rounded_box(ax, (0.29 + idx * 0.19, 0.755), 0.17, 0.045, "#241B3D", edge="#6D5BD0", radius=0.018)
        ax.text(0.305 + idx * 0.19, 0.770, label, color="#EDE9FE", fontsize=6.8, weight="bold")
    for idx, (label, value) in enumerate([("CSV bắt buộc", "3"), ("CSV đã có", "3"), ("CSV còn thiếu", "0")]):
        x = 0.29 + idx * 0.21
        rounded_box(ax, (x, 0.61), 0.18, 0.10, "#10121E", radius=0.02)
        ax.text(x + 0.018, 0.675, label, color="#9CA3AF", fontsize=7)
        ax.text(x + 0.018, 0.635, value, color="white", fontsize=18, weight="bold")
    rounded_box(ax, (0.29, 0.12), 0.61, 0.42, "#10121E", radius=0.025)
    ax.text(0.32, 0.49, "Kiểm tra 3 file CSV trong data/", color="white", fontsize=11, weight="bold")
    rows = [("vietnam_macro_2020_2025.csv", "OK", "6 x 9"), ("vietnam_sectors_2024.csv", "OK", "10 x 9"), ("vietnam_regions_2024.csv", "OK", "6 x 9")]
    for idx, (filename, status, shape) in enumerate(rows):
        y = 0.41 - idx * 0.09
        rounded_box(ax, (0.32, y), 0.52, 0.055, "#151827", radius=0.012)
        ax.text(0.34, y + 0.020, filename, color="#E5E7EB", fontsize=7.5)
        ax.text(0.70, y + 0.020, status, color="#34D399", fontsize=7.5, weight="bold")
        ax.text(0.77, y + 0.020, shape, color="#CBD5E1", fontsize=7.5)
    fig.text(0.08, 0.085, wrap("Hình được dựng từ cấu trúc UI hiện tại của Streamlit app nhằm mô tả giao diện webapp trong báo cáo.", 105), fontsize=8.5, color=MUTED)
    return save_pdf_page(pdf, fig, page, "webapp_home_overview.png")


def webapp_bai12_page(pdf: PdfPages, page: int, kpi_df: pd.DataFrame) -> int:
    """Draw a dashboard-style image describing Bai 12 after running software."""
    best = kpi_df.sort_values("Overall_score", ascending=False).iloc[0]
    fig = new_page()
    header(fig, "Hình 2. Giao diện Bài 12 sau khi chạy phần mềm", "Minh họa KPI cards, tabs, biểu đồ và diễn giải chính sách")
    ax = fig.add_axes([0.07, 0.12, 0.86, 0.72])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rounded_box(ax, (0, 0), 1, 1, "#060711", edge="#060711", radius=0.02)
    rounded_box(ax, (0.02, 0.04), 0.20, 0.92, "#0D0F1A", radius=0.025)
    ax.text(0.05, 0.90, "Bài 12", color="white", fontsize=13, weight="bold")
    ax.text(0.05, 0.865, "Dashboard tích hợp", color="#9CA3AF", fontsize=8)
    ax.text(0.27, 0.88, "Dashboard tích hợp AIDEOM-VN", color="white", fontsize=18, weight="bold")
    for idx, label in enumerate(["Tổng quan", "5 kịch bản", "Phân bổ", "Lao động", "Rủi ro", "Khuyến nghị"]):
        x = 0.27 + idx * 0.105
        rounded_box(ax, (x, 0.805), 0.095, 0.040, "#241B3D" if idx == 0 else "#10121E", radius=0.018)
        ax.text(x + 0.010, 0.818, label, color="#EDE9FE" if idx == 0 else "#CBD5E1", fontsize=5.9)
    cards = [
        ("Kịch bản tốt nhất", str(best["scenario"])),
        ("Tên kịch bản", str(best["scenario_name"])),
        ("Overall score", f"{best['Overall_score']:.1f}/100"),
        ("Budget", f"{best['budget']:,.0f}"),
    ]
    for idx, (label, value) in enumerate(cards):
        x = 0.27 + idx * 0.155
        rounded_box(ax, (x, 0.66), 0.140, 0.10, "#10121E", radius=0.02)
        ax.text(x + 0.010, 0.724, label, color="#9CA3AF", fontsize=5.9)
        ax.text(x + 0.010, 0.685, value, color="white", fontsize=7.1, weight="bold")
    rounded_box(ax, (0.27, 0.16), 0.34, 0.42, "#10121E", radius=0.022)
    ax.text(0.30, 0.535, "Overall_score theo kịch bản", color="white", fontsize=9.5, weight="bold")
    sorted_df = kpi_df.sort_values("Overall_score")
    max_score = sorted_df["Overall_score"].max()
    for idx, row in enumerate(sorted_df.itertuples()):
        y = 0.23 + idx * 0.055
        width = 0.25 * row.Overall_score / max_score
        ax.text(0.30, y + 0.010, row.scenario, color="#CBD5E1", fontsize=7)
        rounded_box(ax, (0.35, y), 0.25, 0.028, "#151827", edge="#151827", radius=0.008)
        rounded_box(ax, (0.35, y), width, 0.028, PRIMARY if row.scenario == "S5" else SECONDARY, edge="none", radius=0.008)
        ax.text(0.35 + width + 0.008, y + 0.006, f"{row.Overall_score:.1f}", color="#E5E7EB", fontsize=6.5)
    rounded_box(ax, (0.65, 0.16), 0.24, 0.42, "#10121E", radius=0.022)
    ax.text(0.68, 0.535, "Diễn giải chính sách", color="white", fontsize=9.5, weight="bold")
    for idx, text in enumerate(["S5 cân bằng tốt nhất.", "S3 dẫn đầu về AI.", "S4 mạnh về việc làm.", "Cần tham vấn chuyên gia."]):
        ax.text(0.68, 0.47 - idx * 0.075, "- " + text, color="#CBD5E1", fontsize=7.0)
    fig.text(0.08, 0.085, wrap("Hình minh họa kết quả thu được khi chạy scenario_engine của Bài 12 với ngân sách mặc định 50.000 tỷ VND.", 105), fontsize=8.5, color=MUTED)
    return save_pdf_page(pdf, fig, page, "webapp_bai12_results.png")


def chart_page(pdf: PdfPages, page: int, title: str, subtitle: str, draw, note: str, png_name: str) -> int:
    """Render a chart page and save its PNG copy."""
    fig = new_page()
    header(fig, title, subtitle)
    ax = fig.add_axes([0.10, 0.20, 0.80, 0.62], projection=getattr(draw, "projection", None))
    draw(ax)
    fig.text(0.08, 0.115, wrap(note, 105), fontsize=8.5, color=MUTED)
    return save_pdf_page(pdf, fig, page, png_name)


def draw_overall(kpi_df: pd.DataFrame):
    """Draw Overall_score chart."""
    def inner(ax):
        df = kpi_df.sort_values("Overall_score")
        colors = [PRIMARY if code == "S5" else "#93C5FD" for code in df["scenario"]]
        ax.barh(df["scenario"] + " - " + df["scenario_name"], df["Overall_score"], color=colors)
        ax.set_xlabel("Overall score (0-100)")
        ax.set_title("Xếp hạng tổng hợp 5 kịch bản")
        ax.grid(axis="x", color=GRID, alpha=0.55)
        for i, value in enumerate(df["Overall_score"]):
            ax.text(value + 1, i, f"{value:.1f}", va="center", fontsize=9)
    return inner


def draw_allocation(kpi_df: pd.DataFrame):
    """Draw stacked allocation shares."""
    def inner(ax):
        x = np.arange(len(kpi_df))
        bottom = np.zeros(len(kpi_df))
        for col, label, color in zip(["K_share", "D_share", "AI_share", "H_share"], ["K", "D", "AI", "H"], ["#93C5FD", SECONDARY, PRIMARY, GOOD]):
            values = kpi_df[col].to_numpy() * 100
            ax.bar(x, values, bottom=bottom, color=color, label=label)
            bottom += values
        ax.set_xticks(x)
        ax.set_xticklabels(kpi_df["scenario"])
        ax.set_ylabel("Tỷ trọng (%)")
        ax.set_title("Cấu trúc phân bổ ngân sách")
        ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.10))
        ax.grid(axis="y", color=GRID, alpha=0.45)
    return inner


def draw_benefits(kpi_df: pd.DataFrame):
    """Draw benefit KPI chart."""
    def inner(ax):
        x = np.arange(len(kpi_df))
        width = 0.24
        for idx, (col, label, color) in enumerate(zip(["Digital_score", "AI_score", "Human_capital_score"], ["Digital", "AI", "Human capital"], [SECONDARY, PRIMARY, GOOD])):
            ax.bar(x + (idx - 1) * width, kpi_df[col], width=width, color=color, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(kpi_df["scenario"])
        ax.set_ylim(0, 105)
        ax.set_ylabel("Điểm KPI")
        ax.set_title("So sánh năng lực số, AI và vốn nhân lực")
        ax.legend()
        ax.grid(axis="y", color=GRID, alpha=0.45)
    return inner


def draw_risks(kpi_df: pd.DataFrame):
    """Draw risk KPI chart."""
    def inner(ax):
        x = np.arange(len(kpi_df))
        width = 0.24
        for idx, (col, label, color) in enumerate(zip(["Inequality_risk", "Cyber_risk", "Emission_risk"], ["Inequality", "Cyber", "Emission"], [WARN, PRIMARY, DANGER])):
            ax.bar(x + (idx - 1) * width, kpi_df[col], width=width, color=color, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(kpi_df["scenario"])
        ax.set_ylabel("Điểm rủi ro")
        ax.set_title("So sánh rủi ro chính sách")
        ax.legend()
        ax.grid(axis="y", color=GRID, alpha=0.45)
    return inner


def draw_radar(kpi_df: pd.DataFrame):
    """Draw normalized radar chart."""
    def inner(ax):
        labels = ["GDP", "Digital", "AI", "Human", "NetJob", "Equity", "Cyber", "Emission"]
        cols = ["GDP_gain_norm", "Digital_score_norm", "AI_score_norm", "Human_capital_score_norm", "NetJob_norm", "Inequality_risk_norm", "Cyber_risk_norm", "Emission_risk_norm"]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_ylim(0, 1)
        for (_, row), color in zip(kpi_df.iterrows(), ["#94A3B8", SECONDARY, PRIMARY, GOOD, WARN]):
            values = [float(row[col]) for col in cols] + [float(row[cols[0]])]
            ax.plot(angles, values, color=color, linewidth=1.5, label=row["scenario"])
            ax.fill(angles, values, color=color, alpha=0.08)
        ax.set_title("Radar normalized KPI", y=1.10)
        ax.legend(loc="upper right", bbox_to_anchor=(1.20, 1.12))
    inner.projection = "polar"
    return inner


def build_pdf() -> Path:
    """Build the full PDF report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    kpi_df = run_all_scenarios(budget=BUDGET)
    alloc_df = allocation_long(kpi_df)
    tables = result_tables(kpi_df)
    best = kpi_df.sort_values("Overall_score", ascending=False).iloc[0]

    page = 1
    with PdfPages(PDF_PATH) as pdf:
        fig = new_page()
        fig.text(0.08, 0.78, "AIDEOM-VN", fontsize=30, weight="bold", color=TEXT)
        fig.text(0.08, 0.735, "Báo cáo nghiên cứu dashboard mô hình ra quyết định", fontsize=15, color=MUTED)
        fig.text(0.08, 0.66, wrap("Báo cáo tổng hợp 12 mô-đun định lượng cho phân tích chính sách phát triển kinh tế số, AI, phân bổ nguồn lực, lao động và rủi ro tại Việt Nam.", 82), fontsize=11, color=TEXT, linespacing=1.4)
        fig.text(0.08, 0.55, f"Ngân sách mô phỏng Bài 12: {BUDGET:,.0f} tỷ VND", fontsize=10.5, color=TEXT)
        fig.text(0.08, 0.51, f"Kịch bản tổng hợp tốt nhất: {best['scenario']} - {best['scenario_name']}", fontsize=10.5, color=TEXT)
        fig.text(0.08, 0.47, f"Overall score: {best['Overall_score']:.1f}/100", fontsize=10.5, color=TEXT)
        page = save_pdf_page(pdf, fig, page)

        for title, subtitle, blocks in [
            (
                "1. Tóm tắt điều hành",
                "Kết luận chính từ dashboard tích hợp AIDEOM-VN",
                [
                    ("Kết luận tổng hợp", "AIDEOM-VN tích hợp 12 mô-đun định lượng, từ Cobb-Douglas, LP, MIP, TOPSIS, Pareto, tối ưu động, stochastic programming tới Q-learning. Kết quả Bài 12 cho thấy S5 - Tối ưu cân bằng đạt điểm tổng hợp cao nhất trong bộ giả định hiện tại."),
                    ("Thông điệp chính sách", "Không có kịch bản đơn lẻ tối ưu cho mọi mục tiêu. S3 nổi bật về AI, S4 nổi bật về bao trùm và việc làm, còn S5 cân bằng giữa tăng trưởng, số hóa, AI, vốn nhân lực và rủi ro."),
                ],
            ),
            (
                "2-4. Giới thiệu, dữ liệu và kiến trúc hệ thống",
                "Tổng quan nền tảng nghiên cứu",
                [
                    ("2. Giới thiệu", "Đề tài tiếp cận phát triển kinh tế số Việt Nam như một bài toán ra quyết định đa mục tiêu, trong đó nhà hoạch định cân nhắc đồng thời tăng trưởng, năng lực số, năng lực AI, việc làm, công bằng vùng và rủi ro."),
                    ("3. Dữ liệu sử dụng", "Dashboard sử dụng dữ liệu vĩ mô Việt Nam 2020-2025, dữ liệu 10 ngành kinh tế năm 2024 và dữ liệu 6 vùng kinh tế năm 2024. Một số bài bổ sung dữ liệu theo đề như danh mục 15 dự án, dữ liệu lao động, kịch bản stochastic và mapping hành động Q-learning."),
                    ("4. Kiến trúc hệ thống", "Giao diện Streamlit nằm trong pages/, logic tính toán nằm trong src/. data_loader.py nạp dữ liệu tương đối; visualization.py và ui.py cung cấp thành phần trình bày; scenario_engine.py tổng hợp kết quả thành 5 kịch bản chính sách."),
                ],
            ),
        ]:
            fig = new_page()
            header(fig, title, subtitle)
            text_blocks(fig, blocks)
            page = save_pdf_page(pdf, fig, page)

        page = webapp_home_page(pdf, page)
        page = webapp_bai12_page(pdf, page, kpi_df)
        page = table_page(pdf, page, "5. Bảng tóm tắt 12 bài/mô-đun", "Mỗi mô-đun có page riêng và logic riêng trong src/", module_summary_table())

        fig = new_page()
        header(fig, "6. Phương pháp mô hình hóa", "Khung phương pháp định lượng")
        bullet_block(
            fig,
            "Các nhóm phương pháp chính",
            [
                "Cobb-Douglas mở rộng đưa số hóa, AI và vốn nhân lực vào hàm sản xuất để phân rã tăng trưởng.",
                "LP/MIP mô tả phân bổ nguồn lực khan hiếm dưới ràng buộc ngân sách, phụ thuộc dự án, công bằng và an toàn xã hội.",
                "MCDA/TOPSIS hỗ trợ xếp hạng ngành và vùng khi tiêu chí chính sách khác đơn vị đo.",
                "Pareto/NSGA-II và tối ưu động biểu diễn đánh đổi đa mục tiêu và liên thời gian.",
                "Stochastic programming và Q-learning mô phỏng bất định và chính sách thích nghi theo trạng thái.",
                "Bài 12 chuẩn hóa KPI lợi ích/rủi ro và tính Overall_score theo trọng số minh bạch.",
            ],
        )
        page = save_pdf_page(pdf, fig, page)

        fig = new_page()
        header(fig, "7. Kết quả chính", "Kết quả định lượng thu được khi chạy phần mềm")
        bullet_block(
            fig,
            "Các phát hiện nổi bật",
            recommendation_text(kpi_df)
            + [
                f"S5 đạt Overall_score {best['Overall_score']:.1f}/100, cao nhất trong 5 kịch bản.",
                "S3 có AI_score cao nhất nhưng rủi ro mạng tăng mạnh do tỷ trọng AI lớn.",
                "S4 có NetJob và điểm vốn nhân lực cao, thể hiện vai trò của đào tạo lại.",
                "S1 có điểm tổng hợp thấp vì thiên về vốn truyền thống và thiếu động lực số hóa/AI.",
            ],
        )
        page = save_pdf_page(pdf, fig, page)

        page = table_page(pdf, page, "Bảng 1. KPI tổng hợp của 5 kịch bản", "Kết quả chính từ Bài 12", tables["kpi"], "GDP gain và NetJob là đại lượng mô phỏng; Overall_score nằm trên thang 0-100.")
        page = chart_page(pdf, page, "Hình 3. Overall_score theo kịch bản", "So sánh năng lực tổng hợp", draw_overall(kpi_df), "S5 được tô nổi bật vì là phương án cân bằng tốt nhất theo trọng số hiện tại.", "fig03_overall_score.png")
        page = table_page(pdf, page, "Bảng 2. Cơ cấu phân bổ ngân sách", "Tỷ trọng và giá trị phân bổ K, D, AI, H", tables["allocation"], "Đơn vị phân bổ là tỷ VND; tỷ trọng tính trên ngân sách 50.000 tỷ VND.")
        page = chart_page(pdf, page, "Hình 4. Cấu trúc phân bổ ngân sách", "Stacked bar K, D, AI, H", draw_allocation(kpi_df), "S1 thiên về K; S2 thiên về D; S3 thiên về AI; S4 thiên về H; S5 cân bằng hơn.", "fig04_allocation.png")
        page = table_page(pdf, page, "Bảng 3. Ba nhóm rủi ro chính sách", "Bất bình đẳng, rủi ro mạng và phát thải", tables["risk"], "Rủi ro là cost KPI: thấp hơn được xem là tốt hơn trong Overall_score.")
        page = chart_page(pdf, page, "Hình 5. KPI năng lực số, AI và vốn nhân lực", "Kết quả benefit KPI", draw_benefits(kpi_df), "Biểu đồ cho thấy đánh đổi giữa số hóa nhanh, AI dẫn dắt và bao trùm nhân lực.", "fig05_benefit_kpi.png")
        page = chart_page(pdf, page, "Hình 6. Rủi ro chính sách theo kịch bản", "Kết quả risk KPI", draw_risks(kpi_df), "Các kịch bản tăng tốc AI cần đi kèm đầu tư vốn nhân lực và an toàn số.", "fig06_risks.png")
        page = table_page(pdf, page, "Bảng 4. KPI chuẩn hóa dùng tính Overall_score", "Các thành phần benefit đã chuẩn hóa", tables["normalized"], "Các cột norm nằm trong khoảng 0-1; Overall_score là điểm tổng hợp 0-100.")
        page = chart_page(pdf, page, "Hình 7. Radar KPI chuẩn hóa", "Hồ sơ lợi ích và kiểm soát rủi ro", draw_radar(kpi_df), "S5 không luôn đứng đầu từng trục, nhưng có cấu hình cân bằng nhất.", "fig07_radar.png")

        fig = new_page()
        header(fig, "8-12. So sánh kịch bản, chính sách và vận hành", "Diễn giải tổng hợp")
        text_blocks(
            fig,
            [
                ("8. So sánh 5 kịch bản", "S1 là đối chứng truyền thống. S2 mở rộng nền tảng số. S3 tạo đột phá AI nhưng làm tăng rủi ro mạng. S4 nhấn mạnh bao trùm số và việc làm. S5 cân bằng nhiều mục tiêu, phù hợp khi cần tối ưu tổng thể."),
                ("9. Diễn giải chính sách", "Đầu tư AI nên đi cùng đào tạo lại, an toàn số và cơ chế giảm chênh lệch vùng. Dashboard hỗ trợ nhận diện đánh đổi ngân sách và ưu tiên chính sách theo giai đoạn."),
                ("10. Hạn chế", "Các hệ số có tính minh họa, một số biến dùng proxy và fallback solver có thể kém chính xác hơn solver chuyên dụng. Kết quả không thay thế dữ liệu ngân sách thật và tham vấn chuyên gia."),
                ("11. Hướng mở rộng", "Có thể mở rộng bằng dữ liệu tỉnh/thành, cập nhật chuỗi thời gian, AHP/Delphi cho trọng số, mô hình cân bằng tổng thể và pipeline xuất báo cáo tự động."),
                ("12. Hướng dẫn chạy dashboard", "Tạo môi trường Python 3.10+, cài requirements bằng pip install -r requirements.txt, sau đó chạy streamlit run app.py. Khi deploy Streamlit Cloud, chọn entrypoint app.py và commit đủ ba file CSV trong data/."),
            ],
            start_y=0.86,
        )
        page = save_pdf_page(pdf, fig, page)

    kpi_df.to_csv(REPORTS_DIR / "bai12_report_kpi_snapshot.csv", index=False, encoding="utf-8-sig")
    alloc_df.to_csv(REPORTS_DIR / "bai12_report_allocation_snapshot.csv", index=False, encoding="utf-8-sig")
    return PDF_PATH


if __name__ == "__main__":
    print(build_pdf())
