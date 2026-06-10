"""Streamlit components for answering the assignment questions explicitly."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypedDict

import streamlit as st


class AssignmentAnswer(TypedDict, total=False):
    """One explicit answer mapped to a question in the assignment PDF."""

    code: str
    question: str
    answer: str
    evidence: str
    status: str


def render_assignment_answers(
    programming_answers: Iterable[AssignmentAnswer],
    policy_answers: Iterable[AssignmentAnswer],
    note: str | None = None,
) -> None:
    """Render auditable programming and policy answers in two tabs."""
    programming = list(programming_answers)
    policy = list(policy_answers)

    st.header("✅ 6. Trả lời yêu cầu đề bài")
    st.caption(
        "Các câu trả lời dưới đây được tạo từ tham số đang chọn và kết quả mô hình "
        "trên trang. Khi thay đổi tham số, kết luận định lượng cũng được cập nhật."
    )
    if note:
        st.info(note)

    tab_programming, tab_policy = st.tabs(
        ["🧪 Yêu cầu lập trình và kết quả", "🏛️ Câu hỏi thảo luận chính sách"]
    )

    def render_items(items: list[AssignmentAnswer]) -> None:
        for item in items:
            status = item.get("status", "Đã trả lời")
            with st.container(border=True):
                st.markdown(f"**{item['code']} — {item['question']}**")
                st.write(item["answer"])
                if item.get("evidence"):
                    st.caption(f"Căn cứ kiểm chứng: {item['evidence']}")
                if status != "Đã trả lời":
                    st.warning(status)

    with tab_programming:
        render_items(programming)
    with tab_policy:
        render_items(policy)
