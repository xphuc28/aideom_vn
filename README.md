# AIDEOM-VN

AIDEOM-VN là dashboard Streamlit cho đồ án môn **Các mô hình ra quyết định**. Project mô phỏng một bộ công cụ hỗ trợ phân tích chính sách phát triển kinh tế số và AI tại Việt Nam, kết hợp các mô hình tối ưu, xếp hạng đa tiêu chí, mô phỏng động, stochastic programming và reinforcement learning.

Dashboard này là công cụ hỗ trợ ra quyết định định lượng. Kết quả không thay thế quyết định chính trị-xã hội, tham vấn chuyên gia, phân tích ngân sách thực tế hoặc đánh giá tác động phân phối.

## Giao diện dashboard

Ứng dụng dùng giao diện dark theme theo phong cách dashboard phân tích: sidebar điều hướng rõ ràng, tiêu đề lớn, badge độ khó cho từng bài, KPI cards, biểu đồ Plotly toàn chiều rộng, các tab kết quả và khối diễn giải chính sách trong `st.info`/`st.success`. Nếu cần ảnh minh họa khi nộp báo cáo hoặc đăng GitHub, chạy app bằng `streamlit run app.py`, mở `http://localhost:8501`, chụp trang chủ hoặc Bài 12 rồi đặt ảnh vào `reports/` hoặc phần README của repository.

## Cấu trúc thư mục

```text
aideom_vn/
├── app.py
├── data/
│   ├── vietnam_macro_2020_2025.csv
│   ├── vietnam_sectors_2024.csv
│   └── vietnam_regions_2024.csv
├── pages/
│   ├── 01_📈_Bai_1_Cobb_Douglas_AI.py
│   ├── 02_💰_Bai_2_LP_Ngan_Sach_So.py
│   ├── 03_🏭_Bai_3_Priority_10_Nganh.py
│   ├── 04_🗺️_Bai_4_LP_Nganh_Vung.py
│   ├── 05_📦_Bai_5_MIP_15_Du_An.py
│   ├── 06_📍_Bai_6_TOPSIS_6_Vung.py
│   ├── 07_🧬_Bai_7_NSGAII_Pareto.py
│   ├── 08_⏳_Bai_8_Dong_2026_2035.py
│   ├── 09_👷_Bai_9_Lao_Dong_AI.py
│   ├── 10_🎲_Bai_10_Stochastic_SP.py
│   ├── 11_🤖_Bai_11_Q_Learning_RL.py
│   └── 12_🇻🇳_Bai_12_AIDEOM_Tich_Hop.py
├── src/
│   ├── data_loader.py
│   ├── visualization.py
│   ├── scenario_engine.py
│   └── bai01...bai11 modules
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── models/
├── reports/
│   ├── aideom_vn_research_report.pdf
│   ├── project_summary.md
│   ├── figures/
│   ├── bai12_report_kpi_snapshot.csv
│   └── bai12_report_allocation_snapshot.csv
├── scripts/
│   ├── count_lines.py
│   └── generate_research_report.py
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

## Cài đặt trên macOS

Yêu cầu Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Nếu một số solver như PuLP/CBC, CVXPY, Pyomo hoặc pymoo chưa cài được, các page liên quan vẫn chạy bằng fallback SciPy hoặc random search khi có thể.

## Chạy local

```bash
streamlit run app.py
```

Sau đó mở địa chỉ Streamlit in ra, thường là `http://localhost:8501`.

## Kiểm thử

```bash
python -m compileall src pages app.py
python -m pytest
python scripts/count_lines.py
```

Ghi chú môi trường: trên máy hiện tại, Anaconda Python 3.13 bị segmentation fault khi nạp pytest/debugging trước lúc collect tests. Các assertion chính đã được kiểm tra trực tiếp bằng Python trong quá trình phát triển. Khi dùng venv Python 3.10+ sạch, hãy chạy lại `python -m pytest`.

## Báo cáo nghiên cứu

Báo cáo PDF theo yêu cầu đề bài nằm tại `reports/aideom_vn_research_report.pdf`. Báo cáo có văn phong học thuật tiếng Việt, bảng tóm tắt 12 mô-đun, hình minh họa webapp, hình kết quả sau khi chạy phần mềm và 4 bảng kết quả chính từ Bài 12. Có thể tái tạo báo cáo bằng:

```bash
python scripts/generate_research_report.py
```

## Deploy Streamlit Cloud

1. Đẩy toàn bộ project lên GitHub.
2. Vào Streamlit Community Cloud và tạo app mới từ repository.
3. Chọn file entrypoint là `app.py`.
4. Đảm bảo ba file CSV trong `data/` đã được commit.
5. Streamlit Cloud sẽ tự đọc `requirements.txt` để cài dependency.
6. Nếu dependency solver nặng không cài được, dashboard vẫn có fallback cho các bài đã thiết kế.

## Chuẩn bị GitHub

Repo đã loại trừ môi trường ảo, cache Python, cache pytest, file macOS `.DS_Store`, secrets Streamlit và model binary nặng trong `outputs/models/`. Không có đường dẫn tuyệt đối tới máy cá nhân trong mã nguồn; các file dữ liệu được đọc tương đối qua `data/`.

## Mô tả 12 bài

- **Bài 1:** Cobb-Douglas mở rộng với AI, số hóa và vốn nhân lực; ước lượng TFP, phân rã tăng trưởng và forecast 2030.
- **Bài 2:** LP phân bổ ngân sách số giữa hạ tầng, AI/dữ liệu, nhân lực và R&D.
- **Bài 3:** Xếp hạng ưu tiên 10 ngành bằng min-max normalization, trọng số chính sách và sensitivity AI.
- **Bài 4:** LP phân bổ ngân sách theo vùng-hạng mục với ràng buộc sàn/trần vùng, nhân lực và công bằng số.
- **Bài 5:** MIP lựa chọn danh mục 15 dự án chuyển đổi số với ràng buộc ngân sách, phụ thuộc dự án và số lượng dự án.
- **Bài 6:** TOPSIS xếp hạng 6 vùng đầu tư AI bằng trọng số expert và entropy.
- **Bài 7:** Tối ưu đa mục tiêu Pareto/NSGA-II cho tăng trưởng, bất bình đẳng, phát thải và rủi ro mạng, có fallback random feasible search.
- **Bài 8:** Tối ưu động 2026-2035 bằng SLSQP cho phân bổ đầu tư liên thời gian vào K, D, AI, H.
- **Bài 9:** LP đánh giá tác động AI tới lao động, việc làm mới, việc làm nâng cấp, displaced jobs và năng lực đào tạo lại.
- **Bài 10:** Quy hoạch ngẫu nhiên hai giai đoạn với 4 kịch bản, VSS/EVPI và minimax regret optional.
- **Bài 11:** Q-learning cho chính sách kinh tế thích nghi với 81 trạng thái và 5 hành động chính sách.
- **Bài 12:** Dashboard tích hợp 5 kịch bản AIDEOM-VN, tổng hợp KPI tăng trưởng, số hóa, AI, lao động và rủi ro.

## Nguồn dữ liệu

Project dùng ba file CSV trong `data/`:

- `vietnam_macro_2020_2025.csv`: chỉ tiêu vĩ mô Việt Nam 2020-2025.
- `vietnam_sectors_2024.csv`: dữ liệu 10 ngành năm 2024.
- `vietnam_regions_2024.csv`: dữ liệu 6 vùng kinh tế năm 2024.

Một số bài dùng bộ dữ liệu hard-code theo đề bài, ví dụ 15 dự án ở Bài 5, 8 ngành lao động ở Bài 9, scenario stochastic ở Bài 10 và action/state mapping ở Bài 11.

## Hạn chế mô hình

- Nhiều hệ số là giả định phục vụ mô phỏng đồ án, không phải ước lượng chính thức.
- Một số biến như vốn K, năng lực AI hoặc vốn nhân lực được proxy khi CSV không có cột trực tiếp.
- Các solver ngoài có thể không khả dụng trên mọi môi trường; fallback được thiết kế để dashboard vẫn chạy nhưng có thể kém chính xác hơn solver chuyên dụng.
- Kết quả không phản ánh đầy đủ ràng buộc thể chế, hành vi doanh nghiệp, phân phối vùng sâu hoặc tác động xã hội dài hạn.
- Dashboard nên được xem là khung minh họa ra quyết định, cần hiệu chỉnh bằng dữ liệu thật và phản biện chuyên gia trước khi dùng cho chính sách thực tế.
