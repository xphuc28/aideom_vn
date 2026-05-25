# Tóm tắt đồ án AIDEOM-VN

## Mục tiêu

AIDEOM-VN là dashboard Streamlit cho môn Các mô hình ra quyết định, tập trung vào bài toán phát triển kinh tế số, AI, phân bổ ngân sách, lựa chọn dự án và quản trị rủi ro chính sách tại Việt Nam. Project tích hợp 12 bài mô hình hóa thành một ứng dụng web có thể chạy local hoặc deploy Streamlit Cloud.

## Thành phần chính

- `app.py`: trang chủ, kiểm tra dữ liệu đầu vào và giới thiệu dashboard.
- `pages/`: 12 page Streamlit, mỗi page tương ứng một bài trong đề.
- `src/`: logic mô hình tách khỏi giao diện, gồm Cobb-Douglas, LP, MIP, TOPSIS, Pareto, dynamic optimization, stochastic programming, Q-learning và scenario engine.
- `data/`: ba file CSV nền gồm macro, ngành và vùng.
- `tests/`: unit tests cho các mô hình chính.
- `.streamlit/config.toml`: cấu hình dark theme.

## 12 mô hình tích hợp

1. Cobb-Douglas mở rộng với AI, số hóa và vốn nhân lực.
2. LP phân bổ ngân sách số giữa hạ tầng, AI/dữ liệu, nhân lực và R&D.
3. Xếp hạng ưu tiên 10 ngành bằng chuẩn hóa min-max và trọng số chính sách.
4. LP phân bổ ngân sách số theo vùng-hạng mục với ràng buộc công bằng.
5. MIP lựa chọn 15 dự án chuyển đổi số.
6. TOPSIS xếp hạng 6 vùng đầu tư AI.
7. Pareto/NSGA-II cho tối ưu đa mục tiêu tăng trưởng, công bằng, phát thải và rủi ro mạng.
8. Tối ưu động phân bổ đầu tư 2026-2035.
9. LP tác động AI tới lao động và đào tạo lại.
10. Quy hoạch ngẫu nhiên hai giai đoạn với VSS/EVPI.
11. Q-learning cho chính sách kinh tế thích nghi.
12. Dashboard tích hợp 5 kịch bản AIDEOM-VN.

## Giao diện

Dashboard dùng dark theme, sidebar điều hướng, badge độ khó, KPI cards, Plotly charts toàn chiều rộng, section có icon và các tab kết quả. Mỗi bài có phần diễn giải chính sách trong `st.info` hoặc `st.success` để chuyển kết quả định lượng thành thông điệp chính sách.

## Dữ liệu

Project dùng ba file CSV:

- `data/vietnam_macro_2020_2025.csv`
- `data/vietnam_sectors_2024.csv`
- `data/vietnam_regions_2024.csv`

Một số bài có dữ liệu giả lập hoặc hard-code theo đề để bảo đảm dashboard chạy độc lập.

## Kiểm tra kỹ thuật

- Mã nguồn Python vượt yêu cầu 1.500 dòng.
- Logic mô hình nằm trong `src/`, giao diện nằm trong `pages/`.
- Không dùng đường dẫn tuyệt đối tới máy cá nhân.
- Không có secrets/API keys trong source.
- Có fallback khi solver hoặc thư viện tối ưu nặng không khả dụng.

## Hạn chế

Các hệ số, trọng số và bộ dữ liệu trong đồ án có tính minh họa, không phải ước lượng chính thức. Kết quả chỉ nên dùng như công cụ hỗ trợ tư duy ra quyết định và cần được hiệu chỉnh bằng dữ liệu thật, thẩm định chuyên gia và phân tích tác động xã hội trước khi dùng cho chính sách thực tế.
