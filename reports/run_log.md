# AIDEOM-VN run log

Ngày chạy: 2026-05-31  
Thư mục làm việc: `/Users/dongphuc/Desktop/aideom_vn`

## Lệnh chạy app Streamlit xác định được

```bash
streamlit run app.py
```

Lệnh kiểm tra server local đã chạy:

```bash
streamlit run app.py --server.headless true --server.fileWatcherType none --server.port 8501
curl -I http://localhost:8501
pkill -f "streamlit run app.py"
```

Kết quả: Streamlit khởi động được, trả `HTTP/1.1 200 OK` tại `http://localhost:8501`.

## Lệnh đã chạy để đọc cấu trúc và kiểm tra requirements

```bash
sed -n '1,220p' requirements.txt
sed -n '1,180p' README.md
rg -n "streamlit run|run_all_scenarios|solve_|run_bai|optimize_|train_q_learning|random_feasible|compare_" README.md scripts src tests
find reports -maxdepth 3 -type f | sort
```

Kết luận: `requirements.txt` có các thư viện chính gồm Streamlit, pandas, numpy, Plotly, matplotlib, scipy, PuLP, CVXPY, gymnasium, pymoo, pyomo và pytest.

Kiểm tra import môi trường local:

```bash
python -c "import streamlit,pandas,numpy,plotly,matplotlib,scipy; import pulp,cvxpy; print('core requirements import ok')"
```

Kết quả: môi trường local hiện tại thiếu `pulp` (`ModuleNotFoundError: No module named 'pulp'`). File `requirements.txt` vẫn có `pulp>=2.8`; các module liên quan đã dùng fallback khi solver/thư viện không khả dụng.

Kiểm tra phiên bản Streamlit:

```bash
python -m streamlit --version
```

Kết quả: `Streamlit, version 1.45.1`.

## Lệnh biên dịch và chạy artifact script

Biên dịch script và app:

```bash
python -m compileall scripts/export_report_artifacts.py src pages app.py
```

Tạo artifact báo cáo:

```bash
python scripts/export_report_artifacts.py
```

Trong quá trình chạy, script ban đầu phát hiện hai lỗi xuất artifact không thuộc logic mô hình:

1. Bài 6 dùng tên cột biểu đồ `score` trong khi output thực tế là `topsis_score`.
2. `stress_test_risk()` của Bài 9 trả về dict, cần lấy `allocation_df` trước khi export.

Sau khi sửa script export, lệnh chạy thành công:

```text
Exported report artifacts to /Users/dongphuc/Desktop/aideom_vn/reports/figures
```

Script được chạy lại lần cuối sau khi loại file ẩn khỏi manifest:

```bash
python scripts/export_report_artifacts.py
```

## Lệnh kiểm tra artifact đã sinh

```bash
find reports/figures -maxdepth 1 -type f | sort | wc -l
find reports/figures -maxdepth 1 -type f | sort | head -80
sed -n '1,80p' reports/figures/00_model_run_summary.md
sed -n '1,120p' reports/figures/00_artifact_manifest.md
find reports/figures -maxdepth 1 -type f -name '*.png' | wc -l
find reports/figures -maxdepth 1 -type f -name '*.csv' | wc -l
find reports/figures -maxdepth 1 -type f -name '*.md' | wc -l
find reports/figures -maxdepth 1 -type f | sort | tail -60
```

Kết quả artifact cuối:

- PNG: 20 file.
- CSV: 53 file.
- Markdown: 54 file.
- Tổng file trong `reports/figures`: 128 file, bao gồm cả `.DS_Store` có sẵn trên macOS.

## Tóm tắt kết quả chạy mô hình

Nguồn: `reports/figures/00_model_run_summary.md`

| module | main_result | status |
|:--|:--|:--|
| Bài 1 | MAPE=5.2383 | completed |
| Bài 2 | objective=112.25 | optimal |
| Bài 3 | Thông tin-Truyền thông-CNTT | completed |
| Bài 4 | objective=53851.16849999999 | optimal |
| Bài 5 | selected=9 | feasible |
| Bài 6 | Đông Nam Bộ | completed |
| Bài 7 | pareto=122 | random |
| Bài 8 | welfare=48.4217 | optimal |
| Bài 9 | objective=40611.09298531811 | optimal |
| Bài 10 | objective=98575.0 | optimal |
| Bài 11 | last100=8.5575 | completed |
| Bài 12 | Tối ưu cân bằng | completed |

## Ghi chú

- Không thay đổi logic mô hình trong `src/`.
- Chỉ thêm script xuất artifact `scripts/export_report_artifacts.py` và file log này.
- Kết quả Bài 7 chạy bằng `random` fallback trong môi trường hiện tại.
- Do môi trường local thiếu `pulp`, các mô-đun có fallback đã tự dùng solver/phương pháp khả dụng.
