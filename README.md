# Vietnamese Text Style Classifier

Phân loại phong cách văn bản tiếng Việt sử dụng các mô hình encoder-only (BERT) từ HuggingFace.

## 5 Phong cách văn bản

| ID | Phong cách | Mô tả |
|---|---|---|
| `bao_chi` | Báo chí - Công luận | Tin tức, phóng sự, bài báo |
| `hanh_chinh` | Hành chính - Công vụ | Văn bản pháp luật, công văn, nghị định |
| `khoa_hoc` | Khoa học | Bài nghiên cứu, luận văn, tài liệu khoa học |
| `chinh_luan` | Chính luận | Bài bình luận chính trị, xã luận |
| `sinh_hoat` | Sinh hoạt hàng ngày | Giao tiếp đời thường, mạng xã hội |

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cấu trúc dữ liệu

Dữ liệu đầu vào là file JSONL hoặc thư mục chứa các file JSONL. Mỗi dòng có dạng:

```json
{"text": "Nội dung văn bản...", "label": "bao_chi"}
```

## Sử dụng

### 1. Tạo dữ liệu mẫu

```bash
python scripts/generate_sample_data.py
```

### 2. Huấn luyện

```bash
python train.py --config configs/default.yaml
```

### 3. Dự đoán

Chế độ tương tác:
```bash
python predict.py --model_path outputs/best_model
```

Dự đoán từ file:
```bash
python predict.py --model_path outputs/best_model --input data.jsonl --output predictions.jsonl
```

## Pipeline

1. **Nạp dữ liệu** - Đọc file JSONL hoặc thư mục JSONL
2. **Tiền xử lý** - Làm sạch văn bản tiếng Việt (URL, HTML, Unicode), tách từ (underthesea)
3. **Khử trùng lặp** - MinHash LSH (datasketch) với ngưỡng Jaccard similarity
4. **Chunking** - Chia văn bản dài thành các đoạn chồng lấp, tổng hợp kết quả (mean/max/majority_vote)
5. **Huấn luyện** - Fine-tune mô hình encoder-only (mặc định: PhoBERT v2)
6. **Đánh giá** - Accuracy, F1-macro, F1-weighted, classification report

## Cấu hình

Chỉnh sửa `configs/default.yaml` để thay đổi:
- Mô hình (`model.name`): bất kỳ mô hình encoder-only nào trên HuggingFace
- Tham số dedup: `dedup.threshold`, `dedup.num_perm`
- Chunking: `chunking.max_length`, `chunking.stride`, `chunking.aggregation`
- Training: learning rate, batch size, epochs, early stopping
