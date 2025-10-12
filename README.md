# Handwritten Mathematical Expression Recognition

Hệ thống nhận dạng biểu thức toán học viết tay, được xây dựng bằng Python và PyTorch, sử dụng kiến trúc **Vision Transformer (ViT)** kết hợp với một **Transformer Decoder**. Mô hình này được thiết kế để chuyển đổi hình ảnh của các biểu thức toán học viết tay thành chuỗi LaTeX tương ứng.

## 🎯 Tính năng chính

-   **Pre-trained Vision Transformer**: Sử dụng mô hình `vit-small-patch16-224` đã được huấn luyện trước trên ImageNet-21k làm bộ mã hóa (encoder), giúp trích xuất đặc trưng hình ảnh mạnh mẽ.
-   **Transformer Decoder**: Sinh ra chuỗi LaTeX từ các đặc trưng hình ảnh đã được mã hóa.
-   **Data Augmentation**: Tích hợp các kỹ thuật tăng cường dữ liệu on-the-fly (xoay, thay đổi độ sáng/tương phản) để cải thiện độ bền của mô hình.
-   **Preprocessing Pipeline**: Cung cấp một pipeline tiền xử lý hiệu quả để chuẩn hóa và làm sạch dữ liệu đầu vào.
-   **Beam Search Inference**: Tùy chọn sử dụng beam search để cải thiện chất lượng của chuỗi LaTeX được sinh ra.
-   **Evaluation Metrics**: Đánh giá hiệu suất mô hình bằng các độ đo phổ biến như BLEU score và Edit Distance.

## 📁 Cấu trúc dự án

```
.
├── data/                     # Dữ liệu train_test
├── config.py                 # File cấu hình tập trung cho toàn bộ dự án
├── dataset.py                # Định nghĩa Dataset class và DataLoader
├── debug_check_labels.py     # (Dev) Script kiểm tra nhãn dữ liệu
├── debug_sanity_check.py     # (Dev) Script kiểm tra sự hợp lệ của một batch
├── demo.py                   # Script chạy demo tương tác
├── inference.py              # Script để thực thi nhận dạng & đánh giá
├── model.py                  # Định nghĩa kiến trúc Vision Transformer
├── preprocess.py             # Script để chạy pipeline tiền xử lý dữ liệu
├── test_image.py             # Script đơn giản để test một ảnh đơn lẻ
├── train.py                  # Script chính để huấn luyện mô hình
├── utils.py                  # Các hàm và lớp tiện ích chung
└── requirements.txt          # Danh sách các thư viện Python cần thiết

# --- CÁC THƯ MỤC SAU SẼ ĐƯỢC TẠO TỰ ĐỘNG (NẾU CHƯA CÓ) ---
# ├─- checkpoints/              # Model checkpoints (bị git ignore)
# ├─- logs/                     # TensorBoard logs (bị git ignore)
# └─- outputs/                  # Kết quả inference (bị git ignore)
```

## 🚀 Quy trình làm việc (Workflow)

Quy trình chuẩn để huấn luyện và sử dụng mô hình được chia thành 3 bước chính:

### 1. Cài đặt Môi trường

Đầu tiên, hãy chắc chắn rằng bạn đã cài đặt đầy đủ các thư viện cần thiết.

```bash
# Cài đặt toàn bộ dependencies
pip install -r requirements.txt
```

### 2. Tiền xử lý Dữ liệu

Trước khi huấn luyện, tất cả các ảnh trong bộ dữ liệu cần được xử lý (binarize, resize, padding) để đưa về một định dạng chuẩn. Quá trình này giúp tăng tốc độ training một cách đáng kể vì các tác vụ nặng đã được thực hiện trước.

Chạy script sau để bắt đầu:

```bash
# Script này sẽ đọc ảnh từ off_image_*, xử lý và lưu vào *_processed/
python preprocess.py
```

### 3. Huấn luyện Mô hình

Sau khi dữ liệu đã được tiền xử lý, bạn có thể bắt đầu quá trình huấn luyện.

```bash
# Bắt đầu training, Dataloader sẽ đọc trực tiếp từ thư mục *_processed/
python train.py
```

-   **Theo dõi quá trình training**: Sử dụng TensorBoard để xem các biểu đồ loss, accuracy, và learning rate.
    ```bash
    tensorboard --logdir logs
    ```
-   Checkpoints của mô hình sẽ được tự động lưu vào thư mục `checkpoints/`.

## 🔍 Inference (Dự đoán)

Sử dụng script `inference.py` để nhận dạng biểu thức trên một ảnh mới hoặc trên toàn bộ tập test.

### Nhận dạng trên một ảnh đơn lẻ

```bash
python inference.py --checkpoint checkpoints/best.pth --image path/to/your/image.bmp
```

### Đánh giá trên toàn bộ tập Test

```bash
python inference.py --checkpoint checkpoints/best.pth --evaluate
```

## ⚙️ Tùy chỉnh và Cấu hình

Toàn bộ các tham số quan trọng của mô hình và quá trình huấn luyện được quản lý tập trung tại `config.py`. Bạn có thể dễ dàng thay đổi các giá trị như:

-   `IMAGE_SIZE`, `PATCH_SIZE`
-   `EMBED_DIM`, `ENCODER_LAYERS`, `DECODER_LAYERS`
-   `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`

---

*Dự án này được phát triển cho mục đích nghiên cứu và học tập. Để triển khai trong môi trường production, có thể cần thêm các tối ưu hóa về hiệu suất và độ tin cậy.*


