# Handwritten Mathematical Expression Recognition

Hệ thống nhận dạng biểu thức toán học viết tay, được xây dựng bằng Python và PyTorch, sử dụng kiến trúc **Vision Transformer (ViT)** kết hợp với một **Transformer Decoder**. Mô hình này được thiết kế để chuyển đổi hình ảnh của các biểu thức toán học viết tay thành chuỗi LaTeX tương ứng.

## Tính năng chính

-   **Vision Transformer patch-2**: khởi tạo encoder từ classifier ký tự đơn, tiếp tục học chuỗi ký tự nhìn thấy, rồi fine-tune trên công thức đầy đủ.
-   **Transformer Decoder**: Sinh ra chuỗi LaTeX từ các đặc trưng hình ảnh đã được mã hóa.
-   **Data Augmentation**: Tích hợp các kỹ thuật tăng cường dữ liệu on-the-fly (xoay, thay đổi độ sáng/tương phản) để cải thiện độ bền của mô hình.
-   **Dữ liệu CROHME**: Tiền xử lý ảnh trước khi train; giữ nguyên tỉ lệ và không resize trong vòng lặp huấn luyện.
-   **Beam Search Inference**: Tùy chọn sử dụng beam search để cải thiện chất lượng của chuỗi LaTeX được sinh ra.
-   **Evaluation Metrics**: Đánh giá hiệu suất mô hình bằng các độ đo phổ biến như BLEU score và Edit Distance.

## Cấu trúc dự án

```
.
├── formula_symbol_training/  # Stage chuỗi ký hiệu và dữ liệu CROHME dùng chung
├── single_symbol_training/   # Stage classifier ký tự đơn và dữ liệu đã xử lý
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

## Quy trình làm việc (Workflow)

Quy trình huấn luyện hiện gồm ba giai đoạn. Hai giai đoạn đầu có source riêng; giai đoạn cuối dùng `train.py` ở thư mục gốc.

### 1. Cài đặt Môi trường

Đầu tiên, hãy chắc chắn rằng bạn đã cài đặt đầy đủ các thư viện cần thiết.

```bash
# Cài đặt toàn bộ dependencies
pip install -r requirements.txt
```

### 2. Dữ liệu

Pipeline dùng dữ liệu ký tự đơn trong `single_symbol_training/data/` và CROHME trong `formula_symbol_training/data/crohme_whywhs_source/`. Trước khi train, chạy `prepare_formula_data.py` để đảo nền và giới hạn cạnh dài tối đa 448 px, giữ nguyên tỉ lệ ảnh. Train đọc các ảnh đã xử lý, không resize trong lúc học. Tập train được chia cố định thành train/validation, còn 986 ảnh test được giữ riêng.

Lưu ý giấy phép: điều khoản CROHME hiện ghi dữ liệu chỉ dành cho nghiên cứu, không dùng thương mại. Cần xin quyền riêng trước khi đưa mô hình hoặc dữ liệu này vào sản phẩm thương mại.

### 3. Huấn luyện Mô hình

Bây giờ bạn có thể bắt đầu quá trình huấn luyện.

```powershell
# Giai đoạn 1: classifier ký tự đơn patch 2 (nếu cần train lại)
.\.venv\Scripts\python.exe -m single_symbol_training.train --epochs 1000

# Giai đoạn 2: học chuỗi ký tự nhìn thấy trong ảnh công thức
.\.venv\Scripts\python.exe -m formula_symbol_training.train --epochs 150 --patience 15

# Giai đoạn 3: học chuỗi LaTeX đầy đủ; mặc định nạp encoder patch 2 từ stage 2
.\.venv\Scripts\python.exe train.py --stage formula --epochs 150
```

Giai đoạn 3 nạp `vision_encoder_state_dict` từ checkpoint sinh chuỗi tốt nhất `formula_symbol_training/checkpoints/vit_patch2_formula_symbols_grid44/best_generation.pth`; decoder LaTeX được khởi tạo mới vì vocab chuỗi ký hiệu phẳng khác vocab LaTeX đầy đủ. Stage này train trên caption LaTeX gốc, thêm loss phụ dự đoán vai trò main/superscript/subscript từ cú pháp caption, cùng coverage attention lấy cảm hứng từ CoMER. Checkpoint, log và output được lưu theo run `vit_patch2_grid44_full_formula_position_coverage` trong các thư mục tương ứng.

-   **Theo dõi quá trình training**: Sử dụng TensorBoard để xem các biểu đồ loss, accuracy, và learning rate.
    ```bash
    tensorboard --logdir logs
    ```
-   Checkpoints stage 1 và 2 nằm trong source tương ứng; stage 3 lưu tại `checkpoints/vit_patch2_grid44_full_formula_position_coverage/`.

## Inference (Dự đoán)

Sử dụng script `inference.py` để nhận dạng biểu thức trên một ảnh mới hoặc trên toàn bộ tập test.
Có thể tải về và sử dụng các tệp trọng số đã train sẵn theo đường link này: https://drive.google.com/file/d/1MqvF4QpjNeEvj__FAJki1yXzQ6vMGNtD/view?usp=sharing
sau đó giải nén vào thư mục checkpoints
tệp best_91_base là tệp tốt nhất

### Nhận dạng trên một ảnh đơn lẻ

```bash
python inference.py --checkpoint checkpoints/vit_patch2_grid44_full_formula_position_coverage/best.pth --image path/to/your/image.png --balance-brackets
```

### Đánh giá trên toàn bộ tập Test

```bash
python inference.py --checkpoint checkpoints/vit_patch2_grid44_full_formula_position_coverage/best.pth --evaluate
```

## ⚙️ Tùy chỉnh và Cấu hình

Toàn bộ các tham số quan trọng của mô hình và quá trình huấn luyện được quản lý tập trung tại `config.py`. Bạn có thể dễ dàng thay đổi các giá trị như:

-   Encoder full-formula dùng patch 2, ảnh tiền xử lý tối đa 448 px và lưới đặc trưng tối đa 44×44. Checkpoint patch 4 không tương thích. Giai đoạn cuối khởi tạo encoder từ checkpoint patch-2 của stage chuỗi ký hiệu; decoder và optimizer bắt đầu mới.

-   `IMAGE_SIZE`, `PATCH_SIZE`
-   `EMBED_DIM`, `ENCODER_LAYERS`, `DECODER_LAYERS`
-   `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`

---

*Dự án này được phát triển cho mục đích nghiên cứu và học tập. Để triển khai trong môi trường production, có thể cần thêm các tối ưu hóa về hiệu suất và độ tin cậy.*


## Bộ dữ liệu sồ và chữ (dùng ByClass)
https://www.nist.gov/itl/products-and-services/emnist-dataset
