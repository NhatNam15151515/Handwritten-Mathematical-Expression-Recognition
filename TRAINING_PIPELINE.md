# Pipeline huấn luyện ViT patch 2

Pipeline hiện có ba giai đoạn. Mỗi giai đoạn giữ checkpoint trong source của mình; stage 3 chạy từ `train.py` ở thư mục gốc.

## 1. Classifier ký tự đơn

Dữ liệu đã xử lý nằm tại `single_symbol_training/data/single_symbol_processed/`. Nếu cần train mới:

```powershell
.\.venv\Scripts\python.exe -m single_symbol_training.train --epochs 1000
```

## 2. Học chuỗi ký tự nhìn thấy trong ảnh công thức

Giai đoạn này học đọc các ký tự nhìn thấy và thứ tự tuyến tính, bỏ nhãn vị trí như mũ/chỉ số dưới. Nó khởi tạo encoder từ stage 1:

```powershell
.\.venv\Scripts\python.exe -m formula_symbol_training.train --epochs 150 --patience 15
```

Checkpoint stage 2 nằm tại `formula_symbol_training/checkpoints/vit_patch2_formula_symbols_grid44/`. Sau khi stage này hoàn tất, `latest.pth` chứa encoder patch 2 để chuyển sang stage 3. Stage 2 và stage 3 dùng chung ảnh CROHME đã chuẩn bị trong `formula_symbol_training/data/crohme_whywhs_source/processed_448/`.

## 3. Fine-tune trên chuỗi LaTeX đầy đủ

```powershell
.\.venv\Scripts\python.exe train.py --stage formula --epochs 150
```

Mặc định stage 3 nạp `vision_encoder_state_dict` từ `formula_symbol_training/checkpoints/vit_patch2_formula_symbols_grid44/best_generation.pth`. Nó không nạp decoder của stage 2 vì vocab chuỗi ký hiệu phẳng khác vocab LaTeX đầy đủ; decoder, optimizer và scheduler bắt đầu mới. Cấu hình dùng ảnh tối đa 448 px, patch 2, lưới 44×44, nhãn phụ main/upper/lower suy ra từ caption, và coverage attention CoMER-inspired.

Validation chạy greedy decoding theo cách thường và thêm một lượt có ràng buộc ngoặc cân bằng; cả hai BLEU/ExpRate được ghi vào lịch sử, còn file preview mỗi 5 epoch lưu cả hai dự đoán để so sánh. Việc này chỉ tác động đến giải mã/đánh giá, không thay đổi loss hay cập nhật trọng số. Hai lượt giải mã làm validation lâu hơn.

Checkpoint stage 3 lưu tại `checkpoints/vit_patch2_grid44_full_formula_position_coverage/`. Để tiếp tục sau khi bị ngắt, dùng:

```powershell
.\.venv\Scripts\python.exe train.py --stage formula --epochs 150 --resume-checkpoint checkpoints\vit_patch2_grid44_full_formula_position_coverage\latest.pth
```

Inference có thể bật ràng buộc ngoặc bằng `--balance-brackets`.

Tập test CROHME được giữ riêng và không dùng để chọn checkpoint.
