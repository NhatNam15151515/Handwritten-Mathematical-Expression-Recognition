# Handwritten Mathematical Expression Recognition với Vision Transformer

Hệ thống nhận dạng biểu thức toán học viết tay sử dụng mô hình Vision Transformer (ViT) với kiến trúc encoder-decoder.

## 🎯 Tính năng chính

- **Vision Transformer Encoder**: Mã hóa ảnh thành sequence embeddings
- **Transformer Decoder**: Sinh ra LaTeX sequence từ image features
- **Data Augmentation**: Tăng cường dữ liệu với rotation, brightness, contrast
- **Beam Search**: Tìm kiếm tối ưu cho inference
- **Attention Visualization**: Trực quan hóa vùng chú ý của model
- **BLEU Score Evaluation**: Đánh giá chất lượng với BLEU metric

## 📁 Cấu trúc dự án

```
├── data/                          # Dữ liệu training/testing
│   ├── off_image_train/          # Ảnh training (8835 images)
│   ├── off_image_test/           # Ảnh testing (986 images)
│   ├── train_caption.txt         # LaTeX labels cho training
│   ├── test_caption.txt          # LaTeX labels cho testing
│   └── dictionary.txt            # Từ điển 112 tokens LaTeX
├── config.py                     # Cấu hình model và training
├── dataset.py                    # Dataset class và data loading
├── model.py                      # Vision Transformer architecture
├── train.py                      # Script training
├── inference.py                  # Script inference và evaluation
├── utils.py                      # Utility functions
├── requirements.txt              # Dependencies
├── checkpoints/                  # Model checkpoints
├── logs/                         # Tensorboard logs
└── outputs/                      # Kết quả inference
```

## ⚙️ Cài đặt

1. **Cài đặt dependencies:**
```bash
pip install -r requirements.txt
```

2. **Kiểm tra dữ liệu:**
```bash
python dataset.py
```

## 🚀 Training

### Cấu hình Model
- **Image Size**: 224x224
- **Patch Size**: 16x16 (196 patches)
- **Embed Dim**: 768
- **Encoder Layers**: 12
- **Decoder Layers**: 6
- **Attention Heads**: 12
- **Max Sequence Length**: 256 tokens

### Bắt đầu training:
```bash
python train.py
```

### Các tham số training chính:
- **Batch Size**: 16
- **Learning Rate**: 1e-4 (với CosineAnnealingLR)
- **Weight Decay**: 1e-4
- **Gradient Clipping**: 1.0
- **Epochs**: 100

### Theo dõi training:
```bash
tensorboard --logdir logs
```

## 🔍 Inference

### Nhận dạng ảnh đơn lẻ:
```bash
python inference.py --checkpoint checkpoints/best.pth --image path/to/image.bmp --visualize
```

### Batch processing:
```bash
python inference.py --checkpoint checkpoints/best.pth --image_dir data/off_image_test --output_dir outputs
```

### Đánh giá trên test set:
```bash
python inference.py --checkpoint checkpoints/best.pth --evaluate
```

### Beam search inference:
```bash
python inference.py --checkpoint checkpoints/best.pth --image image.bmp --beam_size 5 --temperature 0.8
```

## 📊 Kiến trúc Model

### Vision Transformer Encoder
```python
Input Image (224x224x3)
    ↓
Patch Embedding (196x768)
    ↓
+ Position Embedding
    ↓
12x Transformer Blocks
    ↓
Layer Norm
    ↓
Visual Features (197x768)  # +1 for CLS token
```

### Transformer Decoder
```python
LaTeX Tokens
    ↓
Token Embedding (768)
    ↓
+ Positional Encoding
    ↓
6x Decoder Blocks (Self-Attention + Cross-Attention)
    ↓
Layer Norm
    ↓
Linear Head (vocab_size)
    ↓
LaTeX Sequence
```

## 📈 Metrics và Evaluation

- **Cross-Entropy Loss**: Với ignore padding tokens
- **Token Accuracy**: Accuracy trên từng token (loại trừ padding)
- **BLEU Score**: Đánh giá chất lượng sequence generation
- **Edit Distance**: Khoảng cách chỉnh sửa giữa prediction và target

## 🎨 Data Augmentation

- **Rotation**: ±5 degrees
- **Brightness**: ±20%
- **Contrast**: ±20%
- **Gaussian Noise**: Thêm noise ngẫu nhiên
- **Normalization**: ImageNet statistics

## 💡 Tối ưu hóa

### Memory Optimization:
- Gradient checkpointing (có thể thêm)
- Mixed precision training (có thể thêm)
- Efficient attention mechanisms

### Speed Optimization:
- DataLoader với num_workers=4
- Pin memory cho GPU
- Batch processing cho inference

## 📝 Ví dụ sử dụng

```python
from inference import MathExpressionInference
from pathlib import Path

# Load model
inference = MathExpressionInference(Path("checkpoints/best.pth"))

# Predict single image
latex, tokens, confidence = inference.predict(Path("image.bmp"))
print(f"LaTeX: {latex}")
print(f"Confidence: {confidence.mean():.3f}")

# Visualize prediction
inference.visualize_prediction(Path("image.bmp"), save_path=Path("result.png"))
```

## 🔧 Tùy chỉnh

### Thay đổi cấu hình trong `config.py`:
```python
# Model parameters
IMAGE_SIZE = 224
PATCH_SIZE = 16
EMBED_DIM = 768
NUM_LAYERS = 12

# Training parameters  
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
```

### Thêm augmentation mới trong `dataset.py`:
```python
A.ElasticTransform(p=0.2),
A.GridDistortion(p=0.2),
A.OpticalDistortion(p=0.2)
```

## 📊 Kết quả mong đợi

Với dataset hiện tại (8835 train + 986 test samples):
- **Training Accuracy**: ~85-95%
- **Validation Accuracy**: ~80-90%
- **BLEU Score**: ~0.6-0.8
- **Training Time**: ~10-20 giờ trên GPU

## 🐛 Troubleshooting

### Memory issues:
- Giảm `BATCH_SIZE` trong config
- Giảm `IMAGE_SIZE` hoặc `EMBED_DIM`

### Convergence issues:
- Giảm `LEARNING_RATE`
- Tăng `WARMUP_STEPS`
- Kiểm tra data quality

### Inference issues:
- Kiểm tra image format (BMP, PNG, JPG)
- Đảm bảo image không bị corrupt
- Kiểm tra vocabulary consistency

## 📚 Tài liệu tham khảo

- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Mathematical Expression Recognition](https://arxiv.org/abs/2207.04410)

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📄 License

Dự án này được phân phối dưới MIT License. Xem `LICENSE` file để biết thêm chi tiết.

---

**Lưu ý**: Đây là implementation cho mục đích nghiên cứu và học tập. Để sử dụng trong production, cần thêm các tối ưu hóa về performance và robustness.


