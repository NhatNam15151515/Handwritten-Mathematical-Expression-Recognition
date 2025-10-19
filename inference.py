# inference.py
import textwrap

import torch
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import các thành phần cần thiết từ project của anh
from config import Config
from dataset import MathExpressionDataset
from model import MathExpressionVisionTransformer
from preprocess import _preprocess_image as standalone_preprocess
from preprocess import _resize_with_padding as standalone_resize

# --- Chế độ DEBUG ---
DEBUG = True


# ===================================================================================
# HÀM 1: HIỂN THỊ TENSOR ĐÃ XỬ LÝ
# ===================================================================================
def show_preprocessed_tensor(tensor: torch.Tensor, title: str = "Preprocessed Image (Input to Model)"):
    """
    Hủy chuẩn hóa, chuyển vị và hiển thị một image tensor.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = tensor.squeeze(0).cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    img_unnormalized = std * img_np + mean
    img_clipped = np.clip(img_unnormalized, 0, 1)

    plt.figure()
    plt.imshow(img_clipped)
    plt.title(title)
    plt.axis('off')
    plt.show(block=False)  # block=False để không chặn cửa sổ kết quả chính

def visualize_prediction(image_path: Path, latex_str: str):
    """
    Hiển thị ảnh gốc và in chuỗi kết quả dự đoán (không render LaTeX).
    Tự động loại bỏ dấu cách giữa các ký tự.
    """
    try:
        # Làm sạch chuỗi: bỏ dấu cách thừa và ký tự xuống dòng
        cleaned_text = latex_str.replace(" ", "").replace("\n", "")

        # Hiển thị ảnh
        image = Image.open(image_path)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)
        ax.axis("off")

        # Hiển thị chuỗi kết quả phía dưới ảnh
        fig.text(0.5, 0.05, cleaned_text,
                 ha='center', fontsize=12, color='black',
                 bbox={"facecolor": "white", "alpha": 0.9, "pad": 6})

        plt.suptitle("Inference Result", fontsize=18)
        plt.show()

        # In ra console cho dễ kiểm tra
        print(f"\nKết quả dự đoán: {cleaned_text}")

    except Exception as e:
        print(f"Lỗi khi hiển thị ảnh: {e}")

class InferenceEngine:
    def __init__(self, config: Config, checkpoint_path: Path):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        temp_dataset = MathExpressionDataset(
            processed_image_dir=config.TRAIN_PROCESSED_IMAGE_DIR,
            caption_file=config.TRAIN_CAPTION_FILE,
            dictionary_file=config.DICTIONARY_FILE,
            is_train=False
        )
        self.vocab = temp_dataset.vocab
        self.idx_to_token = temp_dataset.idx_to_token
        self.vocab_size = len(self.vocab)

        self.model = MathExpressionVisionTransformer(config=self.config, vocab_size=self.vocab_size)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _preprocess_image(self, image_path: Path) -> torch.Tensor:
        """Hàm này chỉ làm một nhiệm vụ: xử lý ảnh và trả về tensor."""
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise IOError(f"Không thể đọc ảnh tại: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        preprocessed_img = standalone_preprocess(image_rgb)
        resized_img = standalone_resize(preprocessed_img, self.config.IMAGE_SIZE)
        image_tensor = self.transform(resized_img)

        return image_tensor.unsqueeze(0).to(self.device)

    def predict(self, image_path: Path, top_k: int, temperature: float, visualize: bool = False):
        """Hàm dự đoán chính, có gọi hàm hiển thị nếu cần."""
        with torch.no_grad():
            # 1. Xử lý ảnh
            image_tensor = self._preprocess_image(image_path)

            # 2. Hiển thị ảnh đã xử lý NẾU có cờ visualize
            if visualize:
                show_preprocessed_tensor(image_tensor)

            # 3. Chạy dự đoán
            sos_id = self.vocab[Config.SOS_TOKEN]
            eos_id = self.vocab[Config.EOS_TOKEN]
            output_indices = self.model.generate(
                images=image_tensor, sos_token_id=sos_id, eos_token_id=eos_id,
                max_length=self.config.MAX_SEQ_LENGTH, temperature=temperature, top_k=top_k
            ).squeeze(0).cpu().numpy()

            # 4. Giải mã kết quả
            tokens = [self.idx_to_token.get(idx, Config.UNK_TOKEN) for idx in output_indices]
            if tokens and tokens[0] == Config.SOS_TOKEN: tokens = tokens[1:]
            if Config.EOS_TOKEN in tokens: tokens = tokens[:tokens.index(Config.EOS_TOKEN)]

            return " ".join(tokens)


def main(config: Config, args):
    engine = InferenceEngine(config, Path(args.checkpoint))

    if args.image:
        prediction = engine.predict(
            Path(args.image),
            args.top_k,
            args.temperature,
            visualize=args.visualize
        )
        print("\n--- DỰ ĐOÁN ---")
        print(f"Ảnh: {args.image}")
        print(f"Kết quả: {prediction}")

        if args.visualize:
            visualize_prediction(Path(args.image), prediction)
            plt.show()  # Lệnh này sẽ hiển thị tất cả các figure đã được tạo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy dự đoán cho mô hình HMER.")
    parser.add_argument('--checkpoint', type=str, help="Đường dẫn đến file checkpoint model (.pth).")
    parser.add_argument('--image', type=str, help="Đường dẫn đến một ảnh đơn lẻ để dự đoán.")
    parser.add_argument('--top_k', type=int, default=5, help="Sử dụng top-k sampling.")
    parser.add_argument('--temperature', type=float, default=1.0, help="Nhiệt độ cho sampling.")
    parser.add_argument('--visualize', action='store_true', help="Hiển thị ảnh gốc, ảnh đã xử lý và kết quả.")

    cfg = Config()

    if DEBUG:
        #"data/test_image/35_em_18.bmp"
        example_image_path = "data/test_image/516_em_383.bmp"

        debug_args = [
            '--checkpoint', 'checkpoints/best_88.pth',
            '--image', str(example_image_path),
            '--visualize'
        ]
        args = parser.parse_args(debug_args)
    else:
        args = parser.parse_args()

    if not args.checkpoint and not DEBUG:
        parser.error("Tham số --checkpoint là bắt buộc khi không ở chế độ DEBUG.")

    main(cfg, args)