# split_dataset.py
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
from config import Config


def split_data(config: Config):
    """
    Đọc một file caption lớn và chia nó thành 2 file train và validation.
    """
    print("Bắt đầu quá trình chia dữ liệu...")

    # Sử dụng file caption gốc (chưa có CLEAN_) hoặc file đã được dọn dẹp
    # Anh hãy đảm bảo đường dẫn này đúng với file caption tổng của anh
    source_caption_file = config.DATA_DIR / "train_caption.txt"

    if not source_caption_file.exists():
        print(f"Lỗi: Không tìm thấy file caption nguồn tại '{source_caption_file}'")
        return

    # Đọc tất cả các dòng từ file nguồn
    with open(source_caption_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    lines = [line for line in lines if line.strip()]  # Loại bỏ các dòng trống

    print(f"Đã đọc được {len(lines)} dòng từ file nguồn.")

    # Sử dụng train_test_split của sklearn để chia (đảm bảo tính ngẫu nhiên)
    # random_state=42 để đảm bảo lần nào chạy lại script này cũng cho ra cùng một kết quả chia
    train_lines, val_lines = train_test_split(
        lines,
        test_size=config.VAL_SPLIT,
        random_state=42
    )

    print(f"Chia thành {len(train_lines)} mẫu train và {len(val_lines)} mẫu validation.")

    # Ghi ra các file mới
    train_caption_path = config.DATA_DIR / "train_caption.txt"
    val_caption_path = config.DATA_DIR / "val_caption.txt"

    with open(train_caption_path, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    print(f"Đã lưu file train caption tại: {train_caption_path}")

    with open(val_caption_path, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    print(f"Đã lưu file validation caption tại: {val_caption_path}")

    print("\nHoàn tất! cập nhật config.py để sử dụng 2 file mới này.")


if __name__ == "__main__":
    cfg = Config()
    split_data(cfg)