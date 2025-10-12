# preprocess.py
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# Giả sử file config.py nằm cùng cấp
from config import Config


# =========================================================================================
# CÁC HÀM XỬ LÝ ẢNH (COPY TỪ FILE DATASET.PY CŨ)
# Chúng được chuyển ra đây để trở thành các hàm độc lập, không phụ thuộc vào class.
# =========================================================================================

def _preprocess_image(image: np.ndarray) -> np.ndarray:
    """Chuyển ảnh sang dạng nhị phân (nền trắng, chữ đen)."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Dùng THRESH_OTSU để tự động tìm ngưỡng.
    # Nền trắng, chữ đen sẽ có giá trị 0.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Đảo ngược lại để có nền trắng (255), chữ đen (0) cho nhất quán
    # Bước này có thể không cần thiết nếu logic sau đã xử lý
    # Nhưng để đảm bảo, chúng ta kiểm tra tỉ lệ màu trắng
    white_ratio = (binary > 127).mean()
    if white_ratio < 0.5:
        binary = 255 - binary

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


def _resize_with_padding(image: np.ndarray, target_size: int) -> np.ndarray:
    """Resize ảnh về kích thước vuông, giữ nguyên tỉ lệ và thêm padding trắng."""
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_size, target_size, 3), 255, dtype=np.uint8)

    scale = target_size / w
    new_w = target_size
    new_h = max(1, int(h * scale))

    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    if new_h >= target_size:
        return cv2.resize(resized_image, (target_size, target_size), interpolation=cv2.INTER_AREA)

    final_image = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
    pad_top = (target_size - new_h) // 2
    final_image[pad_top: pad_top + new_h, :] = resized_image
    return final_image


def process_and_save(image_info: tuple, config: Config, dest_dir: Path):
    """
    Hàm thực hiện toàn bộ pipeline cho một ảnh và lưu lại.
    Đây là hàm sẽ được chạy song song trên nhiều nhân CPU.
    """
    image_name, source_dir = image_info

    # Tìm file ảnh với các extension khả dụng
    possible_extensions = ['.bmp', '.png', '.jpg', '.jpeg']
    found_path = None
    for ext in possible_extensions:
        p = (source_dir / image_name).with_suffix(ext)
        if p.exists():
            found_path = p
            break

    if not found_path:
        # print(f"Cảnh báo: Không tìm thấy ảnh {image_name} trong {source_dir}")
        return

    try:
        # BƯỚC 1: Đọc ảnh
        image = cv2.imread(str(found_path))
        if image is None: return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # BƯỚC 2: Tiền xử lý (binarize)
        if config.APPLY_PREPROCESSING:
            image = _preprocess_image(image)

        # BƯỚC 3: Resize giữ tỉ lệ
        image = _resize_with_padding(image, config.IMAGE_SIZE)

        # BƯỚC 4: Lưu ảnh đã xử lý (dùng định dạng PNG để không mất chất lượng)
        output_path = (dest_dir / image_name).with_suffix('.png')
        cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {found_path}: {e}")


def get_image_list_from_caption(caption_file: Path, source_dir: Path) -> list:
    """Đọc danh sách tên ảnh từ file caption."""
    image_infos = []
    with open(caption_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 1:
                image_name = parts[0]
                image_infos.append((image_name, source_dir))
    return image_infos


def main():
    """Hàm chính để chạy toàn bộ quá trình tiền xử lý."""
    start_time = time.time()
    config = Config()

    # Cấu hình các thư mục
    datasets_to_process = {
        "train": (config.TRAIN_IMAGE_DIR, config.TRAIN_CAPTION_FILE, config.TRAIN_PROCESSED_IMAGE_DIR),
        "test": (config.TEST_IMAGE_DIR, config.TEST_CAPTION_FILE, config.TEST_PROCESSED_IMAGE_DIR)
    }

    for name, (source_dir, caption_file, dest_dir) in datasets_to_process.items():
        print(f"\nBắt đầu xử lý bộ dữ liệu: {name.upper()}")
        print(f"Thư mục nguồn: {source_dir}")
        print(f"Thư mục đích:  {dest_dir}")

        # Tạo thư mục đích nếu chưa có
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Lấy danh sách ảnh cần xử lý
        image_infos = get_image_list_from_caption(caption_file, source_dir)
        print(f"Tìm thấy {len(image_infos)} ảnh để xử lý.")

        # Sử dụng multiprocessing để tăng tốc
        # Lấy số nhân CPU, trừ đi 1 để hệ thống không bị treo
        num_processes = max(1, cpu_count() - 1)
        print(f"Sử dụng {num_processes} nhân CPU để xử lý song song...")

        # Tạo một phiên bản của hàm process_and_save với các tham số cố định
        processor = partial(process_and_save, config=config, dest_dir=dest_dir)

        with Pool(processes=num_processes) as pool:
            # Dùng imap_unordered để xử lý ngay khi có kết quả, bọc ngoài bằng tqdm để có progress bar
            list(tqdm(pool.imap_unordered(processor, image_infos), total=len(image_infos)))

    end_time = time.time()
    print(f"\nHoàn tất toàn bộ quá trình tiền xử lý trong {end_time - start_time:.2f} giây.")


if __name__ == "__main__":
    main()