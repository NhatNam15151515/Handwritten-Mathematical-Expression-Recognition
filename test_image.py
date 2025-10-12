# visualize.py

from pathlib import Path
from config import Config
from dataset import visualize_preprocessing, compare_resize_methods  # Import hàm tiện ích của anh

# 1. Khởi tạo cấu hình
config = Config()

# 2. Chỉ định đường dẫn đến ảnh test
#    (Đặt file này và ảnh test_image.png trong cùng thư mục gốc)
image_path = Path("test_1.bmp")
# 3. Kiểm tra xem file có tồn tại không
if not image_path.exists():
    print(f"Lỗi: Không tìm thấy file ảnh tại '{image_path}'.")
    print("Hãy chắc chắn bạn đã lưu ảnh và đặt đúng tên.")
else:
    print(f"Đang xử lý và hiển thị ảnh: {image_path}")
    # 4. Gọi hàm visualize mà anh đã viết
    visualize_preprocessing(image_path, config)
    print(f"{compare_resize_methods(image_path)}")
