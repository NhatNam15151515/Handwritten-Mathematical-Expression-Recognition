# debug_check_labels.py
import matplotlib.pyplot as plt
import random
import numpy as np

from config import Config
from dataset import MathExpressionDataset

def un_normalize_image(tensor):
    """Chuyển tensor đã normalize về lại ảnh xem được."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Chuyển từ (C, H, W) sang (H, W, C)
    tensor = tensor.permute(1, 2, 0)

    # Đảo ngược quá trình normalize
    image = tensor.numpy() * std + mean
    image = np.clip(image, 0, 1)
    return image


def check_labels(num_samples=20):
    config = Config()

    # Tải dataset (không cần Dataloader)
    # is_train=False để tắt augmentation khi kiểm tra
    train_dataset = MathExpressionDataset(
        processed_image_dir=config.TRAIN_PROCESSED_IMAGE_DIR,
        caption_file=config.TRAIN_CAPTION_FILE,
        dictionary_file=config.DICTIONARY_FILE,
        is_train=False
    )

    print(f"Displaying {num_samples} random samples for visual inspection...")

    for i in range(num_samples):
        # Lấy một mẫu ngẫu nhiên
        idx = random.randint(0, len(train_dataset) - 1)
        sample = train_dataset[idx]

        image_tensor = sample['image']
        caption = sample['caption']

        # Hiển thị ảnh
        image_to_show = un_normalize_image(image_tensor)

        plt.imshow(image_to_show)
        plt.title(f"Caption: {caption}", fontsize=10)
        plt.axis('off')
        print(f"Sample {i + 1}/{num_samples} - Showing image for caption: '{caption}'")
        plt.show()


if __name__ == "__main__":
    check_labels()