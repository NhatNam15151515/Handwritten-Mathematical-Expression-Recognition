# dataset.py (Phiên bản đã tối ưu để đọc dữ liệu tiền xử lý)
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple

from config import Config
from preprocess import _preprocess_image, _resize_with_padding


class MathExpressionDataset(Dataset):
    """
    Dataset này đọc trực tiếp các ảnh ĐÃ ĐƯỢC TIỀN XỬ LÝ.
    Nó chỉ thực hiện các tác vụ nhẹ on-the-fly như Data Augmentation.
    """

    def __init__(self, processed_image_dir: Path, caption_file: Path, dictionary_file: Path, is_train: bool = True):
        self.image_dir = Path(processed_image_dir)  # THAY ĐỔI: Trỏ đến thư mục đã xử lý
        self.is_train = is_train

        self.vocab = self._load_vocabulary(dictionary_file)
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        self.data = self._load_captions(caption_file)

        # Transform chỉ bao gồm augmentation, không còn resize
        self.transform = self._get_transform()

    # CÁC HÀM HELPER GIỮ NGUYÊN
    def _load_vocabulary(self, dictionary_file: Path) -> Dict[str, int]:
        vocab = {Config.PAD_TOKEN: 0, Config.SOS_TOKEN: 1, Config.EOS_TOKEN: 2, Config.UNK_TOKEN: 3}
        with open(dictionary_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '\t' in line:
                    token, idx = line.split('\t')
                    vocab[token] = int(idx) + 4
        return vocab

    def _load_captions(self, caption_file: Path) -> List[Tuple[str, str]]:
        data = []
        with open(caption_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    data.append((parts[0], '\t'.join(parts[1:])))
        return data

    # BỎ: Các hàm _resize_with_padding và _preprocess_image đã được chuyển sang preprocess.py
    #      vì chúng không còn cần thiết trong quá trình training nữa.

    def _get_transform(self) -> A.Compose:
        normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.is_train:
            return A.Compose([
                A.Rotate(limit=Config.ROTATION_RANGE, p=Config.AUGMENT_PROB, border_mode=cv2.BORDER_CONSTANT,
                         value=[255, 255, 255]),
                A.RandomBrightnessContrast(brightness_limit=Config.BRIGHTNESS_RANGE,
                                           contrast_limit=Config.CONTRAST_RANGE, p=Config.AUGMENT_PROB),
                normalize,
                ToTensorV2()
            ])
        else:
            return A.Compose([normalize, ToTensorV2()])

    def _tokenize_caption(self, caption: str) -> List[int]:
        tokens = caption.split()
        indices = [self.vocab[Config.SOS_TOKEN]] + [self.vocab.get(t, self.vocab[Config.UNK_TOKEN]) for t in tokens] + [
            self.vocab[Config.EOS_TOKEN]]
        return indices

    def _pad_sequence(self, sequence: List[int], max_length: int) -> List[int]:
        if len(sequence) >= max_length: return sequence[:max_length]
        return sequence + [self.vocab[Config.PAD_TOKEN]] * (max_length - len(sequence))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        image_name, caption = self.data[idx]
        image_path = self.image_dir / f"{image_name}.png"

        try:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except (IOError, cv2.error):
            # Fallback nếu ảnh không tồn tại hoặc lỗi
            image = np.ones((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8) * 255

        transformed = self.transform(image=image)
        image_tensor = transformed['image']

        # Xử lý caption (giữ nguyên)
        token_indices = self._tokenize_caption(caption)
        input_seq = self._pad_sequence(token_indices[:-1], Config.MAX_SEQ_LENGTH)
        target_seq = self._pad_sequence(token_indices[1:], Config.MAX_SEQ_LENGTH)

        return {
            'image': image_tensor,
            'input_seq': torch.tensor(input_seq, dtype=torch.long),
            'target_seq': torch.tensor(target_seq, dtype=torch.long),
            'caption': caption,
            'image_name': image_name
        }


def _pad_collate_fn(batch):
    """
    Phiên bản đơn giản hóa: vì tất cả ảnh đã được pad thành kích thước vuông
    trong __getitem__, hàm này chỉ cần gom các tensor lại.
    """
    # Lấy danh sách các ảnh từ batch
    images = [item['image'] for item in batch]

    # Gom chúng lại thành một tensor duy nhất
    # PyTorch sẽ tự động xử lý vì chúng có cùng kích thước
    images_tensor = torch.stack(images, dim=0)

    # Trả về batch đã được định dạng
    return {
        'image': images_tensor,
        'input_seq': torch.stack([item['input_seq'] for item in batch]),
        'target_seq': torch.stack([item['target_seq'] for item in batch]),
        'caption': [item['caption'] for item in batch],
        'image_name': [item['image_name'] for item in batch]
    }


def create_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader]:
    print("Tạo dataloader từ dữ liệu ĐÃ TIỀN XỬ LÝ.")

    train_dataset = MathExpressionDataset(
        # THAY ĐỔI: Sử dụng thư mục ảnh đã xử lý
        processed_image_dir=config.TRAIN_PROCESSED_IMAGE_DIR,
        caption_file=config.TRAIN_CAPTION_FILE,
        dictionary_file=config.DICTIONARY_FILE,
        is_train=True
    )
    test_dataset = MathExpressionDataset(
        # THAY ĐỔI: Sử dụng thư mục ảnh đã xử lý
        processed_image_dir=config.TEST_PROCESSED_IMAGE_DIR,
        caption_file=config.TEST_CAPTION_FILE,
        dictionary_file=config.DICTIONARY_FILE,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=True,
        collate_fn=_pad_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        collate_fn=_pad_collate_fn
    )

    return train_loader, test_loader


def visualize_preprocessing(image_path, config):
    """
    Visualize the preprocessing steps.
    This function now uses the standalone processing functions.
    """
    import matplotlib.pyplot as plt

    # Load original image
    original = cv2.imread(str(image_path))
    if original is None:
        print(f"Error: Could not load image at {image_path}")
        return None
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Apply preprocessing using imported functions
    preprocessed = _preprocess_image(original_rgb)
    resized = _resize_with_padding(preprocessed, config.IMAGE_SIZE)

    # Calculate stats
    h, w = original_rgb.shape[:2]
    aspect_ratio = w / h if h > 0 else 0

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 12))

    axes[0].imshow(original_rgb)
    axes[0].set_title(f'Original: {w}x{h}\nAspect ratio: {aspect_ratio:.2f}')
    axes[0].axis('off')

    axes[1].imshow(preprocessed)
    axes[1].set_title('Preprocessed\n(Binarization)')
    axes[1].axis('off')

    axes[2].imshow(resized)
    axes[2].set_title(f'Final: {resized.shape[1]}x{resized.shape[0]}\n(Resized with padding)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    return resized


def compare_resize_methods(image_path, target_size=224):
    """
    Compare different resize methods.
    This function now uses the standalone resize function.
    """
    import matplotlib.pyplot as plt

    original = cv2.imread(str(image_path))
    if original is None:
        print(f"Error: Could not load image at {image_path}")
        return
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    h, w = original_rgb.shape[:2]

    # Method 1: Direct resize (BAD - distorts aspect ratio)
    direct_resize = cv2.resize(original_rgb, (target_size, target_size))

    # Method 2: Crop to square then resize (BAD - loses information)
    size = min(h, w)
    start_h = (h - size) // 2
    start_w = (w - size) // 2
    cropped = original_rgb[start_h:start_h + size, start_w:start_w + size]
    crop_resize = cv2.resize(cropped, (target_size, target_size))

    # Method 3: Our method (GOOD - preserves aspect ratio) using the imported function
    aspect_preserve = _resize_with_padding(original_rgb, target_size)

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title(f'Original: {w}x{h}\nAspect: {w / h}')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(direct_resize)
    axes[0, 1].set_title('Direct resize\n(DISTORTED)')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(crop_resize)
    axes[1, 0].set_title('Crop then resize\n(INFORMATION LOST)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(aspect_preserve)
    axes[1, 1].set_title(f'Aspect-preserving resize\n(RECOMMENDED) \n{aspect_preserve.shape[0]}x{aspect_preserve.shape[1]}\nAspect: {w / h}')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test dataset
    config = Config()
    train_loader, test_loader = create_dataloaders(config)
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    print(f"Vocabulary size: {len(train_loader.dataset.vocab)}")
    batch = next(iter(train_loader))
    print(f"Image shape: {batch['image'].shape}")
    print(f"Sample caption: {batch['caption'][0]}")
