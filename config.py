"""
Configuration file for Handwritten Mathematical Expression Recognition
Optimized for GTX 1660 Super (6GB VRAM) with ~9000 samples
"""
import torch
from pathlib import Path

class Config:
    # Dataset paths
    DATA_DIR = Path("./data")
    TRAIN_IMAGE_DIR = DATA_DIR / "off_image_train"
    TEST_IMAGE_DIR = DATA_DIR / "off_image_test"
    TRAIN_CAPTION_FILE = DATA_DIR / "train_caption.txt"
    TEST_CAPTION_FILE = DATA_DIR / "test_caption.txt"
    DICTIONARY_FILE = DATA_DIR / "dictionary.txt"
    TRAIN_PROCESSED_IMAGE_DIR = DATA_DIR / "train_processed"
    TEST_PROCESSED_IMAGE_DIR = DATA_DIR / "test_processed"

    # Model parameters
    IMAGE_SIZE = 224
    PATCH_SIZE = 16            # Ít patch hơn (18x18 grid), nhanh hơn
    EMBED_DIM = 256           # Giữ embedding vừa phải
    NUM_HEADS = 8
    NUM_LAYERS = 2             # Encoder 4, decoder ~2-3 (đã set trong model)
    MLP_RATIO = 4.0
    DROPOUT = 0.35            # Tăng nhẹ để giảm overfit
    ATTENTION_DROPOUT = 0.25   # Tăng nhẹ để regularize
    APPLY_PREPROCESSING = True

    # Sequence parameters
    MAX_SEQ_LENGTH = 100
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    # Training parameters
    BATCH_SIZE = 6             # Điều chỉnh theo VRAM
    ACCUMULATION_STEPS = 3     # Tăng tích lũy để batch hiệu quả lớn hơn
    LEARNING_RATE = 1.5e-4       # Học nhanh hơn một chút, có cosine warmup
    MIN_LR = 1e-6
    WEIGHT_DECAY = 2e-4        # Nhẹ hơn để tránh quá regularize với dropout
    NUM_EPOCHS = 60            # Thêm epoch để BLEU lên đều (nhanh/epoch)
    WARMUP_EPOCHS = 4
    GRADIENT_CLIP = 0.6

    # Mixed precision training để tiết kiệm memory
    USE_MIXED_PRECISION = True

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths for saving
    CHECKPOINT_DIR = Path("checkpoints")
    LOG_DIR = Path("logs")
    OUTPUT_DIR = Path("outputs")

    # Create directories
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Data augmentation
    AUGMENT_PROB = 0.9
    ROTATION_RANGE = 12
    SCALE_RANGE = (0.9, 1.1)
    TRANSLATION_RANGE = 0.05
    SHEAR_RANGE = 5
    BRIGHTNESS_RANGE = 0.35
    CONTRAST_RANGE = 0.35
    NOISE_PROB = 0.2
    BLUR_PROB = 0.1

    # Regularization
    LABEL_SMOOTHING = 0.1      # Dùng với CE để giảm overfit
    MIXUP_ALPHA = 0.2
    CUTMIX_PROB = 0.3

    # Evaluation
    BEAM_SIZE = 5
    EVAL_EVERY_N_EPOCHS = 5  # Evaluate mỗi 5 epochs
    SAVE_EVERY_N_EPOCHS = 10  # Save checkpoint mỗi 10 epochs

    # Early stopping
    EARLY_STOPPING_PATIENCE = 5  # Dừng nếu không improve sau 20 epochs

    # Learning rate schedule
    SCHEDULER = "cosine_warmup"
    STEP_SIZE = 30
    GAMMA = 0.5

    # Dataset split
    TRAIN_SPLIT = 0.85  # 85% cho training
    VAL_SPLIT = 0.15   # 15% cho validation

    # Memory optimization
    GRADIENT_CHECKPOINTING = True  # Tiết kiệm memory
    PIN_MEMORY = True              # Tăng tốc copy H2D
    NUM_WORKERS = 2               # 4 worker cho 1660S hợp lý

    def __repr__(self):
        return (f"Config(device={self.DEVICE}, batch_size={self.BATCH_SIZE}, "
                f"lr={self.LEARNING_RATE}, epochs={self.NUM_EPOCHS})")

    def get_effective_batch_size(self):
        """Get effective batch size với gradient accumulation"""
        return self.BATCH_SIZE * self.ACCUMULATION_STEPS

    def print_config(self):
        """Print configuration details"""
        print("="*60)
        print("CONFIGURATION")
        print("="*60)
        print(f"Device: {self.DEVICE}")
        print(f"Image Size: {self.IMAGE_SIZE}x{self.IMAGE_SIZE}")
        print(f"Patch Size: {self.PATCH_SIZE}")
        print(f"Model Dimension: {self.EMBED_DIM}")
        print(f"Number of Layers: {self.NUM_LAYERS}")
        print(f"Number of Heads: {self.NUM_HEADS}")
        print(f"Batch Size: {self.BATCH_SIZE} (Effective: {self.get_effective_batch_size()})")
        print(f"Learning Rate: {self.LEARNING_RATE}")
        print(f"Number of Epochs: {self.NUM_EPOCHS}")
        print(f"Mixed Precision: {self.USE_MIXED_PRECISION}")
        print(f"Gradient Checkpointing: {self.GRADIENT_CHECKPOINTING}")
        print("="*60)