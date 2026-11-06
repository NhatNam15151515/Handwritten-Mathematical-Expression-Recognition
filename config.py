"""Configuration for the patch-2 ViT full-formula training pipeline."""
import torch
from pathlib import Path

class Config:
    # Dataset paths
    DATA_DIR = (Path(__file__).resolve().parent / "formula_symbol_training" / "data")
    CROHME_DIR = DATA_DIR / "crohme_whywhs_source"
    CROHME_PROCESSED_DIR = CROHME_DIR / "processed_448"
    CROHME_TRAIN_IMAGE_DIR = CROHME_PROCESSED_DIR / "train"
    CROHME_TEST_IMAGE_DIR = CROHME_PROCESSED_DIR / "test"

    TRAIN_IMAGE_DIR = CROHME_TRAIN_IMAGE_DIR
    TEST_IMAGE_DIR = CROHME_TEST_IMAGE_DIR
    VAL_IMAGE_DIR = CROHME_TRAIN_IMAGE_DIR

    TRAIN_CAPTION_FILE = CROHME_DIR / "splits" / "train_caption.txt"
    TEST_CAPTION_FILE = CROHME_DIR / "test_caption.txt"
    VAL_CAPTION_FILE = CROHME_DIR / "splits" / "val_caption.txt"

    DICTIONARY_FILE = CROHME_DIR / "dictionary.txt"
    # Formula images are resized offline; training only reads prepared PNGs.
    TRAIN_PROCESSED_IMAGE_DIR = CROHME_TRAIN_IMAGE_DIR
    TEST_PROCESSED_IMAGE_DIR = CROHME_TEST_IMAGE_DIR
    VAL_PROCESSED_IMAGE_DIR = CROHME_TRAIN_IMAGE_DIR
    SINGLE_SYMBOL_DATA_DIR = (Path(__file__).resolve().parent / "single_symbol_training" /
                              "data" / "single_symbol_native")
    TRAIN_STAGE = "formula"
    PROJECT_ROOT = Path(__file__).resolve().parent
    # Initialize the full-formula encoder from the preceding visible-symbol stage.
    INIT_CHECKPOINT = (PROJECT_ROOT / "formula_symbol_training" / "checkpoints" /
                       "vit_patch2_formula_symbols_grid44" / "best_generation.pth")

    # Model parameters
    # Full formulas continue from the patch-2 encoder trained on visible symbols.
    IMAGE_SIZE = 448
    PATCH_SIZE = 2
    # Retain the learned positional parameter shape and use the same capped grid
    # as the preceding formula-symbol stage.
    POSITION_EMBED_GRID = 14
    MAX_PATCH_GRID = 44
    RUN_NAME = f"vit_patch2_grid{MAX_PATCH_GRID}_full_formula_position_coverage"
    EMBED_DIM = 256
    NUM_HEADS = 8
    NUM_LAYERS = 4
    MLP_RATIO = 4.0
    DROPOUT = 0.15
    ATTENTION_DROPOUT = 0.1
    APPLY_PREPROCESSING = True
    SYMBOL_SAMPLES_PER_CLASS_PER_EPOCH = 128
    RANDOM_SEED = 20261023
    SYMBOL_BATCH_SIZE = 32
    # Keep the physical batch aligned with the 6 GB-safe patch-2 formula run.
    FORMULA_BATCH_SIZE = 4
    USE_POSITION_AUXILIARY = True
    POSITION_LOSS_WEIGHT = 0.2
    USE_COVERAGE = True
    # Compare ordinary and bracket-constrained greedy decoding on validation.
    COMPARE_BALANCED_BRACKET_DECODING = True
    PREVIEW_EVERY = 5

    # Sequence parameters
    MAX_SEQ_LENGTH = 100
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    IN_CHANNELS = 3

    # Training parameters
    BATCH_SIZE = 8             # Thử tận dụng VRAM với ViT Small
    ACCUMULATION_STEPS = 1     # Batch hiệu dụng vẫn là 8
    ENCODER_LR = 1e-4
    DECODER_LR = 3e-4
    FINETUNE_ENCODER_LR = 5e-6
    FINETUNE_DECODER_LR = 1e-4
    SYMBOL_ENCODER_LR = 3e-4
    SYMBOL_DECODER_LR = 3e-4
    SYMBOL_NUM_EPOCHS = 30
    MIN_LR = 1e-6
    WEIGHT_DECAY = 5e-4        # Nhẹ hơn để tránh quá regularize với dropout
    NUM_EPOCHS = 150            # Mức trần; early stopping quyết định khi nào dừng
    WARMUP_EPOCHS = 4
    GRADIENT_CLIP = 0.6

    # Mixed precision training để tiết kiệm memory
    USE_MIXED_PRECISION = True

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths for saving
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / RUN_NAME
    LOG_DIR = PROJECT_ROOT / "logs" / RUN_NAME
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / RUN_NAME

    # Create directories
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Data augmentation
    AUGMENT_PROB = 0.9
    ROTATION_RANGE = 10
    SCALE_RANGE = (0.85, 1.15)
    TRANSLATION_RANGE = 0.1
    SHEAR_RANGE = 5
    BRIGHTNESS_RANGE = 0.4
    CONTRAST_RANGE = 0.4
    NOISE_PROB = 0.4
    BLUR_PROB = 0.3

    # Regularization
    LABEL_SMOOTHING = 0.1      # Dùng với CE để giảm overfit
    MIXUP_ALPHA = 0.2
    CUTMIX_PROB = 0.3

    # Evaluation
    BEAM_SIZE = 5
    EVAL_EVERY_N_EPOCHS = 5  # Evaluate mỗi 5 epochs
    SAVE_EVERY_N_EPOCHS = 10  # Save checkpoint mỗi 10 epochs

    # Early stopping
    EARLY_STOPPING_METRIC = "val_exp_rate"
    EARLY_STOPPING_PATIENCE = 15
    EARLY_STOPPING_MIN_DELTA = 1e-4

    # Learning rate schedule
    SCHEDULER = "cosine_warmup"
    STEP_SIZE = 30
    GAMMA = 0.5


    # Memory optimization
    GRADIENT_CHECKPOINTING = False  # Mô hình nhỏ; tắt recompute để train nhanh hơn
    PIN_MEMORY = True              # Tăng tốc copy H2D
    NUM_WORKERS = 4               # 4 worker cho 1660S hợp lý

    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.1
    TRAIN_SPLIT = 0.75

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
        print(f"Maximum ViT feature grid: {self.MAX_PATCH_GRID}x{self.MAX_PATCH_GRID}")
        print(f"Model Dimension: {self.EMBED_DIM}")
        print(f"Number of Layers: {self.NUM_LAYERS}")
        print(f"Number of Heads: {self.NUM_HEADS}")
        print(f"Batch Size: {self.BATCH_SIZE} (Effective: {self.get_effective_batch_size()})")
        print(f"Learning Rate encoder: {self.ENCODER_LR}")
        print(f"Learning Rate decoder: {self.DECODER_LR}")
        print(f"Number of Epochs: {self.NUM_EPOCHS}")
        print(f"Mixed Precision: {self.USE_MIXED_PRECISION}")
        print(f"Gradient Checkpointing: {self.GRADIENT_CHECKPOINTING}")
        print("="*60)
