"""
Configuration file for Handwritten Mathematical Expression Recognition
using Vision Transformer
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
    
    # Model parameters
    IMAGE_SIZE = 224  # Input image size
    PATCH_SIZE = 12  # Patch size for Vision Transformer
    EMBED_DIM = 512   # Embedding dimension
    NUM_HEADS = 8    # Number of attention heads
    NUM_LAYERS = 8   # Number of transformer layers
    MLP_RATIO = 4.0   # MLP expansion ratio
    DROPOUT = 0.1     # Dropout rate
    
    # Sequence parameters
    MAX_SEQ_LENGTH = 256  # Maximum sequence length for LaTeX
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 50
    WARMUP_STEPS = 1000
    GRADIENT_CLIP = 1.0
    
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
    AUGMENT_PROB = 0.5
    ROTATION_RANGE = 5
    BRIGHTNESS_RANGE = 0.2
    CONTRAST_RANGE = 0.2
    
    # Evaluation
    BEAM_SIZE = 5  # For beam search during inference
    
    def __repr__(self):
        return f"Config(device={self.DEVICE}, batch_size={self.BATCH_SIZE}, lr={self.LEARNING_RATE})"


