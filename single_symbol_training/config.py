"""Configuration for isolated patch-2 single-symbol classifier training."""
from pathlib import Path

from config import Config


class SingleSymbolConfig(Config):
    IMAGE_SIZE = 32
    PATCH_SIZE = 2
    MAX_PATCH_GRID = 30
    DATA_ROOT = Path(__file__).resolve().parent / "data"
    SINGLE_SYMBOL_IMAGE_DIR = DATA_ROOT / "single_symbol_native"
    SINGLE_SYMBOL_DATA_DIR = DATA_ROOT / "single_symbol_processed"
    TRAIN_STAGE = "symbols_classifier"
    RUN_NAME = "vit_patch2_single_symbol_classifier_processed"
    # Map selected capital-letter labels to lowercase labels for this classifier.
    # Formula captions retain their original capitalization labels.
    CLASS_ALIASES = {
        "C": "c",
        "P": "p",
        "S": "s",
        "V": "v",
        "X": "x",
        "Y": "y",
        "o": "0",
    }

    EPOCHS = 1000
    EARLY_STOPPING_PATIENCE = 12
    PREVIEW_EVERY = 5
    LEARNING_RATE = 3e-4
    GRADIENT_CLIP = 1.0

    CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints" / RUN_NAME
    LOG_DIR = Path("logs") / RUN_NAME
    OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / RUN_NAME
