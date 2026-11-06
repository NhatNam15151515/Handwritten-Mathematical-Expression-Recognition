"""Settings for the flat visible-symbol sequence training stage."""
from pathlib import Path

from config import Config


class FormulaSymbolConfig(Config):
    TRAIN_STAGE = "flat_formula_symbols"
    RUN_NAME = "vit_patch2_formula_symbols_grid44"
    FLAT_DATA_DIR = (Path(__file__).resolve().parent / "data" / "crohme_whywhs_source" /
                     "flat_symbol_sequences")
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    INIT_CHECKPOINT = (PROJECT_ROOT / "single_symbol_training" / "checkpoints" /
                       "vit_patch2_single_symbol_classifier_processed" / "best.pth")

    IMAGE_SIZE = 448
    PATCH_SIZE = 2
    POSITION_EMBED_GRID = 14
    MAX_PATCH_GRID = 44
    EMBED_DIM = 256
    NUM_HEADS = 8
    NUM_LAYERS = 4
    MLP_RATIO = 4.0
    DROPOUT = 0.15
    ATTENTION_DROPOUT = 0.1

    FORMULA_BATCH_SIZE = 4
    BATCH_SIZE = FORMULA_BATCH_SIZE
    ACCUMULATION_STEPS = 2
    NUM_EPOCHS = 150
    # Decay learning rates over the full planned training window, even when
    # early stopping is enabled.
    LR_SCHEDULER_T_MAX = 150
    ENCODER_LR = 2e-5
    DECODER_LR = 1e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 5e-4
    GRADIENT_CLIP = 0.6
    EARLY_STOPPING_PATIENCE = 15
    EARLY_STOPPING_MIN_DELTA = 1e-4
    GENERATION_METRIC_MIN_DELTA = 1e-4
    PREVIEW_EVERY = 5

    CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints" / RUN_NAME
    LOG_DIR = Path("logs") / RUN_NAME
    OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / RUN_NAME
