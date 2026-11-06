"""Formula image loading with derived flat-symbol target sequences."""
from functools import partial

from torch.utils.data import DataLoader

from dataset import MathExpressionDataset, _pad_collate_fn


def create_dataloaders(config):
    data_dir = config.FLAT_DATA_DIR
    train_dataset = MathExpressionDataset(
        config.CROHME_TRAIN_IMAGE_DIR,
        data_dir / "train_caption.txt",
        data_dir / "dictionary.txt",
        is_train=True,
    )
    val_dataset = MathExpressionDataset(
        config.CROHME_TRAIN_IMAGE_DIR,
        data_dir / "val_caption.txt",
        data_dir / "dictionary.txt",
        is_train=False,
    )
    if train_dataset.vocab != val_dataset.vocab:
        raise ValueError("Train and validation vocabularies differ.")

    collate = partial(_pad_collate_fn, target_size=None)
    common = {
        "batch_size": config.FORMULA_BATCH_SIZE,
        "num_workers": config.NUM_WORKERS,
        "pin_memory": config.PIN_MEMORY,
        "persistent_workers": config.NUM_WORKERS > 0,
        "collate_fn": collate,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader
