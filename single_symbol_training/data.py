"""Single-symbol datasets, class mapping, and balanced DataLoaders."""
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from dataset import BalancedClassSampler, NativeSingleSymbolDataset


class SymbolClassDataset(Dataset):
    """Adapt the project's native 32x32 symbol dataset to class labels."""

    def __init__(self, base: NativeSingleSymbolDataset, class_to_idx: Dict[str, int]):
        self.base = base
        self.class_to_idx = class_to_idx
        self.labels = [caption for _, caption in base.data]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.base[index]
        return sample["image"], self.class_to_idx[sample["caption"]]


def create_dataloaders(config):
    data_dir = config.SINGLE_SYMBOL_DATA_DIR
    image_dir = config.SINGLE_SYMBOL_IMAGE_DIR
    train_base = NativeSingleSymbolDataset(
        image_dir / "train_images.npy",
        data_dir / "train_captions.txt",
        config.DICTIONARY_FILE,
        is_train=True,
    )
    val_base = NativeSingleSymbolDataset(
        image_dir / "val_images.npy",
        data_dir / "val_captions.txt",
        config.DICTIONARY_FILE,
        is_train=False,
    )
    class_names = sorted({caption for _, caption in train_base.data})
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    unknown_val_classes = sorted({caption for _, caption in val_base.data} - set(class_to_idx))
    if unknown_val_classes:
        raise ValueError(f"Validation has classes absent from training: {unknown_val_classes}")

    train_set = SymbolClassDataset(train_base, class_to_idx)
    val_set = SymbolClassDataset(val_base, class_to_idx)
    sampler = BalancedClassSampler(
        train_set.labels,
        samples_per_class=config.SYMBOL_SAMPLES_PER_CLASS_PER_EPOCH,
        seed=config.RANDOM_SEED,
    )

    def loader(dataset, train: bool, class_sampler=None):
        return DataLoader(
            dataset,
            batch_size=config.SYMBOL_BATCH_SIZE,
            shuffle=train and class_sampler is None,
            sampler=class_sampler,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            persistent_workers=config.NUM_WORKERS > 0,
            drop_last=train,
        )

    return loader(train_set, True, sampler), loader(val_set, False), class_names
