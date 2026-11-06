"""Prepare native-resolution single-symbol arrays for the shared ViT warm-up."""
from __future__ import annotations

import csv
import gzip
import struct
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
EMNIST_DIR = ROOT / "data" / "number_letter" / "filtered_54_cap1800"
HASY_DIR = ROOT / "data" / "math_symbols" / "filtered_34"
OUT_DIR = ROOT / "single_symbol_training" / "data" / "single_symbol_native"


def read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic, count, rows, cols = struct.unpack(">IIII", stream.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid IDX image file: {path}")
        return np.frombuffer(stream.read(), dtype=np.uint8).reshape(count, rows, cols)


def read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid IDX label file: {path}")
        return np.frombuffer(stream.read(), dtype=np.uint8).copy()


def read_symbol_rows(caption_path: Path) -> list[tuple[str, str]]:
    rows = []
    with caption_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            name, token = line.rstrip("\n").split("\t", 1)
            rows.append((name, token))
    return rows


def emnist_to_32(images: np.ndarray) -> np.ndarray:
    # EMNIST IDX pixels are stored transposed and use white strokes on black.
    images = images.transpose(0, 2, 1)
    images = 255 - images
    padded = np.full((len(images), 32, 32), 255, dtype=np.uint8)
    padded[:, 2:30, 2:30] = images
    return padded


def write_emnist_split(
    images: np.ndarray,
    labels: np.ndarray,
    label_to_token: dict[int, str],
    output: np.memmap,
    caption_stream,
    start_index: int,
) -> tuple[int, dict[str, int]]:
    images = emnist_to_32(images)
    counts: dict[str, int] = {}
    for offset, (image, label) in enumerate(zip(images, labels)):
        token = label_to_token[int(label)]
        output[start_index + offset] = image
        caption_stream.write(f"{start_index + offset}\t{token}\n")
        counts[token] = counts.get(token, 0) + 1
    return start_index + len(images), counts


def write_hasy_split(
    rows: list[tuple[str, str]],
    image_dir: Path,
    output: np.memmap,
    caption_stream,
    start_index: int,
) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = {}
    for offset, (name, token) in enumerate(tqdm(rows, desc=f"Packing {image_dir.name}")):
        path = image_dir / f"{name}.png"
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (32, 32):
            raise ValueError(f"Missing or invalid HASY image: {path}")
        index = start_index + offset
        output[index] = image
        caption_stream.write(f"{index}\t{token}\n")
        counts[token] = counts.get(token, 0) + 1
    return start_index + len(rows), counts


def merge_counts(*parts: dict[str, int]) -> dict[str, int]:
    combined: dict[str, int] = {}
    for part in parts:
        for token, count in part.items():
            combined[token] = combined.get(token, 0) + count
    return combined


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (EMNIST_DIR / "mapping.tsv").open("r", encoding="utf-8", newline="") as stream:
        mapping = list(csv.DictReader(stream, delimiter="\t"))
    label_to_token = {int(row["new_label"]): row["character"] for row in mapping}
    with (ROOT / "data" / "dictionary.txt").open("r", encoding="utf-8") as stream:
        dictionary_tokens = {line.rstrip("\n").split("\t", 1)[0] for line in stream if "\t" in line}
    hasy_train_rows = read_symbol_rows(HASY_DIR / "train_captions.txt")
    hasy_val_rows = read_symbol_rows(HASY_DIR / "test_captions.txt")
    unexpected = (set(label_to_token.values()) | {token for _, token in hasy_train_rows + hasy_val_rows}) - dictionary_tokens
    if unexpected:
        raise ValueError(f"Tokens absent from data/dictionary.txt: {sorted(unexpected)}")

    emnist_train_images = read_idx_images(EMNIST_DIR / "emnist-byclass-cap1800-train-images-idx3-ubyte.gz")
    emnist_train_labels = read_idx_labels(EMNIST_DIR / "emnist-byclass-cap1800-train-labels-idx1-ubyte.gz")
    emnist_val_images = read_idx_images(EMNIST_DIR / "emnist-byclass-cap1800-val-images-idx3-ubyte.gz")
    emnist_val_labels = read_idx_labels(EMNIST_DIR / "emnist-byclass-cap1800-val-labels-idx1-ubyte.gz")
    n_train = len(emnist_train_images) + len(hasy_train_rows)
    n_val = len(emnist_val_images) + len(hasy_val_rows)
    train_array = np.lib.format.open_memmap(
        OUT_DIR / "train_images.npy", mode="w+", dtype=np.uint8, shape=(n_train, 32, 32)
    )
    val_array = np.lib.format.open_memmap(
        OUT_DIR / "val_images.npy", mode="w+", dtype=np.uint8, shape=(n_val, 32, 32)
    )

    with (OUT_DIR / "train_captions.txt").open("w", encoding="utf-8", newline="") as train_caps:
        offset, emnist_train_counts = write_emnist_split(
            emnist_train_images, emnist_train_labels, label_to_token, train_array, train_caps, 0
        )
        _, hasy_train_counts = write_hasy_split(
            hasy_train_rows, HASY_DIR / "train_images", train_array, train_caps, offset
        )

    with (OUT_DIR / "val_captions.txt").open("w", encoding="utf-8", newline="") as val_caps:
        offset, emnist_val_counts = write_emnist_split(
            emnist_val_images, emnist_val_labels, label_to_token, val_array, val_caps, 0
        )
        _, hasy_val_counts = write_hasy_split(
            hasy_val_rows, HASY_DIR / "test_images", val_array, val_caps, offset
        )

    train_counts = merge_counts(emnist_train_counts, hasy_train_counts)
    val_counts = merge_counts(emnist_val_counts, hasy_val_counts)
    tokens = sorted(set(train_counts) | set(val_counts))
    with (OUT_DIR / "class_counts.tsv").open("w", encoding="utf-8", newline="") as stream:
        stream.write("token\ttrain_count\tval_count\n")
        for token in tokens:
            stream.write(f"{token}\t{train_counts.get(token, 0)}\t{val_counts.get(token, 0)}\n")

    train_array.flush()
    val_array.flush()
    (OUT_DIR / "README.txt").write_text(
        "Native-resolution single-symbol data for the shared patch-4 ViT.\n"
        f"Train: {n_train} images; validation: {n_val} images; {len(tokens)} tokens.\n"
        "EMNIST ByClass: 1,620 train and 180 validation images per character, transposed to upright orientation.\n"
        "HASYv2: all fold-1 train images and official fold-1 test images for the 34 dictionary-matched math tokens.\n"
        "All samples are grayscale 32x32 arrays stored as uint8 in NPY memmaps.\n"
        "Training uses a class-balanced sampler with replacement, so abundant classes are retained and rare classes are not silently discarded.\n"
        "HASYv2 is published under ODbL; see https://doi.org/10.5281/zenodo.259444.\n",
        encoding="utf-8",
    )
    print(f"Wrote {n_train} train and {n_val} validation images across {len(tokens)} tokens to {OUT_DIR}")


if __name__ == "__main__":
    main()
