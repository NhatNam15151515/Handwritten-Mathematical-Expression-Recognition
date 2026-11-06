"""Prepare the derived single-symbol captions with the configured label mapping."""
import csv
import json
from collections import Counter
from pathlib import Path


def _read_caption_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid caption row at {path}:{line_number}")
            rows.append((parts[0], parts[1]))
    return rows


def prepare_processed_data(config) -> dict:
    source_dir = Path(config.SINGLE_SYMBOL_IMAGE_DIR)
    output_dir = Path(config.SINGLE_SYMBOL_DATA_DIR)
    aliases = dict(config.CLASS_ALIASES)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts = {}
    class_counts = Counter()

    for split in ("train", "val"):
        rows = _read_caption_rows(source_dir / f"{split}_captions.txt")
        remapped = [(image_id, aliases.get(label, label)) for image_id, label in rows]
        target_path = output_dir / f"{split}_captions.txt"
        with target_path.open("w", encoding="utf-8", newline="") as stream:
            for image_id, label in remapped:
                stream.write(f"{image_id}\t{label}\n")
        split_counts[split] = len(remapped)
        class_counts.update(label for _, label in remapped)

    with (output_dir / "class_counts.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["token", "train_count", "val_count"])
        train_counts = Counter(label for _, label in _read_caption_rows(output_dir / "train_captions.txt"))
        val_counts = Counter(label for _, label in _read_caption_rows(output_dir / "val_captions.txt"))
        for label in sorted(class_counts):
            writer.writerow([label, train_counts[label], val_counts[label]])

    manifest = {
        "source_image_arrays": str(source_dir.resolve()),
        "derived_caption_dir": str(output_dir.resolve()),
        "class_aliases": aliases,
        "split_image_counts": split_counts,
        "class_count": len(class_counts),
        "images_copied": False,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest


if __name__ == "__main__":
    from .config import SingleSymbolConfig

    result = prepare_processed_data(SingleSymbolConfig())
    print(f"Prepared {result['class_count']} classes: {result['derived_caption_dir']}")
    print(f"Train/val images: {result['split_image_counts']}")
