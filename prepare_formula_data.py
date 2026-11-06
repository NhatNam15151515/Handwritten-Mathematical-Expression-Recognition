"""Create resized, white-background CROHME images for formula training."""
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "formula_symbol_training" / "data" / "crohme_whywhs_source"
OUTPUT_DIR = DATA_DIR / "processed_448"
MAX_EDGE = 448


def load_ids(caption_path: Path) -> list[str]:
    ids = []
    with caption_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            image_id = line.split("\t", 1)[0]
            if not image_id:
                raise ValueError(f"Empty image ID at {caption_path}:{line_number}")
            ids.append(image_id)
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate image IDs in {caption_path}")
    return ids


def prepare_split(name: str, source_dir: Path, output_dir: Path, ids: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = []

    for index, image_id in enumerate(ids, start=1):
        source_path = source_dir / f"{image_id}_0.bmp"
        output_path = output_dir / f"{image_id}.png"
        image = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            missing.append(str(source_path))
            continue

        # CROHME source bitmaps have white strokes on black. Store them as
        # black strokes on white, matching the single-symbol training images.
        image = cv2.bitwise_not(image)
        height, width = image.shape
        scale = min(1.0, MAX_EDGE / max(height, width))
        if scale < 1.0:
            new_width = max(1, round(width * scale))
            new_height = max(1, round(height * scale))
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        if not cv2.imwrite(str(output_path), image):
            raise OSError(f"Could not write processed image: {output_path}")

        if index % 1000 == 0 or index == len(ids):
            print(f"{name}: {index}/{len(ids)}")

    if missing:
        raise FileNotFoundError(
            f"{name}: {len(missing)} source images missing; first examples: {missing[:5]}"
        )


def main() -> None:
    split_dir = DATA_DIR / "splits"
    train_ids = load_ids(split_dir / "train_caption.txt")
    val_ids = load_ids(split_dir / "val_caption.txt")
    test_ids = load_ids(DATA_DIR / "test_caption.txt")

    if set(train_ids) & set(val_ids):
        raise ValueError("Train and validation IDs overlap")

    prepare_split(
        "train+val", DATA_DIR / "off_image_train", OUTPUT_DIR / "train", train_ids + val_ids
    )
    prepare_split("test", DATA_DIR / "off_image_test", OUTPUT_DIR / "test", test_ids)
    print(f"Prepared CROHME images at {OUTPUT_DIR} (max edge {MAX_EDGE}px).")


if __name__ == "__main__":
    main()
