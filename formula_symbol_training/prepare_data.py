"""Create derived flat-symbol caption files without copying or editing images."""
import json
from collections import Counter
from pathlib import Path

from .target_format import visible_symbol_tokens, visible_vocabulary


SOURCE_DIR = Path(__file__).resolve().parent / "data" / "crohme_whywhs_source"
OUTPUT_DIR = SOURCE_DIR / "flat_symbol_sequences"


def read_dictionary(path: Path):
    tokens = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        token = line.split("\t", 1)[0].strip()
        if token and token != "<eol>":
            tokens.append(token)
    return tokens


def transform_split(source_path: Path, output_path: Path, vocabulary: set[str]):
    image_ids = set()
    token_counts = Counter()
    output_lines = []
    for line_number, line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            raise ValueError(f"Malformed caption at {source_path}:{line_number}")
        image_id, caption = parts
        if image_id in image_ids:
            raise ValueError(f"Duplicate image ID {image_id!r} in {source_path}")
        image_ids.add(image_id)
        tokens = visible_symbol_tokens(caption)
        unknown = sorted(set(tokens) - vocabulary)
        if unknown:
            raise ValueError(f"Unknown cleaned tokens at {source_path}:{line_number}: {unknown}")
        token_counts.update(tokens)
        output_lines.append(f"{image_id}\t{' '.join(tokens)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return {"images": len(output_lines), "tokens": sum(token_counts.values()),
            "unique_tokens": len(token_counts), "token_counts": token_counts}


def prepare_data() -> Path:
    dictionary_tokens = read_dictionary(SOURCE_DIR / "dictionary.txt")
    vocabulary = visible_vocabulary(dictionary_tokens)
    vocabulary_set = set(vocabulary)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split_sources = {
        "train": SOURCE_DIR / "splits" / "train_caption.txt",
        "val": SOURCE_DIR / "splits" / "val_caption.txt",
        "test": SOURCE_DIR / "test_caption.txt",
    }
    summary = {}
    for split, source_path in split_sources.items():
        result = transform_split(
            source_path,
            OUTPUT_DIR / f"{split}_caption.txt",
            vocabulary_set,
        )
        summary[split] = {
            "images": result["images"],
            "tokens": result["tokens"],
            "unique_tokens": result["unique_tokens"],
        }

    dictionary_path = OUTPUT_DIR / "dictionary.txt"
    dictionary_path.write_text(
        "".join(f"{token}\t{index}\n" for index, token in enumerate(vocabulary)),
        encoding="utf-8",
    )
    manifest = {
        "image_source": "../processed_448/{train,test}/",
        "caption_transformation": "flat visible-symbol sequence; original captions preserved",
        "vocabulary_size": len(vocabulary),
        "vocabulary": vocabulary,
        "splits": summary,
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Derived captions saved under: {OUTPUT_DIR}")
    print(f"Visible-symbol vocabulary: {len(vocabulary)} tokens")
    for split, values in summary.items():
        print(f"{split}: {values['images']} images, {values['tokens']} target tokens, "
              f"{values['unique_tokens']} unique classes")
    return OUTPUT_DIR


if __name__ == "__main__":
    prepare_data()
