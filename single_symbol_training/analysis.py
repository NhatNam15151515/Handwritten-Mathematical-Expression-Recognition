"""Per-class validation error and confusion reporting."""
import csv
from pathlib import Path
from typing import List

import torch


def save_prediction_records(records: List[dict], class_names: List[str], output_path: Path) -> Path:
    """Write one validation prediction per image for later error analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image_id", "true_class", "predicted_class", "confidence", "correct"]
    with output_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "image_id": record["image_id"],
                "true_class": class_names[record["true_index"]],
                "predicted_class": class_names[record["predicted_index"]],
                "confidence": f"{record['confidence']:.6f}",
                "correct": record["correct"],
            })
    return output_path


def append_class_error_analysis(confusion: torch.Tensor, class_names: List[str],
                                output_path: Path, epoch: int, run_id: int) -> List[dict]:
    confusion = confusion.cpu()
    rows = []
    for class_index, class_name in enumerate(class_names):
        total = int(confusion[class_index].sum().item())
        correct = int(confusion[class_index, class_index].item())
        wrong = total - correct
        other_predictions = confusion[class_index].clone()
        other_predictions[class_index] = 0
        confused_index = int(other_predictions.argmax().item()) if wrong else -1
        confused_count = int(other_predictions[confused_index].item()) if wrong else 0
        rows.append({
            "run_id": run_id,
            "epoch": epoch,
            "class": class_name,
            "validation_count": total,
            "correct": correct,
            "wrong": wrong,
            "error_rate": wrong / total if total else 0.0,
            "most_confused_as": class_names[confused_index] if confused_index >= 0 else "",
            "confused_count": confused_count,
        })

    rows.sort(key=lambda row: (-row["wrong"], -row["error_rate"], row["class"]))
    fields = ["run_id", "epoch", "class", "validation_count", "correct", "wrong",
              "error_rate", "most_confused_as", "confused_count"]
    file_exists = output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    return rows
