"""Save fixed validation prediction boards for qualitative review."""
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image, ImageDraw, ImageFont

from .data import SymbolClassDataset


@torch.inference_mode()
def save_validation_preview(model, dataset: SymbolClassDataset, class_names: List[str],
                            device: torch.device, output_dir: Path, epoch: int) -> Path:
    first_index_by_class: Dict[str, int] = {}
    for index, label in enumerate(dataset.labels):
        first_index_by_class.setdefault(label, index)

    indices = [first_index_by_class[name] for name in class_names if name in first_index_by_class]
    columns = 8
    tile_width, tile_height = 176, 164
    rows = (len(indices) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    model.eval()

    for start in range(0, len(indices), 32):
        batch_indices = indices[start:start + 32]
        images, labels = zip(*(dataset[index] for index in batch_indices))
        logits = model(torch.stack(images).to(device))
        confidences, predictions = logits.softmax(dim=1).max(dim=1)

        for offset, _ in enumerate(batch_indices):
            image_tensor = images[offset].cpu() * std + mean
            gray = image_tensor.mean(dim=0).clamp(0, 1).mul(255).byte().numpy()
            thumbnail = Image.fromarray(gray, mode="L").convert("RGB")
            thumbnail = thumbnail.resize((128, 128), Image.Resampling.NEAREST)
            cell = start + offset
            x = (cell % columns) * tile_width
            y = (cell // columns) * tile_height
            canvas.paste(thumbnail, (x + 24, y + 2))
            truth = class_names[int(labels[offset])]
            predicted = class_names[int(predictions[offset].item())]
            confidence = confidences[offset].item()
            draw.text((x + 5, y + 133), f"true: {truth}", fill="black", font=font)
            draw.text((x + 5, y + 147),
                      f"pred: {predicted}  p={confidence:.2f}", fill="black", font=font)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"validation_epoch_{epoch:03d}.png"
    canvas.save(output_path)
    return output_path
