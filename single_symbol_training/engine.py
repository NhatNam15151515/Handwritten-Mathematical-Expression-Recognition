"""Train and validation epoch loops plus classification metrics."""
from typing import Dict

import torch
from tqdm import tqdm


def run_epoch(model, loader, optimizer, scaler, criterion, device, amp_enabled,
              train: bool, num_classes: int, collect_predictions: bool = False) -> Dict[str, object]:
    model.train(train)
    loss_sum = 0.0
    sample_count = 0
    correct_per_class = torch.zeros(num_classes, dtype=torch.long)
    count_per_class = torch.zeros(num_classes, dtype=torch.long)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    prediction_records = []

    progress = tqdm(loader, desc="Train" if train else "Validation", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

        if not train and collect_predictions:
            confidences, predictions = logits.softmax(dim=1).max(dim=1)
        else:
            confidences = None
            predictions = logits.argmax(dim=1)
        loss_sum += loss.detach().item() * labels.numel()
        sample_count += labels.numel()
        labels_cpu = labels.cpu()
        predictions_cpu = predictions.cpu()
        matches = predictions_cpu.eq(labels_cpu)
        count_per_class += torch.bincount(labels_cpu, minlength=num_classes)
        correct_per_class += torch.bincount(labels_cpu[matches], minlength=num_classes)
        confusion += torch.bincount(
            labels_cpu * num_classes + predictions_cpu,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)
        if not train and collect_predictions:
            dataset = loader.dataset
            base_index = sample_count - labels.numel()
            for offset, (truth, predicted, confidence) in enumerate(zip(
                labels_cpu.tolist(), predictions_cpu.tolist(), confidences.cpu().tolist()
            )):
                image_index = dataset.base.data[base_index + offset][0]
                prediction_records.append({
                    "image_id": image_index,
                    "true_index": truth,
                    "predicted_index": predicted,
                    "confidence": confidence,
                    "correct": truth == predicted,
                })
        progress.set_postfix(loss=f"{loss_sum / max(1, sample_count):.4f}")

    per_class_accuracy = correct_per_class.float() / count_per_class.clamp_min(1)
    return {
        "loss": loss_sum / max(1, sample_count),
        "accuracy": correct_per_class.sum().item() / max(1, sample_count),
        "macro_accuracy": per_class_accuracy.mean().item(),
        "confusion": confusion,
        "prediction_records": prediction_records,
    }
