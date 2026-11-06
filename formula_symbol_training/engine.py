"""Training, validation-loss, and greedy-generation loops."""
from typing import Dict

import torch
from tqdm import tqdm


def train_epoch(model, loader, optimizer, scaler, criterion, config) -> Dict[str, float]:
    model.train()
    loss_sum = 0.0
    token_correct = 0
    token_count = 0
    accumulation = config.ACCUMULATION_STEPS
    pending = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        images = batch["image"].to(config.DEVICE, non_blocking=config.PIN_MEMORY)
        input_seq = batch["input_seq"].to(config.DEVICE, non_blocking=config.PIN_MEMORY)
        targets = batch["target_seq"].to(config.DEVICE, non_blocking=config.PIN_MEMORY)
        amp_enabled = config.DEVICE.type == "cuda" and config.USE_MIXED_PRECISION
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(images, input_seq)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            continue

        scaler.scale(loss / accumulation).backward()
        pending += 1
        if pending == accumulation or batch_index + 1 == len(loader):
            scaler.unscale_(optimizer)
            if pending < accumulation:
                correction = accumulation / pending
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            pending = 0

        mask = targets != config.PAD_TOKEN_ID
        predictions = logits.argmax(dim=-1)
        token_correct += ((predictions == targets) & mask).sum().item()
        token_count += mask.sum().item()
        loss_sum += loss.detach().item() * images.size(0)

    return {
        "loss": loss_sum / max(1, len(loader.dataset)),
        "token_accuracy": token_correct / max(1, token_count),
    }


@torch.inference_mode()
def validate_loss(model, loader, criterion, config) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    token_correct = 0
    token_count = 0
    for batch in tqdm(loader, desc="Validation loss", leave=False):
        images = batch["image"].to(config.DEVICE, non_blocking=config.PIN_MEMORY)
        input_seq = batch["input_seq"].to(config.DEVICE, non_blocking=config.PIN_MEMORY)
        targets = batch["target_seq"].to(config.DEVICE, non_blocking=config.PIN_MEMORY)
        amp_enabled = config.DEVICE.type == "cuda" and config.USE_MIXED_PRECISION
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(images, input_seq)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        mask = targets != config.PAD_TOKEN_ID
        predictions = logits.argmax(dim=-1)
        token_correct += ((predictions == targets) & mask).sum().item()
        token_count += mask.sum().item()
        loss_sum += loss.item() * images.size(0)
    return {
        "loss": loss_sum / max(1, len(loader.dataset)),
        "token_accuracy": token_correct / max(1, token_count),
    }


@torch.inference_mode()
def generate_greedy(model, images, config) -> torch.Tensor:
    model.eval()
    features = model.vision_encoder.forward_features(images)
    features = model.encoder_projection(features)
    generated = torch.full(
        (images.size(0), 1), config.SOS_TOKEN_ID, dtype=torch.long, device=images.device
    )
    finished = torch.zeros(images.size(0), dtype=torch.bool, device=images.device)
    for _ in range(config.MAX_SEQ_LENGTH - 1):
        logits = model.text_decoder(generated, features)[:, -1, :]
        logits[:, config.PAD_TOKEN_ID] = float("-inf")
        logits[:, config.SOS_TOKEN_ID] = float("-inf")
        next_tokens = logits.argmax(dim=-1)
        next_tokens = torch.where(finished, config.PAD_TOKEN_ID, next_tokens)
        generated = torch.cat((generated, next_tokens[:, None]), dim=1)
        finished |= next_tokens.eq(config.EOS_TOKEN_ID)
        if finished.all():
            break
    return generated
