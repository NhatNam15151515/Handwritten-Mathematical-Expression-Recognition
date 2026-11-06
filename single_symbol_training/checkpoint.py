"""Atomic checkpoint writes for interruption-safe training."""
import os
from pathlib import Path

import torch


def save_checkpoint(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
