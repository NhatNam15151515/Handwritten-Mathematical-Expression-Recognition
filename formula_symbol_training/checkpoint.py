"""Atomic checkpoint writes for the flat formula-symbol training stage."""
import os
from pathlib import Path

import torch


def save_checkpoint(payload: dict, destination: Path) -> None:
    """Save a checkpoint atomically so interruption cannot leave a partial file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
