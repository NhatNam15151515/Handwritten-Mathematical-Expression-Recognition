"""Training orchestration, validation, early stopping, and checkpointing."""
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .analysis import append_class_error_analysis, save_prediction_records
from .checkpoint import save_checkpoint
from .engine import run_epoch
from .model import Patch2SymbolClassifier
from .visualization import save_validation_preview


class Trainer:
    def __init__(self, config, train_loader, val_loader, val_dataset, class_names,
                 epochs: int, patience: int, preview_every: int, learning_rate: float,
                 resume_path: Path | None = None, init_encoder_path: Path | None = None):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_dataset = val_dataset
        self.class_names = class_names
        self.class_to_idx = {name: index for index, name in enumerate(class_names)}
        self.epochs = epochs
        self.patience = patience
        self.preview_every = preview_every
        self.device = config.DEVICE
        self.model = Patch2SymbolClassifier(config, len(class_names)).to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=config.WEIGHT_DECAY,
            fused=(self.device.type == "cuda"),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=config.MIN_LR
        )
        self.amp_enabled = self.device.type == "cuda" and config.USE_MIXED_PRECISION
        self.scaler = torch.amp.GradScaler(enabled=self.amp_enabled)
        self.start_epoch = 0
        self.best_macro_accuracy = 0.0
        self.stale_epochs = 0
        self.analysis_run_id = int(time.time())

        self.checkpoint_dir = Path(config.CHECKPOINT_DIR)
        self.log_dir = Path(config.LOG_DIR)
        self.output_dir = Path(config.OUTPUT_DIR)
        for directory in (self.checkpoint_dir, self.log_dir, self.output_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.analysis_path = self.log_dir / "class_error_analysis.csv"

        if resume_path:
            self._resume(resume_path)
        elif init_encoder_path:
            self._initialize_encoder(init_encoder_path)

    def _initialize_encoder(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if checkpoint.get("model_arch") != "shared_patch_vit_patch2_classifier":
            raise ValueError("Initialization checkpoint is not a patch-2 single-symbol classifier.")
        if checkpoint.get("patch_size") != self.config.PATCH_SIZE:
            raise ValueError("Initialization checkpoint patch size does not match this stage.")
        self.model.vision_encoder.load_state_dict(checkpoint["vision_encoder_state_dict"])
        print(f"Initialized ViT encoder from {checkpoint_path}; classifier head and optimizer start fresh.")

    def _resume(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if checkpoint.get("model_arch") != "shared_patch_vit_patch2_classifier":
            raise ValueError("Resume checkpoint is not from this patch-2 classifier.")
        if checkpoint.get("class_names") != self.class_names:
            raise ValueError("Resume checkpoint classes do not match the current dataset.")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_epoch = checkpoint["epoch"]
        self.best_macro_accuracy = checkpoint.get("best_macro_accuracy", 0.0)
        self.stale_epochs = checkpoint.get("stale_epochs", 0)
        self.analysis_run_id = checkpoint.get("analysis_run_id", self.analysis_run_id)
        print(f"Resumed at epoch {self.start_epoch}; best macro accuracy="
              f"{self.best_macro_accuracy:.4f}")

    def _save_training_checkpoint(self, epoch: int, val_metrics: dict) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_arch": "shared_patch_vit_patch2_classifier",
            "training_stage": "symbols_classifier",
            "patch_size": self.config.PATCH_SIZE,
            "image_size": self.config.IMAGE_SIZE,
            "class_names": self.class_names,
            "class_to_idx": self.class_to_idx,
            "class_aliases": dict(self.config.CLASS_ALIASES),
            "model_state_dict": self.model.state_dict(),
            "vision_encoder_state_dict": self.model.vision_encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_macro_accuracy": self.best_macro_accuracy,
            "stale_epochs": self.stale_epochs,
            "analysis_run_id": self.analysis_run_id,
            "validation_metrics": val_metrics,
        }
        save_checkpoint(checkpoint, self.checkpoint_dir / "latest.pth")
        if val_metrics["macro_accuracy"] >= self.best_macro_accuracy:
            save_checkpoint(checkpoint, self.checkpoint_dir / "best.pth")

    def train(self) -> None:
        history_path = self.log_dir / f"history_{int(time.time())}.csv"
        fields = ["epoch", "train_loss", "train_accuracy", "train_macro_accuracy",
                  "val_loss", "val_accuracy", "val_macro_accuracy", "lr"]
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            print(f"Device: {self.device}; classes: {len(self.class_names)}; "
                  f"patch: {self.config.PATCH_SIZE}; input: {self.config.IMAGE_SIZE}x"
                  f"{self.config.IMAGE_SIZE}")
            print(f"Balanced samples/epoch: {len(self.train_loader.dataset.class_to_idx) * self.config.SYMBOL_SAMPLES_PER_CLASS_PER_EPOCH}")
            print(f"Checkpoint directory: {self.checkpoint_dir}")

            for epoch_index in range(self.start_epoch, self.epochs):
                epoch = epoch_index + 1
                collect_predictions = self.preview_every > 0 and (
                    epoch % self.preview_every == 0 or self.stale_epochs + 1 >= self.patience
                )
                train_metrics = run_epoch(
                    self.model, self.train_loader, self.optimizer, self.scaler,
                    self.criterion, self.device, self.amp_enabled, True,
                    len(self.class_names),
                )
                val_metrics = run_epoch(
                    self.model, self.val_loader, self.optimizer, self.scaler,
                    self.criterion, self.device, self.amp_enabled, False,
                    len(self.class_names),
                    collect_predictions=collect_predictions,
                )
                error_rows = append_class_error_analysis(
                    val_metrics["confusion"], self.class_names, self.analysis_path,
                    epoch, self.analysis_run_id,
                )
                self.scheduler.step()
                row = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "train_macro_accuracy": train_metrics["macro_accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_accuracy": val_metrics["macro_accuracy"],
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
                writer.writerow(row)
                stream.flush()
                print(
                    f"Epoch {epoch}/{self.epochs}: train loss={row['train_loss']:.4f}, "
                    f"acc={row['train_accuracy']:.4f}, macro={row['train_macro_accuracy']:.4f}; "
                    f"val loss={row['val_loss']:.4f}, acc={row['val_accuracy']:.4f}, "
                    f"macro={row['val_macro_accuracy']:.4f}"
                )

                top_errors = ", ".join(
                    f"{entry['class']} ({entry['wrong']}/{entry['validation_count']})"
                    for entry in error_rows[:5] if entry["wrong"] > 0
                ) or "none"
                print(f"Most misclassified classes: {top_errors}")

                is_best = val_metrics["macro_accuracy"] > self.best_macro_accuracy
                if is_best:
                    self.best_macro_accuracy = val_metrics["macro_accuracy"]
                    self.stale_epochs = 0
                else:
                    self.stale_epochs += 1

                self._save_training_checkpoint(epoch, val_metrics)
                should_preview = self.preview_every > 0 and (
                    epoch % self.preview_every == 0 or self.stale_epochs >= self.patience
                )
                if should_preview:
                    preview_path = save_validation_preview(
                        self.model, self.val_dataset, self.class_names, self.device,
                        self.output_dir, epoch,
                    )
                    print(f"Validation prediction board saved: {preview_path}")
                    prediction_path = save_prediction_records(
                        val_metrics["prediction_records"], self.class_names,
                        self.output_dir / "validation_predictions" / f"epoch_{epoch:04d}.csv",
                    )
                    print(f"All validation predictions saved: {prediction_path}")

                if self.stale_epochs >= self.patience:
                    print(f"Early stopping after {self.patience} epochs without macro-accuracy improvement.")
                    break
