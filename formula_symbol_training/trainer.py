"""Fine-tune the ViT on flat visible-symbol sequences from formula images."""
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import calculate_bleu_score

from .checkpoint import save_checkpoint
from .engine import generate_greedy, train_epoch, validate_loss
from .model import create_model


class Trainer:
    def __init__(self, config, train_loader, val_loader, vocab, epochs, patience,
                 preview_every, resume_path=None, init_checkpoint=None,
                 from_scratch: bool = False):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.idx_to_token = {index: token for token, index in vocab.items()}
        self.config.PAD_TOKEN_ID = vocab[config.PAD_TOKEN]
        self.config.SOS_TOKEN_ID = vocab[config.SOS_TOKEN]
        self.config.EOS_TOKEN_ID = vocab[config.EOS_TOKEN]
        self.epochs = epochs
        self.patience = patience
        self.preview_every = preview_every
        self.device = config.DEVICE
        self.model = create_model(config, len(vocab)).to(self.device)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=vocab[config.PAD_TOKEN], label_smoothing=config.LABEL_SMOOTHING
        )
        encoder_params = list(self.model.vision_encoder.parameters())
        decoder_params = list(self.model.text_decoder.parameters())
        self.optimizer = AdamW(
            [
                {"params": encoder_params, "lr": config.ENCODER_LR},
                {"params": decoder_params, "lr": config.DECODER_LR},
            ],
            weight_decay=config.WEIGHT_DECAY,
            fused=(self.device.type == "cuda"),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.LR_SCHEDULER_T_MAX, eta_min=config.MIN_LR
        )
        self.scaler = torch.amp.GradScaler(
            enabled=self.device.type == "cuda" and config.USE_MIXED_PRECISION
        )
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.stale_epochs = 0
        self.best_generation_bleu4 = float("-inf")
        self.best_generation_exact = float("-inf")
        self.best_generation_epoch = 0
        self.last_generation_improvement_epoch = 0
        self.run_id = int(time.time())
        for directory in (config.CHECKPOINT_DIR, config.LOG_DIR, config.OUTPUT_DIR):
            Path(directory).mkdir(parents=True, exist_ok=True)
        self.prediction_dir = Path(config.OUTPUT_DIR) / "validation_predictions"

        if resume_path:
            self._resume(resume_path)
        elif init_checkpoint:
            self._initialize_encoder(init_checkpoint)
        elif not from_scratch:
            raise ValueError(
                "Provide the patch-2 single-symbol checkpoint with --init-checkpoint, "
                "or use --from-scratch explicitly."
            )

    def _initialize_encoder(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if checkpoint.get("model_arch") != "shared_patch_vit_patch2_classifier":
            raise ValueError("Initialization checkpoint is not a patch-2 single-symbol classifier.")
        if checkpoint.get("patch_size") != self.config.PATCH_SIZE:
            raise ValueError("Initialization checkpoint patch size does not match this stage.")
        self.model.vision_encoder.load_state_dict(checkpoint["vision_encoder_state_dict"])
        print(f"Initialized ViT encoder from {checkpoint_path}; formula decoder starts fresh.")

    def _resume(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if checkpoint.get("training_stage") != "flat_formula_symbols":
            raise ValueError("Resume checkpoint is not from flat formula-symbol training.")
        if checkpoint.get("vocab") != self.vocab:
            raise ValueError("Resume checkpoint vocabulary differs from derived captions.")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # Older checkpoints used T_max=1000 because the CLI epoch cap was 1000.
        # Continue from the restored LR but decay it over the configured 150-epoch plan.
        self.scheduler.T_max = self.config.LR_SCHEDULER_T_MAX
        if checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.stale_epochs = checkpoint.get("stale_epochs", 0)
        self.run_id = checkpoint.get("run_id", self.run_id)
        self._restore_generation_history()
        print(
            f"Resumed at epoch {self.start_epoch}; best validation loss={self.best_val_loss:.4f}; "
            f"best generated BLEU-4={self.best_generation_bleu4:.4f} "
            f"(epoch {self.best_generation_epoch})"
        )

    def _history_path(self) -> Path:
        return Path(self.config.LOG_DIR) / f"history_{self.run_id}.csv"

    def _restore_generation_history(self):
        """Recover best free-running validation metrics when continuing an old run."""
        history_path = self._history_path()
        if not history_path.is_file():
            return
        with history_path.open("r", newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if not row.get("bleu_4") or not row.get("exact_match"):
                    continue
                try:
                    epoch = int(row["epoch"])
                    bleu4 = float(row["bleu_4"])
                    exact = float(row["exact_match"])
                except (TypeError, ValueError):
                    continue
                if self._generation_score_is_better(bleu4, exact):
                    self.best_generation_bleu4 = bleu4
                    self.best_generation_exact = exact
                    self.best_generation_epoch = epoch
                    self.last_generation_improvement_epoch = epoch

        if self.last_generation_improvement_epoch:
            self.stale_epochs = max(
                0, self.start_epoch - self.last_generation_improvement_epoch
            )

        # Epoch 50 was the strongest generated result in the previous run, while
        # best.pth still points to epoch 35 because that file tracked val loss.
        latest_path = Path(self.config.CHECKPOINT_DIR) / "latest.pth"
        best_generation_path = Path(self.config.CHECKPOINT_DIR) / "best_generation.pth"
        if (self.best_generation_epoch == self.start_epoch and latest_path.is_file()
                and not best_generation_path.exists()):
            state = dict(torch.load(latest_path, map_location=self.device, weights_only=False))
            state["best_generation_bleu4"] = self.best_generation_bleu4
            state["best_generation_exact"] = self.best_generation_exact
            state["best_generation_epoch"] = self.best_generation_epoch
            save_checkpoint(state, best_generation_path)

    def _generation_score_is_better(self, bleu4: float, exact: float) -> bool:
        delta = self.config.GENERATION_METRIC_MIN_DELTA
        return (bleu4 > self.best_generation_bleu4 + delta or
                (abs(bleu4 - self.best_generation_bleu4) <= delta and
                 exact > self.best_generation_exact + delta))

    def _decode(self, sequence):
        tokens = []
        for index in sequence:
            token = self.idx_to_token.get(int(index), self.config.UNK_TOKEN)
            if token == self.config.EOS_TOKEN:
                break
            if token not in (self.config.PAD_TOKEN, self.config.SOS_TOKEN):
                tokens.append(token)
        return " ".join(tokens)

    @torch.inference_mode()
    def evaluate_generations(self, epoch: int):
        self.model.eval()
        predictions, references = [], []
        sample_rows = []
        exact_matches = 0
        total = 0
        for batch in self.val_loader:
            images = batch["image"].to(self.device, non_blocking=self.config.PIN_MEMORY)
            generated = generate_greedy(self.model, images, self.config)
            targets = batch["target_seq"]
            captions = batch["caption"]
            image_names = batch["image_name"]
            for index in range(len(captions)):
                prediction = self._decode(generated[index].tolist())
                reference = self._decode(targets[index].tolist())
                predictions.append(prediction)
                references.append([reference])
                exact = prediction == reference
                exact_matches += int(exact)
                total += 1
                sample_rows.append({
                    "image_id": image_names[index],
                    "reference_flat_symbols": reference,
                    "predicted_flat_symbols": prediction,
                    "exact_match": exact,
                })

        bleu = calculate_bleu_score(predictions, references)
        metrics = {**bleu, "exact_match": exact_matches / max(1, total)}
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        output = self.prediction_dir / f"epoch_{epoch:03d}.csv"
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "image_id", "reference_flat_symbols", "predicted_flat_symbols", "exact_match"
            ])
            writer.writeheader()
            writer.writerows(sample_rows)
        return metrics, output

    def _save_checkpoint(self, epoch: int, val_metrics: dict, is_best_loss: bool,
                         is_best_generation: bool, generation_metrics: dict | None):
        state = {
            "epoch": epoch,
            "model_arch": "shared_patch_vit_flat_symbols",
            "training_stage": "flat_formula_symbols",
            "patch_size": self.config.PATCH_SIZE,
            "max_patch_grid": self.config.MAX_PATCH_GRID,
            "vocab": self.vocab,
            "model_state_dict": self.model.state_dict(),
            "vision_encoder_state_dict": self.model.vision_encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "stale_epochs": self.stale_epochs,
            "best_generation_bleu4": self.best_generation_bleu4,
            "best_generation_exact": self.best_generation_exact,
            "best_generation_epoch": self.best_generation_epoch,
            "run_id": self.run_id,
            "validation_metrics": val_metrics,
            "generation_metrics": generation_metrics,
        }
        save_checkpoint(state, Path(self.config.CHECKPOINT_DIR) / "latest.pth")
        if is_best_loss:
            save_checkpoint(state, Path(self.config.CHECKPOINT_DIR) / "best.pth")
        if is_best_generation:
            save_checkpoint(state, Path(self.config.CHECKPOINT_DIR) / "best_generation.pth")

    def train(self):
        history_path = self._history_path()
        fields = ["epoch", "train_loss", "train_token_accuracy", "val_loss",
                  "val_token_accuracy", "bleu_1", "bleu_2", "bleu_3", "bleu_4",
                  "exact_match", "encoder_lr", "decoder_lr"]
        write_header = not history_path.exists()
        with history_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if write_header:
                writer.writeheader()
            for epoch_index in range(self.start_epoch, self.epochs):
                epoch = epoch_index + 1
                train_metrics = train_epoch(
                    self.model, self.train_loader, self.optimizer, self.scaler,
                    self.criterion, self.config,
                )
                val_metrics = validate_loss(
                    self.model, self.val_loader, self.criterion, self.config
                )
                self.scheduler.step()
                generated_metrics = {
                    "bleu-1": "", "bleu-2": "", "bleu-3": "", "bleu-4": "",
                    "exact_match": "",
                }
                if self.preview_every > 0 and epoch % self.preview_every == 0:
                    generated_metrics, prediction_path = self.evaluate_generations(epoch)
                    print(f"Saved validation sequence samples: {prediction_path}")

                row = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_token_accuracy": train_metrics["token_accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_token_accuracy": val_metrics["token_accuracy"],
                    "bleu_1": generated_metrics["bleu-1"],
                    "bleu_2": generated_metrics["bleu-2"],
                    "bleu_3": generated_metrics["bleu-3"],
                    "bleu_4": generated_metrics["bleu-4"],
                    "exact_match": generated_metrics["exact_match"],
                    "encoder_lr": self.optimizer.param_groups[0]["lr"],
                    "decoder_lr": self.optimizer.param_groups[1]["lr"],
                }
                writer.writerow(row)
                stream.flush()
                print(
                    f"Epoch {epoch}/{self.epochs}: train loss={row['train_loss']:.4f}, "
                    f"token acc={row['train_token_accuracy']:.4f}; "
                    f"val loss={row['val_loss']:.4f}, token acc={row['val_token_accuracy']:.4f}"
                )
                if generated_metrics["bleu-4"] != "":
                    print(f"Flat-sequence BLEU-4={generated_metrics['bleu-4']:.4f}, "
                          f"exact match={generated_metrics['exact_match']:.4f}")

                is_best_loss = val_metrics["loss"] < self.best_val_loss - self.config.EARLY_STOPPING_MIN_DELTA
                if is_best_loss:
                    self.best_val_loss = val_metrics["loss"]
                is_best_generation = False
                if generated_metrics["bleu-4"] != "":
                    bleu4 = float(generated_metrics["bleu-4"])
                    exact = float(generated_metrics["exact_match"])
                    is_best_generation = self._generation_score_is_better(bleu4, exact)
                    if is_best_generation:
                        self.best_generation_bleu4 = bleu4
                        self.best_generation_exact = exact
                        self.best_generation_epoch = epoch
                        self.last_generation_improvement_epoch = epoch
                    self.stale_epochs = epoch - self.last_generation_improvement_epoch

                self._save_checkpoint(
                    epoch, val_metrics, is_best_loss, is_best_generation,
                    generated_metrics if generated_metrics["bleu-4"] != "" else None,
                )

                if (generated_metrics["bleu-4"] != "" and self.stale_epochs >= self.patience):
                    print(
                        f"Early stopping after {self.stale_epochs} epochs without "
                        "improvement in generated BLEU-4/exact match."
                    )
                    break
