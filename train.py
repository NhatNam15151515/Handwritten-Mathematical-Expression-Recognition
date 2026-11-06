# train.py (Phiên bản đơn giản hóa, không còn fine-tuning 2 giai đoạn)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from tqdm import tqdm
import numpy as np
from typing import Dict
import csv
import argparse
from pathlib import Path
import sys

from config import Config
from dataset import create_dataloaders
from model import MathExpressionVisionTransformer
from utils import AverageMeter, calculate_bleu_score, save_checkpoint


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE
        self.train_loader, self.val_loader = create_dataloaders(config)

        self.vocab_size = len(self.train_loader.dataset.vocab)
        self.vocab = self.train_loader.dataset.vocab
        self.idx_to_token = self.train_loader.dataset.idx_to_token
        bracket_token_pairs = (("{", "}"), ("(", ")"), ("[", "]"), (r"\{", r"\}"))
        self.bracket_pairs = {
            self.vocab[open_token]: self.vocab[close_token]
            for open_token, close_token in bracket_token_pairs
            if open_token in self.vocab and close_token in self.vocab
        }
        self.compare_bracket_decoding = (
            config.TRAIN_STAGE == "formula"
            and getattr(config, "COMPARE_BALANCED_BRACKET_DECODING", False)
        )

        self.model = MathExpressionVisionTransformer(config, self.vocab_size).to(self.device)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.vocab[config.PAD_TOKEN],
            label_smoothing=config.LABEL_SMOOTHING
        )
        self.use_position_auxiliary = (
            config.TRAIN_STAGE == "formula"
            and getattr(config, "USE_POSITION_AUXILIARY", False)
        )
        self.position_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        encoder_params = list(self.model.vision_encoder.parameters())
        if hasattr(self.model, "encoder_norm"):
            encoder_params.extend(self.model.encoder_norm.parameters())
        decoder_params = [
            {'params': self.model.encoder_projection.parameters()},
            {'params': self.model.text_decoder.parameters()}
        ]
        if self.use_position_auxiliary:
            decoder_params.append({'params': self.model.position_head.parameters()})

        # Tạo danh sách các nhóm tham số cho optimizer
        param_groups = [
            {'params': list(encoder_params), 'lr': config.ENCODER_LR},
            {'params': [p for group in decoder_params for p in group['params']], 'lr': config.DECODER_LR}
        ]

        print(f"Optimizer: Using differential LR -> Encoder LR: {config.ENCODER_LR}, Decoder LR: {config.DECODER_LR}")

        self.optimizer = optim.AdamW(
            param_groups,  # Truyền các nhóm tham số đã được định nghĩa
            weight_decay=config.WEIGHT_DECAY,
            fused=(self.device.type == 'cuda')
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.NUM_EPOCHS,
            eta_min=config.MIN_LR
        )

        self.writer = SummaryWriter(config.LOG_DIR / f"run_onthefly_{int(time.time())}")
        self.epoch = 0
        self.start_epoch = 0
        self.best_bleu = 0.0
        self.best_metric = -1.0
        self.best_loss = float('inf')
        self.stale_epochs = 0
        self.scaler = torch.amp.GradScaler(enabled=self.config.USE_MIXED_PRECISION)
        self.prediction_dir = Path(config.OUTPUT_DIR) / "validation_predictions"

        self._init_history_log()

        print(f"Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Training on {self.device}")

    def _init_history_log(self):
        """Khởi tạo file log để ghi lại lịch sử huấn luyện."""
        self.history_log_path = self.config.LOG_DIR / f"train_history_{int(time.time())}.csv"

        # Tự động trích xuất các siêu tham số từ config
        hyperparams = {}
        for k in dir(self.config):
            if not k.startswith('_'):
                v = getattr(self.config, k)
                if isinstance(v, (int, float, str, bool)):
                    hyperparams[k] = v

        self.hyperparam_keys = sorted(hyperparams.keys())
        self.hyperparam_values = [hyperparams[k] for k in self.hyperparam_keys]

        try:
            self.history_file = open(self.history_log_path, 'w', newline='', encoding='utf-8')
            self.history_writer = csv.writer(self.history_file)

            # Tạo header với các siêu tham số
            headers = ['epoch', 'train_loss', 'train_position_loss',
                       'train_accuracy',
                       'val_loss',
                       'val_position_loss',
                       'val_accuracy',
                       'val_bleu_1', 'val_bleu_2', 'val_bleu_3', 'val_bleu_4',
                       'val_exp_rate',
                       'val_balanced_bleu_1', 'val_balanced_bleu_2',
                       'val_balanced_bleu_3', 'val_balanced_bleu_4',
                       'val_balanced_exp_rate',
                       'encoder_lr',
                       'decoder_lr'] + self.hyperparam_keys

            self.history_writer.writerow(headers)
            print(f"Lịch sử huấn luyện sẽ được ghi vào: {self.history_log_path}")
        except IOError as e:
            print(f"Lỗi: Không thể mở file log lịch sử. {e}")
            self.history_file = None

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        losses, position_losses, accuracies = AverageMeter(), AverageMeter(), AverageMeter()
        scaler = self.scaler
        accumulated_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}/{self.config.NUM_EPOCHS} [Training]")

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device, non_blocking=self.config.PIN_MEMORY)
            input_seq = batch['input_seq'].to(self.device, non_blocking=self.config.PIN_MEMORY)
            target_seq = batch['target_seq'].to(self.device, non_blocking=self.config.PIN_MEMORY)

            with torch.amp.autocast(device_type='cuda', enabled=self.config.USE_MIXED_PRECISION):
                if self.use_position_auxiliary:
                    position_targets = batch['position_targets'].to(
                        self.device, non_blocking=self.config.PIN_MEMORY
                    )
                    logits, position_logits = self.model(
                        images, input_seq, return_auxiliary=True
                    )
                    sequence_loss = self.criterion(
                        logits.reshape(-1, self.vocab_size), target_seq.reshape(-1)
                    )
                    position_loss = self.position_criterion(
                        position_logits.reshape(-1, 3), position_targets.reshape(-1)
                    )
                    loss = sequence_loss + self.config.POSITION_LOSS_WEIGHT * position_loss
                else:
                    logits = self.model(images, input_seq)
                    loss = self.criterion(
                        logits.reshape(-1, self.vocab_size), target_seq.reshape(-1)
                    )
                    position_loss = loss.new_zeros(())

            if not torch.isfinite(loss): continue

            mask = target_seq != self.vocab[self.config.PAD_TOKEN]
            if self.config.TRAIN_STAGE == "symbols":
                mask &= target_seq != self.vocab[self.config.EOS_TOKEN]
            predictions = torch.argmax(logits, dim=-1)
            accuracy = ((predictions == target_seq) & mask).sum().float() / mask.sum().float()

            scaler.scale(loss / self.config.ACCUMULATION_STEPS).backward()
            accumulated_batches += 1

            if (batch_idx + 1) % self.config.ACCUMULATION_STEPS == 0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0

            losses.update(loss.item(), images.size(0))
            position_losses.update(position_loss.item(), images.size(0))
            accuracies.update(accuracy.item(), images.size(0))
            pbar.set_postfix({'Loss': f'{losses.avg:.4f}', 'Acc': f'{accuracies.avg:.4f}'})

        # Flush the final incomplete accumulation group once, after the loop.
        if accumulated_batches:
            scaler.unscale_(self.optimizer)
            scaling_factor = self.config.ACCUMULATION_STEPS / accumulated_batches
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.mul_(scaling_factor)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        return {
            'loss': losses.avg,
            'position_loss': position_losses.avg,
            'accuracy': accuracies.avg,
        }

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        losses, position_losses, accuracies = AverageMeter(), AverageMeter(), AverageMeter()
        all_predictions, all_targets = [], []
        balanced_predictions, balanced_exact_matches = [], 0
        prediction_rows = []
        exact_matches = 0
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="[Evaluating]")
            for batch in pbar:
                images = batch['image'].to(self.device, non_blocking=self.config.PIN_MEMORY)
                input_seq = batch['input_seq'].to(self.device, non_blocking=self.config.PIN_MEMORY)
                target_seq = batch['target_seq'].to(self.device, non_blocking=self.config.PIN_MEMORY)

                with torch.amp.autocast(device_type='cuda', enabled=self.config.USE_MIXED_PRECISION):
                    if self.use_position_auxiliary:
                        position_targets = batch['position_targets'].to(
                            self.device, non_blocking=self.config.PIN_MEMORY
                        )
                        logits, position_logits = self.model(
                            images, input_seq, return_auxiliary=True
                        )
                        sequence_loss = self.criterion(
                            logits.reshape(-1, self.vocab_size), target_seq.reshape(-1)
                        )
                        position_loss = self.position_criterion(
                            position_logits.reshape(-1, 3), position_targets.reshape(-1)
                        )
                        loss = sequence_loss + self.config.POSITION_LOSS_WEIGHT * position_loss
                    else:
                        logits = self.model(images, input_seq)
                        loss = self.criterion(
                            logits.reshape(-1, self.vocab_size), target_seq.reshape(-1)
                        )
                        position_loss = loss.new_zeros(())

                if not torch.isfinite(loss): continue

                mask = target_seq != self.vocab[self.config.PAD_TOKEN]
                if self.config.TRAIN_STAGE == "symbols":
                    mask &= target_seq != self.vocab[self.config.EOS_TOKEN]
                predictions = torch.argmax(logits, dim=-1)
                accuracy = ((predictions == target_seq) & mask).sum().float() / mask.sum().float()

                losses.update(loss.item(), images.size(0))
                position_losses.update(position_loss.item(), images.size(0))
                accuracies.update(accuracy.item(), images.size(0))

                if self.config.TRAIN_STAGE != "symbols":
                    gen = self.model.generate(
                        images, self.vocab[self.config.SOS_TOKEN], self.vocab[self.config.EOS_TOKEN],
                        max_length=self.config.MAX_SEQ_LENGTH, top_k=1
                    )
                    balanced_gen = None
                    if self.compare_bracket_decoding:
                        balanced_gen = self.model.generate(
                            images,
                            self.vocab[self.config.SOS_TOKEN],
                            self.vocab[self.config.EOS_TOKEN],
                            max_length=self.config.MAX_SEQ_LENGTH,
                            top_k=1,
                            enforce_balanced_brackets=True,
                            bracket_pairs=self.bracket_pairs,
                        )
                    for i in range(gen.size(0)):
                        pred_str = self._indices_to_tokens(gen[i].cpu().numpy())
                        tgt_str = self._indices_to_tokens(target_seq[i].cpu().numpy())
                        exact = pred_str == tgt_str
                        exact_matches += int(exact)
                        all_predictions.append(pred_str)
                        all_targets.append([tgt_str])
                        row = {
                            "image_id": batch["image_name"][i],
                            "reference_latex_tokens": tgt_str,
                            "predicted_latex_tokens": pred_str,
                            "exact_match": exact,
                        }
                        if balanced_gen is not None:
                            balanced_pred_str = self._indices_to_tokens(
                                balanced_gen[i].cpu().numpy()
                            )
                            balanced_exact = balanced_pred_str == tgt_str
                            balanced_exact_matches += int(balanced_exact)
                            balanced_predictions.append(balanced_pred_str)
                            row["balanced_predicted_latex_tokens"] = balanced_pred_str
                            row["balanced_exact_match"] = balanced_exact
                        prediction_rows.append(row)

        bleu_scores = calculate_bleu_score(all_predictions, all_targets) if all_predictions else {}
        eval_results = {
            'loss': losses.avg,
            'position_loss': position_losses.avg,
            'accuracy': accuracies.avg,
        }
        if all_predictions:
            eval_results['exp_rate'] = exact_matches / len(all_predictions)
        eval_results.update(bleu_scores)  # Gộp dictionary bleu_scores vào
        if balanced_predictions:
            balanced_bleu = calculate_bleu_score(
                balanced_predictions, all_targets
            )
            eval_results['balanced_exp_rate'] = balanced_exact_matches / len(balanced_predictions)
            eval_results.update({f"balanced_{key}": value for key, value in balanced_bleu.items()})
        epoch_number = self.epoch + 1
        preview_every = int(getattr(self.config, "PREVIEW_EVERY", 5))
        if prediction_rows and preview_every > 0 and epoch_number % preview_every == 0:
            self.prediction_dir.mkdir(parents=True, exist_ok=True)
            prediction_path = self.prediction_dir / f"epoch_{epoch_number:03d}.csv"
            with prediction_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=prediction_rows[0].keys())
                writer.writeheader()
                writer.writerows(prediction_rows)
            print(f"Validation predictions saved: {prediction_path}")
        return eval_results

    def _indices_to_tokens(self, indices: np.ndarray) -> str:
        tokens = []
        for idx in indices:
            if idx == self.vocab[self.config.EOS_TOKEN]: break
            if idx not in [self.vocab[self.config.PAD_TOKEN], self.vocab[self.config.SOS_TOKEN]]:
                tokens.append(self.idx_to_token.get(idx, self.config.UNK_TOKEN))
        return ' '.join(tokens)

    def generate_samples(self, num_samples: int = 5):
        self.model.eval()
        with torch.no_grad():
            try:
                batch = next(iter(self.val_loader))
            except StopIteration:
                print("Không thể lấy batch từ val_loader để tạo mẫu.")
                return
            images = batch['image'][:num_samples].to(self.device)
            captions = batch['caption'][:num_samples]

            # Lấy số lượng mẫu thực tế có trong batch này
            actual_batch_size = images.size(0)

            # Nếu không có mẫu nào thì dừng lại
            if actual_batch_size == 0:
                print("Không có mẫu nào trong batch để tạo.")
                return

            # Tạo chuỗi dự đoán
            generated = self.model.generate(
                images, self.vocab[self.config.SOS_TOKEN], self.vocab[self.config.EOS_TOKEN],
                max_length=self.config.MAX_SEQ_LENGTH
            )

            print("\n" + "=" * 80 + "\nSAMPLE PREDICTIONS:\n" + "=" * 80)

            # Chạy vòng lặp dựa trên số lượng mẫu THỰC TẾ
            for i in range(actual_batch_size):
                # Dòng này bây giờ sẽ hoàn toàn an toàn
                print(
                    f"\nSample {i + 1}:\nTarget:     {captions[i]}\nPredicted:  {self._indices_to_tokens(generated[i].cpu().numpy())}\n" + "-" * 60)

    def _log_epoch_history(self, epoch: int, train_metrics: Dict, eval_metrics: Dict):
        """Ghi lại thông số của một epoch vào file log."""
        if self.history_file:
            try:
                encoder_lr = self.optimizer.param_groups[0]['lr']
                decoder_lr = self.optimizer.param_groups[1]['lr']

                # Thêm giá trị siêu tham số vào mỗi dòng
                row = [
                          epoch + 1,
                          train_metrics['loss'],
                          train_metrics.get('position_loss', 0.0),
                          train_metrics['accuracy'],
                          eval_metrics['loss'],
                          eval_metrics.get('position_loss', 0.0),
                          eval_metrics['accuracy'],
                          eval_metrics.get('bleu-1', 0.0),
                          eval_metrics.get('bleu-2', 0.0),
                          eval_metrics.get('bleu-3', 0.0),
                          eval_metrics.get('bleu-4', 0.0),
                          eval_metrics.get('exp_rate', 0.0),
                          eval_metrics.get('balanced_bleu-1', 0.0),
                          eval_metrics.get('balanced_bleu-2', 0.0),
                          eval_metrics.get('balanced_bleu-3', 0.0),
                          eval_metrics.get('balanced_bleu-4', 0.0),
                          eval_metrics.get('balanced_exp_rate', 0.0),
                          encoder_lr,
                          decoder_lr
                      ] + self.hyperparam_values

                self.history_writer.writerow(row)
                self.history_file.flush()
            except IOError as e:
                print(f"Lỗi: Không thể ghi vào file log lịch sử. {e}")

    def train(self):
        print(f"\nBắt đầu huấn luyện trong {self.config.NUM_EPOCHS} epochs...")
        patience_counter = self.stale_epochs
        for epoch in range(self.start_epoch, self.config.NUM_EPOCHS):
            self.epoch = epoch
            train_metrics = self.train_epoch()
            eval_metrics = self.evaluate()

            self._log_epoch_history(epoch, train_metrics, eval_metrics)

            self.scheduler.step()

            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}:\n"
                  f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Position Loss: {train_metrics.get('position_loss', 0.0):.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.4f}\n"
                  f"Val Loss: {eval_metrics['loss']:.4f}, "
                  f"Position Loss: {eval_metrics.get('position_loss', 0.0):.4f}, "
                  f"Val Acc: {eval_metrics['accuracy']:.4f}\n"
                  f"Val ExpRate: {eval_metrics.get('exp_rate', 0.0):.4f}, "
                  f"Val BLEU-1: {eval_metrics.get('bleu-1', 0.0):.4f}, BLEU-2: {eval_metrics.get('bleu-2', 0.0):.4f}, "
                  f"BLEU-3: {eval_metrics.get('bleu-3', 0.0):.4f}, BLEU-4: {eval_metrics.get('bleu-4', 0.0):.4f}")
            if 'balanced_exp_rate' in eval_metrics:
                print(
                    f"Balanced decode -> Val ExpRate: {eval_metrics['balanced_exp_rate']:.4f}, "
                    f"BLEU-4: {eval_metrics.get('balanced_bleu-4', 0.0):.4f}"
                )

            if self.config.TRAIN_STAGE == "symbols":
                current_metric = eval_metrics["accuracy"]
                metric_name = "validation accuracy"
                is_best = current_metric > self.best_metric
                if is_best:
                    self.best_metric = current_metric
            else:
                current_metric = eval_metrics.get("exp_rate", 0.0)
                current_bleu = eval_metrics.get("bleu-4", 0.0)
                metric_name = "validation ExpRate"
                min_delta = float(getattr(self.config, "EARLY_STOPPING_MIN_DELTA", 1e-4))
                is_best = (
                    current_metric > self.best_metric + min_delta
                    or (
                        abs(current_metric - self.best_metric) <= min_delta
                        and current_bleu > self.best_bleu + min_delta
                    )
                )
                if is_best:
                    self.best_metric = current_metric
                    self.best_bleu = current_bleu
                    print(f"🎉 New best model found with {metric_name}: {current_metric:.4f}")
                val_loss = float(eval_metrics['loss'])
                self.best_loss = min(self.best_loss, val_loss)

            if is_best:
                patience_counter = 0
                if self.config.TRAIN_STAGE == "symbols":
                    print(f"🎉 New best model found with {metric_name}: {current_metric:.4f}")
            else:
                patience_counter += 1
            self.stale_epochs = patience_counter

            model_arch = (
                "shared_patch_vit_patch2_symbols"
                if self.config.TRAIN_STAGE == "symbols"
                else "shared_patch_vit_patch2_full_formula_position_coverage"
            )
            save_checkpoint({
                'epoch': epoch + 1,
                'model_arch': model_arch,
                'patch_size': self.config.PATCH_SIZE,
                'max_patch_grid': self.config.MAX_PATCH_GRID,
                'model_state_dict': self.model.state_dict(),
                'vision_encoder_state_dict': self.model.vision_encoder.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'best_bleu': self.best_bleu,
                'best_metric': self.best_metric,
                'best_loss': self.best_loss,
                'best_metric_name': metric_name,
                'stale_epochs': patience_counter,
                'use_position_auxiliary': self.use_position_auxiliary,
                'position_loss_weight': getattr(self.config, 'POSITION_LOSS_WEIGHT', 0.0),
                'use_coverage': getattr(self.config, 'USE_COVERAGE', False),
                'training_stage': self.config.TRAIN_STAGE,
                'vocab': self.vocab
            }, is_best, self.config.CHECKPOINT_DIR)

            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(
                    f"\nEarly stopping sau {epoch + 1} epochs vì không cải thiện "
                    f"{metric_name} trong {self.config.EARLY_STOPPING_PATIENCE} epochs."
                )
                break

        print(
            f"\nHoàn tất huấn luyện! Best metric ({metric_name}): "
            f"{self.best_metric:.4f}; associated BLEU-4: {self.best_bleu:.4f}"
        )

        if self.history_file:
            self.history_file.close()
            print(f"Đã lưu lịch sử huấn luyện vào {self.history_log_path}")

    def resume_from_checkpoint(self, checkpoint_path: Path):
        """Resume at an epoch boundary with the checkpoint's optimizer and LR state."""
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        if checkpoint.get('training_stage') != self.config.TRAIN_STAGE:
            raise ValueError(
                f"Checkpoint stage {checkpoint.get('training_stage')!r} does not match "
                f"requested stage {self.config.TRAIN_STAGE!r}."
            )
        if checkpoint.get('patch_size', self.config.PATCH_SIZE) != self.config.PATCH_SIZE:
            raise ValueError(
                f"Checkpoint patch size {checkpoint.get('patch_size')} does not match "
                f"configured patch size {self.config.PATCH_SIZE}."
            )
        if checkpoint.get('use_position_auxiliary', self.use_position_auxiliary) != self.use_position_auxiliary:
            raise ValueError("Resume checkpoint position-auxiliary setting differs from the current run.")
        if checkpoint.get('use_coverage', getattr(self.config, 'USE_COVERAGE', False)) != getattr(
            self.config, 'USE_COVERAGE', False
        ):
            raise ValueError("Resume checkpoint coverage setting differs from the current run.")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        saved_epoch = int(checkpoint.get('epoch', 0))
        if saved_epoch >= self.config.NUM_EPOCHS:
            raise ValueError(
                f"Checkpoint is already at epoch {saved_epoch}; --epochs must be larger."
            )
        self.start_epoch = saved_epoch
        remaining_epochs = self.config.NUM_EPOCHS - self.start_epoch
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            # Older checkpoints lack scheduler state; preserve their per-group LRs.
            for group in self.optimizer.param_groups:
                group['initial_lr'] = group['lr']
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_epochs,
                eta_min=self.config.MIN_LR,
            )
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.best_metric = float(checkpoint.get('best_metric', 0.0))
        self.best_bleu = float(checkpoint.get('best_bleu', 0.0))
        self.best_loss = float(checkpoint.get('best_loss', float('inf')))
        self.stale_epochs = int(checkpoint.get('stale_epochs', 0))
        current_lrs = [group['lr'] for group in self.optimizer.param_groups]
        print(
            f"Resumed from {checkpoint_path} at epoch {saved_epoch}; "
            f"best metric: {self.best_metric:.4f}; kept optimizer LRs {current_lrs}."
        )

def main():
    # Windows PowerShell may start Python with a legacy code page that cannot
    # encode the Vietnamese training messages.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Train the patch-2 ViT on full formulas.")
    parser.add_argument("--stage", choices=("symbols", "formula"), default=None)
    parser.add_argument(
        "--init-checkpoint", type=Path, default=None,
        help="Warm-start model weights from a compatible ViT checkpoint; optimizer starts fresh.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--resume-checkpoint", type=Path, default=None,
        help="Resume a training stage from an epoch-boundary checkpoint.",
    )
    args = parser.parse_args()
    config = Config()
    stage = args.stage or config.TRAIN_STAGE
    config.TRAIN_STAGE = stage
    init_checkpoint = args.init_checkpoint
    if not args.resume_checkpoint and stage == "formula" and init_checkpoint is None:
        init_checkpoint = config.INIT_CHECKPOINT
    if args.resume_checkpoint and args.init_checkpoint:
        raise ValueError("Use either a resume checkpoint or an initialization checkpoint, not both.")
    if init_checkpoint and not init_checkpoint.is_file():
        raise FileNotFoundError(f"Initialization checkpoint does not exist: {init_checkpoint}")
    if stage == "symbols":
        config.RUN_NAME = "vit_patch2_symbols_legacy"
        config.IMAGE_SIZE = 32
        config.BATCH_SIZE = config.SYMBOL_BATCH_SIZE
        config.ACCUMULATION_STEPS = 1
        config.USE_POSITION_AUXILIARY = False
        config.USE_COVERAGE = False
        config.NUM_EPOCHS = args.epochs or config.SYMBOL_NUM_EPOCHS
        config.ENCODER_LR = config.SYMBOL_ENCODER_LR
        config.DECODER_LR = config.SYMBOL_DECODER_LR
    else:
        config.RUN_NAME = f"vit_patch2_grid{config.MAX_PATCH_GRID}_full_formula_position_coverage"
        config.IMAGE_SIZE = 448
        config.BATCH_SIZE = config.FORMULA_BATCH_SIZE
        config.ACCUMULATION_STEPS = 2
        config.NUM_EPOCHS = args.epochs or config.NUM_EPOCHS
        if init_checkpoint:
            config.ENCODER_LR = config.FINETUNE_ENCODER_LR
            config.DECODER_LR = config.FINETUNE_DECODER_LR

    config.CHECKPOINT_DIR = config.PROJECT_ROOT / "checkpoints" / config.RUN_NAME
    config.LOG_DIR = config.PROJECT_ROOT / "logs" / config.RUN_NAME
    config.OUTPUT_DIR = config.PROJECT_ROOT / "outputs" / config.RUN_NAME
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.print_config()

    trainer = Trainer(config)
    if args.resume_checkpoint:
        trainer.resume_from_checkpoint(args.resume_checkpoint)
    if init_checkpoint:
        checkpoint = torch.load(init_checkpoint, map_location=config.DEVICE, weights_only=False)
        allowed_architectures = {
            "shared_patch_vit_patch2_classifier",
            "shared_patch_vit_flat_symbols",
        }
        if checkpoint.get("model_arch") not in allowed_architectures:
            raise ValueError(
                "Initialization checkpoint must come from the patch-2 single-symbol "
                "classifier or flat formula-symbol stage."
            )
        if checkpoint.get("patch_size") != config.PATCH_SIZE:
            raise ValueError(
                f"Checkpoint patch size {checkpoint.get('patch_size')} does not match "
                f"this model's patch size {config.PATCH_SIZE}."
            )
        encoder_state = checkpoint.get("vision_encoder_state_dict")
        if not encoder_state:
            raise ValueError("Initialization checkpoint has no vision_encoder_state_dict.")
        trainer.model.vision_encoder.load_state_dict(encoder_state, strict=True)
        print(f"Initialized only the patch-2 ViT encoder from: {init_checkpoint}")
        print("The full-formula decoder, optimizer, and scheduler start fresh.")
    trainer.train()


if __name__ == "__main__":
    main()
