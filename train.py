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

from config import Config
from dataset import create_dataloaders
from model import MathExpressionVisionTransformer
from utils import AverageMeter, calculate_bleu_score, save_checkpoint

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE
        self.train_loader, self.test_loader = create_dataloaders(config)

        self.vocab_size = len(self.train_loader.dataset.vocab)
        self.vocab = self.train_loader.dataset.vocab
        self.idx_to_token = self.train_loader.dataset.idx_to_token

        self.model = MathExpressionVisionTransformer(config, self.vocab_size).to(self.device)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.vocab[config.PAD_TOKEN],
            label_smoothing=config.LABEL_SMOOTHING
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.NUM_EPOCHS,
            eta_min=config.MIN_LR
        )

        self.writer = SummaryWriter(config.LOG_DIR / f"run_onthefly_{int(time.time())}")
        self.epoch = 0
        self.best_bleu = 0.0
        self.best_loss = float('inf')

        print(f"Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Training on {self.device}")

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        losses, accuracies = AverageMeter(), AverageMeter()
        scaler = torch.amp.GradScaler(enabled=self.config.USE_MIXED_PRECISION)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}/{self.config.NUM_EPOCHS} [Training]")

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            input_seq = batch['input_seq'].to(self.device)
            target_seq = batch['target_seq'].to(self.device)

            with torch.amp.autocast(device_type='cuda', enabled=self.config.USE_MIXED_PRECISION):
                logits = self.model(images, input_seq)
                loss = self.criterion(logits.view(-1, self.vocab_size), target_seq.view(-1))

            if not torch.isfinite(loss): continue

            mask = target_seq != self.vocab[self.config.PAD_TOKEN]
            predictions = torch.argmax(logits, dim=-1)
            accuracy = ((predictions == target_seq) & mask).sum().float() / mask.sum().float()

            scaler.scale(loss / self.config.ACCUMULATION_STEPS).backward()

            if (batch_idx + 1) % self.config.ACCUMULATION_STEPS == 0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

            losses.update(loss.item(), images.size(0))
            accuracies.update(accuracy.item(), images.size(0))
            pbar.set_postfix({'Loss': f'{losses.avg:.4f}', 'Acc': f'{accuracies.avg:.4f}'})

        return {'loss': losses.avg, 'accuracy': accuracies.avg}

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        losses, accuracies = AverageMeter(), AverageMeter()
        all_predictions, all_targets = [], []

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="[Evaluating]")
            for batch in pbar:
                images = batch['image'].to(self.device)
                input_seq = batch['input_seq'].to(self.device)
                target_seq = batch['target_seq'].to(self.device)

                with torch.amp.autocast(device_type='cuda', enabled=self.config.USE_MIXED_PRECISION):
                    logits = self.model(images, input_seq)
                    loss = self.criterion(logits.view(-1, self.vocab_size), target_seq.view(-1))

                if not torch.isfinite(loss): continue

                mask = target_seq != self.vocab[self.config.PAD_TOKEN]
                predictions = torch.argmax(logits, dim=-1)
                accuracy = ((predictions == target_seq) & mask).sum().float() / mask.sum().float()

                losses.update(loss.item(), images.size(0))
                accuracies.update(accuracy.item(), images.size(0))

                gen = self.model.generate(
                    images, self.vocab[self.config.SOS_TOKEN], self.vocab[self.config.EOS_TOKEN],
                    max_length=self.config.MAX_SEQ_LENGTH
                )
                for i in range(gen.size(0)):
                    pred_str = self._indices_to_tokens(gen[i].cpu().numpy())
                    tgt_str = self._indices_to_tokens(target_seq[i].cpu().numpy())
                    all_predictions.append(pred_str)
                    all_targets.append([tgt_str])

        bleu = calculate_bleu_score(all_predictions, all_targets)
        return {'loss': losses.avg, 'accuracy': accuracies.avg, 'bleu': bleu}

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
            batch = next(iter(self.test_loader))
            images = batch['image'][:num_samples].to(self.device)
            captions = batch['caption'][:num_samples]

            generated = self.model.generate(
                images, self.vocab[self.config.SOS_TOKEN], self.vocab[self.config.EOS_TOKEN],
                max_length=self.config.MAX_SEQ_LENGTH
            )

            print("\n" + "="*80 + "\nSAMPLE PREDICTIONS:\n" + "="*80)
            for i in range(num_samples):
                print(f"\nSample {i+1}:\nTarget:     {captions[i]}\nPredicted:  {self._indices_to_tokens(generated[i].cpu().numpy())}\n" + "-" * 60)

    def train(self):
        print(f"\nBắt đầu huấn luyện trong {self.config.NUM_EPOCHS} epochs...")
        for epoch in range(self.config.NUM_EPOCHS):
            self.epoch = epoch
            train_metrics = self.train_epoch()
            eval_metrics = self.evaluate()
            self.scheduler.step()

            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}:\n"
                  f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}\n"
                  f"Val Loss: {eval_metrics['loss']:.4f}, Val Acc: {eval_metrics['accuracy']:.4f}\n"
                  f"Val BLEU: {eval_metrics['bleu']:.4f}")

            is_best = eval_metrics['bleu'] > self.best_bleu
            if is_best: self.best_bleu = eval_metrics['bleu']

            save_checkpoint({
                'epoch': epoch + 1, 'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(), 'best_bleu': self.best_bleu
            }, is_best, self.config.CHECKPOINT_DIR)

            if (epoch + 1) % 10 == 0: self.generate_samples()

        print(f"\nHoàn tất huấn luyện! Best BLEU: {self.best_bleu:.4f}")

def main():
    config = Config()
    config.print_config()
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()