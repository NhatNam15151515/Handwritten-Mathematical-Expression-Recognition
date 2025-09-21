"""
Training script for Handwritten Mathematical Expression Recognition
using Vision Transformer
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple

from config import Config
from dataset import create_dataloaders, MathExpressionDataset
from model import MathExpressionVisionTransformer
from utils import AverageMeter, calculate_bleu_score, save_checkpoint, load_checkpoint
import torch

# Check có GPU không
print("CUDA available:", torch.cuda.is_available())

# Xem tên GPU
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("Training on CPU")

# Đặt device cho model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE
        
        # Create dataloaders
        self.train_loader, self.test_loader = create_dataloaders(config)
        
        # Get vocabulary size
        self.vocab_size = len(self.train_loader.dataset.vocab)
        self.vocab = self.train_loader.dataset.vocab
        self.idx_to_token = self.train_loader.dataset.idx_to_token
        
        # Create model
        self.model = MathExpressionVisionTransformer(config, self.vocab_size).to(self.device)
        
        # Loss function (ignore padding tokens)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab[config.PAD_TOKEN])
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.NUM_EPOCHS,
            eta_min=config.LEARNING_RATE * 0.01
        )
        
        # Tensorboard
        self.writer = SummaryWriter(config.LOG_DIR / f"run_{int(time.time())}")
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        self.best_bleu = 0.0
        
        print(f"Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Training on {self.device}")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}/{self.config.NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            input_seq = batch['input_seq'].to(self.device)
            target_seq = batch['target_seq'].to(self.device)
            
            # Forward pass
            logits = self.model(images, input_seq)
            
            # Calculate loss
            loss = self.criterion(logits.view(-1, self.vocab_size), target_seq.view(-1))
            
            # Calculate accuracy (excluding padding tokens)
            mask = target_seq != self.vocab[self.config.PAD_TOKEN]
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == target_seq) & mask
            accuracy = correct.sum().float() / mask.sum().float()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
            
            self.optimizer.step()
            
            # Update metrics
            losses.update(loss.item(), images.size(0))
            accuracies.update(accuracy.item(), images.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{accuracies.avg:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log to tensorboard
            global_step = self.epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/Accuracy', accuracy.item(), global_step)
            self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], global_step)
            
        return {
            'loss': losses.avg,
            'accuracy': accuracies.avg
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on test set"""
        self.model.eval()
        
        losses = AverageMeter()
        accuracies = AverageMeter()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Evaluating")
            
            for batch in pbar:
                images = batch['image'].to(self.device)
                input_seq = batch['input_seq'].to(self.device)
                target_seq = batch['target_seq'].to(self.device)
                
                # Forward pass
                logits = self.model(images, input_seq)
                
                # Calculate loss
                loss = self.criterion(logits.view(-1, self.vocab_size), target_seq.view(-1))
                
                # Calculate accuracy
                mask = target_seq != self.vocab[self.config.PAD_TOKEN]
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == target_seq) & mask
                accuracy = correct.sum().float() / mask.sum().float()
                
                # Update metrics
                losses.update(loss.item(), images.size(0))
                accuracies.update(accuracy.item(), images.size(0))
                
                # Collect predictions for BLEU score
                for i in range(predictions.size(0)):
                    pred_tokens = self._indices_to_tokens(predictions[i].cpu().numpy())
                    target_tokens = self._indices_to_tokens(target_seq[i].cpu().numpy())
                    all_predictions.append(pred_tokens)
                    all_targets.append([target_tokens])  # BLEU expects list of references
                
                pbar.set_postfix({
                    'Loss': f'{losses.avg:.4f}',
                    'Acc': f'{accuracies.avg:.4f}'
                })
        
        # Calculate BLEU score
        bleu_score = calculate_bleu_score(all_predictions, all_targets)
        
        return {
            'loss': losses.avg,
            'accuracy': accuracies.avg,
            'bleu': bleu_score
        }
    
    def _indices_to_tokens(self, indices: np.ndarray) -> str:
        """Convert token indices to string"""
        tokens = []
        for idx in indices:
            if idx == self.vocab[self.config.EOS_TOKEN]:
                break
            if idx not in [self.vocab[self.config.PAD_TOKEN], self.vocab[self.config.SOS_TOKEN]]:
                token = self.idx_to_token.get(idx, self.config.UNK_TOKEN)
                tokens.append(token)
        return ' '.join(tokens)
    
    def generate_samples(self, num_samples: int = 5):
        """Generate sample predictions"""
        self.model.eval()
        
        with torch.no_grad():
            batch = next(iter(self.test_loader))
            images = batch['image'][:num_samples].to(self.device)
            captions = batch['caption'][:num_samples]
            
            # Generate predictions
            generated = self.model.generate(
                images,
                sos_token_id=self.vocab[self.config.SOS_TOKEN],
                eos_token_id=self.vocab[self.config.EOS_TOKEN],
                max_length=self.config.MAX_SEQ_LENGTH
            )
            
            print("\n" + "="*80)
            print("SAMPLE PREDICTIONS:")
            print("="*80)
            
            for i in range(num_samples):
                pred_tokens = self._indices_to_tokens(generated[i].cpu().numpy())
                target_caption = captions[i]
                
                print(f"\nSample {i+1}:")
                print(f"Target:     {target_caption}")
                print(f"Predicted:  {pred_tokens}")
                print("-" * 60)
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.config.NUM_EPOCHS} epochs...")
        
        for epoch in range(self.config.NUM_EPOCHS):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Evaluate
            eval_metrics = self.evaluate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            self.writer.add_scalar('Epoch/Train_Loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Epoch/Train_Accuracy', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Epoch/Val_Loss', eval_metrics['loss'], epoch)
            self.writer.add_scalar('Epoch/Val_Accuracy', eval_metrics['accuracy'], epoch)
            self.writer.add_scalar('Epoch/Val_BLEU', eval_metrics['bleu'], epoch)
            
            # Print results
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}:")
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {eval_metrics['loss']:.4f}, Val Acc: {eval_metrics['accuracy']:.4f}")
            print(f"Val BLEU: {eval_metrics['bleu']:.4f}")
            
            # Save checkpoint
            is_best = eval_metrics['bleu'] > self.best_bleu
            if is_best:
                self.best_bleu = eval_metrics['bleu']
                self.best_loss = eval_metrics['loss']
            
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_bleu': self.best_bleu,
                'best_loss': self.best_loss,
                'config': self.config,
                'vocab': self.vocab
            }, is_best, self.config.CHECKPOINT_DIR)
            
            # Generate samples every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.generate_samples()
        
        print(f"\nTraining completed!")
        print(f"Best BLEU score: {self.best_bleu:.4f}")
        print(f"Best validation loss: {self.best_loss:.4f}")
        
        self.writer.close()

def main():
    config = Config()
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()


