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

        self.model = MathExpressionVisionTransformer(config, self.vocab_size).to(self.device)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.vocab[config.PAD_TOKEN],
            label_smoothing=config.LABEL_SMOOTHING
        )
        encoder_params = self.model.vision_encoder.parameters()
        decoder_params = [
            {'params': self.model.encoder_projection.parameters()},
            {'params': self.model.text_decoder.parameters()}
        ]

        # Tạo danh sách các nhóm tham số cho optimizer
        param_groups = [
            {'params': list(encoder_params), 'lr': config.ENCODER_LR},
            {'params': [p for group in decoder_params for p in group['params']], 'lr': config.DECODER_LR}
        ]

        print(f"Optimizer: Using differential LR -> Encoder LR: {config.ENCODER_LR}, Decoder LR: {config.DECODER_LR}")

        self.optimizer = optim.AdamW(
            param_groups,  # Truyền các nhóm tham số đã được định nghĩa
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
            headers = ['epoch', 'train_loss',
                       'train_accuracy',
                       'val_loss',
                       'val_accuracy',
                       'val_bleu_1', 'val_bleu_2', 'val_bleu_3', 'val_bleu_4',
                       'encoder_lr',
                       'decoder_lr'] + self.hyperparam_keys

            self.history_writer.writerow(headers)
            print(f"Lịch sử huấn luyện sẽ được ghi vào: {self.history_log_path}")
        except IOError as e:
            print(f"Lỗi: Không thể mở file log lịch sử. {e}")
            self.history_file = None

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        losses, accuracies = AverageMeter(), AverageMeter()
        scaler = torch.amp.GradScaler(enabled=self.config.USE_MIXED_PRECISION)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}/{self.config.NUM_EPOCHS} [Training]")

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
            pbar = tqdm(self.val_loader, desc="[Evaluating]")
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

        bleu_scores = calculate_bleu_score(all_predictions, all_targets)
        eval_results = {'loss': losses.avg, 'accuracy': accuracies.avg}
        eval_results.update(bleu_scores)  # Gộp dictionary bleu_scores vào
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
                          train_metrics['accuracy'],
                          eval_metrics['loss'],
                          eval_metrics['accuracy'],
                          eval_metrics.get('bleu-1', 0.0),
                          eval_metrics.get('bleu-2', 0.0),
                          eval_metrics.get('bleu-3', 0.0),
                          eval_metrics.get('bleu-4', 0.0),
                          encoder_lr,
                          decoder_lr
                      ] + self.hyperparam_values

                self.history_writer.writerow(row)
                self.history_file.flush()
            except IOError as e:
                print(f"Lỗi: Không thể ghi vào file log lịch sử. {e}")

    def train(self):
        print(f"\nBắt đầu huấn luyện trong {self.config.NUM_EPOCHS} epochs...")
        patience_counter = 0
        for epoch in range(self.config.NUM_EPOCHS):
            self.epoch = epoch
            train_metrics = self.train_epoch()
            eval_metrics = self.evaluate()

            self._log_epoch_history(epoch, train_metrics, eval_metrics)

            self.scheduler.step()

            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}:\n"
                  f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}\n"
                  f"Val Loss: {eval_metrics['loss']:.4f}, Val Acc: {eval_metrics['accuracy']:.4f}\n"
                  f"Val BLEU-1: {eval_metrics.get('bleu-1', 0.0):.4f}, BLEU-2: {eval_metrics.get('bleu-2', 0.0):.4f}, "
                  f"BLEU-3: {eval_metrics.get('bleu-3', 0.0):.4f}, BLEU-4: {eval_metrics.get('bleu-4', 0.0):.4f}")

            current_bleu = eval_metrics.get('bleu-4', 0.0)
            is_best = current_bleu > self.best_bleu
            if is_best:
                self.best_bleu = current_bleu
                patience_counter = 0
                print(f"🎉 New best model found with BLEU-4: {self.best_bleu:.4f}")
            else:
                patience_counter += 1

            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_bleu': self.best_bleu,
                'vocab': self.vocab
            }, is_best, self.config.CHECKPOINT_DIR)

            if (epoch + 1) % 10 == 0: self.generate_samples()

            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(
                    f"\nEarly stopping sau {epoch + 1} epochs vì không có cải thiện trong {self.config.EARLY_STOPPING_PATIENCE} epochs.")
                break

        print(f"\nHoàn tất huấn luyện! Best BLEU: {self.best_bleu:.4f}")

        if self.history_file:
            self.history_file.close()
            print(f"Đã lưu lịch sử huấn luyện vào {self.history_log_path}")

def main():
    config = Config()
    config.print_config()

    # --- BẮT ĐẦU LOGIC FINE-TUNING ---

    # 1. Khởi tạo Trainer như bình thường
    trainer = Trainer(config)

    # 2. Tải lên checkpoint tốt nhất đã có
    CHECKPOINT_TO_LOAD = "best.pth"  # Hoặc best_89_base.pth,...
    checkpoint_path = config.CHECKPOINT_DIR / CHECKPOINT_TO_LOAD
    FINE_TUNING = False

    if FINE_TUNING and checkpoint_path.exists():
        print("\n" + "#" * 60)
        print(f"### TIẾN HÀNH FINE-TUNING TỪ CHECKPOINT: {CHECKPOINT_TO_LOAD} ###")
        print("#" * 60 + "\n")

        # Tải trọng số mô hình
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])

        # (Tùy chọn nhưng khuyến khích) Tải lại trạng thái optimizer để tiếp tục mượt mà hơn
        if 'optimizer_state_dict' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Cập nhật các thông số từ checkpoint
        trainer.epoch = checkpoint.get('epoch', 0)
        trainer.best_bleu = checkpoint.get('best_bleu', 0.0)

        print(f"Đã tải thành công checkpoint từ epoch {trainer.epoch}. Best BLEU hiện tại: {trainer.best_bleu:.4f}")

        # 3. THIẾT LẬP CÁC THAM SỐ CHO GIAI ĐOẠN FINE-TUNING

        # Cực kỳ quan trọng: Sử dụng learning rate rất thấp!
        FINETUNE_ENCODER_LR = 5e-6  # Rất thấp để "lay nhẹ" encoder
        FINETUNE_DECODER_LR = 1e-5  # Thấp hơn so với training ban đầu

        print(
            f"Cập nhật optimizer cho fine-tuning. Encoder LR: {FINETUNE_ENCODER_LR}, Decoder LR: {FINETUNE_DECODER_LR}")

        # Ghi đè learning rate trong optimizer đã tải
        trainer.optimizer.param_groups[0]['lr'] = FINETUNE_ENCODER_LR
        trainer.optimizer.param_groups[1]['lr'] = FINETUNE_DECODER_LR

        # Thiết lập số epoch cho giai đoạn fine-tuning này
        NUM_FINETUNE_EPOCHS = 20

        # Reset scheduler với số epoch mới
        trainer.scheduler = CosineAnnealingLR(
            trainer.optimizer,
            T_max=NUM_FINETUNE_EPOCHS,
            eta_min=config.MIN_LR
        )

        # Ghi đè lại NUM_EPOCHS trong config để thanh tiến trình hiển thị đúng
        trainer.config.NUM_EPOCHS = trainer.epoch + NUM_FINETUNE_EPOCHS

        # 4. Bắt đầu huấn luyện tiếp
        trainer.train()

    else:
        # Nếu không tìm thấy checkpoint, chạy huấn luyện từ đầu như bình thường
        print("\n" + "#" * 60)
        print(f"BẮT ĐẦU HUẤN LUYỆN TỪ ĐẦU")
        print("#" * 60 + "\n")
        trainer.train()


if __name__ == "__main__":
    main()