# debug_sanity_check.py
import torch
import torch.nn as nn
from torch.optim import AdamW

from config import Config
from model import MathExpressionVisionTransformer
from dataset import create_dataloaders


def calculate_accuracy(logits, targets, pad_idx):
    """Tính accuracy, bỏ qua các vị trí padding."""
    # Lấy token có xác suất cao nhất
    preds = logits.argmax(dim=-1)

    # Tạo mask để không tính các vị trí padding
    non_pad_mask = (targets != pad_idx)

    # Số token dự đoán đúng (và không phải padding)
    correct_tokens = ((preds == targets) & non_pad_mask).sum()

    # Tổng số token không phải padding
    total_non_pad_tokens = non_pad_mask.sum()

    # Tránh chia cho 0
    if total_non_pad_tokens == 0:
        return 0.0

    return (correct_tokens.float() / total_non_pad_tokens.float()).item() * 100


def run_sanity_check():
    config = Config()
    train_loader, _ = create_dataloaders(config)

    # Lấy vocab để biết PAD_TOKEN index
    vocab = train_loader.dataset.vocab
    pad_idx = vocab[Config.PAD_TOKEN]

    model = MathExpressionVisionTransformer(config, vocab_size=len(vocab))
    model.to(config.DEVICE)

    # --- BƯỚC QUAN TRỌNG: LẤY 1 BATCH DUY NHẤT ---
    print("Fetching one batch for the sanity check...")
    fixed_batch = next(iter(train_loader))

    # Đưa dữ liệu của batch cố định lên device
    images = fixed_batch['image'].to(config.DEVICE)
    input_seq = fixed_batch['input_seq'].to(config.DEVICE)
    target_seq = fixed_batch['target_seq'].to(config.DEVICE)

    # Optimizer và Loss function (RẤT QUAN TRỌng: ignore_index=pad_idx)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    print("Starting sanity check training loop...")
    model.train()
    for epoch in range(1, 201):
        optimizer.zero_grad()

        # Forward pass chỉ với batch cố định
        logits = model(images, input_seq)

        # Reshape để tính loss
        # logits: (B, SeqLen, VocabSize) -> (B*SeqLen, VocabSize)
        # target_seq: (B, SeqLen) -> (B*SeqLen)
        loss = criterion(logits.view(-1, logits.size(-1)), target_seq.view(-1))

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            # Tính accuracy trên chính batch đó
            acc = calculate_accuracy(logits, target_seq, pad_idx)
            print(f"Epoch [{epoch:03d}/200] | Loss: {loss.item():.4f} | Accuracy on this batch: {acc:.2f}%")


if __name__ == "__main__":
    run_sanity_check()