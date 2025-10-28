# evaluate.py (Phiên bản đã cập nhật để hiển thị BLEU chi tiết)
import torch
from tqdm import tqdm
import numpy as np
from typing import Dict

from config import Config
from dataset import MathExpressionDataset, _pad_collate_fn
from torch.utils.data import DataLoader
from model import MathExpressionVisionTransformer
from utils import AverageMeter, calculate_bleu_score


def evaluate_on_test_set(config: Config):
    """
    Tải checkpoint tốt nhất và đánh giá trên tập test cuối cùng.
    """
    print("=" * 60)
    print("Bắt đầu quá trình đánh giá cuối cùng trên TẬP TEST")
    print("=" * 60)

    device = config.DEVICE

    print(f"Đang tải dữ liệu từ: {config.TEST_CAPTION_FILE}")
    test_dataset = MathExpressionDataset(
        processed_image_dir=config.TEST_PROCESSED_IMAGE_DIR,
        caption_file=config.TEST_CAPTION_FILE,
        dictionary_file=config.DICTIONARY_FILE,
        is_train=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        collate_fn=_pad_collate_fn
    )

    vocab_size = len(test_dataset.vocab)
    vocab = test_dataset.vocab
    idx_to_token = test_dataset.idx_to_token

    # 2. TẢI LÊN MÔ HÌNH VÀ CHECKPOINT
    print("Đang khởi tạo mô hình...")
    model = MathExpressionVisionTransformer(config, vocab_size).to(device)

    checkpoint_path = config.CHECKPOINT_DIR / "best_89_base.pth"
    if not checkpoint_path.exists():
        print(f"Lỗi: Không tìm thấy checkpoint tại '{checkpoint_path}'.")
        print("Hãy chắc chắn đã huấn luyện và lưu lại checkpoint tốt nhất.")
        return

    print(f"Đang tải checkpoint từ: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Tải checkpoint thành công!")

    # 3. CHẠY ĐÁNH GIÁ
    model.eval()
    accuracies = AverageMeter()
    all_predictions, all_targets = [], []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="[Final Evaluation on TEST set]")
        for batch in pbar:
            images = batch['image'].to(device)
            target_seq = batch['target_seq'].to(device)

            # --- Tính Accuracy (tùy chọn) ---
            input_seq = batch['input_seq'].to(device)
            with torch.amp.autocast('cuda', enabled=config.USE_MIXED_PRECISION):
                logits = model(images, input_seq)
            mask = target_seq != vocab[config.PAD_TOKEN]
            predictions = torch.argmax(logits, dim=-1)
            accuracy = ((predictions == target_seq) & mask).sum().float() / mask.sum().float()
            accuracies.update(accuracy.item(), images.size(0))

            # --- Tính BLEU ---
            gen = model.generate(
                images, vocab[config.SOS_TOKEN], vocab[config.EOS_TOKEN],
                max_length=config.MAX_SEQ_LENGTH
            )
            for i in range(gen.size(0)):
                pred_str = ' '.join([idx_to_token.get(idx, config.UNK_TOKEN) for idx in gen[i].cpu().numpy() if
                                     idx not in [vocab[config.PAD_TOKEN], vocab[config.SOS_TOKEN],
                                                 vocab[config.EOS_TOKEN]]])
                tgt_str = ' '.join([idx_to_token.get(idx, config.UNK_TOKEN) for idx in target_seq[i].cpu().numpy() if
                                    idx not in [vocab[config.PAD_TOKEN], vocab[config.SOS_TOKEN],
                                                vocab[config.EOS_TOKEN]]])
                all_predictions.append(pred_str)
                all_targets.append([tgt_str])

    # <<< THAY ĐỔI 1: Nhận dictionary chứa các điểm BLEU >>>
    bleu_scores = calculate_bleu_score(all_predictions, all_targets)

    # 4. IN KẾT QUẢ CUỐI CÙNG
    print("\n" + "=" * 60)
    print("KẾT QUẢ CUỐI CÙNG TRÊN TẬP TEST")
    print("=" * 60)
    print(f"Test Accuracy: {accuracies.avg:.4f}")

    # <<< THAY ĐỔI 2: In ra bảng điểm BLEU chi tiết >>>
    print("-" * 30)
    print("Detailed BLEU Scores:")
    if bleu_scores:
        print(f"  - BLEU-1 (Individual tokens): {bleu_scores.get('bleu-1', 0.0):.4f}")
        print(f"  - BLEU-2 (Token pairs):       {bleu_scores.get('bleu-2', 0.0):.4f}")
        print(f"  - BLEU-3 (Token triplets):    {bleu_scores.get('bleu-3', 0.0):.4f}")
        print(f"  - BLEU-4 (Standard):          {bleu_scores.get('bleu-4', 0.0):.4f}")
    else:
        print("  - Không thể tính toán điểm BLEU.")
    print("=" * 60)


if __name__ == "__main__":
    cfg = Config()
    evaluate_on_test_set(cfg)