# evaluate.py (Phiên bản đã cập nhật để hiển thị BLEU chi tiết)
import torch
from matplotlib.mathtext import MathTextParser
from tqdm import tqdm

from config import Config
from dataset import MathExpressionDataset, _pad_collate_fn
from torch.utils.data import DataLoader
from model import MathExpressionVisionTransformer
from utils import AverageMeter, calculate_bleu_score, calculate_edit_distance, tokens_to_latex


def check_latex_valid(latex_str: str) -> bool:
    """Kiểm tra xem chuỗi LaTeX có thể compile được không (dùng matplotlib parser)"""
    if not latex_str.strip():
        return False
    try:
        # Cố gắng parse chuỗi LaTeX
        MathTextParser('path').parse(f"${latex_str}$")
        return True
    except Exception:
        # Nếu có lỗi (thiếu ngoặc, cú pháp sai...) -> Không hợp lệ
        return False


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

    checkpoint_path = config.CHECKPOINT_DIR / "best.pth"
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
    exact_matches = 0
    total_samples = 0
    total_edit_distance = 0
    valid_latex_count = 0

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
                pred_tokens = [idx_to_token.get(idx, config.UNK_TOKEN) for idx in gen[i].cpu().numpy() if
                               idx not in [vocab[config.PAD_TOKEN], vocab[config.EOS_TOKEN],
                                           vocab[config.SOS_TOKEN]]]
                tgt_tokens = [idx_to_token.get(idx, config.UNK_TOKEN) for idx in target_seq[i].cpu().numpy() if
                              idx not in [vocab[config.PAD_TOKEN], vocab[config.SOS_TOKEN],
                                          vocab[config.EOS_TOKEN]]]
                pred_str = tokens_to_latex(pred_tokens)
                tgt_str = tokens_to_latex(tgt_tokens)
                all_predictions.append(pred_str)
                all_targets.append([tgt_str])

                total_samples += 1

                if pred_str == tgt_str:
                    exact_matches += 1
                ned = calculate_edit_distance(pred_tokens, tgt_tokens)
                total_edit_distance += ned

                if check_latex_valid(pred_str):
                    valid_latex_count += 1

    # <<< THAY ĐỔI 1: Nhận dictionary chứa các điểm BLEU >>>
    bleu_scores = calculate_bleu_score(all_predictions, all_targets)

    # 4. IN KẾT QUẢ CUỐI CÙNG
    # Tính toán phần trăm
    exact_match_rate = exact_matches / max(1, total_samples)
    avg_edit_distance = total_edit_distance / max(1, total_samples)
    latex_valid_rate = valid_latex_count / max(1, total_samples)

    print("\n" + "=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ CHẤT LƯỢNG NÂNG CAO")
    print("=" * 60)
    print(f"Exact Match (Công thức đúng 100%): {exact_match_rate:.4f} ({exact_matches}/{total_samples})")
    print(f"Normalized Edit Distance (Càng thấp càng tốt): {avg_edit_distance:.4f}")
    print(f"Tỷ lệ Compile LaTeX hợp lệ: {latex_valid_rate:.4f} ({valid_latex_count}/{total_samples})")
    print("=" * 60)


if __name__ == "__main__":
    cfg = Config()
    evaluate_on_test_set(cfg)
