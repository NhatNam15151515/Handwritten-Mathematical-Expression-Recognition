# error_analysis.py
import torch
from tqdm import tqdm
from typing import Dict, List
from collections import Counter
import difflib
import csv

from config import Config
from dataset import MathExpressionDataset, _pad_collate_fn
from torch.utils.data import DataLoader
from model import MathExpressionVisionTransformer


class ErrorAnalyzer:
    """
    Một class chuyên dụng để phân tích sâu các lỗi của mô hình
    trên tập validation.
    """

    def __init__(self, config: Config, checkpoint_path: str):
        self.config = config
        self.device = config.DEVICE

        # --- 1. Tải dữ liệu ---
        print("Bắt đầu tải dữ liệu validation...")
        val_dataset = MathExpressionDataset(
            processed_image_dir=config.VAL_PROCESSED_IMAGE_DIR,
            caption_file=config.VAL_CAPTION_FILE,
            dictionary_file=config.DICTIONARY_FILE,
            is_train=False
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
            num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
            collate_fn=_pad_collate_fn
        )
        self.vocab = val_dataset.vocab
        self.idx_to_token = val_dataset.idx_to_token

        # --- 2. Tải mô hình và checkpoint ---
        print("Đang khởi tạo mô hình...")
        self.model = MathExpressionVisionTransformer(config, len(self.vocab)).to(self.device)

        ckpt_path = config.CHECKPOINT_DIR / checkpoint_path
        print(f"Đang tải checkpoint từ: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("Tải mô hình thành công!")

        # --- 3. Khởi tạo các bộ đếm lỗi ---
        self.substitution_errors = Counter()  # Lỗi nhầm lẫn: ('target', 'pred')
        self.insertion_errors = Counter()  # Lỗi thừa: ('token_thừa')
        self.deletion_errors = Counter()  # Lỗi thiếu: ('token_thiếu')
        self.error_samples = []  # Lưu các ví dụ sai
        self.total_samples = 0
        self.correct_samples = 0

    def _indices_to_tokens(self, indices) -> List[str]:
        """Chuyển indices thành list các token (không join)."""
        tokens = []
        for idx in indices:
            if idx == self.vocab[self.config.EOS_TOKEN]: break
            if idx not in [self.vocab[self.config.PAD_TOKEN], self.vocab[self.config.SOS_TOKEN]]:
                tokens.append(self.idx_to_token.get(idx, self.config.UNK_TOKEN))
        return tokens

    def analyze_pair(self, target_tokens: List[str], pred_tokens: List[str]):
        """
        Sử dụng difflib để so sánh và ghi nhận lỗi.
        """
        matcher = difflib.SequenceMatcher(None, target_tokens, pred_tokens)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue
            elif tag == 'replace':
                # Ghi nhận lỗi nhầm lẫn
                for t_tok, p_tok in zip(target_tokens[i1:i2], pred_tokens[j1:j2]):
                    if t_tok != p_tok:
                        self.substitution_errors[(t_tok, p_tok)] += 1
            elif tag == 'delete':
                # Ghi nhận lỗi thiếu (token có trong target nhưng không có trong pred)
                for t_tok in target_tokens[i1:i2]:
                    self.deletion_errors[t_tok] += 1
            elif tag == 'insert':
                # Ghi nhận lỗi thừa (token có trong pred nhưng không có trong target)
                for p_tok in pred_tokens[j1:j2]:
                    self.insertion_errors[p_tok] += 1

    def run_analysis(self):
        """Chạy phân tích trên toàn bộ tập validation."""
        print("\nBắt đầu phân tích lỗi trên tập validation...")
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="[Analyzing Errors]")
            for batch in pbar:
                self.total_samples += batch['image'].size(0)
                images = batch['image'].to(self.device)
                target_seq = batch['target_seq'].to(self.device)

                generated_seq = self.model.generate(
                    images, self.vocab[self.config.SOS_TOKEN], self.vocab[self.config.EOS_TOKEN],
                    max_length=self.config.MAX_SEQ_LENGTH
                )

                for i in range(images.size(0)):
                    pred_tokens = self._indices_to_tokens(generated_seq[i].cpu().numpy())
                    target_tokens = self._indices_to_tokens(target_seq[i].cpu().numpy())

                    if pred_tokens == target_tokens:
                        self.correct_samples += 1
                    else:
                        # Chỉ phân tích các cặp bị lỗi
                        self.analyze_pair(target_tokens, pred_tokens)
                        self.error_samples.append({
                            'target': ' '.join(target_tokens),
                            'predicted': ' '.join(pred_tokens)
                        })

    def report_results(self, top_n: int = 20):
        """In ra báo cáo tổng hợp và lưu lại file."""
        print("\n" + "=" * 80)
        print("### BÁO CÁO PHÂN TÍCH LỖI ###")
        print("=" * 80)

        # --- Thống kê chung ---
        error_rate = (1 - self.correct_samples / self.total_samples) * 100
        print(f"\nTổng số mẫu: {self.total_samples}")
        print(f"Số mẫu dự đoán đúng 100% (Exact Match): {self.correct_samples} ({100 - error_rate:.2f}%)")
        print(f"Tỷ lệ lỗi (Error Rate): {error_rate:.2f}%")

        # --- Các lỗi phổ biến nhất ---
        print("\n" + "-" * 40)
        print(f"TOP {top_n} LỖI NHẦM LẪN PHỔ BIẾN NHẤT (SUBSTITUTION)")
        print("(Target -> Predicted): Count")
        print("-" * 40)
        for (target, pred), count in self.substitution_errors.most_common(top_n):
            print(f"'{target}' -> '{pred}': {count} lần")

        print("\n" + "-" * 40)
        print(f"TOP {top_n} TOKEN BỊ BỎ SÓT NHIỀU NHẤT (DELETION)")
        print("Token: Count")
        print("-" * 40)
        for token, count in self.deletion_errors.most_common(top_n):
            print(f"'{token}': {count} lần")

        print("\n" + "-" * 40)
        print(f"TOP {top_n} TOKEN BỊ THÊM VÀO NHIỀU NHẤT (INSERTION)")
        print("Token: Count")
        print("-" * 40)
        for token, count in self.insertion_errors.most_common(top_n):
            print(f"'{token}': {count} lần")

        # --- Lưu ví dụ lỗi ra file ---
        error_file_path = self.config.OUTPUT_DIR / "error_samples.csv"
        try:
            with open(error_file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['target', 'predicted'])
                writer.writeheader()
                writer.writerows(self.error_samples)
            print(f"\nĐã lưu {len(self.error_samples)} ví dụ lỗi vào file: {error_file_path}")
        except IOError as e:
            print(f"\nLỗi: Không thể ghi file ví dụ lỗi. {e}")

        print("\n" + "=" * 80)


def main():
    config = Config()

    # Anh hãy điền tên file checkpoint tốt nhất của mình vào đây
    CHECKPOINT_FILENAME = "best_90_base_fine_tuning.pth"

    analyzer = ErrorAnalyzer(config, CHECKPOINT_FILENAME)
    analyzer.run_analysis()
    analyzer.report_results()


if __name__ == "__main__":
    main()