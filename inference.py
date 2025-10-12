"""
Interface for Handwritten Mathematical Expression Recognition
- Single image prediction
- Batch prediction
- Visualization
- Evaluation on test set
- Save results as JSON
"""
import torch
import argparse
from pathlib import Path
import json
import cv2
import matplotlib.pyplot as plt

from config import Config
from model import MathExpressionVisionTransformer
from dataset import MathExpressionDataset
from utils import (
    load_checkpoint,
    preprocess_image_for_inference,
    tokens_to_latex,
    calculate_bleu_score,
    calculate_edit_distance
)

class MathExpressionInference:
    def __init__(self, checkpoint_path: Path, config: Config = None):
        self.config = config or Config()
        self.device = self.config.DEVICE

        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.vocab = checkpoint['vocab']
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        self.model = MathExpressionVisionTransformer(self.config, self.vocab_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully! Vocabulary size: {self.vocab_size}")
        print(f"Best BLEU from training: {checkpoint.get('best_bleu', 'N/A')}")

    def predict(self, image_path: Path, max_length: int = 256, beam_size: int = 1, temperature: float = 1.0):
        """Predict LaTeX from image"""
        image_tensor = preprocess_image_for_inference(image_path, self.config.IMAGE_SIZE)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            if beam_size > 1:
                tokens, confidence = self._beam_search(image_tensor, max_length, beam_size, temperature)
            else:
                tokens, confidence = self._greedy_search(image_tensor, max_length, temperature)

        latex = tokens_to_latex(tokens)
        return latex, tokens, confidence

    def _greedy_search(self, image_tensor, max_length, temperature):
        encoder_output = self.model.vision_encoder(image_tensor)
        sos_id = self.vocab[self.config.SOS_TOKEN]
        eos_id = self.vocab[self.config.EOS_TOKEN]
        pad_id = self.vocab[self.config.PAD_TOKEN]

        generated = torch.full((1, 1), sos_id, device=self.device, dtype=torch.long)
        confidences = []

        for _ in range(max_length - 1):
            logits = self.model.text_decoder(generated, encoder_output)
            next_logits = logits[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(next_logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            confidences.append(probs.max(dim=-1)[0].item())
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == eos_id:
                break

        tokens = [self.idx_to_token.get(tid, self.config.UNK_TOKEN)
                  for tid in generated[0].cpu().numpy() if tid not in [sos_id, pad_id, eos_id]]
        return tokens, torch.tensor(confidences)

    def _beam_search(self, image_tensor, max_length, beam_size, temperature):
        encoder_output = self.model.vision_encoder(image_tensor)
        sos_id = self.vocab[self.config.SOS_TOKEN]
        eos_id = self.vocab[self.config.EOS_TOKEN]
        pad_id = self.vocab[self.config.PAD_TOKEN]

        sequences = [[sos_id]]
        scores = [0.0]

        for _ in range(max_length - 1):
            candidates = []
            for i, seq in enumerate(sequences):
                if seq[-1] == eos_id:
                    candidates.append((seq, scores[i]))
                    continue
                input_seq = torch.tensor([seq], device=self.device, dtype=torch.long)
                logits = self.model.text_decoder(input_seq, encoder_output)
                next_logits = logits[:, -1, :] / temperature
                probs = torch.nn.functional.log_softmax(next_logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, beam_size)
                for j in range(beam_size):
                    new_seq = seq + [topk_indices[0, j].item()]
                    new_score = scores[i] + topk_probs[0, j].item()
                    candidates.append((new_seq, new_score))
            candidates.sort(key=lambda x: x[1], reverse=True)
            sequences = [c[0] for c in candidates[:beam_size]]
            scores = [c[1] for c in candidates[:beam_size]]
            if all(seq[-1] == eos_id for seq in sequences):
                break

        best_seq = sequences[0]
        tokens = [self.idx_to_token.get(tid, self.config.UNK_TOKEN)
                  for tid in best_seq if tid not in [sos_id, pad_id, eos_id]]
        avg_confidence = torch.exp(torch.tensor(scores[0] / len(best_seq)))
        return tokens, avg_confidence

    def predict_batch(self, image_paths, max_length=256, beam_size=1, temperature=1.0):
        results = []
        for path in image_paths:
            try:
                latex, tokens, confidence = self.predict(path, max_length, beam_size, temperature)
                avg_conf = confidence.mean().item() if len(confidence) > 0 else 0.0
                results.append((latex, tokens, avg_conf))
            except Exception as e:
                print(f"Error {path.name}: {e}")
                results.append(("", [], 0.0))
        return results

    def visualize_prediction(self, image_path: Path, latex: str, tokens: list, confidence: float, save_path: Path = None):
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].axis('off')
        axes[0].set_title("Input Image")

        axes[1].text(0.1, 0.7, f"LaTeX: {latex}", fontsize=12, wrap=True, transform=axes[1].transAxes)
        axes[1].text(0.1, 0.5, f"Tokens: {' '.join(tokens)}", fontsize=10, wrap=True, transform=axes[1].transAxes)
        axes[1].text(0.1, 0.3, f"Avg Confidence: {confidence:.3f}", fontsize=10, transform=axes[1].transAxes)
        axes[1].axis('off')
        axes[1].set_title("Prediction")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def evaluate_test_set(self):
        dataset = MathExpressionDataset(
            image_dir=self.config.TEST_IMAGE_DIR,
            caption_file=self.config.TEST_CAPTION_FILE,
            dictionary_file=self.config.DICTIONARY_FILE,
            is_train=False
        )
        predictions, references, edit_distances = [], [], []
        for i, sample in enumerate(dataset):
            image_path = self.config.TEST_IMAGE_DIR / f"{sample['image_name']}.bmp"
            if not image_path.exists():
                image_path = self.config.TEST_IMAGE_DIR / sample['image_name']
            try:
                pred_latex, pred_tokens, _ = self.predict(image_path)
                target_tokens = sample['caption'].split()
                predictions.append(pred_latex)
                references.append([sample['caption']])
                edit_distances.append(calculate_edit_distance(pred_tokens, target_tokens))
                if i % 100 == 0:
                    print(f"Processed {i}/{len(dataset)}")
            except Exception as e:
                print(f"Error {sample['image_name']}: {e}")

        bleu = calculate_bleu_score(predictions, references)
        avg_edit = float(sum(edit_distances)/len(edit_distances))
        metrics = {"bleu_score": bleu, "edit_distance": avg_edit, "num_samples": len(predictions)}
        print(metrics)
        return metrics

    @staticmethod
    def save_single(output_dir: Path, image_name: str, latex: str, tokens: list, confidence: float):
        output_dir.mkdir(exist_ok=True, parents=True)
        with open(output_dir / f"{Path(image_name).stem}_prediction.json", 'w', encoding='utf-8') as f:
            json.dump({"image_name": image_name, "latex": latex, "tokens": tokens, "confidence": confidence}, f, indent=2, ensure_ascii=False)

    @staticmethod
    def save_batch(output_dir: Path, image_paths: list, results: list):
        output_dir.mkdir(exist_ok=True, parents=True)
        batch_data = []
        for path, (latex, tokens, confidence) in zip(image_paths, results):
            batch_data.append({"image_name": path.name, "latex": latex, "tokens": tokens, "confidence": confidence})
        with open(output_dir / 'batch_predictions.json', 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    config = Config()
    engine = MathExpressionInterface(Path(args.checkpoint), config)
    output_dir = Path(args.output_dir)

    if args.evaluate:
        metrics = engine.evaluate_test_set()
        with open(output_dir / 'evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    elif args.image:
        image_path = Path(args.image)
        latex, tokens, confidence = engine.predict(image_path, beam_size=args.beam_size, temperature=args.temperature)
        print(f"LaTeX: {latex}\nTokens: {' '.join(tokens)}\nConfidence: {confidence.mean():.3f}")
        engine.save_single(output_dir, image_path.name, latex, tokens, confidence.mean().item())
        if args.visualize:
            engine.visualize_prediction(image_path, latex, tokens, confidence.mean().item(), save_path=output_dir / f"{image_path.stem}_prediction.png")

    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_paths = list(image_dir.glob("*.bmp")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        results = engine.predict_batch(image_paths, beam_size=args.beam_size, temperature=args.temperature)
        engine.save_batch(output_dir, image_paths, results)

    else:
        print("Specify --image, --image_dir, or --evaluate")


if __name__ == "__main__":
    main()
