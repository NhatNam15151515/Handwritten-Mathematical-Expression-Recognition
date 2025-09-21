"""
Inference script for Handwritten Mathematical Expression Recognition
"""
import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import json

from config import Config
from model import MathExpressionVisionTransformer
from utils import (
    load_checkpoint, 
    preprocess_image_for_inference, 
    tokens_to_latex,
    visualize_attention
)

class MathExpressionInference:
    def __init__(self, checkpoint_path: Path, config: Optional[Config] = None):
        self.config = config or Config()
        self.device = self.config.DEVICE
        
        # Load checkpoint
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get vocabulary
        self.vocab = checkpoint['vocab']
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        # Create model
        self.model = MathExpressionVisionTransformer(self.config, self.vocab_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Best BLEU from training: {checkpoint.get('best_bleu', 'N/A')}")
    
    def predict(
        self, 
        image_path: Path, 
        max_length: int = 256,
        temperature: float = 1.0,
        beam_size: int = 1
    ) -> Tuple[str, List[str], torch.Tensor]:
        """
        Predict mathematical expression from image
        
        Returns:
            latex_string: Generated LaTeX string
            tokens: List of predicted tokens
            confidence: Confidence scores for each token
        """
        # Preprocess image
        image_tensor = preprocess_image_for_inference(image_path, self.config.IMAGE_SIZE)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            if beam_size > 1:
                tokens, confidence = self._beam_search(
                    image_tensor, max_length, beam_size, temperature
                )
            else:
                tokens, confidence = self._greedy_search(
                    image_tensor, max_length, temperature
                )
        
        # Convert to LaTeX
        latex_string = tokens_to_latex(tokens)
        
        return latex_string, tokens, confidence
    
    def _greedy_search(
        self, 
        image_tensor: torch.Tensor, 
        max_length: int, 
        temperature: float
    ) -> Tuple[List[str], torch.Tensor]:
        """Greedy decoding"""
        batch_size = image_tensor.size(0)
        
        # Encode image
        encoder_output = self.model.vision_encoder(image_tensor)
        
        # Initialize with SOS token
        sos_token_id = self.vocab[self.config.SOS_TOKEN]
        eos_token_id = self.vocab[self.config.EOS_TOKEN]
        pad_token_id = self.vocab[self.config.PAD_TOKEN]
        
        generated = torch.full((batch_size, 1), sos_token_id, device=self.device, dtype=torch.long)
        confidences = []
        
        for _ in range(max_length - 1):
            # Get logits
            logits = self.model.text_decoder(generated, encoder_output)
            
            # Apply temperature
            next_token_logits = logits[:, -1, :] / temperature
            
            # Get probabilities and confidence
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            confidence = probs.max(dim=-1)[0]
            
            confidences.append(confidence.item())
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS token
            if next_token.item() == eos_token_id:
                break
        
        # Convert to tokens
        tokens = []
        for token_id in generated[0].cpu().numpy():
            if token_id == eos_token_id:
                break
            if token_id not in [sos_token_id, pad_token_id]:
                token = self.idx_to_token.get(token_id, self.config.UNK_TOKEN)
                tokens.append(token)
        
        return tokens, torch.tensor(confidences)
    
    def _beam_search(
        self, 
        image_tensor: torch.Tensor, 
        max_length: int, 
        beam_size: int,
        temperature: float
    ) -> Tuple[List[str], torch.Tensor]:
        """Beam search decoding"""
        # Encode image
        encoder_output = self.model.vision_encoder(image_tensor)
        
        sos_token_id = self.vocab[self.config.SOS_TOKEN]
        eos_token_id = self.vocab[self.config.EOS_TOKEN]
        pad_token_id = self.vocab[self.config.PAD_TOKEN]
        
        # Initialize beam
        sequences = [[sos_token_id]]
        scores = [0.0]
        
        for step in range(max_length - 1):
            candidates = []
            
            for i, seq in enumerate(sequences):
                if seq[-1] == eos_token_id:
                    candidates.append((seq, scores[i]))
                    continue
                
                # Get logits for current sequence
                input_seq = torch.tensor([seq], device=self.device, dtype=torch.long)
                logits = self.model.text_decoder(input_seq, encoder_output)
                
                # Apply temperature and get probabilities
                next_token_logits = logits[:, -1, :] / temperature
                probs = F.log_softmax(next_token_logits, dim=-1)
                
                # Get top k candidates
                top_k_probs, top_k_indices = torch.topk(probs, beam_size)
                
                for j in range(beam_size):
                    token_id = top_k_indices[0, j].item()
                    token_score = top_k_probs[0, j].item()
                    
                    new_seq = seq + [token_id]
                    new_score = scores[i] + token_score
                    
                    candidates.append((new_seq, new_score))
            
            # Select top beam_size candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            sequences = [cand[0] for cand in candidates[:beam_size]]
            scores = [cand[1] for cand in candidates[:beam_size]]
            
            # Check if all sequences ended
            if all(seq[-1] == eos_token_id for seq in sequences):
                break
        
        # Get best sequence
        best_seq = sequences[0]
        
        # Convert to tokens
        tokens = []
        for token_id in best_seq:
            if token_id == eos_token_id:
                break
            if token_id not in [sos_token_id, pad_token_id]:
                token = self.idx_to_token.get(token_id, self.config.UNK_TOKEN)
                tokens.append(token)
        
        # Calculate average confidence (approximation)
        avg_confidence = torch.exp(torch.tensor(scores[0] / len(best_seq)))
        
        return tokens, avg_confidence
    
    def predict_batch(
        self, 
        image_paths: List[Path], 
        max_length: int = 256,
        temperature: float = 1.0
    ) -> List[Tuple[str, List[str], float]]:
        """Predict on batch of images"""
        results = []
        
        for image_path in image_paths:
            try:
                latex, tokens, confidence = self.predict(
                    image_path, max_length, temperature
                )
                avg_confidence = confidence.mean().item() if len(confidence) > 0 else 0.0
                results.append((latex, tokens, avg_confidence))
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append(("", [], 0.0))
        
        return results
    
    def visualize_prediction(
        self, 
        image_path: Path, 
        save_path: Optional[Path] = None,
        show_attention: bool = False
    ):
        """Visualize prediction with original image"""
        # Load and preprocess image
        original_image = cv2.imread(str(image_path))
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Make prediction
        latex, tokens, confidence = self.predict(image_path)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2 if not show_attention else 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Prediction text
        axes[1].text(0.1, 0.7, f"LaTeX: {latex}", fontsize=12, wrap=True, 
                    verticalalignment='top', transform=axes[1].transAxes)
        axes[1].text(0.1, 0.5, f"Tokens: {' '.join(tokens)}", fontsize=10, wrap=True,
                    verticalalignment='top', transform=axes[1].transAxes)
        axes[1].text(0.1, 0.3, f"Avg Confidence: {confidence.mean():.3f}", fontsize=10,
                    transform=axes[1].transAxes)
        axes[1].set_title('Prediction')
        axes[1].axis('off')
        
        # Attention visualization (if requested)
        if show_attention and len(axes) > 2:
            # This would require modifying the model to return attention weights
            axes[2].text(0.5, 0.5, "Attention visualization\nrequires model modification", 
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Attention Map')
            axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    def evaluate_on_test_set(self, test_data_path: Path) -> dict:
        """Evaluate model on test set"""
        from dataset import MathExpressionDataset
        from utils import calculate_bleu_score, calculate_edit_distance
        
        # Create test dataset
        test_dataset = MathExpressionDataset(
            image_dir=self.config.TEST_IMAGE_DIR,
            caption_file=self.config.TEST_CAPTION_FILE,
            dictionary_file=self.config.DICTIONARY_FILE,
            is_train=False
        )
        
        predictions = []
        references = []
        edit_distances = []
        
        print(f"Evaluating on {len(test_dataset)} test samples...")
        
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            image_name = sample['image_name']
            target_caption = sample['caption']
            
            # Find image path
            image_path = self.config.TEST_IMAGE_DIR / f"{image_name}.bmp"
            if not image_path.exists():
                image_path = self.config.TEST_IMAGE_DIR / image_name
            
            if not image_path.exists():
                continue
            
            try:
                # Make prediction
                pred_latex, pred_tokens, _ = self.predict(image_path)
                target_tokens = target_caption.split()
                
                predictions.append(pred_latex)
                references.append([target_caption])
                
                # Calculate edit distance
                edit_dist = calculate_edit_distance(pred_tokens, target_tokens)
                edit_distances.append(edit_dist)
                
                if i % 100 == 0:
                    print(f"Processed {i}/{len(test_dataset)} samples")
                    
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                continue
        
        # Calculate metrics
        bleu_score = calculate_bleu_score(predictions, references)
        avg_edit_distance = np.mean(edit_distances)
        
        metrics = {
            'bleu_score': bleu_score,
            'edit_distance': avg_edit_distance,
            'num_samples': len(predictions)
        }
        
        print(f"\nEvaluation Results:")
        print(f"BLEU Score: {bleu_score:.4f}")
        print(f"Average Edit Distance: {avg_edit_distance:.4f}")
        print(f"Number of samples: {len(predictions)}")
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description='Mathematical Expression Recognition Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, 
                       help='Path to input image')
    parser.add_argument('--image_dir', type=str,
                       help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--beam_size', type=int, default=1,
                       help='Beam size for beam search')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for sampling')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate on test set')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization')
    
    args = parser.parse_args()
    
    # Create inference engine
    config = Config()
    inference = MathExpressionInference(Path(args.checkpoint), config)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.evaluate:
        # Evaluate on test set
        metrics = inference.evaluate_on_test_set(config.TEST_CAPTION_FILE)
        
        # Save results
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    
    elif args.image:
        # Single image prediction
        image_path = Path(args.image)
        latex, tokens, confidence = inference.predict(
            image_path, 
            beam_size=args.beam_size,
            temperature=args.temperature
        )
        
        print(f"\nPrediction for {image_path.name}:")
        print(f"LaTeX: {latex}")
        print(f"Tokens: {' '.join(tokens)}")
        print(f"Confidence: {confidence.mean():.3f}")
        
        if args.visualize:
            inference.visualize_prediction(
                image_path, 
                save_path=output_dir / f"{image_path.stem}_prediction.png"
            )
    
    elif args.image_dir:
        # Batch prediction
        image_dir = Path(args.image_dir)
        image_paths = list(image_dir.glob("*.bmp")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        
        print(f"Processing {len(image_paths)} images...")
        
        results = inference.predict_batch(image_paths)
        
        # Save results
        output_file = output_dir / 'batch_predictions.txt'
        with open(output_file, 'w') as f:
            for i, (latex, tokens, confidence) in enumerate(results):
                f.write(f"{image_paths[i].name}\t{latex}\t{confidence:.3f}\n")
        
        print(f"Results saved to {output_file}")
    
    else:
        print("Please specify --image, --image_dir, or --evaluate")

if __name__ == "__main__":
    main()
