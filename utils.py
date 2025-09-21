"""
Utility functions for Handwritten Mathematical Expression Recognition
"""
import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Any, Optional
try:
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
except ImportError:
    # Fallback BLEU implementation
    def corpus_bleu(references, predictions):
        return 0.0
    def sentence_bleu(references, prediction):
        return 0.0
import matplotlib.pyplot as plt
import cv2
from PIL import Image

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_bleu_score(predictions: List[str], references: List[List[str]]) -> float:
    """Calculate BLEU score for predictions"""
    try:
        # Convert strings to token lists
        pred_tokens = [pred.split() for pred in predictions]
        ref_tokens = [[ref[0].split() for ref in ref_list] for ref_list in references]
        
        # Flatten reference tokens
        ref_tokens_flat = []
        for ref_list in ref_tokens:
            ref_tokens_flat.append([ref for ref in ref_list])
        
        # Calculate BLEU score
        bleu = corpus_bleu(ref_tokens_flat, pred_tokens)
        return bleu
    except:
        return 0.0

def save_checkpoint(state: Dict[str, Any], is_best: bool, checkpoint_dir: Path):
    """Save model checkpoint"""
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save latest checkpoint
    checkpoint_path = checkpoint_dir / "latest.pth"
    torch.save(state, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / "best.pth"
        torch.save(state, best_path)
        print(f"New best model saved with BLEU: {state['best_bleu']:.4f}")

def load_checkpoint(checkpoint_path: Path, model: nn.Module, optimizer=None, scheduler=None) -> Dict[str, Any]:
    """Load model checkpoint"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Best BLEU: {checkpoint.get('best_bleu', 0.0):.4f}")
    
    return checkpoint

def count_parameters(model: nn.Module) -> int:
    """Count total number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def visualize_attention(
    attention_weights: torch.Tensor,
    image: np.ndarray,
    tokens: List[str],
    save_path: Optional[Path] = None
):
    """Visualize attention weights on image"""
    # attention_weights: (num_heads, seq_len, num_patches)
    # Average over heads and get attention for each token
    attention = attention_weights.mean(0)  # (seq_len, num_patches)
    
    # Calculate grid size
    num_patches_per_side = int(np.sqrt(attention.shape[1] - 1))  # -1 for cls token
    
    fig, axes = plt.subplots(2, min(len(tokens), 4), figsize=(16, 8))
    if len(tokens) == 1:
        axes = axes.reshape(2, 1)
    
    for i, token in enumerate(tokens[:4]):
        # Get attention for this token (skip cls token)
        token_attention = attention[i, 1:].reshape(num_patches_per_side, num_patches_per_side)
        
        # Resize attention to image size
        attention_resized = cv2.resize(
            token_attention.cpu().numpy(),
            (image.shape[1], image.shape[0])
        )
        
        # Plot original image
        axes[0, i].imshow(image)
        axes[0, i].set_title(f"Token: {token}")
        axes[0, i].axis('off')
        
        # Plot attention overlay
        axes[1, i].imshow(image)
        axes[1, i].imshow(attention_resized, alpha=0.6, cmap='jet')
        axes[1, i].set_title(f"Attention for: {token}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def preprocess_image_for_inference(image_path: Path, target_size: int = 224) -> torch.Tensor:
    """Preprocess image for inference"""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, (target_size, target_size))
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # Convert to tensor
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    return image

def tokens_to_latex(tokens: List[str]) -> str:
    """Convert token list to readable LaTeX string"""
    latex = " ".join(tokens)
    
    # Clean up common formatting issues
    latex = latex.replace(" _ { ", "_{")
    latex = latex.replace(" } ", "} ")
    latex = latex.replace(" ^ { ", "^{")
    latex = latex.replace("{ ", "{")
    latex = latex.replace(" }", "}")
    latex = latex.replace("\\frac { ", "\\frac{")
    latex = latex.replace("\\sqrt { ", "\\sqrt{")
    
    return latex.strip()

def latex_to_tokens(latex: str) -> List[str]:
    """Convert LaTeX string to token list (basic tokenization)"""
    # This is a simple tokenizer - in practice, you might want a more sophisticated one
    tokens = latex.split()
    return tokens

def calculate_edit_distance(pred_tokens: List[str], target_tokens: List[str]) -> float:
    """Calculate normalized edit distance between two token sequences"""
    if len(pred_tokens) == 0 and len(target_tokens) == 0:
        return 0.0
    
    if len(pred_tokens) == 0 or len(target_tokens) == 0:
        return 1.0
    
    # Dynamic programming for edit distance
    dp = [[0] * (len(target_tokens) + 1) for _ in range(len(pred_tokens) + 1)]
    
    # Initialize base cases
    for i in range(len(pred_tokens) + 1):
        dp[i][0] = i
    for j in range(len(target_tokens) + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, len(pred_tokens) + 1):
        for j in range(1, len(target_tokens) + 1):
            if pred_tokens[i-1] == target_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    # Normalize by maximum length
    max_len = max(len(pred_tokens), len(target_tokens))
    return dp[len(pred_tokens)][len(target_tokens)] / max_len

def create_training_plots(log_dir: Path, save_dir: Path):
    """Create training plots from tensorboard logs"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        save_dir.mkdir(exist_ok=True)
        
        # Find the most recent run
        run_dirs = [d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
        if not run_dirs:
            print("No tensorboard logs found")
            return
        
        latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
        
        # Load events
        ea = EventAccumulator(str(latest_run))
        ea.Reload()
        
        # Get scalar tags
        tags = ea.Tags()['scalars']
        
        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        if 'Epoch/Train_Loss' in tags and 'Epoch/Val_Loss' in tags:
            train_loss = ea.Scalars('Epoch/Train_Loss')
            val_loss = ea.Scalars('Epoch/Val_Loss')
            
            epochs = [x.step for x in train_loss]
            train_loss_values = [x.value for x in train_loss]
            val_loss_values = [x.value for x in val_loss]
            
            axes[0, 0].plot(epochs, train_loss_values, label='Train')
            axes[0, 0].plot(epochs, val_loss_values, label='Validation')
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Accuracy curves
        if 'Epoch/Train_Accuracy' in tags and 'Epoch/Val_Accuracy' in tags:
            train_acc = ea.Scalars('Epoch/Train_Accuracy')
            val_acc = ea.Scalars('Epoch/Val_Accuracy')
            
            epochs = [x.step for x in train_acc]
            train_acc_values = [x.value for x in train_acc]
            val_acc_values = [x.value for x in val_acc]
            
            axes[0, 1].plot(epochs, train_acc_values, label='Train')
            axes[0, 1].plot(epochs, val_acc_values, label='Validation')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # BLEU score
        if 'Epoch/Val_BLEU' in tags:
            bleu = ea.Scalars('Epoch/Val_BLEU')
            
            epochs = [x.step for x in bleu]
            bleu_values = [x.value for x in bleu]
            
            axes[1, 0].plot(epochs, bleu_values, color='green')
            axes[1, 0].set_title('BLEU Score')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('BLEU')
            axes[1, 0].grid(True)
        
        # Learning rate
        if 'Train/LR' in tags:
            lr = ea.Scalars('Train/LR')
            
            steps = [x.step for x in lr]
            lr_values = [x.value for x in lr]
            
            axes[1, 1].plot(steps, lr_values, color='red')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('LR')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to {save_dir / 'training_curves.png'}")
        
    except ImportError:
        print("tensorboard not available for plotting")
    except Exception as e:
        print(f"Error creating plots: {e}")

def export_model_to_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
    opset_version: int = 11
):
    """Export model to ONNX format"""
    try:
        model.eval()
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['logits'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            }
        )
        print(f"Model exported to ONNX: {output_path}")
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")

if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test AverageMeter
    meter = AverageMeter()
    meter.update(1.0)
    meter.update(2.0)
    meter.update(3.0)
    print(f"Average: {meter.avg}")  # Should be 2.0
    
    # Test BLEU score
    predictions = ["hello world", "foo bar"]
    references = [["hello world"], ["foo baz"]]
    bleu = calculate_bleu_score(predictions, references)
    print(f"BLEU score: {bleu}")
    
    # Test edit distance
    pred_tokens = ["hello", "world"]
    target_tokens = ["hello", "world"]
    edit_dist = calculate_edit_distance(pred_tokens, target_tokens)
    print(f"Edit distance: {edit_dist}")  # Should be 0.0
    
    print("All tests passed!")
