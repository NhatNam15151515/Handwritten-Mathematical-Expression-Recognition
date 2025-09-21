"""
Demo script for Handwritten Mathematical Expression Recognition
Chạy demo đơn giản để test model
"""
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

from config import Config
from dataset import create_dataloaders
from model import MathExpressionVisionTransformer
from inference import MathExpressionInference
from utils import preprocess_image_for_inference

def test_dataset():
    """Test dataset loading"""
    print("🔍 Testing dataset loading...")
    
    config = Config()
    train_loader, test_loader = create_dataloaders(config)
    
    print(f"✅ Train dataset: {len(train_loader.dataset)} samples")
    print(f"✅ Test dataset: {len(test_loader.dataset)} samples")
    print(f"✅ Vocabulary size: {len(train_loader.dataset.vocab)}")
    
    # Show sample
    batch = next(iter(train_loader))
    print(f"✅ Batch shapes:")
    print(f"   - Images: {batch['image'].shape}")
    print(f"   - Input sequences: {batch['input_seq'].shape}")
    print(f"   - Target sequences: {batch['target_seq'].shape}")
    
    # Show sample data
    print(f"\n📝 Sample caption: {batch['caption'][0]}")
    print(f"📝 Sample image name: {batch['image_name'][0]}")
    
    return train_loader, test_loader

def test_model():
    """Test model creation and forward pass"""
    print("\n🧠 Testing model...")
    
    config = Config()
    model = MathExpressionVisionTransformer(config, vocab_size=116)
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    input_seq = torch.randint(0, 116, (batch_size, 50))
    
    with torch.no_grad():
        logits = model(images, input_seq)
        print(f"✅ Forward pass successful: {logits.shape}")
        
        # Test generation
        generated = model.generate(
            images, 
            sos_token_id=1, 
            eos_token_id=2, 
            max_length=50
        )
        print(f"✅ Generation successful: {generated.shape}")
    
    return model

def visualize_sample_data(train_loader, num_samples=4):
    """Visualize sample training data"""
    print(f"\n🖼️  Visualizing {num_samples} training samples...")
    
    batch = next(iter(train_loader))
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    
    for i in range(num_samples):
        # Get image and denormalize
        image = batch['image'][i].permute(1, 2, 0).numpy()
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        # Show image
        axes[0, i].imshow(image)
        axes[0, i].set_title(f"Image {i+1}")
        axes[0, i].axis('off')
        
        # Show caption
        caption = batch['caption'][i]
        axes[1, i].text(0.1, 0.5, caption, fontsize=8, wrap=True,
                       verticalalignment='center', transform=axes[1, i].transAxes)
        axes[1, i].set_title(f"LaTeX {i+1}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/sample_data.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Sample visualization saved to outputs/sample_data.png")

def test_inference_pipeline():
    """Test inference pipeline (without trained model)"""
    print("\n🔮 Testing inference pipeline...")
    
    # Create dummy checkpoint for testing
    config = Config()
    model = MathExpressionVisionTransformer(config, vocab_size=116)
    
    # Create dummy vocabulary
    vocab = {
        '<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3,
        'x': 4, 'y': 5, '=': 6, '+': 7, '-': 8, '1': 9, '2': 10
    }
    
    # Save dummy checkpoint
    checkpoint_path = Path('checkpoints/dummy.pth')
    checkpoint_path.parent.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'best_bleu': 0.5,
        'epoch': 1
    }, checkpoint_path)
    
    print("✅ Dummy checkpoint created")
    
    # Test loading
    try:
        inference = MathExpressionInference(checkpoint_path, config)
        print("✅ Inference engine loaded successfully")
        
        # Test with dummy image
        dummy_image_path = Path('outputs/dummy_image.png')
        dummy_image = np.ones((224, 224, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(dummy_image_path), dummy_image)
        
        latex, tokens, confidence = inference.predict(dummy_image_path)
        print(f"✅ Prediction successful:")
        print(f"   LaTeX: {latex}")
        print(f"   Tokens: {tokens}")
        print(f"   Confidence: {confidence.mean():.3f}")
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
    
    # Cleanup
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    if dummy_image_path.exists():
        dummy_image_path.unlink()

def show_model_architecture():
    """Show model architecture summary"""
    print("\n🏗️  Model Architecture:")
    print("=" * 60)
    
    config = Config()
    
    print(f"Vision Encoder:")
    print(f"  - Input Size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}x3")
    print(f"  - Patch Size: {config.PATCH_SIZE}x{config.PATCH_SIZE}")
    print(f"  - Num Patches: {(config.IMAGE_SIZE // config.PATCH_SIZE)**2}")
    print(f"  - Embed Dim: {config.EMBED_DIM}")
    print(f"  - Num Layers: {config.NUM_LAYERS}")
    print(f"  - Num Heads: {config.NUM_HEADS}")
    
    print(f"\nText Decoder:")
    print(f"  - Vocab Size: ~116 tokens")
    print(f"  - Max Seq Length: {config.MAX_SEQ_LENGTH}")
    print(f"  - Embed Dim: {config.EMBED_DIM}")
    print(f"  - Num Layers: {config.NUM_LAYERS // 2}")
    print(f"  - Num Heads: {config.NUM_HEADS}")
    
    print(f"\nTraining Config:")
    print(f"  - Batch Size: {config.BATCH_SIZE}")
    print(f"  - Learning Rate: {config.LEARNING_RATE}")
    print(f"  - Epochs: {config.NUM_EPOCHS}")
    print(f"  - Device: {config.DEVICE}")

def main():
    """Run complete demo"""
    print("🚀 Handwritten Mathematical Expression Recognition Demo")
    print("=" * 60)
    
    # Create output directory
    Path('outputs').mkdir(exist_ok=True)
    
    try:
        # Test 1: Dataset
        train_loader, test_loader = test_dataset()
        
        # Test 2: Model
        model = test_model()
        
        # Test 3: Architecture info
        show_model_architecture()
        
        # Test 4: Visualize data
        visualize_sample_data(train_loader)
        
        # Test 5: Inference pipeline
        test_inference_pipeline()
        
        print("\n✅ All tests passed! Hệ thống sẵn sàng để training.")
        print("\n📋 Các bước tiếp theo:")
        print("1. Chạy training: python train.py")
        print("2. Theo dõi training: tensorboard --logdir logs")
        print("3. Test inference: python inference.py --checkpoint checkpoints/best.pth --image data/off_image_test/[image_name].bmp")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


