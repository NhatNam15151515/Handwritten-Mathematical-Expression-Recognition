"""
Dataset class for Handwritten Mathematical Expression Recognition
"""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional
import re

from config import Config

class MathExpressionDataset(Dataset):
    def __init__(
        self, 
        image_dir: Path, 
        caption_file: Path, 
        dictionary_file: Path,
        transform=None,
        is_train: bool = True
    ):
        self.image_dir = Path(image_dir)
        self.is_train = is_train
        
        # Load vocabulary
        self.vocab = self._load_vocabulary(dictionary_file)
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        
        # Load captions
        self.data = self._load_captions(caption_file)
        
        # Setup transforms
        self.transform = transform if transform is not None else self._get_default_transform()
        
    def _load_vocabulary(self, dictionary_file: Path) -> Dict[str, int]:
        """Load vocabulary from dictionary file"""
        vocab = {
            Config.PAD_TOKEN: 0,
            Config.SOS_TOKEN: 1,
            Config.EOS_TOKEN: 2,
            Config.UNK_TOKEN: 3
        }
        
        with open(dictionary_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    token, idx = line.split('\t')
                    vocab[token] = int(idx) + 4  # Offset by special tokens
                    
        return vocab
    
    def _load_captions(self, caption_file: Path) -> List[Tuple[str, str]]:
        """Load image-caption pairs"""
        data = []
        with open(caption_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        image_name = parts[0]
                        caption = '\t'.join(parts[1:])  # Handle captions with tabs
                        data.append((image_name, caption))
        return data
    
    def _get_default_transform(self):
        """Get default image transforms"""
        if self.is_train:
            return A.Compose([
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.Rotate(limit=Config.ROTATION_RANGE, p=Config.AUGMENT_PROB),
                A.RandomBrightnessContrast(
                    brightness_limit=Config.BRIGHTNESS_RANGE,
                    contrast_limit=Config.CONTRAST_RANGE,
                    p=Config.AUGMENT_PROB
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def _tokenize_caption(self, caption: str) -> List[int]:
        """Tokenize LaTeX caption to token indices"""
        # Split by spaces and handle special LaTeX tokens
        tokens = caption.split()
        
        # Convert to indices
        indices = [self.vocab[Config.SOS_TOKEN]]
        for token in tokens:
            if token in self.vocab:
                indices.append(self.vocab[token])
            else:
                indices.append(self.vocab[Config.UNK_TOKEN])
        indices.append(self.vocab[Config.EOS_TOKEN])
        
        return indices
    
    def _pad_sequence(self, sequence: List[int], max_length: int) -> List[int]:
        """Pad sequence to max_length"""
        if len(sequence) >= max_length:
            return sequence[:max_length]
        else:
            return sequence + [self.vocab[Config.PAD_TOKEN]] * (max_length - len(sequence))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_name, caption = self.data[idx]
        
        # Load image
        image_path = self.image_dir / f"{image_name}_0.bmp"

        if not image_path.exists():
            # Fallback to original name if not found
            image_path = self.image_dir / image_name
            
        image = cv2.imread(str(image_path))
        if image is None:
            # Create dummy image if file not found
            image = np.ones((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8) * 255
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Tokenize caption
        token_indices = self._tokenize_caption(caption)
        
        # Create input (without EOS) and target (without SOS) sequences
        input_seq = self._pad_sequence(token_indices[:-1], Config.MAX_SEQ_LENGTH)
        target_seq = self._pad_sequence(token_indices[1:], Config.MAX_SEQ_LENGTH)
        
        return {
            'image': image,
            'input_seq': torch.tensor(input_seq, dtype=torch.long),
            'target_seq': torch.tensor(target_seq, dtype=torch.long),
            'caption': caption,
            'image_name': image_name
        }

def create_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders"""
    
    # Create datasets
    train_dataset = MathExpressionDataset(
        image_dir=config.TRAIN_IMAGE_DIR,
        caption_file=config.TRAIN_CAPTION_FILE,
        dictionary_file=config.DICTIONARY_FILE,
        is_train=True
    )
    
    test_dataset = MathExpressionDataset(
        image_dir=config.TEST_IMAGE_DIR,
        caption_file=config.TEST_CAPTION_FILE,
        dictionary_file=config.DICTIONARY_FILE,
        is_train=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader

if __name__ == "__main__":
    # Test dataset
    config = Config()
    train_loader, test_loader = create_dataloaders(config)
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    print(f"Vocabulary size: {len(train_loader.dataset.vocab)}")
    
    # Test a batch
    batch = next(iter(train_loader))
    print(f"Image shape: {batch['image'].shape}")
    print(f"Input sequence shape: {batch['input_seq'].shape}")
    print(f"Target sequence shape: {batch['target_seq'].shape}")
    print(f"Sample caption: {batch['caption'][0]}")


