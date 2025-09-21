"""
Vision Transformer model for Handwritten Mathematical Expression Recognition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange, repeat

from config import Config

class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings"""
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, n_patches**0.5, n_patches**0.5)
        x = rearrange(x, 'b e h w -> b (h w) e')  # (batch_size, n_patches, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x

class MLP(nn.Module):
    """Multi-layer perceptron"""
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class VisionTransformer(nn.Module):
    """Vision Transformer backbone"""
    def __init__(
        self, 
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches + 1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        return x

class TransformerDecoder(nn.Module):
    """Transformer decoder for sequence generation"""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_length: int = 256
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_length)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask for autoregressive generation"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    def forward(
        self, 
        tgt: torch.Tensor, 
        encoder_output: torch.Tensor, 
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        seq_len = tgt.size(1)
        
        # Token embedding + positional encoding
        x = self.token_embed(tgt) * math.sqrt(self.embed_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Create causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self.create_causal_mask(seq_len).to(tgt.device)
            
        # Apply decoder blocks
        for block in self.blocks:
            x = block(x, encoder_output, tgt_mask)
            
        x = self.norm(x)
        x = self.head(x)
        
        return x

class TransformerDecoderBlock(nn.Module):
    """Transformer decoder block with cross-attention"""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        
        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: torch.Tensor, 
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        x = x + self.self_attn(self.norm1(x), tgt_mask)
        
        # Cross-attention
        x = x + self.cross_attn(self.norm2(x), encoder_output)
        
        # MLP
        x = x + self.mlp(self.norm3(x))
        
        return x

class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention mechanism"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(embed_dim, embed_dim)
        self.kv = nn.Linear(embed_dim, embed_dim * 2)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        B, N_q, C = query.shape
        N_kv = key_value.shape[1]
        
        q = self.q(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(key_value).reshape(B, N_kv, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        
        return x

class MathExpressionVisionTransformer(nn.Module):
    """Complete model for handwritten mathematical expression recognition"""
    def __init__(self, config: Config, vocab_size: int):
        super().__init__()
        
        self.config = config
        self.vocab_size = vocab_size
        
        # Vision encoder
        self.vision_encoder = VisionTransformer(
            img_size=config.IMAGE_SIZE,
            patch_size=config.PATCH_SIZE,
            embed_dim=config.EMBED_DIM,
            num_layers=config.NUM_LAYERS,
            num_heads=config.NUM_HEADS,
            mlp_ratio=config.MLP_RATIO,
            dropout=config.DROPOUT
        )
        
        # Text decoder
        self.text_decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=config.EMBED_DIM,
            num_layers=config.NUM_LAYERS // 2,  # Use fewer layers for decoder
            num_heads=config.NUM_HEADS,
            mlp_ratio=config.MLP_RATIO,
            dropout=config.DROPOUT,
            max_seq_length=config.MAX_SEQ_LENGTH
        )
        
    def forward(
        self, 
        images: torch.Tensor, 
        input_seq: torch.Tensor, 
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Encode images
        encoder_output = self.vision_encoder(images)  # (B, n_patches + 1, embed_dim)
        
        # Decode sequences
        logits = self.text_decoder(input_seq, encoder_output, tgt_mask)
        
        return logits
    
    def generate(
        self, 
        images: torch.Tensor, 
        sos_token_id: int,
        eos_token_id: int,
        max_length: int = 256,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Generate sequences using greedy decoding"""
        batch_size = images.size(0)
        device = images.device
        
        # Encode images
        encoder_output = self.vision_encoder(images)
        
        # Initialize with SOS token
        generated = torch.full((batch_size, 1), sos_token_id, device=device, dtype=torch.long)
        
        for _ in range(max_length - 1):
            # Get logits for current sequence
            logits = self.text_decoder(generated, encoder_output)
            
            # Get next token (greedy)
            next_token_logits = logits[:, -1, :] / temperature
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences have EOS token
            if (next_token == eos_token_id).all():
                break
                
        return generated

if __name__ == "__main__":
    # Test model
    config = Config()
    model = MathExpressionVisionTransformer(config, vocab_size=116)  # 112 + 4 special tokens
    
    # Test forward pass
    images = torch.randn(2, 3, 224, 224)
    input_seq = torch.randint(0, 116, (2, 50))
    
    logits = model(images, input_seq)
    print(f"Output shape: {logits.shape}")  # Should be (2, 50, 116)
    
    # Test generation
    generated = model.generate(images, sos_token_id=1, eos_token_id=2)
    print(f"Generated shape: {generated.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
