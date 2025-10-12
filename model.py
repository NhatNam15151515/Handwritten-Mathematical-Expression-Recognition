"""
Vision Transformer model optimized for GTX 1660 Super and small dataset
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from typing import Optional, Tuple
from einops import rearrange, repeat
import random

from config import Config

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings"""
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.projection(x)  # (B, E, H', W')
        h, w = x.shape[2], x.shape[3]
        x = rearrange(x, 'b e h w -> b (h w) e')
        x = self.norm(x)
        return x, h, w

class MultiHeadAttention(nn.Module):
    """
    Lớp này DÀNH RIÊNG cho Self-Attention.
    Nó chỉ nhận MỘT đầu vào 'x' và tự tạo ra Q, K, V từ đó.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, attn_dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask_fill = torch.finfo(attn.dtype).min
            attn = attn.masked_fill(mask == 0, mask_fill)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    """MLP với GELU activation và dropout"""
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block với DropPath và gradient checkpointing"""
    def __init__(
        self,
        embed_dim: int,
        encoder_embed_dim: int,  # <--- THÊM THAM SỐ NÀY
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        drop_path: float = 0.
    ):
        super().__init__()
        self.mlp_ratio = mlp_ratio
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout, attn_dropout)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x), mask))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer với regularization techniques"""
    def __init__(
        self,
        img_size: int = 192,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 3.0,
        dropout: float = 0.15,
        attn_dropout: float = 0.1,
        drop_path_rate: float = 0.2
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        # base positional embedding grid at (img_size/patch_size)
        self.base_grid_size = (img_size // patch_size, img_size // patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, (self.base_grid_size[0] * self.base_grid_size[1]) + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Stochastic depth - tăng dần drop rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, mlp_ratio,
                dropout, attn_dropout, drop_path=dpr[i]
            )
            for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding
        x, h_p, w_p = self.patch_embed(x)


        # Add positional embedding (interpolate 2D grid to current h_p x w_p)
        pos_tokens = self.pos_embed[:, 1:, :]  # (1, H0*W0, D)
        H0, W0 = self.base_grid_size
        pos_tokens = pos_tokens.reshape(1, H0, W0, -1).permute(0, 3, 1, 2)  # (1, D, H0, W0)
        pos_tokens = F.interpolate(pos_tokens, size=(h_p, w_p), mode='bilinear', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, h_p * w_p, -1)  # (1, H'*W', D)
        x = x + pos_tokens
        x = self.pos_drop(x)

        # Apply transformer blocks với gradient checkpointing
        for block in self.blocks:
            if self.training and hasattr(torch.utils.checkpoint, 'checkpoint'):
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)

        x = self.norm(x)

        return x

class TransformerDecoder(nn.Module):
    """Transformer decoder với label smoothing và regularization"""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 384,
        encoder_embed_dim: int = 768,
        num_layers: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 3.0,
        dropout: float = 0.15,
        attn_dropout: float = 0.1,
        max_seq_length: int = 200
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(
                embed_dim=embed_dim,
                encoder_embed_dim=encoder_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def create_causal_mask(self, seq_len: int, device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, seq_len = tgt.shape

        # Token embedding với scaling
        x = self.token_embed(tgt) * math.sqrt(self.embed_dim)
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.dropout(x)

        # Create causal mask
        if tgt_mask is None:
            tgt_mask = self.create_causal_mask(seq_len, tgt.device)

        # Apply decoder blocks
        for block in self.blocks:
            x = block(x, encoder_output, tgt_mask)

        x = self.norm(x)
        x = self.head(x)

        return x

class TransformerDecoderBlock(nn.Module):
    """Decoder block với cross-attention"""
    def __init__(
        self,
        embed_dim: int,
        encoder_embed_dim: int, # <--- THÊM THAM SỐ NÀY
        num_heads: int,
        mlp_ratio: float = 3.0,
        dropout: float = 0.15,
        attn_dropout: float = 0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout, attn_dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        # Truyền cả hai chiều dữ liệu vào cross_attn
        self.cross_attn = MultiHeadCrossAttention(
            decoder_dim=embed_dim,          # <--- CHIỀU CỦA DECODER
            encoder_dim=encoder_embed_dim,  # <--- CHIỀU CỦA ENCODER
            num_heads=num_heads,
            dropout=dropout,
            attn_dropout=attn_dropout
        )
        self.drop2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention với residual connection
        x = x + self.drop1(self.self_attn(self.norm1(x), tgt_mask))

        # Cross-attention
        x = x + self.drop2(self.cross_attn(self.norm2(x), encoder_output))

        # MLP
        x = x + self.drop3(self.mlp(self.norm3(x)))

        return x


class MultiHeadCrossAttention(nn.Module):
    """
    Lớp này DÀNH RIÊNG cho Cross-Attention.
    Nó nhận HAI đầu vào: 'query' (từ decoder) và 'key_value' (từ encoder).
    """

    def __init__(self, decoder_dim: int, encoder_dim: int, num_heads: int, dropout: float = 0.15,
                 attn_dropout: float = 0.1):
        super().__init__()
        assert decoder_dim % num_heads == 0

        self.embed_dim = decoder_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(decoder_dim, self.embed_dim, bias=False)
        self.kv = nn.Linear(encoder_dim, self.embed_dim * 2, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        B, N_q, C_q = query.shape
        B, N_kv, C_kv = key_value.shape

        q = self.q(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        kv = self.kv(key_value).reshape(B, N_kv, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, self.embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MathExpressionVisionTransformer(nn.Module):
    """Complete model với regularization"""
    def __init__(self, config: Config, vocab_size: int):
        super().__init__()

        self.config = config
        self.vocab_size = vocab_size

        # Vision encoder
        self.vision_encoder = timm.create_model(
            # 'vit_base_patch16_224.augreg_in21k'
            # 'vit_small_patch16_224.augreg_in21k'
            # 'vit_tiny_patch16_224.augreg_in21k'
            'vit_small_patch16_224.augreg_in21k',
            pretrained=True,
            num_classes=0,  # Quan trọng: Bỏ lớp phân loại cuối cùng
            img_size=config.IMAGE_SIZE,
        )
        pretrained_embed_dim = self.vision_encoder.embed_dim
        self.encoder_projection = nn.Linear(pretrained_embed_dim, config.EMBED_DIM)
        # Text decoder - ít layers hơn encoder
        self.text_decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=config.EMBED_DIM,
            encoder_embed_dim=config.EMBED_DIM,  # <--- QUAN TRỌNG: Truyền vào chiều của encoder SAU KHI chiếu
            num_layers=max(config.NUM_LAYERS // 2, 3),
            num_heads=config.NUM_HEADS,
            mlp_ratio=config.MLP_RATIO,
            dropout=config.DROPOUT,
            attn_dropout=config.ATTENTION_DROPOUT,
            max_seq_length=config.MAX_SEQ_LENGTH
        )

    def forward(
        self,
        images: torch.Tensor,
        input_seq: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        mixup_lambda: Optional[float] = None,
        mixup_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Apply mixup to images if provided
        if mixup_lambda is not None and mixup_index is not None:
            images = mixup_lambda * images + (1 - mixup_lambda) * images[mixup_index]

        # Encode images
        encoder_output = self.vision_encoder.forward_features(images)
        encoder_output = self.encoder_projection(encoder_output)

        # Decode sequences
        logits = self.text_decoder(input_seq, encoder_output, tgt_mask)

        return logits

    def generate(
        self,
        images: torch.Tensor,
        sos_token_id: int,
        eos_token_id: int,
        max_length: int = 200,
        temperature: float = 0.8,
        top_k: int = 50
    ) -> torch.Tensor:
        """Generate với top-k sampling"""
        batch_size = images.size(0)
        device = images.device

        # Encode images
        encoder_output = self.vision_encoder.forward_features(images)
        encoder_output = self.encoder_projection(encoder_output)

        # Initialize với SOS token
        generated = torch.full((batch_size, 1), sos_token_id, device=device, dtype=torch.long)

        for _ in range(max_length - 1):
            # Get logits
            logits = self.text_decoder(generated, encoder_output)
            next_token_logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        return generated

if __name__ == "__main__":
    # Test model
    config = Config()
    config.print_config()

    model = MathExpressionVisionTransformer(config, vocab_size=116)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (float32)")

    # Test forward pass
    images = torch.randn(4, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    input_seq = torch.randint(0, 116, (4, 50))

    logits = model(images, input_seq)
    print(f"\nOutput shape: {logits.shape}")

    # Memory usage estimation
    print(f"\nEstimated memory per batch:")
    print(f"Batch size {config.BATCH_SIZE}: ~{(total_params * 4 + config.BATCH_SIZE * 3 * config.IMAGE_SIZE * config.IMAGE_SIZE * 4) / 1024**3:.2f} GB")