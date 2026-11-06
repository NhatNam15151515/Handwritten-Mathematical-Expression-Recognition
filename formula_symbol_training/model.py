"""ViT encoder plus autoregressive decoder for flattened visible symbols."""
from model import MathExpressionVisionTransformer


def create_model(config, vocab_size: int):
    return MathExpressionVisionTransformer(config, vocab_size)
