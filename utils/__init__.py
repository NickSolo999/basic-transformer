# utils/__init__.py
# Makes utils/ a Python package so notebooks can do:
#   import sys; sys.path.append('..')
#   from utils.model import GPTModel
#   from utils.visualisation import plot_attention_heatmap

from .model import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
    GPTModel,
    DEFAULT_CONFIG,
)

from .visualisation import (
    plot_attention_heatmap,
    plot_multi_head_attention,
    plot_positional_encoding,
    plot_loss_curves,
    plot_embedding_pca,
    plot_attention_rollout,
)

__all__ = [
    # Model components
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "FeedForward",
    "TransformerBlock",
    "GPTModel",
    "DEFAULT_CONFIG",
    # Visualisation helpers
    "plot_attention_heatmap",
    "plot_multi_head_attention",
    "plot_positional_encoding",
    "plot_loss_curves",
    "plot_embedding_pca",
    "plot_attention_rollout",
]
