"""
utils/model.py
==============
Full GPT-style transformer model, built entirely from scratch in PyTorch.

Architecture overview (decoder-only transformer):
  Input tokens
      │
  Token Embedding  +  Positional Encoding
      │
  ┌───▼─────────────────┐
  │   Transformer Block  │  × n_layers
  │  ┌─────────────────┐ │
  │  │ CausalMHA       │ │  ← masked multi-head self-attention
  │  └────────┬────────┘ │
  │     Add & LayerNorm  │  ← residual connection
  │  ┌─────────────────┐ │
  │  │ FeedForward     │ │  ← position-wise FFN (expand → GELU → project)
  │  └────────┬────────┘ │
  │     Add & LayerNorm  │  ← residual connection
  └───────────┬──────────┘
  Final LayerNorm
      │
  Linear projection  → logits over vocabulary

All modules save their attention weights as attributes so notebook 06
can extract them for visualisation without re-running the forward pass.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ---------------------------------------------------------------------------
# Default hyperparameter config (kept small for CPU training)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "vocab_size": 50257,      # GPT-2 tokenizer vocabulary size
    "d_model":    64,          # Embedding / hidden dimension
    "n_heads":    4,           # Number of attention heads
    "n_layers":   2,           # Number of transformer blocks
    "d_ff":       256,         # Feed-forward inner dimension (4 × d_model)
    "max_seq_len": 128,        # Maximum context length
    "dropout":    0.1,         # Dropout probability
}


# ---------------------------------------------------------------------------
# 1. Scaled Dot-Product Attention
# ---------------------------------------------------------------------------
class ScaledDotProductAttention(nn.Module):
    """
    The core attention operation:

        Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) ) · V

    Q : (batch, heads, seq_q, d_k)
    K : (batch, heads, seq_k, d_k)
    V : (batch, heads, seq_v, d_v)   where seq_k == seq_v always

    Returns:
        output        : (batch, heads, seq_q, d_v)
        attn_weights  : (batch, heads, seq_q, seq_k)  — stored for viz
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Store the last computed weights so notebooks can inspect them
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # d_k is the per-head key/query dimension
        d_k = q.size(-1)  # (batch, heads, seq, d_k)

        # Step 1: Compute raw attention scores  QK^T
        # q @ k.transpose(-2, -1) → (batch, heads, seq_q, seq_k)
        scores = torch.matmul(q, k.transpose(-2, -1))

        # Step 2: Scale to prevent vanishing gradients in softmax
        # dividing by sqrt(d_k) keeps variance ≈ 1 regardless of d_k
        scores = scores / math.sqrt(d_k)

        # Step 3: Apply mask (for causal / padding masking)
        # mask is True where positions should be IGNORED
        if mask is not None:
            # Fill masked positions with -inf → softmax gives ≈ 0
            scores = scores.masked_fill(mask, float("-inf"))

        # Step 4: Softmax over key dimension → attention weights
        # (batch, heads, seq_q, seq_k)  each row sums to 1
        attn_weights = F.softmax(scores, dim=-1)

        # Handle the all-masked case (NaN from -inf softmax)
        # This can happen at the very first position where all keys are masked
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Step 5: Apply dropout to attention weights (regularisation)
        attn_weights_dropped = self.dropout(attn_weights)

        # Step 6: Weighted sum of value vectors
        # (batch, heads, seq_q, seq_k) @ (batch, heads, seq_v, d_v)
        # → (batch, heads, seq_q, d_v)
        output = torch.matmul(attn_weights_dropped, v)

        # Cache the undropped weights for visualisation
        self.last_attn_weights = attn_weights.detach()

        return output, attn_weights


# ---------------------------------------------------------------------------
# 2. Multi-Head Attention
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention splits the d_model dimension across n_heads
    heads, each learning to attend to different aspects of the input.

        MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W_O

    where each head_i = Attention(Q W_Qi, K W_Ki, V W_Vi)

    This implementation uses a single (d_model × d_model) weight matrix
    for all heads combined, then reshapes — equivalent to n_heads separate
    (d_model × d_k) matrices but more efficient.

    Input/output shape: (batch, seq, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads   # per-head key/query dimension
        self.d_v = d_model // n_heads   # per-head value dimension

        # Projection matrices: Q, K, V each project d_model → d_model
        # (all heads concatenated, so shape is d_model × d_model)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection: recombines all head outputs
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Core attention function (shared across all heads)
        self.attention = ScaledDotProductAttention(dropout=dropout)

        # Dropout on the final output
        self.dropout = nn.Dropout(dropout)

        # Store per-head attention weights for visualisation
        # Shape after forward: (batch, n_heads, seq, seq)
        self.last_attn_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape from (batch, seq, d_model) → (batch, n_heads, seq, d_k)

        Strategy: split the last dimension into (n_heads, d_k), then
        transpose so heads dimension is dim-1 (for batched matmul).
        """
        batch, seq, d_model = x.shape
        # (batch, seq, d_model) → (batch, seq, n_heads, d_k)
        x = x.view(batch, seq, self.n_heads, self.d_k)
        # (batch, seq, n_heads, d_k) → (batch, n_heads, seq, d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of _split_heads:
        (batch, n_heads, seq, d_k) → (batch, seq, d_model)
        """
        batch, n_heads, seq, d_k = x.shape
        # (batch, n_heads, seq, d_k) → (batch, seq, n_heads, d_k)
        x = x.transpose(1, 2).contiguous()
        # (batch, seq, n_heads, d_k) → (batch, seq, d_model)
        return x.view(batch, seq, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Self-attention: Q, K, V all come from the same input x.

        Args:
            x    : (batch, seq, d_model)
            mask : (1, 1, seq, seq) causal or padding mask

        Returns:
            (batch, seq, d_model)
        """
        # Step 1: Project input to Q, K, V
        # Each: (batch, seq, d_model)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Step 2: Split into multiple heads
        # Each: (batch, n_heads, seq, d_k)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Step 3: Scaled dot-product attention (all heads in parallel)
        # out: (batch, n_heads, seq, d_k)
        # attn_w: (batch, n_heads, seq, seq)
        out, attn_weights = self.attention(q, k, v, mask=mask)

        # Cache attention weights for visualisation
        self.last_attn_weights = attn_weights.detach()

        # Step 4: Merge heads back
        # (batch, n_heads, seq, d_k) → (batch, seq, d_model)
        out = self._merge_heads(out)

        # Step 5: Final output projection
        out = self.W_o(out)
        out = self.dropout(out)

        return out


# ---------------------------------------------------------------------------
# 3. Position-wise Feed-Forward Network
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    """
    Two-layer MLP applied independently to each position:

        FFN(x) = GELU( x W_1 + b_1 ) W_2 + b_2

    The inner dimension d_ff is typically 4 × d_model.
    GELU (Gaussian Error Linear Unit) is used instead of ReLU — it is
    smoother and performs slightly better in practice for transformers.

    Input/output: (batch, seq, d_model)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()

        # Expand: d_model → d_ff  (linear projection upward)
        self.linear1 = nn.Linear(d_model, d_ff)

        # Contract: d_ff → d_model  (linear projection back down)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        x = self.linear1(x)         # → (batch, seq, d_ff)
        x = F.gelu(x)               # smooth nonlinearity
        x = self.dropout(x)
        x = self.linear2(x)         # → (batch, seq, d_model)
        return x


# ---------------------------------------------------------------------------
# 4. Transformer Block (Decoder-style)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """
    A single transformer decoder block (GPT-style, no cross-attention):

        x  →  LayerNorm  →  CausalMHA  →  + x   (residual)
           →  LayerNorm  →  FFN        →  + x   (residual)

    Note: Pre-norm ordering (LN before sub-layer) as used in GPT-2.
    This is more stable during training than the original post-norm.

    Input/output: (batch, seq, d_model)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Layer normalisation before each sub-layer (pre-norm)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Multi-head self-attention
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)

        # Position-wise feed-forward
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)

        # Dropout on residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pre-norm transformer block:

            h   = x + Dropout( MHA( LN(x) ) )
            out = h + Dropout( FFN( LN(h) ) )
        """
        # --- Sub-layer 1: Multi-head self-attention with residual ---
        # Apply layer norm first (pre-norm)
        normed = self.ln1(x)
        # Self-attention
        attn_out = self.attn(normed, mask=mask)
        # Residual connection: adds back the original input
        x = x + self.dropout(attn_out)

        # --- Sub-layer 2: Feed-forward with residual ---
        normed = self.ln2(x)
        ff_out = self.ff(normed)
        x = x + self.dropout(ff_out)

        return x


# ---------------------------------------------------------------------------
# 5. Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    Adds fixed sinusoidal position information to token embeddings.

    For position pos and dimension i:
        PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
        PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )

    This encoding is:
    - Deterministic (no learned parameters)
    - Bounded (-1 to 1) — doesn't blow up with long sequences
    - Allows the model to infer relative positions via linear combinations
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build the (max_seq_len, d_model) encoding table once at init
        pe = torch.zeros(max_seq_len, d_model)  # (max_seq_len, d_model)

        # Position indices: (max_seq_len, 1)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()

        # Frequency divisors: 10000^(2i / d_model) for each even dimension i
        # Computed in log space for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )  # shape: (d_model/2,)

        # Even indices → sin,  Odd indices → cos
        pe[:, 0::2] = torch.sin(position * div_term)   # (max_seq_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)   # (max_seq_len, d_model/2)

        # Add batch dimension and register as buffer (saved in state_dict,
        # but NOT a trainable parameter)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq, d_model)
        Adds the first seq positions of the precomputed PE.
        """
        seq_len = x.size(1)
        # self.pe[:, :seq_len, :] broadcasts over the batch dimension
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 6. Full GPT-style Model
# ---------------------------------------------------------------------------
class GPTModel(nn.Module):
    """
    A minimal GPT-style language model.

    Architecture:
        Token Embedding  →  Positional Encoding
            →  N × TransformerBlock (causal)
            →  LayerNorm
            →  Linear head (d_model → vocab_size)

    Usage:
        model = GPTModel(config)
        logits = model(token_ids)          # (batch, seq, vocab_size)

    For generation, use model.generate(...)
    For visualisation, use model.get_attention_weights() after a forward pass.
    """

    def __init__(self, config: dict = None):
        super().__init__()
        cfg = config or DEFAULT_CONFIG

        self.vocab_size  = cfg["vocab_size"]
        self.d_model     = cfg["d_model"]
        self.n_heads     = cfg["n_heads"]
        self.n_layers    = cfg["n_layers"]
        self.d_ff        = cfg["d_ff"]
        self.max_seq_len = cfg["max_seq_len"]
        self.dropout_p   = cfg["dropout"]

        # Token embedding: maps integer token IDs → d_model vectors
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)

        # Positional encoding: adds position information
        self.pos_encoding = PositionalEncoding(
            self.d_model, self.max_seq_len, dropout=self.dropout_p
        )

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model  = self.d_model,
                n_heads  = self.n_heads,
                d_ff     = self.d_ff,
                dropout  = self.dropout_p,
            )
            for _ in range(self.n_layers)
        ])

        # Final layer normalisation (before the language model head)
        self.ln_final = nn.LayerNorm(self.d_model)

        # Language model head: projects d_model → vocabulary logits
        # Weight tying: share weights with token_embedding (GPT-2 trick)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # Causal mask cache (avoids recomputing on every forward pass)
        self._causal_mask_cache: dict = {}

        # Initialise weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialise weights following GPT-2 conventions:
        - Embeddings: N(0, 0.02)
        - Linear layers: N(0, 0.02)
        - Residual projections scaled by 1/sqrt(2 * n_layers) to prevent
          gradient blow-up with depth.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Build (or retrieve from cache) an upper-triangular causal mask.

        The mask is True where attention should be BLOCKED (future positions).
        Shape: (1, 1, seq_len, seq_len) — broadcasts over batch & head dims.

        Example for seq_len=4:
            [[F, T, T, T],
             [F, F, T, T],
             [F, F, F, T],
             [F, F, F, F]]
        Position 0 can only attend to position 0.
        Position 3 can attend to positions 0-3.
        """
        key = (seq_len, device)
        if key not in self._causal_mask_cache:
            # torch.triu with diagonal=1 gives upper triangle above main diagonal
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                diagonal=1
            )
            # Expand to (1, 1, seq_len, seq_len) for broadcasting
            mask = mask.unsqueeze(0).unsqueeze(0)
            self._causal_mask_cache[key] = mask
        return self._causal_mask_cache[key]

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the full model.

        Args:
            token_ids : (batch, seq)  — integer token indices

        Returns:
            logits    : (batch, seq, vocab_size) — unnormalised log-probs
        """
        batch, seq_len = token_ids.shape
        assert seq_len <= self.max_seq_len, (
            f"Sequence length {seq_len} exceeds model max {self.max_seq_len}"
        )

        # Step 1: Token embedding  (batch, seq) → (batch, seq, d_model)
        x = self.token_embedding(token_ids)

        # Scale embeddings (GPT-2 style: multiply by sqrt(d_model))
        x = x * math.sqrt(self.d_model)

        # Step 2: Add positional encoding  (batch, seq, d_model)
        x = self.pos_encoding(x)

        # Step 3: Build causal mask for this sequence length
        causal_mask = self._get_causal_mask(seq_len, token_ids.device)

        # Step 4: Pass through all transformer blocks
        for block in self.blocks:
            x = block(x, mask=causal_mask)

        # Step 5: Final layer normalisation  (batch, seq, d_model)
        x = self.ln_final(x)

        # Step 6: Project to vocabulary logits  (batch, seq, vocab_size)
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        greedy: bool = False,
    ) -> torch.Tensor:
        """
        Autoregressively generate new tokens given a prompt.

        Args:
            prompt_ids   : (1, prompt_len)  — starting token IDs
            max_new_tokens: how many tokens to generate
            temperature  : >1 makes distribution flatter (more random),
                           <1 makes it peakier (more deterministic)
            top_k        : if >0, only sample from the top-k logits
            greedy       : if True, always pick argmax (ignores temperature)

        Returns:
            (1, prompt_len + max_new_tokens) — full sequence with generated tokens
        """
        self.eval()
        ids = prompt_ids.clone()  # (1, current_len)

        for _ in range(max_new_tokens):
            # Truncate context to max_seq_len if it's getting too long
            ids_ctx = ids[:, -self.max_seq_len:]

            # Forward pass → (1, current_len, vocab_size)
            logits = self(ids_ctx)

            # We only care about the LAST position's logits
            # (next token prediction)
            next_logits = logits[:, -1, :]   # (1, vocab_size)

            if greedy:
                # Deterministic: pick the highest-probability token
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                # Stochastic sampling
                # Apply temperature scaling
                next_logits = next_logits / max(temperature, 1e-8)

                # Apply top-k filtering if requested
                if top_k > 0:
                    # Keep only the top-k logits, set rest to -inf
                    values, _ = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)))
                    threshold = values[:, -1].unsqueeze(-1)
                    next_logits = next_logits.masked_fill(
                        next_logits < threshold, float("-inf")
                    )

                # Convert to probabilities and sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Append sampled token to sequence
            ids = torch.cat([ids, next_token], dim=1)

        return ids

    def get_attention_weights(self) -> list[torch.Tensor]:
        """
        Return the cached attention weights from the most recent forward pass.

        Returns:
            List of length n_layers, each tensor:
            (batch, n_heads, seq, seq)
        """
        weights = []
        for block in self.blocks:
            w = block.attn.last_attn_weights
            if w is not None:
                weights.append(w)
        return weights

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
