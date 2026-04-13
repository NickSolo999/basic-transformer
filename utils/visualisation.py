"""
utils/visualisation.py
======================
Shared plotting utilities for the Educational Transformer project.

Every function in this module is designed to be called directly from
the Jupyter notebooks, keeping the notebook cells clean while providing
rich, well-labelled figures.

Functions
---------
plot_attention_heatmap     : Single attention matrix as a seaborn heatmap
plot_multi_head_attention  : Grid of per-head attention heatmaps
plot_positional_encoding   : 2-D heatmap of the positional encoding table
plot_loss_curves           : Training / validation loss over steps
plot_embedding_pca         : PCA scatter of token embedding vectors
plot_attention_rollout     : Accumulated attention across transformer layers
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Colour palette – consistent across all notebooks
# ---------------------------------------------------------------------------
CMAP_ATTENTION = "Blues"       # Light→dark blue for attention weights
CMAP_PE        = "RdYlBu_r"   # Diverging for positional encoding
CMAP_ROLLOUT   = "viridis"    # Perceptually uniform for rollout


# ---------------------------------------------------------------------------
# 1.  plot_attention_heatmap
# ---------------------------------------------------------------------------

def plot_attention_heatmap(
    attn_weights,
    tokens,
    title="Attention Weights",
    ax=None,
    annot=True,
    fmt=".2f",
    cbar=True,
    figsize=(7, 6),
):
    """
    Draw a single attention matrix as a colour-coded heatmap.

    Parameters
    ----------
    attn_weights : array-like, shape (seq_len, seq_len)
        Attention weight matrix.  Row i contains the distribution over
        source positions used when producing output token i.
        Values should sum to 1.0 along axis=-1.
    tokens : list[str]
        Human-readable token strings.  Must have length == seq_len.
    title : str
        Plot title.
    ax : matplotlib.axes.Axes or None
        If provided, draw into this axes; otherwise create a new figure.
    annot : bool
        Whether to print numeric values inside each cell.
    fmt : str
        Format string for the cell annotations (default: two decimals).
    cbar : bool
        Whether to show the colour-bar.
    figsize : tuple
        Figure size in inches (used only when ax is None).

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    # ---- Convert to numpy (handles torch tensors gracefully) ----
    if hasattr(attn_weights, "detach"):
        attn_weights = attn_weights.detach().cpu().numpy()
    attn_weights = np.asarray(attn_weights, dtype=float)

    # ---- If a batch / head dimension slipped through, squeeze it ----
    while attn_weights.ndim > 2:
        attn_weights = attn_weights[0]

    seq_len = attn_weights.shape[0]

    # ---- Truncate / pad token list to match matrix size ----
    if len(tokens) > seq_len:
        tokens = tokens[:seq_len]
    elif len(tokens) < seq_len:
        tokens = list(tokens) + [f"[{i}]" for i in range(len(tokens), seq_len)]

    # ---- Create figure if no axes supplied ----
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)

    # ---- Draw heatmap ----
    sns.heatmap(
        attn_weights,
        ax=ax,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap=CMAP_ATTENTION,
        vmin=0.0,
        vmax=1.0,
        annot=annot,
        fmt=fmt,
        linewidths=0.3,
        linecolor="white",
        cbar=cbar,
        square=True,
    )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Key (source token)", fontsize=10)
    ax.set_ylabel("Query (output token)", fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

    if standalone:
        plt.tight_layout()
        plt.show()

    return ax


# ---------------------------------------------------------------------------
# 2.  plot_multi_head_attention
# ---------------------------------------------------------------------------

def plot_multi_head_attention(
    attn_weights_per_head,
    tokens,
    title="Multi-Head Attention",
    figsize_per_head=(4, 4),
    annot=False,
    max_cols=4,
):
    """
    Draw a grid of per-head attention heatmaps on a single figure.

    Parameters
    ----------
    attn_weights_per_head : array-like, shape (n_heads, seq_len, seq_len)
        Stacked attention matrices, one per head.
    tokens : list[str]
        Token strings (length == seq_len).
    title : str
        Super-title placed above the grid.
    figsize_per_head : tuple
        Width × height (in inches) allocated to each individual head panel.
    annot : bool
        Whether to annotate cells with numeric values (recommend False for
        long sequences to keep the plot readable).
    max_cols : int
        Maximum number of columns in the grid.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # ---- Convert to numpy ----
    if hasattr(attn_weights_per_head, "detach"):
        attn_weights_per_head = attn_weights_per_head.detach().cpu().numpy()
    attn_weights_per_head = np.asarray(attn_weights_per_head, dtype=float)

    # ---- Handle extra leading dimensions (batch, layer, …) ----
    while attn_weights_per_head.ndim > 3:
        attn_weights_per_head = attn_weights_per_head[0]

    n_heads = attn_weights_per_head.shape[0]

    # ---- Compute grid layout ----
    n_cols = min(n_heads, max_cols)
    n_rows = (n_heads + n_cols - 1) // n_cols   # ceiling division

    fig_w = figsize_per_head[0] * n_cols
    fig_h = figsize_per_head[1] * n_rows + 0.6   # extra for suptitle

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))

    # ---- Normalise axes to always be a flat list ----
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = list(axes.flat)
    else:
        axes = [ax for row in axes for ax in row]

    # ---- Draw each head ----
    for h in range(n_heads):
        plot_attention_heatmap(
            attn_weights_per_head[h],   # shape: (seq_len, seq_len)
            tokens=tokens,
            title=f"Head {h}",
            ax=axes[h],
            annot=annot,
            cbar=False,
        )

    # ---- Hide any unused subplots ----
    for idx in range(n_heads, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()

    return fig


# ---------------------------------------------------------------------------
# 3.  plot_positional_encoding
# ---------------------------------------------------------------------------

def plot_positional_encoding(
    pe,
    title="Sinusoidal Positional Encoding",
    max_positions=64,
    max_dims=64,
    figsize=(12, 5),
):
    """
    Visualise the positional encoding matrix as a 2-D heatmap.

    Each row represents one position (token index 0, 1, 2, …).
    Each column represents one embedding dimension.
    The alternating sin/cos pattern is clearly visible as vertical stripes
    of increasing frequency from left (low dim) to right (high dim).

    Parameters
    ----------
    pe : array-like, shape (max_seq_len, d_model)
        The positional encoding matrix.
    title : str
        Plot title.
    max_positions : int
        How many position rows to show (to keep the plot readable).
    max_dims : int
        How many embedding dimensions (columns) to show.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if hasattr(pe, "detach"):
        pe = pe.detach().cpu().numpy()
    pe = np.asarray(pe, dtype=float)

    # Crop to requested window
    pe_crop = pe[:max_positions, :max_dims]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        pe_crop,
        aspect="auto",
        cmap=CMAP_PE,
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Encoding value", fontsize=10)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Embedding dimension", fontsize=11)
    ax.set_ylabel("Position (token index)", fontsize=11)

    # Tick every 8 positions and every 8 dimensions
    ax.xaxis.set_major_locator(ticker.MultipleLocator(8))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(8))

    plt.tight_layout()
    plt.show()

    return fig


# ---------------------------------------------------------------------------
# 4.  plot_loss_curves
# ---------------------------------------------------------------------------

def plot_loss_curves(
    train_losses,
    val_losses=None,
    title="Training Loss",
    xlabel="Step",
    figsize=(9, 4),
    smoothing=0.0,
):
    """
    Plot one or two loss curves (training and optionally validation).

    Parameters
    ----------
    train_losses : list[float]
        Per-step (or per-epoch) training loss values.
    val_losses : list[float] or None
        Validation loss values sampled at the same cadence (optional).
    title : str
        Plot title.
    xlabel : str
        Label for the x-axis (e.g. "Step", "Epoch").
    figsize : tuple
        Figure size in inches.
    smoothing : float in [0, 1)
        Exponential moving-average smoothing factor.  0 = no smoothing.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """

    def _smooth(values, factor):
        """Exponential moving average."""
        if factor <= 0.0:
            return values
        smoothed = []
        last = values[0]
        for v in values:
            last = last * factor + v * (1 - factor)
            smoothed.append(last)
        return smoothed

    train_losses = list(train_losses)
    steps = list(range(len(train_losses)))

    fig, ax = plt.subplots(figsize=figsize)

    # ---- Raw curves (faint) ----
    ax.plot(steps, train_losses, color="steelblue", alpha=0.25,
            linewidth=1, label="_raw_train")

    # ---- Smoothed / display curves ----
    train_smooth = _smooth(train_losses, smoothing)
    ax.plot(steps, train_smooth, color="steelblue", linewidth=2,
            label="Train loss")

    if val_losses is not None:
        val_losses = list(val_losses)
        val_steps = list(range(len(val_losses)))
        ax.plot(val_steps, val_losses, color="tomato", alpha=0.25,
                linewidth=1, label="_raw_val")
        val_smooth = _smooth(val_losses, smoothing)
        ax.plot(val_steps, val_smooth, color="tomato", linewidth=2,
                label="Val loss")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate the final training loss
    final = train_losses[-1]
    ax.annotate(
        f"Final: {final:.4f}",
        xy=(steps[-1], final),
        xytext=(-60, 15),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="steelblue"),
        color="steelblue",
    )

    plt.tight_layout()
    plt.show()

    return fig


# ---------------------------------------------------------------------------
# 5.  plot_embedding_pca
# ---------------------------------------------------------------------------

def plot_embedding_pca(
    embeddings,
    labels,
    title="Token Embedding PCA",
    highlight=None,
    figsize=(9, 7),
    marker_size=60,
    alpha=0.7,
):
    """
    Project token embeddings to 2-D with PCA and draw a labelled scatter.

    Parameters
    ----------
    embeddings : array-like, shape (n_tokens, d_model)
        Embedding vectors.  Typically a subset of the full vocabulary
        (picking interesting tokens to avoid an unreadable cloud).
    labels : list[str]
        Token strings for each row of `embeddings`.
    title : str
        Plot title.
    highlight : list[str] or None
        A subset of `labels` to highlight in a different colour.
        Useful for drawing attention to semantically related words.
    figsize : tuple
        Figure size in inches.
    marker_size : int
        Scatter-plot marker size.
    alpha : float
        Marker opacity.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy()
    embeddings = np.asarray(embeddings, dtype=float)

    # ---- PCA to 2 components ----
    n_components = min(2, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(embeddings)

    # Pad to 2D if only 1 component was possible
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((len(coords), 1))])

    var_exp = pca.explained_variance_ratio_ * 100   # percentage

    # ---- Colours ----
    highlight_set = set(highlight) if highlight else set()
    colours = [
        "tomato" if lbl in highlight_set else "steelblue"
        for lbl in labels
    ]

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colours, s=marker_size, alpha=alpha, edgecolors="white", linewidth=0.5
    )

    # ---- Token labels ----
    for i, lbl in enumerate(labels):
        ax.annotate(
            lbl,
            (coords[i, 0], coords[i, 1]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
            color="black",
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC 1 ({var_exp[0]:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC 2 ({var_exp[1]:.1f}% variance)" if len(var_exp) > 1
                  else "PC 2 (0.0% variance)", fontsize=11)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.grid(True, alpha=0.2)

    if highlight_set:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="tomato",   label="Highlighted tokens"),
            Patch(facecolor="steelblue", label="Other tokens"),
        ]
        ax.legend(handles=legend_elements, fontsize=9)

    plt.tight_layout()
    plt.show()

    return fig


# ---------------------------------------------------------------------------
# 6.  plot_attention_rollout
# ---------------------------------------------------------------------------

def plot_attention_rollout(
    rollout_matrix,
    tokens,
    title="Attention Rollout",
    figsize=(8, 6),
    annot=False,
):
    """
    Visualise an attention-rollout matrix.

    Attention rollout (Abnar & Zuidema, 2020) propagates attention weights
    back through the layers to estimate how much each input token contributed
    to each output position.

    The rollout matrix is computed as follows::

        R_0 = I                       (identity: token i only attends to itself)
        R_l = 0.5 * A_l + 0.5 * I    (add residual connection)
        R   = R_1 @ R_2 @ … @ R_L    (matrix-multiply across layers)

    where A_l is the (seq_len × seq_len) attention matrix for layer l
    (averaged over heads).

    Parameters
    ----------
    rollout_matrix : array-like, shape (seq_len, seq_len)
        Pre-computed rollout matrix.  Row i shows the input-token
        attribution for output token i.
    tokens : list[str]
        Token strings (length == seq_len).
    title : str
        Plot title.
    figsize : tuple
        Figure size in inches.
    annot : bool
        Whether to annotate cells with values.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if hasattr(rollout_matrix, "detach"):
        rollout_matrix = rollout_matrix.detach().cpu().numpy()
    rollout_matrix = np.asarray(rollout_matrix, dtype=float)

    while rollout_matrix.ndim > 2:
        rollout_matrix = rollout_matrix[0]

    seq_len = rollout_matrix.shape[0]

    if len(tokens) > seq_len:
        tokens = tokens[:seq_len]
    elif len(tokens) < seq_len:
        tokens = list(tokens) + [f"[{i}]" for i in range(len(tokens), seq_len)]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        rollout_matrix,
        ax=ax,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap=CMAP_ROLLOUT,
        vmin=0.0,
        annot=annot,
        fmt=".2f",
        linewidths=0.2,
        linecolor="white",
        cbar=True,
        square=True,
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Input token (source)", fontsize=10)
    ax.set_ylabel("Output position (query)", fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

    plt.tight_layout()
    plt.show()

    return fig


# ---------------------------------------------------------------------------
# Helper exposed for notebooks that want to compute rollout themselves
# ---------------------------------------------------------------------------

def compute_attention_rollout(attn_weights_per_layer):
    """
    Compute attention rollout from a list of per-layer attention matrices.

    Parameters
    ----------
    attn_weights_per_layer : list of array-like, each shape
        (n_heads, seq_len, seq_len) or (seq_len, seq_len).
        Attention weights for each transformer layer, in order (layer 0 first).

    Returns
    -------
    rollout : np.ndarray, shape (seq_len, seq_len)
        The cumulative attribution matrix.
    """
    import numpy as np

    rollout = None

    for layer_weights in attn_weights_per_layer:
        if hasattr(layer_weights, "detach"):
            layer_weights = layer_weights.detach().cpu().numpy()
        layer_weights = np.asarray(layer_weights, dtype=float)

        # Average over heads if necessary
        if layer_weights.ndim == 3:
            layer_weights = layer_weights.mean(axis=0)   # (seq_len, seq_len)
        elif layer_weights.ndim == 4:
            layer_weights = layer_weights[0].mean(axis=0) # drop batch dim first

        seq_len = layer_weights.shape[0]

        # Add residual connection (0.5 * A + 0.5 * I) then re-normalise rows
        A_with_residual = 0.5 * layer_weights + 0.5 * np.eye(seq_len)
        # Row-normalise so each row still sums to 1
        row_sums = A_with_residual.sum(axis=-1, keepdims=True)
        A_norm = A_with_residual / (row_sums + 1e-9)

        if rollout is None:
            rollout = A_norm
        else:
            rollout = A_norm @ rollout   # propagate backwards through layers

    if rollout is None:
        raise ValueError("attn_weights_per_layer was empty.")

    return rollout
