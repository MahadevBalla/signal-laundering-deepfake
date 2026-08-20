"""
Centered Kernel Alignment (CKA) for measuring SSL embedding stability.
CKA(X, Y) = 1 means identical representations, 0 means unrelated.

Reference: Kornblith et al., "Similarity of Neural Network Representations
Revisited", ICML 2019.

GPU acceleration
----------------
``linear_cka`` converts its NumPy inputs to float32 CUDA tensors and performs
the N×N Gram matrix multiplications (``X @ X.T``) on the GPU, which is
significantly faster than NumPy for the N=1000 utterance batches used in CKA
analysis.  The public function signature is unchanged — callers continue to
pass NumPy arrays and receive a Python float.

If CUDA is unavailable the computation falls back to the CPU transparently.

Temporal pooling guard
----------------------
``cka_layer_stability`` detects 3-D inputs shaped ``[N, T, D]`` (unpooled
frame sequences) and mean-pools them to ``[N, D]`` before computing CKA.
Operating on unpooled sequences would build an O((N·T)²) Gram matrix which
scales quadratically with sequence length.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
import pickle

import torch


# ── Device selection ──────────────────────────────────────────────────────────
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(X: np.ndarray) -> torch.Tensor:
    """Convert a NumPy array to a float32 tensor on the CKA compute device."""
    return torch.from_numpy(np.asarray(X, dtype=np.float32)).to(_DEVICE)


def _center_gram_cuda(K: torch.Tensor) -> torch.Tensor:
    """Remove mean from rows and columns of a Gram matrix (GPU implementation).

    Equivalent to  H @ K @ H  where  H = I - (1/n) * 11^T,
    implemented without materialising the full n×n centering matrix.
    """
    n = K.shape[0]
    row_mean = K.mean(dim=1, keepdim=True)   # [N, 1]
    col_mean = K.mean(dim=0, keepdim=True)   # [1, N]
    total_mean = K.mean()
    return K - row_mean - col_mean + total_mean


# Keep the original NumPy helper for any downstream callers that import it
# directly (e.g. unit tests or notebooks).
def center_gram(K: np.ndarray) -> np.ndarray:
    """Remove mean from rows and columns of gram matrix (NumPy, CPU)."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between two representation matrices.

    Computation is performed on GPU (CUDA) when available and falls back to
    CPU automatically.  Accepts and returns the same types as before.

    Args:
        X: [N, D1] — embeddings from condition A (e.g. clean)
        Y: [N, D2] — embeddings from condition B (e.g. laundered)
        N must be identical (same utterances, same order)

    Returns:
        CKA score in [0, 1]. Higher = more similar representations.
    """
    assert X.shape[0] == Y.shape[0], "N must match — same utterances"

    Xt = _to_tensor(X)   # [N, D1]  on GPU
    Yt = _to_tensor(Y)   # [N, D2]  on GPU

    # Gram matrices computed on GPU
    K = Xt @ Xt.T        # [N, N]
    L = Yt @ Yt.T        # [N, N]

    # Center using the broadcasting implementation (no N×N H matrix)
    Kc = _center_gram_cuda(K)
    Lc = _center_gram_cuda(L)

    # HSIC estimates via element-wise product + sum
    hsic_kl = torch.sum(Kc * Lc)
    hsic_kk = torch.sum(Kc * Kc)
    hsic_ll = torch.sum(Lc * Lc)

    if hsic_kk == 0 or hsic_ll == 0:
        return 0.0

    return float((hsic_kl / torch.sqrt(hsic_kk * hsic_ll)).item())


def cka_layer_stability(
    clean_embeddings: dict[int, np.ndarray],
    laundered_embeddings: dict[int, np.ndarray],
) -> dict[int, float]:
    """
    Compute CKA per layer between clean and laundered embeddings.

    Args:
        clean_embeddings:    {layer_idx: [N, D]}  or  {layer_idx: [N, T, D]}
        laundered_embeddings:{layer_idx: [N, D]}  or  {layer_idx: [N, T, D]}

    If the arrays are 3-D (unpooled frame sequences), they are mean-pooled over
    the time axis before CKA is computed to avoid an O((N·T)²) Gram matrix.

    Returns:
        {layer_idx: cka_score}
    """
    assert set(clean_embeddings.keys()) == set(laundered_embeddings.keys()), \
        "Layer sets must match"

    result = {}
    for layer in clean_embeddings:
        X = clean_embeddings[layer]
        Y = laundered_embeddings[layer]
        # Temporal-pool guard: collapse [N, T, D] → [N, D]
        if X.ndim == 3:
            X = X.mean(axis=1)
        if Y.ndim == 3:
            Y = Y.mean(axis=1)
        result[layer] = linear_cka(X, Y)
    return result


def cosine_stability(
    clean_embeddings: dict[int, np.ndarray],
    laundered_embeddings: dict[int, np.ndarray],
) -> dict[int, float]:
    """
    Mean cosine similarity per layer between clean and laundered utterance pairs.
    Utterance-level (paired): same N utterances, same order.

    Returns:
        {layer_idx: mean_cosine_similarity}
    """
    results = {}
    for layer in clean_embeddings:
        X = clean_embeddings[layer]    # [N, 768]
        Y = laundered_embeddings[layer]

        # Normalize rows
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)

        # Paired cosine similarity
        cos_sim = np.sum(X_norm * Y_norm, axis=1)   # [N]
        results[layer] = float(np.mean(cos_sim))

    return results


def load_embeddings(pkl_path: str | Path) -> dict[int, np.ndarray]:
    """Load saved layer embeddings from a pickle file."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)
