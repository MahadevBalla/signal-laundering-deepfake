"""
Centered Kernel Alignment (CKA) for measuring SSL embedding stability.
CKA(X, Y) = 1 means identical representations, 0 means unrelated.

Reference: Kornblith et al., "Similarity of Neural Network Representations
Revisited", ICML 2019.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
import pickle


def center_gram(K: np.ndarray) -> np.ndarray:
    """Remove mean from rows and columns of gram matrix."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between two representation matrices.

    Args:
        X: [N, D1] — embeddings from condition A (e.g. clean)
        Y: [N, D2] — embeddings from condition B (e.g. laundered)
        N must be identical (same utterances, same order)

    Returns:
        CKA score in [0, 1]. Higher = more similar representations.
    """
    assert X.shape[0] == Y.shape[0], "N must match — same utterances"

    # Gram matrices
    K = X @ X.T   # [N, N]
    L = Y @ Y.T   # [N, N]

    # Center
    Kc = center_gram(K)
    Lc = center_gram(L)

    # HSIC estimates
    hsic_kl = np.sum(Kc * Lc)          # trace(Kc @ Lc)
    hsic_kk = np.sum(Kc * Kc)
    hsic_ll = np.sum(Lc * Lc)

    if hsic_kk == 0 or hsic_ll == 0:
        return 0.0

    return float(hsic_kl / np.sqrt(hsic_kk * hsic_ll))


def cka_layer_stability(
    clean_embeddings: dict[int, np.ndarray],
    laundered_embeddings: dict[int, np.ndarray],
) -> dict[int, float]:
    """
    Compute CKA per layer between clean and laundered embeddings.

    Args:
        clean_embeddings:    {layer_idx: [N, 768]}
        laundered_embeddings:{layer_idx: [N, 768]}

    Returns:
        {layer_idx: cka_score}
    """
    assert set(clean_embeddings.keys()) == set(laundered_embeddings.keys()), \
        "Layer sets must match"

    return {
        layer: linear_cka(clean_embeddings[layer], laundered_embeddings[layer])
        for layer in clean_embeddings
    }


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
    with open(pkl_path, "rb") as f:
        return pickle.load(f)
