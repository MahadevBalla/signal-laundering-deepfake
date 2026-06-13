"""Frozen SSL frontend with layer-output hooks.

The frontend wraps HuggingFace wav2vec2/hubert/wavlm models and returns
intermediate layer states for downstream backends.

Optionally backed by an on-disk `EmbeddingCache` (see
`src/models/embedding_cache.py`). When a cache is attached and utterance IDs
are provided via `forward_with_ids`, per-utterance sequence outputs for
**clean** inputs are read from / written to disk, skipping the SSL forward
pass entirely on a cache hit. This is only valid for unlaundered inputs —
laundering is applied per-sample before the frontend and changes the input
on every call, so callers must not pass a cache when `launder_fn is not None`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModel

from src.models.embedding_cache import EmbeddingCache, pad_and_stack

SSL_MODEL_IDS = {
    "wav2vec2": "facebook/wav2vec2-base",
    "hubert":   "facebook/hubert-base-ls960",
    "wavlm":    "microsoft/wavlm-base",
}

SSL_NUM_LAYERS = {
    "wav2vec2": 12,
    "hubert":   12,
    "wavlm":    12,
}


class SSLFrontend(nn.Module):
    """Extract hidden states from selected layers of a frozen SSL model."""

    def __init__(
        self,
        model_type: str,
        extract_layers: list[int] | None = None,
        device: str = "cuda",
        cache: EmbeddingCache | None = None,
    ):
        """Load one SSL checkpoint and register forward hooks for selected layers.

        Args:
            model_type: one of "wav2vec2", "hubert", "wavlm".
            extract_layers: transformer layer indices to hook.
            device: torch device for the underlying model.
            cache: optional `EmbeddingCache` for clean-input sequence outputs.
        """
        super().__init__()
        assert model_type in SSL_MODEL_IDS, f"Unknown SSL model: {model_type}"
        self.model_type = model_type
        self.device = device
        model_id = SSL_MODEL_IDS[model_type]
        num_layers = SSL_NUM_LAYERS[model_type]
        self.extract_layers = extract_layers if extract_layers is not None else list(range(num_layers))
        self.processor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.model.eval()
        self._layer_outputs: dict[int, torch.Tensor] = {}
        self._hooks: list = []
        self._register_hooks()
        self.cache = cache

    def _register_hooks(self):
        """Attach hooks that cache hidden states from requested transformer layers."""
        transformer_layers = self.model.encoder.layers
        for layer_idx in self.extract_layers:
            def make_hook(idx):
                def hook(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    self._layer_outputs[idx] = hidden.detach()
                return hook
            h = transformer_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            self._hooks.append(h)

    def remove_hooks(self):
        """Remove all registered hooks from the underlying SSL model."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def forward(self, waveform: torch.Tensor) -> dict[int, torch.Tensor]:
        """Run the SSL model and return layer outputs as a dictionary.

        This path never touches the cache (no utterance IDs available). Use
        `forward_with_ids` for cache-aware extraction on clean inputs.
        """
        self._layer_outputs.clear()
        with torch.no_grad():
            self.model(waveform.to(self.device))
        return dict(self._layer_outputs)

    def forward_with_ids(
        self, waveform: torch.Tensor, utt_ids: list[str]
    ) -> dict[int, torch.Tensor]:
        """Cache-aware forward pass for clean (unlaundered) batches.

        For each utterance, checks `self.cache` for a precomputed
        `{layer: [T, D]}` entry. Utterances with a cache hit skip the SSL
        forward pass; the rest are run through the model in one batch, and
        their outputs are written back to the cache. Results are padded to a
        common sequence length and stacked to match the shape of `forward()`.

        If `self.cache` is None, falls back to a plain `forward()` call.
        """
        if self.cache is None:
            return self.forward(waveform)

        B = waveform.shape[0]
        cached: dict[int, dict[int, torch.Tensor]] = {}
        miss_idx: list[int] = []
        for i, utt_id in enumerate(utt_ids):
            hit = self.cache.get(utt_id, self.extract_layers)
            if hit is not None:
                cached[i] = hit
            else:
                miss_idx.append(i)

        if miss_idx:
            miss_wave = waveform[miss_idx]
            self._layer_outputs.clear()
            with torch.no_grad():
                self.model(miss_wave.to(self.device))
            miss_outputs = dict(self._layer_outputs)  # {layer: [n_miss, T, D]}
            for j, i in enumerate(miss_idx):
                per_utt = {layer: miss_outputs[layer][j].cpu() for layer in self.extract_layers}
                self.cache.put(utt_ids[i], per_utt)
                cached[i] = per_utt

        ordered = [cached[i] for i in range(B)]
        stacked = pad_and_stack(ordered, self.extract_layers)
        return {layer: tensor.to(self.device) for layer, tensor in stacked.items()}

    def mean_pool(self, layer_outputs: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """Mean-pool layer outputs across time to get utterance embeddings."""
        return {idx: emb.mean(dim=1) for idx, emb in layer_outputs.items()}
