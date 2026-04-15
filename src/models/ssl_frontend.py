"""Frozen SSL frontend with layer-output hooks.

The frontend wraps HuggingFace wav2vec2/hubert/wavlm models and returns
intermediate layer states for downstream backends.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModel

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

    def __init__(self, model_type: str, extract_layers: list[int] | None = None, device: str = "cuda"):
        """Load one SSL checkpoint and register forward hooks for selected layers."""
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
        """Run the SSL model and return layer outputs as a dictionary."""
        self._layer_outputs.clear()
        with torch.no_grad():
            self.model(waveform.to(self.device))
        return dict(self._layer_outputs)

    def mean_pool(self, layer_outputs: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """Mean-pool layer outputs across time to get utterance embeddings."""
        return {idx: emb.mean(dim=1) for idx, emb in layer_outputs.items()}
