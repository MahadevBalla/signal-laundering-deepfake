"""
SSL frontend wrapper with layer-wise extraction hooks.
Supports Wav2Vec2, HuBERT, WavLM from HuggingFace transformers.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
)


# Maps model_type -> HuggingFace model ID
SSL_MODEL_IDS = {
    "wav2vec2": "facebook/wav2vec2-base",
    "hubert":   "facebook/hubert-base-ls960",
    "wavlm":    "microsoft/wavlm-base",
}

# Number of transformer layers in each base model
SSL_NUM_LAYERS = {
    "wav2vec2": 12,
    "hubert":   12,
    "wavlm":    12,
}


class SSLFrontend(nn.Module):
    """
    Wraps a HuggingFace SSL model and captures intermediate
    transformer layer outputs via forward hooks.

    Usage:
        frontend = SSLFrontend("wav2vec2", extract_layers=[3, 6, 9, 11])
        frontend.eval()
        with torch.no_grad():
            embeddings = frontend(waveform_tensor)
            # embeddings: dict {layer_idx: Tensor[B, T, D]}
    """

    def __init__(
        self,
        model_type: str,
        extract_layers: list[int] | None = None,
        device: str = "cuda",
    ):
        super().__init__()
        assert model_type in SSL_MODEL_IDS, (
            f"Unknown SSL model: {model_type}. Choose from {list(SSL_MODEL_IDS)}"
        )
        self.model_type = model_type
        self.device = device
        model_id = SSL_MODEL_IDS[model_type]
        num_layers = SSL_NUM_LAYERS[model_type]

        # All layers if not specified
        self.extract_layers = (
            extract_layers if extract_layers is not None
            else list(range(num_layers))
        )

        # Load model + processor
        self.processor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.model.eval()

        # Hook storage
        self._layer_outputs: dict[int, torch.Tensor] = {}
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self):
        """Attach forward hooks to each requested transformer layer."""
        # HuggingFace Wav2Vec2/HuBERT/WavLM all expose .encoder.layers
        transformer_layers = self.model.encoder.layers
        for layer_idx in self.extract_layers:
            layer = transformer_layers[layer_idx]

            def make_hook(idx):
                def hook(module, input, output):
                    # output is a tuple; [0] is the hidden state tensor
                    hidden = output[0] if isinstance(output, tuple) else output
                    self._layer_outputs[idx] = hidden.detach().cpu()
                return hook

            h = layer.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def forward(self, waveform: torch.Tensor) -> dict[int, torch.Tensor]:
        """
        Args:
            waveform: Tensor [B, T] at 16kHz, raw float32
        Returns:
            dict mapping layer_idx -> Tensor [B, T', D]
            where T' is the downsampled time axis after CNN encoder
        """
        self._layer_outputs.clear()

        # Move to device
        waveform = waveform.to(self.device)

        # HuggingFace expects [B, T] float32
        with torch.no_grad():
            self.model(waveform)  # hooks fire during this forward pass

        # Return copy of captured layer outputs
        return dict(self._layer_outputs)

    def mean_pool(self, layer_outputs: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """
        Mean-pool across time axis T' → [B, D] per layer.
        Use this to get utterance-level embeddings.
        """
        return {
            layer_idx: emb.mean(dim=1)   # [B, T', D] → [B, D]
            for layer_idx, emb in layer_outputs.items()
        }