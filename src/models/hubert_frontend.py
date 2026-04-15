"""HuBERT frontend adapter used before RawNet2 in hybrid experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    from transformers import HubertModel
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "transformers is required for HuBERT frontend. Install it with `pip install transformers`."
    ) from exc


@dataclass
class HuBERTConfig:
    """Configuration values for HuBERT feature extraction."""

    model_name: str = "facebook/hubert-base-ls960"
    target_sample_rate: int = 16000


class HuBERTFrontend(torch.nn.Module):
    """Extract HuBERT embeddings and adapt them back to a 1D sequence for RawNet2."""

    def __init__(self, device: str, cfg: HuBERTConfig | None = None):
        """Load the HuBERT model and prepare it for inference."""
        super().__init__()
        self.cfg = cfg or HuBERTConfig()
        self.device = device
        self.model = HubertModel.from_pretrained(self.cfg.model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def forward(self, batch_wave: torch.Tensor) -> torch.Tensor:
        """
        Convert input waveform batch [B, T] -> HuBERT features -> [B, T] proxy sequence.
        RawNet2 expects a 1D waveform-like tensor, so we collapse feature channels and upsample.
        """
        x = batch_wave.to(self.device)
        original_len = x.shape[-1]

        hidden = self.model(input_values=x).last_hidden_state  # [B, frames, dim]
        seq = hidden.mean(dim=-1)  # [B, frames]

        # Match RawNet2 expected temporal length.
        seq = F.interpolate(
            seq.unsqueeze(1),
            size=original_len,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        return seq
