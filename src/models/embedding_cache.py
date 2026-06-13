"""On-disk cache for frozen SSL frontend layer outputs.

The SSL frontends (wav2vec2 / HuBERT / WavLM) are frozen everywhere they're
used — `train_ssl_backend.py` sets `requires_grad_(False)` on all frontend
parameters, and `layer_sweep.py` re-trains a backend per layer over the same
clean inputs many times. In every one of these cases the frontend's output
for a given (model_type, utterance, extract_layers) is deterministic, so
recomputing it on every epoch / every layer-sweep run is pure waste.

This module caches per-utterance, per-layer sequence outputs (the full
`[T, D]` hidden states, pre-batching) to disk as `.pt` files. Only **clean**
(unlaundered) inputs are cacheable, since laundering is applied per-sample
before the frontend and changes the input every time.

Cache layout:

    <cache_dir>/<model_type>/<split>/<utt_id>.pt

Each file stores a dict: ``{layer_idx: Tensor[T, D], "_layers": [..]}``.
If a cached file exists but doesn't cover all currently-requested
``extract_layers``, it's treated as a miss and recomputed+overwritten so the
cache stays self-consistent.
"""

from __future__ import annotations

from pathlib import Path

import torch


class EmbeddingCache:
    """Disk cache for frozen SSL frontend sequence outputs (clean inputs only)."""

    def __init__(self, cache_dir: str | Path, model_type: str, split: str):
        """Set up the cache directory for one (model_type, split) combination."""
        self.dir = Path(cache_dir) / model_type / split
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, utt_id: str) -> Path:
        """Return the on-disk path for one utterance's cached embeddings."""
        return self.dir / f"{utt_id}.pt"

    def get(
        self, utt_id: str, extract_layers: list[int]
    ) -> dict[int, torch.Tensor] | None:
        """Return cached `{layer_idx: Tensor[T, D]}` if present and covers all requested layers."""
        path = self._path(utt_id)
        if not path.exists():
            self.misses += 1
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            self.misses += 1
            return None
        cached_layers = set(payload.get("_layers", []))
        if not set(extract_layers).issubset(cached_layers):
            self.misses += 1
            return None
        self.hits += 1
        return {layer: payload[layer] for layer in extract_layers}

    def put(self, utt_id: str, layer_outputs: dict[int, torch.Tensor]) -> None:
        """Write `{layer_idx: Tensor[T, D]}` (CPU tensors) to disk for one utterance."""
        payload = {
            layer: tensor.detach().cpu() for layer, tensor in layer_outputs.items()
        }
        payload["_layers"] = sorted(layer_outputs.keys())
        torch.save(payload, self._path(utt_id))

    def stats(self) -> dict[str, int]:
        """Return cumulative hit/miss counters for this cache instance."""
        return {"hits": self.hits, "misses": self.misses}


def pad_and_stack(
    seqs: list[dict[int, torch.Tensor]], extract_layers: list[int]
) -> dict[int, torch.Tensor]:
    """Pad a list of per-utterance `{layer: [T_i, D]}` dicts to a common T and stack to `[B, T, D]`.

    Sequence lengths can differ slightly across utterances depending on how
    the SSL model's conv-subsampling rounds the input length. Right-pad with
    zeros to the max length in the batch (matches the effect of padding the
    waveform itself, which downstream pooling/backends already tolerate).
    """
    out: dict[int, torch.Tensor] = {}
    for layer in extract_layers:
        tensors = [s[layer] for s in seqs]
        max_t = max(t.shape[0] for t in tensors)
        padded = [
            torch.nn.functional.pad(t, (0, 0, 0, max_t - t.shape[0]))
            if t.shape[0] < max_t
            else t
            for t in tensors
        ]
        out[layer] = torch.stack(padded, dim=0)
    return out
