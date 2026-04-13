"""Core laundering engine that applies N/M/P pipelines by depth and strength."""

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml

from . import pipeline_N, pipeline_M, pipeline_P

SR = 16000

_PIPELINES = {
    "N": pipeline_N.apply,
    "M": pipeline_M.apply,
    "P": pipeline_P.apply,
}


class LaunderingEngine:
    """Load pipeline configs and provide sample/batch laundering functions."""

    def __init__(self, config_dir: str = "configs"):
        """Read N/M/P parameter YAML files from the config directory."""
        self.params = {
            s: yaml.safe_load((Path(config_dir) / f"{s}_params.yaml").read_text())
            for s in ("N", "M", "P")
        }

    def apply_sample(
        self, wav: np.ndarray, sr: int, pipeline: str, depth: int, strength: str
    ) -> np.ndarray:
        """Apply one laundering pipeline to a single waveform array."""
        if depth == 0:
            return wav
        return _PIPELINES[pipeline](wav, sr, depth, strength, self.params[pipeline])

    def get_batch_fn(self, pipeline: str, depth: int, strength: str) -> Callable | None:
        """Return a batch-wise laundering callable for wrapper evaluation loops."""
        if depth == 0:
            return None

        def _launder(batch: torch.Tensor) -> torch.Tensor:
            arr = batch.cpu().numpy()
            out = np.stack(
                [
                    self.apply_sample(arr[i], SR, pipeline, depth, strength)
                    for i in range(len(arr))
                ]
            )
            return torch.from_numpy(out).float()

        return _launder
