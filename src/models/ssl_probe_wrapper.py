"""Legacy SSL linear-probe evaluation wrapper.

This module is kept for compatibility and side-by-side experiments with the
older mean-pool + linear probe setup.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.ssl_frontend import SSLFrontend
from src.models.dataset import WavDataset
from src.evaluation.metrics import evaluate_scores

# Keep _SimpleWavDataset as an alias so existing code that imports it by name
# from this module doesn't break during transition.
_SimpleWavDataset = WavDataset


class SSLProbeWrapper:
    """
    Legacy linear probe (1 linear layer on mean-pooled embeddings).
    Not used in the main eval flow. SSLEvalWrapper is the current interface.
    Kept because it's architecturally distinct (mean-pool → linear vs
    full-sequence → FFN with temporal attention).
    """

    def __init__(self, config_path: str, data_root: str):
        """Initialize frozen SSL frontend and linear probe classifier."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.data_root = Path(data_root)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.frontend = SSLFrontend(
            model_type=self.config["model_type"],
            extract_layers=self.config.get("extract_layers"),
            device=self.device,
        )
        self.probe_layer = self.config["probe_layer"]
        self.probe = nn.Linear(self.config["embed_dim"], 2).to(self.device)
        self._probe_trained = False
        self._last_eval_result = None

    def load_weights(self, weights_path: Optional[str] = None):
        """Load trained probe weights if available."""
        if weights_path and Path(weights_path).exists():
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.probe.load_state_dict(state)
            self.probe.eval()
            self._probe_trained = True
            print(f"[{self.config['model_type']}] Probe loaded: {weights_path}")
        else:
            print(f"[{self.config['model_type']}] No probe weights — embedding mode only")

    def evaluate(self, output_dir: str = "outputs", launder_fn=None, max_eval: Optional[int] = None) -> tuple[float, float]:
        """Evaluate the legacy probe and compute EER/min-tDCF."""
        if not self._probe_trained:
            raise RuntimeError("Probe weights not loaded. Run train_probe.py first.")

        t0 = time.time()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        dataset = WavDataset(self.data_root, self.config.get("track", "LA"), split="eval",
                             max_len=self.config.get("max_len_samples", 64000))
        if max_eval is not None:
            dataset.trials = dataset.trials[:max_eval]

        loader = DataLoader(dataset, batch_size=self.config.get("batch_size", 8), shuffle=False, num_workers=4)
        fname_list, score_list, src_list, key_list = [], [], [], []

        self.probe.eval()
        with torch.no_grad():
            for batch_x, utt_ids, srcs, keys in tqdm(loader, desc=f"{self.config['model_type']}"):
                if launder_fn is not None:
                    batch_x = launder_fn(batch_x)
                pooled = self.frontend.mean_pool(self.frontend(batch_x))
                emb = pooled[self.probe_layer].to(self.device)
                scores = self.probe(emb)[:, 1].cpu().numpy()
                fname_list.extend(utt_ids)
                score_list.extend(scores.tolist())
                src_list.extend(srcs)
                key_list.extend(keys)

        score_path = out / "eval_scores.txt"
        with open(score_path, "w") as fh:
            for fn, sc, src, key in zip(fname_list, score_list, src_list, key_list):
                fh.write(f"{fn} {src} {key} {sc}\n")

        result = evaluate_scores(score_path, self.data_root / self.config["asv_score_path"])
        self._last_eval_result = result
        print(f"[{self.config['model_type']}] EER={result.eer:.4f}%  [{(time.time()-t0)/60:.1f}min]")
        return result.eer, result.min_tdcf

    def extract_all_layers(self, output_dir: str, launder_fn=None, max_eval: Optional[int] = None, save_embeddings: bool = True) -> dict[int, np.ndarray]:
        """Extract and optionally save embeddings for all configured layers."""
        dataset = WavDataset(self.data_root, self.config.get("track", "LA"), split="eval",
                             max_len=self.config.get("max_len_samples", 64000))
        if max_eval is not None:
            dataset.trials = dataset.trials[:max_eval]

        loader = DataLoader(dataset, batch_size=self.config.get("batch_size", 8), shuffle=False, num_workers=4)
        all_embs: dict[int, list] = {l: [] for l in self.frontend.extract_layers}

        with torch.no_grad():
            for batch_x, *_ in tqdm(loader, desc="Extracting embeddings"):
                if launder_fn is not None:
                    batch_x = launder_fn(batch_x)
                pooled = self.frontend.mean_pool(self.frontend(batch_x))
                for l, emb in pooled.items():
                    all_embs[l].append(emb.cpu().numpy())

        result = {l: np.concatenate(v, axis=0) for l, v in all_embs.items()}
        if save_embeddings:
            import pickle
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(output_dir) / "layer_embeddings.pkl", "wb") as f:
                pickle.dump(result, f)
        return result
