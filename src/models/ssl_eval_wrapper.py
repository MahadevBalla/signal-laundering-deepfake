"""Evaluation wrapper for frozen SSL frontend + trained backend.

This wrapper exposes the same `load_weights()` and `evaluate()` style used by
the waveform wrappers, so `eval_suite.py` can treat all models uniformly.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.ssl_frontend import SSLFrontend
from src.models.dataset import WavDataset
from src.models.backends import FFNBackend, WeightedAggregationBackend, SSLWithAASIST, SSLWithRawNet2
from src.evaluation.metrics import evaluate_scores


class SSLEvalWrapper:
    """Run evaluation for SSL models with FFN, AASIST, or RawNet2 backends."""

    def __init__(self, config_path, data_root, backend_mode="weighted", layer=None, backend_type="ffn"):
        """Build frontend/backend modules from config and backend settings."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        if backend_mode not in ("single", "weighted"):
            raise ValueError(f"backend_mode must be 'single' or 'weighted'")
        if backend_mode == "single" and layer is None:
            raise ValueError("layer required when backend_mode='single'")
        if backend_type in ("aasist", "rawnet2") and backend_mode == "single":
            raise ValueError(f"backend_type='{backend_type}' only supports weighted mode")
        if backend_type not in ("ffn", "aasist", "rawnet2"):
            raise ValueError(f"Unknown backend_type '{backend_type}'")

        self.data_root = Path(data_root)
        self.backend_mode = backend_mode
        self.backend_type = backend_type
        self.layer = layer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        embed_dim = self.config.get("embed_dim", 768)
        num_layers = len(self.config.get("extract_layers", list(range(12))))
        extract = [layer] if backend_mode == "single" else self.config.get("extract_layers", list(range(num_layers)))

        self.frontend = SSLFrontend(model_type=self.config["model_type"], extract_layers=extract, device=self.device)

        if backend_type == "ffn":
            self.backend = FFNBackend(embed_dim=embed_dim, dropout=0.2) if backend_mode == "single" else WeightedAggregationBackend(num_layers=num_layers, embed_dim=embed_dim, dropout=0.2)
        elif backend_type == "aasist":
            self.backend = SSLWithAASIST(num_layers=num_layers, embed_dim=embed_dim)
        elif backend_type == "rawnet2":
            self.backend = SSLWithRawNet2(num_layers=num_layers, embed_dim=embed_dim)

        self.backend = self.backend.to(self.device)
        self._weights_loaded = False
        self._last_eval_result = None

    def load_weights(self, weights_path=None):
        """Load backend checkpoint weights and switch backend to eval mode."""
        if not weights_path or not Path(weights_path).exists():
            hint = (f"--mode single --layer {self.layer}" if self.backend_mode == "single"
                    else f"--mode weighted --backend {self.backend_type}")
            raise FileNotFoundError(
                f"Weights not found: {weights_path}\n"
                f"Train first: python train_ssl_backend.py --model {self.config['model_type']} {hint}"
            )
        self.backend.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.backend.eval()
        self._weights_loaded = True
        print(f"[{self.config['model_type']}|{self._tag()}] Loaded {weights_path}")

    def _tag(self):
        """Return a short text tag describing backend mode/type."""
        if self.backend_mode == "single":
            return f"FFN-L{self.layer}"
        return {"ffn": "FFN-W", "aasist": "AASIST-W", "rawnet2": "RN2-W"}[self.backend_type]

    def _forward(self, layer_states):
        """Dispatch forward pass based on backend mode and type."""
        if self.backend_mode == "single" and self.backend_type == "ffn":
            return self.backend(layer_states[self.layer])
        return self.backend(layer_states)

    def evaluate(self, output_dir="outputs", launder_fn=None, max_eval=None):
        """Evaluate model scores on eval split and compute EER/min-tDCF."""
        if not self._weights_loaded:
            raise RuntimeError("Call load_weights() before evaluate().")
        t0 = time.time()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dataset = WavDataset(self.data_root, self.config.get("track", "LA"), split="eval",
                             max_len=self.config.get("max_len_samples", 64000))
        if max_eval is not None:
            dataset.trials = dataset.trials[:max_eval]
        loader = DataLoader(dataset, batch_size=self.config.get("batch_size", 32),
                            shuffle=False, num_workers=4, pin_memory=(self.device == "cuda"))
        fname_list, score_list, src_list, key_list = [], [], [], []
        self.backend.eval()
        with torch.no_grad():
            for batch_x, utt_ids, srcs, keys in tqdm(loader, desc=f"{self.config['model_type']}|{self._tag()}"):
                if launder_fn is not None:
                    batch_x = launder_fn(batch_x)
                scores = self._forward(self.frontend(batch_x))[:, 1].cpu().numpy()
                fname_list.extend(utt_ids); score_list.extend(scores.tolist())
                src_list.extend(srcs); key_list.extend(keys)
        score_path = out / "eval_scores.txt"
        with open(score_path, "w") as fh:
            for fn, sc, src, key in zip(fname_list, score_list, src_list, key_list):
                fh.write(f"{fn} {src} {key} {sc}\n")
        result = evaluate_scores(score_path, self.data_root / self.config["asv_score_path"])
        self._last_eval_result = result
        print(f"[{self.config['model_type']}|{self._tag()}] EER={result.eer:.4f}%  tDCF={result.min_tdcf:.4f}  [{(time.time()-t0)/60:.1f}min]")
        return result.eer, result.min_tdcf

    def extract_all_layers(self, output_dir, launder_fn=None, max_eval=None, save_embeddings=False):
        """Extract mean-pooled embeddings for all configured SSL layers."""
        all_layers = self.config.get("extract_layers", list(range(12)))
        if self.frontend.extract_layers != all_layers:
            self.frontend.remove_hooks()
            self.frontend.extract_layers = all_layers
            self.frontend._register_hooks()
        dataset = WavDataset(self.data_root, self.config.get("track", "LA"), split="eval",
                             max_len=self.config.get("max_len_samples", 64000))
        if max_eval is not None:
            dataset.trials = dataset.trials[:max_eval]
        loader = DataLoader(dataset, batch_size=self.config.get("batch_size", 32), shuffle=False, num_workers=4)
        all_embs = {l: [] for l in all_layers}
        with torch.no_grad():
            for batch_x, *_ in tqdm(loader, desc="Extracting embeddings"):
                if launder_fn is not None:
                    batch_x = launder_fn(batch_x)
                for l, emb in self.frontend.mean_pool(self.frontend(batch_x)).items():
                    all_embs[l].append(emb.cpu().numpy())
        result = {l: np.concatenate(v, axis=0) for l, v in all_embs.items()}
        if save_embeddings:
            import pickle
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(output_dir) / "layer_embeddings.pkl", "wb") as f:
                pickle.dump(result, f)
        return result

    def get_layer_weights(self):
        """Return learned layer weights when backend exposes them."""
        if hasattr(self.backend, "get_layer_weights"):
            return self.backend.get_layer_weights()
        return None
