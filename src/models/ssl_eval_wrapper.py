"""Evaluation wrapper for frozen SSL frontend + trained backend.

This wrapper exposes the same `load_weights()` and `evaluate()` style used by
the waveform wrappers, so `eval_suite.py` can treat all models uniformly.

SSL frontend caching:
  When `launder_fn is None` (clean / depth=0 conditions, including the CKA
  baseline split), per-utterance frozen-SSL sequence outputs are cached on
  disk under `cache_dir/<model_type>/eval/<utt_id>.pt` and reused across
  `evaluate()` and `extract_all_layers()` calls and across repeated runs.
  Laundered conditions (`launder_fn is not None`) never use the cache, since
  laundering changes the waveform per-sample and the SSL output is no longer
  reusable.
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
from src.models.embedding_cache import EmbeddingCache
from src.models.dataset import WavDataset
from src.models.backends import FFNBackend, WeightedAggregationBackend, SSLWithAASIST, SSLWithRawNet2
from src.evaluation.metrics import evaluate_scores


class _LaunderedDataset(torch.utils.data.Dataset):
    """Applies laundering per sample inside DataLoader worker processes."""

    def __init__(self, base_dataset, launder_fn=None):
        self.base = base_dataset
        self.launder_fn = launder_fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        wav, utt_id, src, key = self.base[idx]
        if self.launder_fn is not None:
            wav = self.launder_fn(wav.unsqueeze(0)).squeeze(0)
        return wav, utt_id, src, key


def _load_state_dict_compat(weights_path, device):
    """Load checkpoint and strip torch.compile() wrapper prefix if present."""
    state = torch.load(weights_path, map_location=device, weights_only=True)
    if not isinstance(state, dict):
        return state

    has_orig_mod = any(k.startswith("_orig_mod.") for k in state.keys())
    if has_orig_mod:
        state = {
            (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v
            for k, v in state.items()
        }
    return state


class SSLEvalWrapper:
    """Run evaluation for SSL models with FFN, AASIST, or RawNet2 backends."""

    def __init__(self, config_path, data_root, backend_mode="weighted", layer=None, backend_type="ffn",
                 cache_dir: str | None = "data/ssl_cache", use_cache: bool = True):
        """Build frontend/backend modules from config and backend settings.

        Args:
            cache_dir: base directory for the frozen-SSL embedding cache
                (used only for clean / launder_fn=None passes).
            use_cache: set False to disable caching entirely.
        """
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

        self.use_cache = use_cache
        self.cache_dir = cache_dir
        # Cache is keyed by (model_type, "eval") and shared across all
        # backend_mode/backend_type combinations for this SSL model, since
        # the cached payload is the raw frontend layer output (pre-backend).
        eval_cache = EmbeddingCache(cache_dir, self.config["model_type"], "eval") if use_cache else None

        self.frontend = SSLFrontend(model_type=self.config["model_type"], extract_layers=extract,
                                     device=self.device, cache=eval_cache)

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
        ckpt = torch.load(weights_path, map_location=self.device, weights_only=True)
        # Strip torch.compile() prefix if present in checkpoint keys.
        prefix = "_orig_mod."
        ckpt = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in ckpt.items()}
        self.backend.load_state_dict(ckpt)
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

    def _frontend_forward(self, batch_x, utt_ids, launder_fn):
        """Run the frontend, using the embedding cache only for clean (launder_fn=None) passes."""
        if launder_fn is None and self.use_cache:
            return self.frontend.forward_with_ids(batch_x, list(utt_ids))
        return self.frontend(batch_x)

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
        loader = DataLoader(_LaunderedDataset(dataset, launder_fn),
                            batch_size=self.config.get("batch_size", 32),
                            shuffle=False, num_workers=8, pin_memory=(self.device == "cuda"),
                            persistent_workers=True, prefetch_factor=4)
        fname_list, score_list, src_list, key_list = [], [], [], []
        self.backend.eval()
        with torch.inference_mode():
            for batch_x, utt_ids, srcs, keys in tqdm(loader, desc=f"{self.config['model_type']}|{self._tag()}"):
                batch_x = batch_x.to(self.device, non_blocking=True)
                layer_states = self._frontend_forward(batch_x, utt_ids, launder_fn)
                scores = self._forward(layer_states)[:, 0].cpu().numpy()
                fname_list.extend(utt_ids); score_list.extend(scores.tolist())
                src_list.extend(srcs); key_list.extend(keys)
        if launder_fn is None and self.use_cache:
            stats = self.frontend.cache.stats()
            print(f"  [CACHE] eval hits={stats['hits']} misses={stats['misses']}")
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
        loader = DataLoader(_LaunderedDataset(dataset, launder_fn),
                            batch_size=self.config.get("batch_size", 32),
                            shuffle=False, num_workers=8,
                            pin_memory=(self.device == "cuda"),
                            persistent_workers=True, prefetch_factor=4)
        all_embs = {l: [] for l in all_layers}
        with torch.inference_mode():
            for batch_x, utt_ids, *_ in tqdm(loader, desc="Extracting embeddings"):
                batch_x = batch_x.to(self.device, non_blocking=True)
                layer_states = self._frontend_forward(batch_x, utt_ids, launder_fn)
                for l, emb in self.frontend.mean_pool(layer_states).items():
                    all_embs[l].append(emb.cpu().numpy())
        if launder_fn is None and self.use_cache:
            stats = self.frontend.cache.stats()
            print(f"  [CACHE] extract_all_layers hits={stats['hits']} misses={stats['misses']}")
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
