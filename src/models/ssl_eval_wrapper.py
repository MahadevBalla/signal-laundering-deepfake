"""Evaluation wrapper for frozen SSL frontend + trained backend.

This wrapper exposes the same `load_weights()` and `evaluate()` style used by
the waveform wrappers, so `eval_suite.py` can treat all models uniformly.

SSL frontend caching:
  When `launder_fn is None` (clean / depth=0 conditions, including the CKA
  baseline split), per-utterance frozen-SSL sequence outputs are cached on
  disk under `cache_dir/<model_type>/eval/<utt_id>.pt` and reused across
  `evaluate()` and `extract_all_layers()` calls and across repeated runs.

  When a `precomputed_root` is provided together with `pipeline`, `depth`, and
  `strength`, the dataset is loaded from the static precomputed audio on disk
  (produced by `precompute_laundering.py`) instead of applying a live
  `launder_fn`.  Because the waveforms are now deterministic, their SSL
  embeddings are also cacheable.  The cache subdirectory for a laundered
  condition is ``eval_{pipeline}_d{depth}_{strength}`` (e.g.
  ``eval_N_d1_M``), keeping it separate from the clean ``eval`` cache so no
  existing files are invalidated.

  Laundered conditions using the legacy live `launder_fn` path (used by CKA
  and any caller that does not supply `precomputed_root`) never use the cache,
  since laundering re-randomises the waveform per-sample.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Optional

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


class _PrecomputedLaunderDataset(torch.utils.data.Dataset):
    """Loads static precomputed laundered audio instead of applying live laundering.

    Reads from the directory layout produced by ``precompute_laundering.py``::

        <precomputed_root>/<pipeline>_d<depth>_<strength>/
            ASVspoof2019_LA_eval/flac/<utt_id>.flac
            ASVspoof2019_LA_cm_protocols/...

    The returned (wav, utt_id, src, key) tuples are identical in type and shape
    to those returned by ``_LaunderedDataset``, so the evaluation loop is
    completely agnostic to which dataset class is in use.

    Args:
        data_root: Original ASVspoof dataset root (used only for the protocol
            file; audio is loaded from the precomputed subdirectory).
        precomputed_root: Root directory written by ``precompute_laundering.py``
            (e.g. ``data/precomputed``).
        track: ASVspoof track identifier (``"LA"``).
        pipeline: Laundering pipeline code (``"N"``, ``"M"``, or ``"P"``).
        depth: Laundering depth (1, 2, or 3).
        strength: Laundering strength (``"L"``, ``"M"``, or ``"H"``).
        max_len: Maximum waveform length in samples (truncated/padded).
    """

    def __init__(
        self,
        data_root: Path,
        precomputed_root: str | Path,
        track: str,
        pipeline: str,
        depth: int,
        strength: str,
        max_len: int = 64000,
    ) -> None:
        variant_dir = Path(precomputed_root) / f"{pipeline}_d{depth}_{strength}"
        if not variant_dir.exists():
            raise FileNotFoundError(
                f"Precomputed variant not found: {variant_dir}\n"
                "Run precompute_laundering.py first, or omit --precomputed_root "
                "to use live laundering."
            )
        # Audio is loaded from the precomputed variant directory, but the
        # protocol file is sourced from the original data_root (it is identical
        # across all variants — precompute_laundering.py copies it verbatim).
        self._dataset = WavDataset(variant_dir, track, split="eval", max_len=max_len)
        # Expose trials so callers can truncate (max_eval support).
        self.trials = self._dataset.trials

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        return self._dataset[idx]


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
                (used for clean passes and precomputed-laundered passes).
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
        # Leave 1-2 cores for the main process / OS instead of hardcoding a
        # worker count that silently ignores whatever SLURM actually granted.
        self.num_workers = max(1, len(os.sched_getaffinity(0)) - 2)

        embed_dim = self.config.get("embed_dim", 768)
        num_layers = len(self.config.get("extract_layers", list(range(12))))
        extract = [layer] if backend_mode == "single" else self.config.get("extract_layers", list(range(num_layers)))

        self.use_cache = use_cache
        self.cache_dir = cache_dir
        # Cache is keyed by (model_type, split) and shared across all
        # backend_mode/backend_type combinations for this SSL model.
        # Clean eval uses split="eval"; precomputed-laundered conditions use
        # split="eval_{pipeline}_d{depth}_{strength}" — see _build_cache().
        self._extract_layers = extract
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

    def _build_cache(self, split_key: str) -> EmbeddingCache | None:
        """Return an EmbeddingCache for the given split key, or None if disabled."""
        if not self.use_cache:
            return None
        return EmbeddingCache(self.cache_dir, self.config["model_type"], split_key)

    def _swap_frontend_cache(self, cache: EmbeddingCache | None) -> None:
        """Replace the frontend's active cache without rebuilding the frontend."""
        self.frontend.cache = cache

    def _frontend_forward(self, batch_x, utt_ids, use_cache_path: bool):
        """Run the frontend, using the cache when ``use_cache_path`` is True."""
        if use_cache_path and self.use_cache:
            return self.frontend.forward_with_ids(batch_x, list(utt_ids))
        return self.frontend(batch_x)

    def evaluate(
        self,
        output_dir: str = "outputs",
        launder_fn: Optional[Callable] = None,
        precomputed_root: Optional[str] = None,
        pipeline: Optional[str] = None,
        depth: Optional[int] = None,
        strength: Optional[str] = None,
        max_eval: Optional[int] = None,
    ):
        """Evaluate model scores on eval split and compute EER/min-tDCF.

        Three mutually-exclusive dataset paths:

        1. **Clean** (``launder_fn=None``, no ``precomputed_root``):
           Loads raw audio; caches embeddings under ``eval/``.

        2. **Precomputed laundered** (``precomputed_root`` + ``pipeline`` +
           ``depth`` + ``strength`` all provided):
           Loads static waveforms from ``data/precomputed/``; caches embeddings
           under ``eval_{pipeline}_d{depth}_{strength}/``.  This path bypasses
           ``LaunderingEngine`` (and FFmpeg) entirely.

        3. **Live laundered** (``launder_fn`` provided, no ``precomputed_root``):
           Applies laundering on-the-fly inside DataLoader workers; embeddings
           are NOT cached (non-deterministic per call).  Used by CKA analysis
           and any legacy caller.

        Args:
            output_dir: Directory for score file and evaluation artifacts.
            launder_fn: Optional per-sample laundering callable (live path).
            precomputed_root: Root directory of precomputed laundered audio.
            pipeline: Laundering pipeline code (required with precomputed_root).
            depth: Laundering depth (required with precomputed_root).
            strength: Laundering strength (required with precomputed_root).
            max_eval: Cap on number of utterances evaluated (None = full set).
        """
        if not self._weights_loaded:
            raise RuntimeError("Call load_weights() before evaluate().")

        use_precomputed = bool(precomputed_root and pipeline and depth is not None and strength)
        use_live_launder = bool(launder_fn is not None) and not use_precomputed

        t0 = time.time()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        track = self.config.get("track", "LA")
        max_len = self.config.get("max_len_samples", 64000)

        # ── Dataset construction ──────────────────────────────────────────────
        if use_precomputed:
            # Path 2: static waveforms from precomputed_laundering.py output.
            dataset = _PrecomputedLaunderDataset(
                data_root=self.data_root,
                precomputed_root=precomputed_root,
                track=track,
                pipeline=pipeline,
                depth=depth,
                strength=strength,
                max_len=max_len,
            )
            # Switch frontend to the condition-specific cache subdirectory.
            split_key = f"eval_{pipeline}_d{depth}_{strength}"
            self._swap_frontend_cache(self._build_cache(split_key))
            use_cache_path = True
        elif use_live_launder:
            # Path 3: live FFmpeg/SciPy laundering (CKA path, legacy callers).
            base_dataset = WavDataset(self.data_root, track, split="eval", max_len=max_len)
            dataset = _LaunderedDataset(base_dataset, launder_fn)
            self._swap_frontend_cache(None)   # never cache live-laundered passes
            use_cache_path = False
        else:
            # Path 1: clean audio.
            dataset = WavDataset(self.data_root, track, split="eval", max_len=max_len)
            self._swap_frontend_cache(self._build_cache("eval"))
            use_cache_path = True

        if max_eval is not None:
            dataset.trials = dataset.trials[:max_eval]

        loader = DataLoader(
            dataset,
            batch_size=self.config.get("batch_size", 32),
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device == "cuda"),
            persistent_workers=True,
            prefetch_factor=4,
        )

        fname_list, score_list, src_list, key_list = [], [], [], []
        self.backend.eval()
        with torch.inference_mode():
            with torch.amp.autocast('cuda', dtype=torch.float16):
                for batch_x, utt_ids, srcs, keys in tqdm(loader, desc=f"{self.config['model_type']}|{self._tag()}"):
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    layer_states = self._frontend_forward(batch_x, utt_ids, use_cache_path)
                    scores = self._forward(layer_states)[:, 0].cpu().numpy()
                    fname_list.extend(utt_ids); score_list.extend(scores.tolist())
                    src_list.extend(srcs); key_list.extend(keys)

        if use_cache_path and self.use_cache and self.frontend.cache is not None:
            stats = self.frontend.cache.stats()
            cache_label = "clean eval" if not use_precomputed else f"{pipeline}_d{depth}_{strength}"
            print(f"  [CACHE] {cache_label} hits={stats['hits']} misses={stats['misses']}")

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
                            shuffle=False, num_workers=self.num_workers,
                            pin_memory=(self.device == "cuda"),
                            persistent_workers=True, prefetch_factor=4)
        all_embs = {l: [] for l in all_layers}
        # CKA always runs on the live path (launder_fn may be non-None).
        # Clean CKA baseline (launder_fn=None) can use the cache.
        use_cache_path = launder_fn is None
        if use_cache_path:
            self._swap_frontend_cache(self._build_cache("eval"))
        else:
            self._swap_frontend_cache(None)
        with torch.inference_mode():
            with torch.amp.autocast('cuda', dtype=torch.float16):
                for batch_x, utt_ids, *_ in tqdm(loader, desc="Extracting embeddings"):
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    layer_states = self._frontend_forward(batch_x, utt_ids, use_cache_path)
                    for l, emb in self.frontend.mean_pool(layer_states).items():
                        all_embs[l].append(emb.cpu().numpy())
        if use_cache_path and self.use_cache and self.frontend.cache is not None:
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
