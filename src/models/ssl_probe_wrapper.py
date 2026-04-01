"""
SSL frontend + linear probe wrapper.
Matches AASISTWrapper / RawNet2Wrapper interface exactly.
evaluate(launder_fn) → (eer, min_tdcf)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
import soundfile as sf
from tqdm import tqdm

from .ssl_frontend import SSLFrontend
from src.evaluation.metrics import evaluate_scores


class _SimpleWavDataset(Dataset):
    """
    Minimal dataset: reads trial file → returns (waveform_tensor, utt_id).
    Does NOT depend on AASIST's get_loader — avoids submodule coupling.
    """
    def __init__(self, data_root: Path, track: str, split: str = "eval", max_len: int = 64000):
        self.data_root = data_root
        self.max_len = max_len
        split_dir_map = {"train": "train", "dev": "dev", "eval": "eval"}
        flac_dir = data_root / f"ASVspoof2019_{track}_{split_dir_map[split]}"
        
        protocol_map = {"train": "trn", "dev": "dev", "eval": "eval.trl"}
        proto_suffix = protocol_map[split]
        protocol = (
            data_root
            / f"ASVspoof2019_{track}_cm_protocols"
            / f"ASVspoof2019.{track}.cm.{proto_suffix}.txt"
        )
        lines = protocol.read_text().strip().splitlines()
        self.trials = []   # (utt_id, src, key)
        self.flac_dir = flac_dir
        for line in lines:
            parts = line.strip().split()
            # format: spk  utt_id  -  src  key
            self.trials.append((parts[1], parts[3], parts[4]))

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        utt_id, src, key = self.trials[idx]
        path = self.flac_dir / "flac" / f"{utt_id}.flac"
        wav, _ = sf.read(str(path), dtype="float32")
        max_len = self.max_len
        if len(wav) > max_len:
            self._truncated_count = getattr(self, "_truncated_count", 0) + 1
            wav = wav[:max_len]
        else:
            wav = np.pad(wav, (0, max_len - len(wav)))
        return torch.tensor(wav, dtype=torch.float32), utt_id, src, key


class SSLProbeWrapper:
    """
    SSL frontend + frozen linear probe for spoof detection.

    The probe is a single linear layer trained on mean-pooled
    embeddings from a chosen layer. Weights are loaded from a
    .pth file saved after probe training (or random init for
    embedding-only analysis).
    """

    def __init__(self, config_path: str, data_root: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.data_root = Path(data_root)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # SSL frontend
        self.frontend = SSLFrontend(
            model_type=self.config["model_type"],
            extract_layers=self.config.get("extract_layers"),
            device=self.device,
        )

        # Linear probe per layer
        embed_dim = self.config["embed_dim"]          # e.g. 768 for base models
        probe_layer = self.config["probe_layer"]       # which layer to score from
        self.probe_layer = probe_layer
        self.probe = nn.Linear(embed_dim, 2).to(self.device)
        self._last_eval_result = None
        self._probe_trained = False

    def load_weights(self, weights_path: str = None):
        if weights_path and Path(weights_path).exists():
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.probe.load_state_dict(state)
            self.probe.eval()
            self._probe_trained = True
            print(f"[SSL:{self.config['model_type']}] Probe loaded: {weights_path}")
        else:
            # No probe weights = run in embedding-extraction-only mode
            print(f"[SSL:{self.config['model_type']}] No probe weights — embedding mode only")

    def evaluate(
        self,
        output_dir: str = "outputs",
        launder_fn=None,
        max_eval: int | None = None,
    ) -> tuple[float, float]:
        if not self._probe_trained:
            raise RuntimeError(
                f"[SSL:{self.config['model_type']}] Probe weights not loaded. "
                "Train probe first via train_probe.py and pass weights path. "
                "Use extract_all_layers() if you only need embeddings."
            )
        start = time.time()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        track = self.config.get("track", "LA")
        dataset = _SimpleWavDataset(self.data_root, track, split="eval",
                                    max_len=self.config.get("max_len_samples", 64000))
        if max_eval is not None:
            dataset.trials = dataset.trials[:max_eval]

        loader = DataLoader(
            dataset,
            batch_size=self.config.get("batch_size", 8),
            shuffle=False,
            num_workers=4,
        )

        eval_score_path = output_dir / "eval_scores.txt"
        asv_score_file = self.data_root / self.config["asv_score_path"]

        fname_list, score_list, src_list, key_list = [], [], [], []

        self.probe.eval()
        with torch.no_grad():
            for batch_x, utt_ids, srcs, keys in tqdm(loader, desc=f"SSL:{self.config['model_type']}"):
                if launder_fn is not None:
                    batch_x = launder_fn(batch_x)

                # Extract layer-wise embeddings
                layer_outputs = self.frontend(batch_x)          # {layer: [B, T', D]}
                pooled = self.frontend.mean_pool(layer_outputs)  # {layer: [B, D]}

                # Score using probe on chosen layer
                emb = pooled[self.probe_layer].to(self.device)  # [B, D]
                logits = self.probe(emb)                         # [B, 2]
                scores = logits[:, 1].cpu().numpy()              # spoof score

                fname_list.extend(utt_ids)
                score_list.extend(scores.tolist())
                src_list.extend(srcs)
                key_list.extend(keys)

        # Write score file (same format as AASIST)
        with open(eval_score_path, "w") as fh:
            for fn, sco, src, key in zip(fname_list, score_list, src_list, key_list):
                fh.write(f"{fn} {src} {key} {sco}\n")

        result = evaluate_scores(eval_score_path, asv_score_file)
        self._last_eval_result = result

        model_name = self.config["model_type"]
        print(f"[SSL:{model_name}] EER: {result.eer:.4f}% | min-tDCF: {result.min_tdcf:.4f}")
        print(f"[SSL:{model_name}] Eval time: {(time.time() - start)/60:.2f} min")
        return result.eer, result.min_tdcf

    def extract_all_layers(
        self,
        output_dir: str,
        launder_fn=None,
        max_eval: int | None = None,
        save_embeddings: bool = True,
    ) -> dict[int, np.ndarray]:
        """
        Extract and optionally save mean-pooled embeddings for ALL
        configured layers. Returns {layer_idx: array[N, D]}.
        Use this for Objective 2 layer stability analysis.
        """
        track = self.config.get("track", "LA")
        dataset = _SimpleWavDataset(self.data_root, track, split="eval",
                                    max_len=self.config.get("max_len_samples", 64000))
        if max_eval is not None:
            dataset.trials = dataset.trials[:max_eval]

        loader = DataLoader(dataset, batch_size=self.config.get("batch_size", 8),
                            shuffle=False, num_workers=4)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_embeddings: dict[int, list] = {
            l: [] for l in self.frontend.extract_layers
        }

        with torch.no_grad():
            for batch_x, utt_ids, srcs, keys in tqdm(loader, desc="Extracting embeddings"):
                if launder_fn is not None:
                    batch_x = launder_fn(batch_x)
                layer_outputs = self.frontend(batch_x)
                pooled = self.frontend.mean_pool(layer_outputs)
                for layer_idx, emb in pooled.items():
                    all_embeddings[layer_idx].append(emb.numpy())

        result = {
            layer_idx: np.concatenate(arrs, axis=0)
            for layer_idx, arrs in all_embeddings.items()
        }

        if save_embeddings:
            import pickle
            out_file = output_dir / "layer_embeddings.pkl"
            with open(out_file, "wb") as f:
                pickle.dump(result, f)
            print(f"Embeddings saved → {out_file}")

        return result