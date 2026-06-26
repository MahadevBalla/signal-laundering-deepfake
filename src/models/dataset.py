"""Dataset utilities for ASVspoof waveform loading.

This module provides a shared dataset class used by training and evaluation
scripts so data handling stays consistent across workflows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


class WavDataset(Dataset):
    """
    Reads ASVspoof2019 trial protocol and returns waveform tensors.
    Works for train / dev / eval splits.
    """

    def __init__(self, data_root: Path, track: str, split: str = "eval", max_len: int = 64000):
        """Load protocol entries and prepare FLAC paths for one split."""
        self.max_len = max_len
        split_map = {"train": "train", "dev": "dev", "eval": "eval"}
        flac_dir = data_root / f"ASVspoof2019_{track}_{split_map[split]}" / "flac"
        protocol_map = {
            "train": "ASVspoof2019.LA.cm.train.trn.txt",
            "dev":   "ASVspoof2019.LA.cm.dev.trl.txt",
            "eval":  "ASVspoof2019.LA.cm.eval.trl.txt",
        }
        protocol = data_root / f"ASVspoof2019_{track}_cm_protocols" / protocol_map[split]
        self.flac_dir = flac_dir
        self.trials = []
        for line in protocol.read_text().strip().splitlines():
            parts = line.strip().split()
            self.trials.append((parts[1], parts[3], parts[4]))

    def __len__(self):
        """Return the number of protocol entries in this split."""
        return len(self.trials)

    def __getitem__(self, idx):
        """Return padded/truncated waveform with metadata for one utterance."""
        utt_id, src, key = self.trials[idx]
        wav, _ = sf.read(str(self.flac_dir / f"{utt_id}.flac"), dtype="float32")
        if len(wav) > self.max_len:
            wav = wav[: self.max_len]
        else:
            wav = np.pad(wav, (0, self.max_len - len(wav)))
        return torch.tensor(wav, dtype=torch.float32), utt_id, src, key


class PairedLaunderDataset(Dataset):
    """
    Returns a (clean, laundered) waveform pair for every utterance index.

    Both waveforms always correspond to the same utterance — alignment is
    structural and holds under any DataLoader shuffle order, because a single
    __getitem__ call loads both files for the same idx.

    Condition assignment and epoch diversity
    ----------------------------------------
    At construction the per-utterance condition (pipeline, depth) is assigned
    for epoch 0. Call set_epoch(epoch) once before each epoch to re-randomise
    the assignment with a seed derived from (base_seed, epoch). This gives a
    different laundering condition per utterance per epoch, matching the
    diversity of the original per-batch online sampling, while preserving
    perfect clean↔laundered alignment.

    The assignment is fully deterministic: given the same seed and epoch number
    the mapping is identical across runs and resumes.

    Directory layout expected from precomputed_root (produced by
    precompute_laundering.py):
        <precomputed_root>/<PIPELINE>_d<DEPTH>_M/

    Each sub-directory must be a valid WavDataset root (same ASVspoof
    protocol layout) containing exactly the same utterances as data_root.
    An AssertionError is raised at construction time if they diverge.

    Returns per item:
        clean_wav   : float32 tensor [max_len]
        laund_wav   : float32 tensor [max_len]
        utt_id      : str
        src         : str  (attack system or "-" for bonafide)
        key         : str  ("bonafide" or "spoof")
    """

    def __init__(
        self,
        data_root: Path,
        track: str,
        precomputed_root: Path,
        pipelines: list[str],
        depths: list[int],
        split: str = "train",
        max_len: int = 64000,
        seed: int = 42,
    ) -> None:
        self._clean_ds = WavDataset(data_root, track, split=split, max_len=max_len)
        # Expose trials so callers can inspect / truncate utterance lists.
        self.trials = self._clean_ds.trials

        # Build one WavDataset per (pipeline, depth) condition and verify
        # that utterance IDs match the clean dataset exactly.
        self._conditions: list[tuple[str, int]] = [
            (p, d) for p in pipelines for d in depths
        ]
        if not self._conditions:
            raise ValueError("pipelines and depths must each be non-empty.")

        self._precomp_ds: dict[tuple[str, int], WavDataset] = {}
        clean_ids = [t[0] for t in self._clean_ds.trials]
        for cond in self._conditions:
            pipeline, depth = cond
            subdir = precomputed_root / f"{pipeline}_d{depth}_M"
            if not subdir.exists():
                raise FileNotFoundError(
                    f"Precomputed dataset not found: {subdir}. "
                    "Run precompute_laundering.py first, or omit "
                    "--precomputed_root to use online laundering."
                )
            ds = WavDataset(subdir, track, split=split, max_len=max_len)
            precomp_ids = [t[0] for t in ds.trials]
            if clean_ids != precomp_ids:
                raise AssertionError(
                    f"Utterance ID mismatch between clean {split} dataset and "
                    f"precomputed dataset at {subdir}. "
                    "Ensure precompute_laundering.py processed the same protocol "
                    "in the same order."
                )
            self._precomp_ds[cond] = ds

        # Pre-assign one condition to each sample for epoch 0.
        # Call set_epoch(epoch) at the start of each training epoch to
        # re-randomise the assignment, giving a different laundering condition
        # per utterance per epoch without breaking clean↔laundered alignment.
        self._base_seed = seed
        self._assign_conditions(epoch=0)

    def _assign_conditions(self, epoch: int) -> None:
        """Sample per-utterance conditions for the given epoch (internal)."""
        rng = np.random.default_rng((self._base_seed, epoch))
        n = len(self._clean_ds)
        cond_indices = rng.integers(0, len(self._conditions), size=n)
        self._assigned: list[tuple[str, int]] = [
            self._conditions[int(i)] for i in cond_indices
        ]

    def set_epoch(self, epoch: int) -> None:
        """Re-randomise the per-utterance condition assignment for a new epoch.

        Call once per epoch (before iterating the DataLoader) to ensure each
        utterance is paired with a different laundering condition every epoch.
        The seed is derived from (base_seed, epoch) so the assignment is
        deterministic and reproducible across runs and resumes.

        Safe to call on a dev dataset with a single condition; the resulting
        assignment is identical regardless of epoch number.
        """
        self._assign_conditions(epoch)

    def __len__(self) -> int:
        return len(self._clean_ds)

    def __getitem__(self, idx: int):
        """Return (clean_wav, laund_wav, utt_id, src, key) for utterance idx."""
        clean_wav, utt_id, src, key = self._clean_ds[idx]
        cond = self._assigned[idx]
        laund_wav, _, _, _ = self._precomp_ds[cond][idx]
        return clean_wav, laund_wav, utt_id, src, key
