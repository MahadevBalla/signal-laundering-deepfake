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
