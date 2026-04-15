"""Pipeline N: network-style laundering stages."""

import numpy as np
from scipy.signal import butter, sosfilt

from .utils import (
    ffmpeg_roundtrip,
    add_noise_at_snr,
    load_noise,
    resolve_strength,
    safe_lowpass_cutoff,
)


def stage_N1(wav: np.ndarray, sr: int, p: dict) -> np.ndarray:
    """Apply Opus encode/decode roundtrip."""
    return ffmpeg_roundtrip(wav, sr, codec="libopus", bitrate=p["bitrate"])


def stage_N2(wav: np.ndarray, sr: int, p: dict) -> np.ndarray:
    """Simulate packet loss and narrowband low-pass filtering."""
    frame_len = int(sr * p["frame_ms"] / 1000)
    n_frames = len(wav) // frame_len
    rng = np.random.default_rng(seed=42)
    drop = rng.random(n_frames) < p["plr"]

    out = wav.copy()
    for i, dropped in enumerate(drop):
        if dropped:
            out[i * frame_len : (i + 1) * frame_len] = 0.0

    sos = butter(
        N=p["lp_order"],
        Wn=safe_lowpass_cutoff(p["lp_cutoff_hz"], sr),
        btype="low",
        fs=sr,
        output="sos",
    )
    return sosfilt(sos, out).astype(np.float32)


def stage_N3(wav: np.ndarray, sr: int, p: dict) -> np.ndarray:
    """Add background noise at a target SNR level."""
    rng = np.random.default_rng(seed=42)
    noise = load_noise(p.get("noise_dir"), rng)
    return add_noise_at_snr(wav, noise, p["snr_db"])


_STAGES = [stage_N1, stage_N2, stage_N3]


def apply(
    wav: np.ndarray, sr: int, depth: int, strength: str, params: dict
) -> np.ndarray:
    """Apply first `depth` stages of pipeline N at selected strength."""
    for i in range(depth):
        p = resolve_strength(params["stages"][f"N{i + 1}"], strength)
        p["noise_dir"] = params.get("noise_dir")
        wav = _STAGES[i](wav, sr, p)
    return wav
