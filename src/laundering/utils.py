"""Shared audio utility functions used by laundering pipelines."""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000

_CODEC_EXT = {
    "libopus": "ogg",
    "libmp3lame": "mp3",
    "aac": "aac",
}


def ffmpeg_roundtrip(wav: np.ndarray, sr: int, codec: str, bitrate: int) -> np.ndarray:
    """Encode and decode audio with ffmpeg, then restore original length."""
    ext = _CODEC_EXT[codec]
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        enc = Path(tmp) / f"enc.{ext}"
        dst = Path(tmp) / "out.wav"

        sf.write(str(src), wav, sr, subtype="PCM_16")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
                "-c:a",
                codec,
                "-b:a",
                str(bitrate),
                str(enc),
            ],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(enc), "-ar", str(sr), "-ac", "1", str(dst)],
            capture_output=True,
            check=True,
        )
        out, _ = sf.read(str(dst), dtype="float32")

    return _match_length(out, len(wav))


def _match_length(wav: np.ndarray, n: int) -> np.ndarray:
    """Trim or pad waveform to exactly `n` samples."""
    if len(wav) >= n:
        return wav[:n]
    return np.pad(wav, (0, n - len(wav))).astype(np.float32)


def add_noise_at_snr(
    signal: np.ndarray, noise: np.ndarray, snr_db: float
) -> np.ndarray:
    """Mix noise into signal at the requested SNR in dB."""
    if len(noise) < len(signal):
        noise = np.tile(noise, int(np.ceil(len(signal) / len(noise))))
    noise = noise[: len(signal)]

    sig_rms = np.sqrt(np.mean(signal**2) + 1e-9)
    noise_rms = np.sqrt(np.mean(noise**2) + 1e-9)
    scale = sig_rms / (noise_rms * (10 ** (snr_db / 20)))

    return np.clip(signal + scale * noise, -1.0, 1.0).astype(np.float32)


def load_noise(noise_dir: str | None, rng: np.random.Generator) -> np.ndarray:
    """Load one random noise file, or return synthetic noise fallback."""
    if noise_dir is None:
        return rng.standard_normal(SR * 5).astype(np.float32)

    files = list(Path(noise_dir).glob("*.wav")) + list(Path(noise_dir).glob("*.flac"))
    if not files:
        return rng.standard_normal(SR * 5).astype(np.float32)

    noise, _ = sf.read(str(rng.choice(files)), dtype="float32")
    return noise[:, 0] if noise.ndim > 1 else noise


def resolve_strength(stage_params: dict, strength: str) -> dict:
    """Flatten strength-keyed dicts to scalar values for the given strength."""
    return {
        k: (v[strength] if isinstance(v, dict) else v) for k, v in stage_params.items()
    }
