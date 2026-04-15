"""Shared audio utility functions used by laundering pipelines."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.io import loadmat
from scipy.signal import resample_poly

SR = 16000

_CODEC_EXT = {
    "libopus": "ogg",
    "libmp3lame": "mp3",
    "aac": "aac",
}


def ffmpeg_roundtrip(wav: np.ndarray, sr: int, codec: str, bitrate: int) -> np.ndarray:
    """Encode and decode audio with ffmpeg, then restore original length."""
    ext = _CODEC_EXT[codec]
    ffmpeg_exe = _resolve_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        enc = Path(tmp) / f"enc.{ext}"
        dst = Path(tmp) / "out.wav"

        sf.write(str(src), wav, sr, subtype="PCM_16")
        subprocess.run(
            [
                ffmpeg_exe,
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
            [ffmpeg_exe, "-y", "-i", str(enc), "-ar", str(sr), "-ac", "1", str(dst)],
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

    root = Path(noise_dir)
    files = (
        list(root.rglob("*.wav"))
        + list(root.rglob("*.flac"))
        + list(root.rglob("*.mat"))
    )
    if not files:
        return rng.standard_normal(SR * 5).astype(np.float32)

    path = Path(rng.choice(files))
    if path.suffix.lower() == ".mat":
        noise, src_sr = _load_mat_noise(path)
    else:
        noise, src_sr = sf.read(str(path), dtype="float32")

    if noise.ndim > 1:
        noise = noise[:, 0]
    if src_sr and src_sr != SR:
        noise = resample_poly(noise, SR, int(src_sr)).astype(np.float32)
    return noise.astype(np.float32)


def _load_mat_noise(path: Path) -> tuple[np.ndarray, int]:
    """Extract a 1-D waveform and optional sample rate from a MATLAB file."""
    data = loadmat(path)
    sample_rate = None

    for key in ("fs", "Fs", "sr", "SR", "sample_rate", "SampleRate"):
        value = data.get(key)
        if value is not None and np.size(value) == 1:
            sample_rate = int(np.squeeze(value))
            break

    candidates: list[np.ndarray] = []
    for key, value in data.items():
        if key.startswith("__") or not isinstance(value, np.ndarray):
            continue
        if not np.issubdtype(value.dtype, np.number):
            continue
        squeezed = np.squeeze(value)
        if squeezed.ndim == 1 and squeezed.size > 1000:
            candidates.append(squeezed.astype(np.float32))
        elif squeezed.ndim == 2 and 1 in squeezed.shape and squeezed.size > 1000:
            candidates.append(squeezed.reshape(-1).astype(np.float32))

    if not candidates:
        raise ValueError(f"No numeric waveform found in MATLAB noise file: {path}")

    if sample_rate is None:
        sample_rate = 19980
    return candidates[0], sample_rate


def _resolve_ffmpeg_exe() -> str:
    """Find a usable ffmpeg executable from PATH, env, or bundled Python package."""
    env_override = os.environ.get("FFMPEG_BINARY")
    if env_override:
        return env_override

    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        return path_ffmpeg

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise FileNotFoundError(
            "ffmpeg was not found on PATH and imageio-ffmpeg is not installed. "
            "Install requirements or set FFMPEG_BINARY."
        ) from exc

    return imageio_ffmpeg.get_ffmpeg_exe()


def safe_lowpass_cutoff(cutoff_hz: float, sr: int) -> float:
    """Clamp a low-pass cutoff to the valid open interval (0, sr/2)."""
    nyquist = sr / 2.0
    return float(min(cutoff_hz, nyquist - 1.0))


def safe_highpass_cutoff(cutoff_hz: float, sr: int) -> float:
    """Clamp a high-pass cutoff to the valid open interval (0, sr/2)."""
    nyquist = sr / 2.0
    return float(max(1.0, min(cutoff_hz, nyquist - 1.0)))


def safe_bandpass_cutoffs(low_hz: float, high_hz: float, sr: int) -> tuple[float, float]:
    """Clamp band-pass cutoffs so SciPy receives a strictly valid interval."""
    nyquist = sr / 2.0
    low = max(1.0, float(low_hz))
    high = min(float(high_hz), nyquist - 1.0)
    if low >= high:
        high = min(nyquist - 1.0, low + 1.0)
    return low, high


def resolve_strength(stage_params: dict, strength: str) -> dict:
    """Flatten strength-keyed dicts to scalar values for the given strength."""
    return {
        k: (v[strength] if isinstance(v, dict) else v) for k, v in stage_params.items()
    }
