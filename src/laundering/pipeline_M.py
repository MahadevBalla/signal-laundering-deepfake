import numpy as np

from .utils import ffmpeg_roundtrip, resolve_strength

# Hardcoded - different psychoacoustic model at each stage is mandatory
_CODEC_CHAIN = ["libmp3lame", "aac", "libopus"]


def apply(
    wav: np.ndarray, sr: int, depth: int, strength: str, params: dict
) -> np.ndarray:
    for i in range(depth):
        p = resolve_strength(params["stages"][f"M{i + 1}"], strength)
        codec = _CODEC_CHAIN[i]
        wav = ffmpeg_roundtrip(wav, sr, codec=codec, bitrate=p["bitrate"])
    return wav
