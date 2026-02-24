import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve

from .utils import add_noise_at_snr, load_noise, resolve_strength


def stage_P1(wav: np.ndarray, sr: int, p: dict) -> np.ndarray:
    """Loudspeaker bandpass coloration - fixed across all strengths."""
    sos = butter(
        N=p["bp_order"],
        Wn=[p["bp_low_hz"], p["bp_high_hz"]],
        btype="band",
        fs=sr,
        output="sos",
    )
    return sosfilt(sos, wav).astype(np.float32)


def stage_P2(wav: np.ndarray, sr: int, p: dict) -> np.ndarray:
    """Room reverberation via pyroomacoustics Image Source Method.
    Only RT60 varies with strength; room geometry is fixed for reproducibility.
    """
    try:
        import pyroomacoustics as pra
    except ImportError:
        raise ImportError("pip install pyroomacoustics")

    room_dim = [6.0, 5.0, 3.0]  # metres - fixed shoebox
    e_abs, max_order = pra.inverse_sabine(p["rt60"], room_dim)

    room = pra.ShoeBox(
        room_dim,
        fs=sr,
        materials=pra.Material(e_abs),
        max_order=max_order,
        air_absorption=True,
        ray_tracing=False,
    )
    room.add_source([2.0, 2.0, 1.5])
    room.add_microphone([4.0, 3.0, 1.5])
    room.compute_rir()

    rir = np.array(room.rir[0][0], dtype=np.float32)
    out = fftconvolve(wav, rir)[: len(wav)].astype(np.float32)

    # Preserve RMS level - convolution amplifies energy
    orig_rms = np.sqrt(np.mean(wav**2) + 1e-9)
    out_rms = np.sqrt(np.mean(out**2) + 1e-9)
    if out_rms > 1e-6:
        out = out * (orig_rms / out_rms)

    return np.clip(out, -1.0, 1.0).astype(np.float32)


def stage_P3(wav: np.ndarray, sr: int, p: dict) -> np.ndarray:
    """Microphone highpass coloration + environmental noise at target SNR."""
    sos = butter(
        N=p["hp_order"], Wn=p["hp_cutoff_hz"], btype="high", fs=sr, output="sos"
    )
    wav = sosfilt(sos, wav).astype(np.float32)
    rng = np.random.default_rng(seed=42)
    return add_noise_at_snr(wav, load_noise(p.get("noise_dir"), rng), p["snr_db"])


_STAGES = [stage_P1, stage_P2, stage_P3]


def apply(
    wav: np.ndarray, sr: int, depth: int, strength: str, params: dict
) -> np.ndarray:
    for i in range(depth):
        p = resolve_strength(params["stages"][f"P{i + 1}"], strength)
        p["noise_dir"] = params.get("noise_dir")
        wav = _STAGES[i](wav, sr, p)
    return wav
