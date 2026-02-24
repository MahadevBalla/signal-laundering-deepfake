"""
Evaluation metrics for the signal laundering framework.

EER and min-tDCF use the ASVspoof 2019 legacy formulation,
identical to AASIST/evaluation.py. Framework metrics (AURC, kc, ℓc)
are defined in the laundering specification.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

# ASVspoof 2019 t-DCF cost model (fixed per eval plan)
_Pspoof = 0.05
_COST = {
    "Pspoof": _Pspoof,
    "Ptar": (1 - _Pspoof) * 0.99,
    "Pnon": (1 - _Pspoof) * 0.01,
    "Cmiss_asv": 1,
    "Cfa_asv": 10,
    "Cmiss_cm": 1,
    "Cfa_cm": 10,
}

# A07–A19: spoof attack types present in ASVspoof 2019 LA eval
ATTACK_TYPES = [f"A{i:02d}" for i in range(7, 20)]

_STRENGTH_ORDER = {"L": 0, "M": 1, "H": 2}


# Core statistical functions
def compute_det_curve(
    target_scores: np.ndarray, nontarget_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )
    idx = np.argsort(all_scores, kind="mergesort")
    labels = labels[idx]
    tar_sums = np.cumsum(labels)
    non_sums = nontarget_scores.size - (np.arange(1, n + 1) - tar_sums)
    frr = np.concatenate((np.atleast_1d(0.0), tar_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1.0), non_sums / nontarget_scores.size))
    thr = np.concatenate((np.atleast_1d(all_scores[idx[0]] - 0.001), all_scores[idx]))
    return frr, far, thr


def compute_eer(
    target_scores: np.ndarray, nontarget_scores: np.ndarray
) -> tuple[float, float]:
    """Returns (EER as fraction [0,1], decision threshold)."""
    frr, far, thr = compute_det_curve(target_scores, nontarget_scores)
    i = np.argmin(np.abs(frr - far))
    return float(np.mean((frr[i], far[i]))), float(thr[i])


def _asv_error_rates(
    tar_asv: np.ndarray,
    non_asv: np.ndarray,
    spoof_asv: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    pfa = float(np.sum(non_asv >= threshold) / non_asv.size)
    pmiss = float(np.sum(tar_asv < threshold) / tar_asv.size)
    pmiss_spoof = (
        float(np.sum(spoof_asv < threshold) / spoof_asv.size)
        if spoof_asv.size > 0
        else None
    )
    return pfa, pmiss, pmiss_spoof


def compute_min_tdcf(
    bona_cm: np.ndarray,
    spoof_cm: np.ndarray,
    pfa_asv: float,
    pmiss_asv: float,
    pmiss_spoof_asv: float,
) -> float:
    """
    ASVspoof 2019 normalized min-tDCF (legacy formulation).
    Matches AASIST evaluation.py exactly.
    """
    c = _COST
    frr, far, _ = compute_det_curve(bona_cm, spoof_cm)
    C1 = (
        c["Ptar"] * (c["Cmiss_cm"] - c["Cmiss_asv"] * pmiss_asv)
        - c["Pnon"] * c["Cfa_asv"] * pfa_asv
    )
    C2 = c["Cfa_cm"] * c["Pspoof"] * (1.0 - pmiss_spoof_asv)
    tdcf = C1 * frr + C2 * far
    return float(np.min(tdcf) / min(C1, C2))


# Score file I/O
def load_cm_scores(
    score_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads: `utt_id  src  key  score`
    Returns: sources, keys, scores (all np.ndarray of str/float).
    """
    data = np.genfromtxt(score_path, dtype=str)
    return data[:, 1], data[:, 2], data[:, 3].astype(float)


def load_asv_scores(
    asv_score_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reads organizer ASV score file: `spk  key  score`"""
    data = np.genfromtxt(asv_score_path, dtype=str)
    keys = data[:, 1]
    scores = data[:, 2].astype(float)
    return (
        scores[keys == "target"],
        scores[keys == "nontarget"],
        scores[keys == "spoof"],
    )


# Full evaluation result
class EvalResult(NamedTuple):
    eer: float  # percent
    min_tdcf: float
    eer_per_attack: dict[str, float]  # A07–A19, percent
    det_frr: np.ndarray
    det_far: np.ndarray
    det_thr: np.ndarray


def evaluate_scores(
    cm_score_file: str | Path,
    asv_score_file: str | Path,
) -> EvalResult:
    """
    Full ASVspoof 2019 LA evaluation.
    Computes pooled EER, min-tDCF, per-attack EER (A07–A19), and DET curve.
    """
    sources, keys, cm_scores = load_cm_scores(cm_score_file)
    tar_asv, non_asv, spoof_asv = load_asv_scores(asv_score_file)

    bona_cm = cm_scores[keys == "bonafide"]
    spoof_cm = cm_scores[keys == "spoof"]

    # Pooled EER
    eer_asv, asv_thr = compute_eer(tar_asv, non_asv)
    eer_cm, _ = compute_eer(bona_cm, spoof_cm)

    # min-tDCF
    pfa_asv, pmiss_asv, pmiss_spoof_asv = _asv_error_rates(
        tar_asv, non_asv, spoof_asv, asv_thr
    )
    tdcf = compute_min_tdcf(bona_cm, spoof_cm, pfa_asv, pmiss_asv, pmiss_spoof_asv)

    # Per-attack EER (A07–A19)
    per_attack: dict[str, float] = {}
    for atk in ATTACK_TYPES:
        spoof_atk = cm_scores[sources == atk]
        per_attack[atk] = (
            compute_eer(bona_cm, spoof_atk)[0] * 100
            if spoof_atk.size > 0
            else float("nan")
        )

    # DET curve (for plotting)
    frr, far, thr = compute_det_curve(bona_cm, spoof_cm)

    return EvalResult(
        eer=eer_cm * 100,
        min_tdcf=tdcf,
        eer_per_attack=per_attack,
        det_frr=frr,
        det_far=far,
        det_thr=thr,
    )


# Framework-level metrics
def compute_aurc(eer_by_depth: dict[int, float]) -> float:
    """
    AURC = (1/K+1) Σ EER(k) over k=0..K at fixed Medium strength.
    Lower = more robust.
    """
    depths = sorted(eer_by_depth)
    return float(np.mean([eer_by_depth[k] for k in depths]))


def compute_aurc_trap(eer_by_depth: dict[int, float]) -> float:
    """
    Trapezoidal AURC - more precise when EER changes non-linearly with depth.
    """
    depths = sorted(eer_by_depth)
    eers = [eer_by_depth[k] for k in depths]
    K = depths[-1] - depths[0]
    if K == 0:
        return float(eers[0])
    total = sum(
        (eers[i] + eers[i + 1]) / 2.0 * (depths[i + 1] - depths[i])
        for i in range(len(depths) - 1)
    )
    return float(total / K)


def compute_collapse_depth(
    eer_by_depth: dict[int, float],
    tau: float | None = None,
    relative_factor: float = 2.0,
) -> int | float:
    """
    kc = min k ∈ {0,1,2,3} s.t. EER(k) ≥ tau.

    tau=None  → relative threshold: tau = relative_factor × EER(k=0).
    Returns float('inf') if threshold never reached within k=0..3.
    """
    depths = sorted(eer_by_depth)
    baseline = eer_by_depth[depths[0]]
    threshold = tau if tau is not None else relative_factor * baseline
    for k in depths:
        if eer_by_depth[k] >= threshold:
            return k
    return float("inf")


def compute_collapse_strength(
    eer_by_strength: dict[str, float],
    tau: float | None = None,
    relative_factor: float = 2.0,
    baseline_eer: float | None = None,
) -> str | None:
    """
    ℓc = min ℓ ∈ {L,M,H} s.t. EER(ℓ) ≥ tau.

    tau=None → relative threshold: tau = relative_factor × baseline_eer.
    Returns None if never collapsed.
    """
    if tau is None:
        if baseline_eer is None:
            raise ValueError("Provide tau or baseline_eer for relative threshold.")
        tau = relative_factor * baseline_eer
    for s in sorted(_STRENGTH_ORDER, key=_STRENGTH_ORDER.__getitem__):
        if s in eer_by_strength and eer_by_strength[s] >= tau:
            return s
    return None
