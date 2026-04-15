"""One-command bootstrap and end-to-end runner for SSL laundering experiments."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REQUIRED_ASVSPOOF_DIRS = [
    "ASVspoof2019_LA_train/flac",
    "ASVspoof2019_LA_dev/flac",
    "ASVspoof2019_LA_eval/flac",
    "ASVspoof2019_LA_cm_protocols",
    "ASVspoof2019_LA_asv_scores",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["wav2vec2", "hubert", "wavlm"], default="wav2vec2")
    parser.add_argument("--backend", choices=["ffn", "aasist", "rawnet2"], default="aasist")
    parser.add_argument("--data-root", default="data/ASVspoof2019/LA")
    parser.add_argument("--noise-mode", choices=["real", "white"], default="white")
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--sync-submodules", action="store_true")
    parser.add_argument("--run-cka", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


def run(cmd: list[str], repo_root: Path, env: dict[str, str] | None = None) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=repo_root, check=True, env=env)


def ensure_deps(repo_root: Path) -> None:
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], repo_root)


def ensure_runtime_deps(repo_root: Path) -> None:
    """Install repo requirements when critical runtime packages are missing."""
    missing = []
    checks = {
        "torch": "torch",
        "scipy": "scipy",
        "soundfile": "soundfile",
        "yaml": "pyyaml",
        "transformers": "transformers",
        "imageio_ffmpeg": "imageio-ffmpeg",
    }
    for module_name, package_name in checks.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        print(f"[DEPS] Missing runtime packages detected: {', '.join(missing)}")
        ensure_deps(repo_root)


def ensure_submodules(repo_root: Path) -> None:
    aasist_root = repo_root / "external" / "aasist"
    if aasist_root.exists():
        return
    run(["git", "submodule", "update", "--init", "--recursive"], repo_root)


def resolve_asvspoof_root(repo_root: Path, requested_root: str) -> Path:
    """Resolve the LA dataset root from the requested path or common local layouts."""
    candidates = [
        (repo_root / requested_root).resolve(),
        (repo_root / "data/LA/LA").resolve(),
        (repo_root / "data/ASVspoof2019/LA").resolve(),
        (repo_root / "data/LA").resolve(),
    ]

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _has_asvspoof_layout(candidate):
            return candidate

    return candidates[0]


def _has_asvspoof_layout(data_root: Path) -> bool:
    """Return True when the expected ASVspoof LA directory structure exists."""
    return all((data_root / rel).exists() for rel in REQUIRED_ASVSPOOF_DIRS)


def ensure_asvspoof(data_root: Path) -> None:
    missing = [str(data_root / rel) for rel in REQUIRED_ASVSPOOF_DIRS if not (data_root / rel).exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "ASVspoof2019 LA is missing required files. "
            "Please download it manually and place it under the requested layout:\n"
            f"{missing_text}"
        )


def ensure_noise(repo_root: Path, max_retries: int) -> None:
    run(
        [
            sys.executable,
            "scripts/noise_setup.py",
            "--datasets",
            "SPIB",
            "QUT",
            "--max-retries",
            str(max_retries),
        ],
        repo_root,
    )


def prefetch_hf_assets(model: str, env: dict[str, str]) -> None:
    """Cache the SSL frontend assets once before subprocess-heavy stages begin."""
    from transformers import AutoFeatureExtractor, AutoModel

    model_ids = {
        "wav2vec2": "facebook/wav2vec2-base",
        "hubert": "facebook/hubert-base-ls960",
        "wavlm": "microsoft/wavlm-base",
    }
    model_id = model_ids[model]
    print(f"[HF] Prefetching {model_id}")
    AutoFeatureExtractor.from_pretrained(model_id)
    AutoModel.from_pretrained(model_id)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"


def write_config_variant(repo_root: Path, noise_mode: str) -> Path:
    source_dir = repo_root / "configs"
    temp_dir = Path(tempfile.mkdtemp(prefix="laundering_cfg_"))

    for src in source_dir.glob("*_params.yaml"):
        text = src.read_text()
        if src.name == "N_params.yaml":
            replacement = "noise_dir: null" if noise_mode == "white" else "noise_dir: data/noise/SPIB"
            text = re.sub(r"^noise_dir:.*$", replacement, text, flags=re.MULTILINE)
        elif src.name == "P_params.yaml":
            replacement = "noise_dir: null" if noise_mode == "white" else "noise_dir: data/noise/QUT-NOISE"
            text = re.sub(r"^noise_dir:.*$", replacement, text, flags=re.MULTILINE)
        (temp_dir / src.name).write_text(text)

    return temp_dir


def eval_model_name(model: str, backend: str) -> str:
    if backend == "ffn":
        return model
    if backend == "aasist":
        return f"{model}_aasist"
    if model == "hubert":
        return "hubert_rawnet2_ssl"
    return f"{model}_rawnet2"


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_root = resolve_asvspoof_root(repo_root, args.data_root)

    try:
        if args.install_deps:
            ensure_deps(repo_root)
        else:
            ensure_runtime_deps(repo_root)
        if args.sync_submodules:
            ensure_submodules(repo_root)
        ensure_asvspoof(data_root)
        print(f"[DATA] Using ASVspoof root: {data_root}")
        if args.noise_mode == "real":
            ensure_noise(repo_root, args.max_retries)

        config_dir = write_config_variant(repo_root, args.noise_mode)
        env = os.environ.copy()
        env.setdefault("PYTHONPATH", str(repo_root))
        env.setdefault("HF_HOME", str(repo_root / ".cache" / "huggingface"))
        Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", env["HF_HOME"])
        os.environ.setdefault("PYTHONPATH", env["PYTHONPATH"])
        prefetch_hf_assets(args.model, env)

        try:
            if not args.skip_sweep:
                cmd = [sys.executable, "layer_sweep.py", "--model", args.model, "--data_root", args.data_root]
                cmd[-1] = str(data_root)
                if args.dry_run:
                    cmd.append("--dry_run")
                run(cmd, repo_root, env)

            if not args.skip_train:
                cmd = [
                    sys.executable,
                    "train_ssl_backend.py",
                    "--model",
                    args.model,
                    "--mode",
                    "weighted",
                    "--backend",
                    args.backend,
                    "--data_root",
                    str(data_root),
                ]
                if args.dry_run:
                    cmd.append("--dry_run")
                run(cmd, repo_root, env)

            cmd = [
                sys.executable,
                "eval_suite.py",
                "--model",
                eval_model_name(args.model, args.backend),
                "--data_root",
                str(data_root),
                "--config_dir",
                str(config_dir),
            ]
            if args.run_cka:
                cmd.append("--run_cka")
            if args.dry_run:
                cmd.append("--dry_run")
            run(cmd, repo_root, env)
        finally:
            shutil.rmtree(config_dir, ignore_errors=True)

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
