"""Cross-platform setup and validation for SPIB and QUT-NOISE datasets."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path


SPIB_URLS = {
    "white.mat": "http://spib.linse.ufsc.br/data/noise/white.mat",
    "pink.mat": "http://spib.linse.ufsc.br/data/noise/pink.mat",
    "babble.mat": "http://spib.linse.ufsc.br/data/noise/babble.mat",
    "factory1.mat": "http://spib.linse.ufsc.br/data/noise/factory1.mat",
    "factory2.mat": "http://spib.linse.ufsc.br/data/noise/factory2.mat",
    "buccaneer1.mat": "http://spib.linse.ufsc.br/data/noise/buccaneer1.mat",
    "buccaneer2.mat": "http://spib.linse.ufsc.br/data/noise/buccaneer2.mat",
    "f16.mat": "http://spib.linse.ufsc.br/data/noise/f16.mat",
    "destroyerengine.mat": "http://spib.linse.ufsc.br/data/noise/destroyerengine.mat",
    "destroyerops.mat": "http://spib.linse.ufsc.br/data/noise/destroyerops.mat",
    "leopard.mat": "http://spib.linse.ufsc.br/data/noise/leopard.mat",
    "m109.mat": "http://spib.linse.ufsc.br/data/noise/m109.mat",
    "machinegun.mat": "http://spib.linse.ufsc.br/data/noise/machinegun.mat",
    "volvo.mat": "http://spib.linse.ufsc.br/data/noise/volvo.mat",
    "hfchannel.mat": "http://spib.linse.ufsc.br/data/noise/hfchannel.mat",
}

QUT_URLS = {
    "qutnoise.zip": "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/8342a090-89e7-4402-961e-1851da11e1aa/download/qutnoise.zip",
    "qutnoisecafe.zip": "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/9b0f10ed-e3f5-40e7-b503-73c2943abfb1/download/qutnoisecafe.zip",
    "qutnoisecar.zip": "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/7412452a-92e9-4612-9d9a-6b00f167dc15/download/qutnoisecar.zip",
    "qutnoisehome.zip": "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/35cd737a-e6ad-4173-9aee-a1768e864532/download/qutnoisehome.zip",
    "qutnoisereverb.zip": "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/164d38a5-c08e-4e20-8272-793534eb10c7/download/qutnoisereverb.zip",
    "qutnoisestreet.zip": "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/10eeceae-9f0c-4556-b33a-dcf35c4f4db9/download/qutnoisestreet.zip",
}

QUT_REFERER = "https://research.qut.edu.au/saivt/databases/qut-noise-databases-and-protocols/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1], type=Path)
    parser.add_argument("--datasets", nargs="+", choices=["SPIB", "QUT"], default=["SPIB", "QUT"])
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--force-redownload", action="store_true")
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()

    print(f"[DOWNLOAD] {destination.name}")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
    }
    if "researchdatafinder.qut.edu.au" in url:
        headers["Referer"] = QUT_REFERER

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    partial.replace(destination)


def validate_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            return bad is None
    except zipfile.BadZipFile:
        return False


def extract_zip(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(destination)


def ensure_spib(root: Path, validate_only: bool, force_redownload: bool) -> None:
    spib_root = root / "data" / "noise" / "SPIB"
    spib_root.mkdir(parents=True, exist_ok=True)
    missing = []
    for filename, url in SPIB_URLS.items():
        target = spib_root / filename
        if target.exists() and not force_redownload:
            continue
        if validate_only:
            missing.append(filename)
            continue
        download_file(url, target)

    if validate_only:
        if missing:
            raise FileNotFoundError(f"Missing SPIB files: {', '.join(missing)}")
        print("[OK] SPIB files present")
    else:
        print(f"[OK] SPIB ready at {spib_root}")


def ensure_qut(root: Path, validate_only: bool, force_redownload: bool, max_retries: int) -> None:
    downloads = root / "data" / "noise" / "downloads" / "QUT-NOISE"
    extract_root = root / "data" / "noise" / "QUT-NOISE"
    downloads.mkdir(parents=True, exist_ok=True)
    extract_root.mkdir(parents=True, exist_ok=True)

    for filename, url in QUT_URLS.items():
        archive = downloads / filename
        valid = archive.exists() and validate_zip(archive)
        if force_redownload and archive.exists():
            archive.unlink()
            valid = False

        if not valid:
            if validate_only:
                raise FileNotFoundError(f"Missing or corrupt QUT archive: {archive}")

            for attempt in range(1, max_retries + 1):
                if archive.exists():
                    archive.unlink()
                download_file(url, archive)
                if validate_zip(archive):
                    break
                print(f"[WARN] Invalid archive after download: {archive.name} (attempt {attempt}/{max_retries})")
                time.sleep(1)
            else:
                raise RuntimeError(f"Could not obtain a valid QUT archive: {archive}")

        print(f"[EXTRACT] {archive.name}")
        extract_zip(archive, extract_root)

    qut_wavs = list(extract_root.rglob("*.wav"))
    if not qut_wavs:
        raise RuntimeError(f"No QUT wav files found after extraction under {extract_root}")
    print(f"[OK] QUT ready at {extract_root} ({len(qut_wavs)} wav files)")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    try:
        if "SPIB" in args.datasets:
            ensure_spib(repo_root, args.validate_only, args.force_redownload)
        if "QUT" in args.datasets:
            ensure_qut(repo_root, args.validate_only, args.force_redownload, args.max_retries)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
