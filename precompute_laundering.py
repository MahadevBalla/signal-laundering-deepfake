"""
precompute_laundering.py

Precompute laundered audio datasets offline so train_stab.py and eval_suite.py
can load them directly, removing the online CPU laundering bottleneck.

All processing is delegated to src.laundering.LaunderingEngine — no laundering
logic is reproduced here. Outputs are bit-identical to the online pipeline.

Output layout mirrors the original dataset exactly (drop-in replacement):
  <output_root>/<pipeline>_d<depth>_<strength>/
    ASVspoof2019_LA_train/flac/*.flac
    ASVspoof2019_LA_dev/flac/*.flac
    ASVspoof2019_LA_eval/flac/*.flac
    ASVspoof2019_LA_cm_protocols/  (copied from original)
    ASVspoof2019_LA_asv_protocols/ (copied from original)
    ASVspoof2019_LA_asv_scores/    (copied from original)
    README.LA.txt                  (copied from original)
    ASVspoof2019_LA_*/LICENSE.txt  (copied from original)

Usage
-----
python precompute_laundering.py \
    --data_root   data/ASVspoof2019/LA \
    --pipeline    N \
    --depths      1 2 3 \
    --strength    M \
    --output_root data/precomputed \
    --config_dir  configs \
    --workers     8
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Iterator, NamedTuple

import soundfile as sf
from tqdm import tqdm

log = logging.getLogger("precompute")

_AUDIO_SUFFIXES: frozenset[str] = frozenset({".flac", ".wav"})

_SPLIT_DIRS: tuple[str, ...] = (
    "ASVspoof2019_LA_train",
    "ASVspoof2019_LA_dev",
    "ASVspoof2019_LA_eval",
)

# Top-level non-audio items to copy into every variant root.
_NON_AUDIO_ITEMS: tuple[str, ...] = (
    "ASVspoof2019_LA_cm_protocols",
    "ASVspoof2019_LA_asv_protocols",
    "ASVspoof2019_LA_asv_scores",
    "README.LA.txt",
)


class WorkItem(NamedTuple):
    src: str
    dst: str


class VariantResult(NamedTuple):
    n_ok: int
    n_skipped: int
    failures: list[tuple[str, str]]


# Worker-local engine singleton — created once per process in _worker_init.
_worker_engine = None


def _worker_init(config_dir: str) -> None:
    # Runs once per worker process. Engine construction (YAML parsing etc.)
    # happens here instead of per-file, saving ~20ms * N_files of overhead.
    global _worker_engine
    from src.laundering import LaunderingEngine  # type: ignore
    _worker_engine = LaunderingEngine(config_dir=config_dir)
    logging.getLogger("src.laundering").setLevel(logging.WARNING)


def _process_chunk(
    items: list[tuple[str, str]],
    pipeline: str,
    depth: int,
    strength: str,
) -> list[tuple[str, str]]:
    # Process a batch of (src, dst) pairs. Returns (src, status) for each.
    # status: "ok" | "skipped" | "failed: <msg>"
    results: list[tuple[str, str]] = []
    engine = _worker_engine
    assert engine is not None

    for src_str, dst_str in items:
        dst = Path(dst_str)
        if dst.exists():
            results.append((src_str, "skipped"))
            continue

        tmp_dst = dst.with_suffix(dst.suffix + ".part")
        try:
            wav, sr = sf.read(src_str, dtype="float32")
            if wav.ndim > 1:
                wav = wav[:, 0]

            laundered = engine.apply_sample(wav, sr, pipeline, depth, strength)

            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(tmp_dst), laundered, sr, format="FLAC", subtype="PCM_16")
            tmp_dst.rename(dst)
            results.append((src_str, "ok"))

        except Exception as exc:  # noqa: BLE001
            tmp_dst.unlink(missing_ok=True)
            results.append((src_str, f"failed: {exc}"))

    return results


def _chunked(items: list[WorkItem], size: int) -> Iterator[list[tuple[str, str]]]:
    for i in range(0, len(items), size):
        yield [(w.src, w.dst) for w in items[i : i + size]]


def collect_audio_files(data_root: Path) -> list[Path]:
    files: list[Path] = []
    for split in _SPLIT_DIRS:
        split_dir = data_root / split
        if not split_dir.is_dir():
            log.warning("Split dir not found, skipping: %s", split_dir)
            continue
        for p in sorted(split_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in _AUDIO_SUFFIXES:
                files.append(p)
    return files


def _copy_non_audio_assets(data_root: Path, variant_root: Path) -> None:
    # Copy protocol dirs, README, and LICENSE files so the variant root is a
    # fully self-contained drop-in. Always copies — no symlinks so rsync/scp
    # to HPC works without any special flags.
    for name in _NON_AUDIO_ITEMS:
        src = data_root / name
        if not src.exists():
            continue
        dst = variant_root / name
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))

    # LICENSE.txt lives inside each split dir
    for lic_src in sorted(data_root.rglob("**/LICENSE.txt")):
        lic_dst = variant_root / lic_src.relative_to(data_root)
        if lic_dst.exists():
            continue
        lic_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(lic_src), str(lic_dst))


def dst_for(src: Path, data_root: Path, variant_root: Path) -> Path:
    return variant_root / src.relative_to(data_root)


def process_variant(
    *,
    all_src: list[Path],
    data_root: Path,
    output_root: Path,
    pipeline: str,
    depth: int,
    strength: str,
    config_dir: str,
    workers: int,
    chunk_size: int,
) -> VariantResult:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    variant_tag = f"{pipeline}_d{depth}_{strength}"
    variant_root = output_root / variant_tag
    variant_root.mkdir(parents=True, exist_ok=True)

    _copy_non_audio_assets(data_root, variant_root)

    work: list[WorkItem] = []
    n_pre_skipped = 0
    for src in all_src:
        dst = dst_for(src, data_root, variant_root)
        if dst.exists():
            n_pre_skipped += 1
        else:
            work.append(WorkItem(str(src), str(dst)))

    log.info(
        "Variant %s | total=%d  to_process=%d  pre-skipped=%d",
        variant_tag, len(all_src), len(work), n_pre_skipped,
    )

    n_ok = 0
    n_skipped = n_pre_skipped
    failures: list[tuple[str, str]] = []

    if not work:
        return VariantResult(n_ok, n_skipped, failures)

    # fork is faster on Linux (no fresh interpreter startup per worker).
    # Only switch to spawn if you see ffmpeg-related deadlocks.
    ctx = mp.get_context("fork")
    t0 = time.monotonic()

    with tqdm(total=len(work), desc=variant_tag, unit="file", dynamic_ncols=True, smoothing=0.1) as pbar:
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_worker_init, initargs=(config_dir,)) as pool:
            futs = {
                pool.submit(_process_chunk, chunk, pipeline, depth, strength): chunk
                for chunk in _chunked(work, chunk_size)
            }

            for fut in as_completed(futs):
                try:
                    chunk_results: list[tuple[str, str]] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    for src_str, _ in futs[fut]:
                        failures.append((src_str, f"worker crash: {exc}"))
                    pbar.update(len(futs[fut]))
                    continue

                for src_str, status in chunk_results:
                    if status == "ok":
                        n_ok += 1
                    elif status == "skipped":
                        n_skipped += 1
                    else:
                        failures.append((src_str, status))
                    pbar.update(1)

                elapsed = time.monotonic() - t0
                done = n_ok + n_skipped + len(failures)
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (len(all_src) - done) / rate if rate > 0 else float("inf")
                pbar.set_postfix(
                    ok=n_ok, skip=n_skipped, fail=len(failures),
                    rate=f"{rate:.1f}f/s",
                    eta=f"{eta:.0f}s" if eta != float("inf") else "?",
                    refresh=False,
                )

    elapsed = time.monotonic() - t0
    rate = (n_ok + n_skipped) / elapsed if elapsed > 0 else 0.0
    log.info(
        "Variant %s done in %.1fs (%.1f f/s) | ok=%d  skipped=%d  failed=%d",
        variant_tag, elapsed, rate, n_ok, n_skipped, len(failures),
    )
    return VariantResult(n_ok, n_skipped, failures)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root",   default="data/ASVspoof2019/LA")
    p.add_argument("--pipeline",    required=True, choices=["N", "M", "P"])
    p.add_argument("--depths",      nargs="+", type=int, default=[1, 2, 3], metavar="D")
    p.add_argument("--strength",    default="M", choices=["L", "M", "H"])
    p.add_argument("--output_root", default="data/precomputed")
    p.add_argument("--config_dir",  default="configs")
    p.add_argument("--workers",     type=int, default=os.cpu_count() or 1)
    p.add_argument("--chunk_size",  type=int, default=32,
                   help="Files per worker dispatch. Larger = less IPC overhead, less load-balance.")
    return p.parse_args(argv)


def _setup_logging() -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(ch)


def main(argv: list[str] | None = None) -> None:
    _setup_logging()
    args = parse_args(argv)

    data_root   = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    config_dir  = str(Path(args.config_dir).resolve())

    if not data_root.is_dir():
        log.error("data_root does not exist: %s", data_root)
        sys.exit(1)

    bad = [d for d in args.depths if not (1 <= d <= 3)]
    if bad:
        log.error("--depths must be in [1, 3]. Invalid: %s", bad)
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    log.info("data_root   : %s", data_root)
    log.info("output_root : %s", output_root)
    log.info("pipeline    : %s", args.pipeline)
    log.info("depths      : %s", args.depths)
    log.info("strength    : %s", args.strength)
    log.info("workers     : %d", args.workers)
    log.info("chunk_size  : %d", args.chunk_size)

    log.info("Scanning %s ...", data_root)
    all_src = collect_audio_files(data_root)
    if not all_src:
        log.error("No audio files found under %s", data_root)
        sys.exit(1)
    log.info("Found %d audio files.", len(all_src))

    grand_ok = grand_skip = grand_fail = 0
    t_total = time.monotonic()

    for depth in args.depths:
        result = process_variant(
            all_src=all_src,
            data_root=data_root,
            output_root=output_root,
            pipeline=args.pipeline,
            depth=depth,
            strength=args.strength,
            config_dir=config_dir,
            workers=args.workers,
            chunk_size=args.chunk_size,
        )
        grand_ok   += result.n_ok
        grand_skip += result.n_skipped
        grand_fail += len(result.failures)

        if result.failures:
            log.warning("Failures for depth=%d (%d total):", depth, len(result.failures))
            for src_path, err in result.failures[:20]:
                log.warning("  %s -> %s", src_path, err)
            if len(result.failures) > 20:
                log.warning("  ... and %d more.", len(result.failures) - 20)

    elapsed = time.monotonic() - t_total
    log.info("=" * 56)
    log.info("Done in %.1fs | ok=%d  skipped=%d  failed=%d", elapsed, grand_ok, grand_skip, grand_fail)
    if grand_fail:
        log.warning("%d files failed. Check logs above.", grand_fail)
        sys.exit(2)


if __name__ == "__main__":
    main()
