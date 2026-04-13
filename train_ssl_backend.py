"""
train_ssl_backend.py

Modes (--mode):
  single    FFN on one transformer layer. Used by layer_sweep.py.
            Saves: models/<model>_ffn_layer<L>.pth
  weighted  All layers with learnable softmax weights.
            Saves: models/<model>_<backend>_weighted.pth

Backends (--backend):
  ffn      Works in single and weighted mode.
  aasist   Weighted mode only.
  rawnet2  Weighted mode only.

Training: Adam lr=1e-4, batch=32, CE loss, 50 epochs,
early stopping patience=10 on dev loss.

Usage:
    python train_ssl_backend.py --model wav2vec2 --mode single --layer 3
    python train_ssl_backend.py --model wav2vec2 --mode weighted --backend ffn
    python train_ssl_backend.py --model hubert --mode weighted --backend aasist
    python train_ssl_backend.py --model wavlm --mode weighted --backend rawnet2 --dry_run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.ssl_frontend import SSLFrontend
from src.models.dataset import WavDataset
from src.models.backends import (
    FFNBackend,
    WeightedAggregationBackend,
    SSLWithAASIST,
    SSLWithRawNet2,
)


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Passing `gradient_checkpointing`")

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

LABEL_MAP = {"bonafide": 0, "spoof": 1}


def _run_epoch(backend, frontend, loader, optimizer, criterion, device, mode, layer, backend_type, train):
    """Run one training or validation epoch and return loss/accuracy."""
    backend.train(train)
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_x, _ids, _srcs, keys in tqdm(loader, desc="train" if train else "dev  ", leave=False):
            with torch.no_grad():
                layer_states = frontend(batch_x)
            labels = torch.tensor([LABEL_MAP[k] for k in keys], dtype=torch.long, device=device)
            logits = backend(layer_states[layer]) if (mode == "single" and backend_type == "ffn") else backend(layer_states)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


def parse_args() -> argparse.Namespace:
    """Parse CLI options for SSL backend training."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",      required=True, choices=["wav2vec2", "hubert", "wavlm"])
    p.add_argument("--mode",       required=True, choices=["single", "weighted"])
    p.add_argument("--backend",    default="ffn", choices=["ffn", "aasist", "rawnet2"])
    p.add_argument("--layer",      type=int,   default=None)
    p.add_argument("--data_root",  default="data/ASVspoof2019/LA")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--patience",   type=int,   default=10)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--dropout",    type=float, default=0.2)
    p.add_argument("--embed_dim",  type=int,   default=768)
    p.add_argument("--num_layers", type=int,   default=12)
    p.add_argument("--max_train",  type=int,   default=None)
    p.add_argument("--max_dev",    type=int,   default=None)
    p.add_argument("--dry_run",    action="store_true")
    return p.parse_args()


def main() -> None:
    """Train the selected backend and save the best checkpoint by dev loss."""
    args = parse_args()

    if args.mode == "single" and args.layer is None:
        raise ValueError("--layer required for --mode single")
    if args.mode == "single" and args.backend != "ffn":
        raise ValueError("--mode single only supports --backend ffn")

    if args.dry_run:
        args.max_train = min(args.max_train or 200, 200)
        args.max_dev   = min(args.max_dev   or 200, 200)
        args.epochs    = min(args.epochs, 2)
        print(f"[DRY RUN] max_train={args.max_train}  max_dev={args.max_dev}  epochs={args.epochs}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  model={args.model}  mode={args.mode}  backend={args.backend}"
          + (f"  layer={args.layer}" if args.mode == "single" else ""))

    extract = [args.layer] if args.mode == "single" else list(range(args.num_layers))
    frontend = SSLFrontend(model_type=args.model, extract_layers=extract, device=device)
    frontend.eval()
    for p in frontend.parameters():
        p.requires_grad_(False)

    if args.backend == "ffn":
        if args.mode == "single":
            backend: nn.Module = FFNBackend(embed_dim=args.embed_dim, dropout=args.dropout).to(device)
            save_name = f"{args.model}_ffn_layer{args.layer}.pth"
        else:
            backend = WeightedAggregationBackend(num_layers=args.num_layers, embed_dim=args.embed_dim, dropout=args.dropout).to(device)
            save_name = f"{args.model}_ffn_weighted.pth"
    elif args.backend == "aasist":
        backend = SSLWithAASIST(num_layers=args.num_layers, embed_dim=args.embed_dim).to(device)
        save_name = f"{args.model}_aasist_weighted.pth"
    elif args.backend == "rawnet2":
        backend = SSLWithRawNet2(num_layers=args.num_layers, embed_dim=args.embed_dim).to(device)
        save_name = f"{args.model}_rawnet2_weighted.pth"
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    print(f"Trainable params: {sum(p.numel() for p in backend.parameters() if p.requires_grad):,}")

    data_root = Path(args.data_root)
    train_ds = WavDataset(data_root, "LA", split="train", max_len=64000)
    dev_ds   = WavDataset(data_root, "LA", split="dev",   max_len=64000)
    if args.max_train:
        train_ds.trials = train_ds.trials[:args.max_train]
    if args.max_dev:
        dev_ds.trials = dev_ds.trials[:args.max_dev]
    print(f"Train: {len(train_ds)}  Dev: {len(dev_ds)}")

    kw = dict(batch_size=args.batch_size, num_workers=4, pin_memory=(device == "cuda"))
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    dev_loader   = DataLoader(dev_ds,   shuffle=False, **kw)

    optimizer = torch.optim.Adam(backend.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    save_path = Path("models") / save_name
    save_path.parent.mkdir(exist_ok=True)

    best_dev_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _run_epoch(backend, frontend, train_loader, optimizer, criterion, device, args.mode, args.layer, args.backend, True)
        dev_loss,   dev_acc   = _run_epoch(backend, frontend, dev_loader,   None,      criterion, device, args.mode, args.layer, args.backend, False)
        print(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}/{train_acc:.4f}  dev={dev_loss:.4f}/{dev_acc:.4f}")

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            patience_counter = 0
            torch.save(backend.state_dict(), save_path)
            print(f"  -> saved {save_path}")
        else:
            patience_counter += 1
            print(f"  -> no improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"[EARLY STOP] epoch {epoch}")
                break

    if hasattr(backend, "get_layer_weights"):
        backend.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        print("\nLayer weights:")
        for l, wt in sorted(backend.get_layer_weights().items()):
            print(f"  Layer {l:2d}: {wt:.4f}  {'|' * max(1, int(wt * 60))}")

    print(f"\n[DONE] {save_path}  (best dev_loss={best_dev_loss:.4f})")


if __name__ == "__main__":
    main()
