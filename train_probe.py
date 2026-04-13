"""
train_probe.py — Legacy linear probe on frozen SSL embeddings.
Not used in the main eval flow. SSLEvalWrapper + train_ssl_backend is the current path.

Usage:
    python train_probe.py --model wav2vec2 --layer 11 --epochs 20
    python train_probe.py --model hubert --layer 8 --max_train 2000
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.ssl_frontend import SSLFrontend
from src.models.dataset import WavDataset


def parse_args():
    """Parse command-line options for legacy linear-probe training."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["wav2vec2", "hubert", "wavlm"])
    p.add_argument("--layer", type=int, default=11)
    p.add_argument("--data_root", default="data/ASVspoof2019/LA")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_train", type=int, default=None)
    return p.parse_args()


def main():
    """Train a simple linear probe on one SSL layer and save weights."""
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frontend = SSLFrontend(model_type=args.model, extract_layers=[args.layer], device=device)
    frontend.eval()

    dataset = WavDataset(data_root=Path(args.data_root), track="LA", split="train", max_len=64000)
    if args.max_train:
        dataset.trials = dataset.trials[:args.max_train]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    probe = nn.Linear(768, 2).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    label_map = {"bonafide": 0, "spoof": 1}

    for epoch in range(args.epochs):
        probe.train()
        total_loss = correct = total = 0
        for batch_x, _ids, _srcs, keys in tqdm(loader, desc=f"Epoch {epoch+1}"):
            with torch.no_grad():
                emb = frontend.mean_pool(frontend(batch_x))[args.layer].to(device)
            labels = torch.tensor([label_map[k] for k in keys], dtype=torch.long, device=device)
            optimizer.zero_grad()
            logits = probe(emb)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
        print(f"Epoch {epoch+1}: loss={total_loss/total:.4f}  acc={correct/total:.4f}")

    out = Path("models") / f"{args.model}_probe_layer{args.layer}.pth"
    out.parent.mkdir(exist_ok=True)
    torch.save(probe.state_dict(), out)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
