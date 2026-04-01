"""
Train a linear probe on frozen SSL embeddings for spoof detection.
Usage:
    python train_probe.py --model wav2vec2 --layer 11 --epochs 20
Saves: models/<model_type>_probe_layer<N>.pth
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.ssl_frontend import SSLFrontend
from src.models.ssl_probe_wrapper import _SimpleWavDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["wav2vec2", "hubert", "wavlm"])
    p.add_argument("--layer", type=int, default=11)
    p.add_argument("--data_root", default="data/ASVspoof2019/LA")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_train", type=int, default=None,
                   help="Cap training samples (useful for quick sweep)")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load SSL frontend (frozen — no grad)
    frontend = SSLFrontend(
        model_type=args.model,
        extract_layers=[args.layer],
        device=device,
    )
    frontend.eval()

    # Dataset — train split
    train_dataset = _SimpleWavDataset(
        data_root=Path(args.data_root),
        track="LA",
        split="train",
        max_len=64000,
    )
    if args.max_train:
        train_dataset.trials = train_dataset.trials[:args.max_train]

    loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)

    # Linear probe
    embed_dim = 768   # all base models
    probe = nn.Linear(embed_dim, 2).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Label map
    label_map = {"bonafide": 0, "spoof": 1}

    for epoch in range(args.epochs):
        probe.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_x, utt_ids, srcs, keys in tqdm(loader, desc=f"Epoch {epoch+1}"):
            # Extract frozen SSL embeddings
            with torch.no_grad():
                layer_outputs = frontend(batch_x)
                pooled = frontend.mean_pool(layer_outputs)
                emb = pooled[args.layer].to(device)   # [B, 768]

            labels = torch.tensor(
                [label_map[k] for k in keys], dtype=torch.long
            ).to(device)

            optimizer.zero_grad()
            logits = probe(emb)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)

        print(f"Epoch {epoch+1}: loss={total_loss/total:.4f}  acc={correct/total:.4f}")

    # Save probe weights only (not the SSL model)
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.model}_probe_layer{args.layer}.pth"
    torch.save(probe.state_dict(), out_path)
    print(f"Probe saved → {out_path}")


if __name__ == "__main__":
    main()
