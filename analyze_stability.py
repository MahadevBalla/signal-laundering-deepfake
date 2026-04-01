"""
Computes CKA + cosine stability across layers and depths for a given SSL model.
Usage:
    python analyze_stability.py --model wav2vec2 --pipeline P --strength M
Outputs:
    outputs/stability/wav2vec2_P_M_cka.json
    outputs/stability/wav2vec2_P_M_cosine.json
"""
import argparse
import json
from pathlib import Path

from src.laundering import LaunderingEngine
from src.models.ssl_probe_wrapper import SSLProbeWrapper
from src.evaluation.cka import cka_layer_stability, cosine_stability


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["wav2vec2", "hubert", "wavlm"])
    p.add_argument("--pipeline", required=True, choices=["N", "M", "P"])
    p.add_argument("--strength", default="M", choices=["L", "M", "H"])
    p.add_argument("--data_root", default="data/ASVspoof2019/LA")
    p.add_argument("--config_dir", default="configs")
    p.add_argument("--max_eval", type=int, default=2000,
                   help="Cap utterances — CKA is O(N^2) in memory")
    return p.parse_args()


def main():
    args = parse_args()
    config_path = f"configs/{args.model}_probe.yaml"
    engine = LaunderingEngine(args.config_dir)

    wrapper = SSLProbeWrapper(config_path=config_path, data_root=args.data_root)
    # No probe weights needed — extract_all_layers() doesn't use probe

    out_dir = Path(f"outputs/stability/{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Extract clean embeddings (depth=0) ---
    print("[STABILITY] Extracting clean embeddings (depth=0)...")
    clean_embs = wrapper.extract_all_layers(
        output_dir=str(out_dir / "clean"),
        launder_fn=None,
        max_eval=args.max_eval,
        save_embeddings=True,
    )

    cka_results = {}     # depth → {layer → cka}
    cosine_results = {}  # depth → {layer → cosine}

    for depth in [1, 2, 3]:
        print(f"\n[STABILITY] Pipeline={args.pipeline} Depth={depth} Strength={args.strength}")
        launder_fn = engine.get_batch_fn(args.pipeline, depth, args.strength)

        laundered_embs = wrapper.extract_all_layers(
            output_dir=str(out_dir / f"{args.pipeline}_k{depth}_{args.strength}"),
            launder_fn=launder_fn,
            max_eval=args.max_eval,
            save_embeddings=True,
        )

        cka_results[depth] = {
            int(k): v for k, v in
            cka_layer_stability(clean_embs, laundered_embs).items()
        }
        cosine_results[depth] = {
            int(k): v for k, v in
            cosine_stability(clean_embs, laundered_embs).items()
        }

        print(f"  CKA per layer: { {l: f'{v:.3f}' for l,v in cka_results[depth].items()} }")

    # Save
    tag = f"{args.model}_{args.pipeline}_{args.strength}"
    with open(out_dir / f"{tag}_cka.json", "w") as f:
        json.dump(cka_results, f, indent=2)
    with open(out_dir / f"{tag}_cosine.json", "w") as f:
        json.dump(cosine_results, f, indent=2)

    print(f"\n[STABILITY] Saved → {out_dir}/{tag}_cka.json")


if __name__ == "__main__":
    main()
