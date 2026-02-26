import argparse
from pathlib import Path

from src.evaluation.plots import plot_det_curve, plot_per_attack_eer
from src.evaluation.results_writer import write_csv
from src.laundering import LaunderingEngine
from src.models.registry import get_model

CONFIGS = {
    "aasist": "external/aasist/config/AASIST.conf",
    "aasist-l": "external/aasist/config/AASIST-L.conf",
    "rawnet2": "external/aasist/config/RawNet2_baseline.conf",
}

WEIGHTS = {
    "aasist": "external/aasist/models/weights/AASIST.pth",
    "aasist-l": "external/aasist/models/weights/AASIST-L.pth",
    "rawnet2": "external/aasist/models/weights/RawNet2.pth",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(CONFIGS.keys()))
    p.add_argument("--data_root", default="data/ASVspoof2019/LA")
    p.add_argument("--pipeline", choices=["N", "M", "P"], default=None)
    p.add_argument("--depth", type=int, choices=[0, 1, 2, 3], default=0)
    p.add_argument("--strength", choices=["L", "M", "H"], default="M")
    p.add_argument("--output", default="outputs")
    p.add_argument("--config_dir", default="configs")
    return p.parse_args()


def main():
    args = parse_args()

    if args.depth > 0 and args.pipeline is None:
        raise ValueError("--pipeline {N,M,P} is required when --depth > 0")

    outdir = (
        Path(args.output)
        / args.model
        / (args.pipeline or "clean")
        / f"k{args.depth}"
        / (args.strength if args.depth > 0 else "")
    )
    outdir.mkdir(parents=True, exist_ok=True)

    # Init model
    model = get_model(
        args.model,
        config_path=CONFIGS[args.model],
        data_root=args.data_root,
    )
    model.load_weights(WEIGHTS[args.model])

    # Laundering engine
    engine = LaunderingEngine(args.config_dir)
    launder_fn = (
        engine.get_batch_fn(args.pipeline or "N", args.depth, args.strength)
        if args.depth > 0
        else None
    )

    # Evaluate
    eer, tdcf = model.evaluate(output_dir=str(outdir), launder_fn=launder_fn)
    result = model._last_eval_result
    if args.depth == 0:
        label = "clean"
    else:
        label = f"{args.pipeline}_k{args.depth}_{args.strength}"

    plot_det_curve(
        {label: result},
        str(outdir),
        args.model,
        condition_label=label,
    )

    plot_per_attack_eer(
        result.eer_per_attack,
        str(outdir),
        args.model,
        condition_label=label,
    )

    # Save results
    write_csv(
        [
            {
                "model": args.model,
                "pipeline": args.pipeline or "clean",
                "depth": args.depth,
                "strength": args.strength if args.depth > 0 else "-",
                "eer": eer,
                "tdcf": tdcf,
            }
        ],
        str(outdir),
    )


if __name__ == "__main__":
    main()
