import argparse

from src.models.registry import get_model

CONFIGS = {
    "aasist": "external/aasist/config/AASIST.conf",
    "aasist-l": "external/aasist/config/AASIST-L.conf",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["aasist", "aasist-l"])
    p.add_argument("--data_root", default="data/ASVspoof2019/LA")
    p.add_argument("--pipeline", choices=["N", "M", "P"], default=None)
    p.add_argument("--depth", type=int, choices=[0, 1, 2, 3], default=0)
    p.add_argument("--strength", choices=["L", "M", "H"], default="M")
    p.add_argument("--output", default="outputs")
    return p.parse_args()


def main():
    args = parse_args()

    model = get_model(
        args.model,
        config_path=CONFIGS[args.model],
        data_root=args.data_root,
    )

    if args.depth == 0 or args.pipeline is None:
        _, _ = model.evaluate(output_dir=args.output)
    else:
        raise NotImplementedError("Laundering pipeline not yet implemented")


if __name__ == "__main__":
    main()
