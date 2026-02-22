import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

_AASIST_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../external/aasist")
)
sys.path.insert(0, _AASIST_ROOT)

from evaluation import calculate_tDCF_EER
from main import get_loader, get_model


class AASISTWrapper:
    def __init__(self, config_path: str, data_root: str):
        with open(config_path) as f:
            self.config = json.load(f)
        self.config["database_path"] = str(Path(data_root).resolve())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = get_model(self.config["model_config"], self.device)

    def load_weights(self, weights_path: str = None):
        path = weights_path or self.config["model_path"]
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        print(f"[AASIST] Loaded: {path}")

    def evaluate(self, output_dir: str = "outputs") -> tuple[float, float]:
        start = time.time()
        print(f"[AASIST] Device: {self.device}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        database_path = Path(self.config["database_path"])
        track = self.config["track"]
        prefix = f"ASVspoof2019.{track}"

        eval_trial_path = (
            database_path
            / f"ASVspoof2019_{track}_cm_protocols"
            / f"{prefix}.cm.eval.trl.txt"
        )
        eval_score_path = output_dir / "eval_scores.txt"

        _, _, eval_loader = get_loader(database_path, seed=1234, config=self.config)
        print(f"[AASIST] Eval batches: {len(eval_loader)}")

        self.model.eval()
        with open(eval_trial_path) as f:
            trial_lines = f.readlines()

        fname_list, score_list = [], []
        with torch.no_grad():
            for batch_x, utt_ids in tqdm(eval_loader, desc="Scoring", unit="batch"):
                batch_x = batch_x.to(self.device)
                _, batch_out = self.model(batch_x)
                scores = batch_out[:, 1].cpu().numpy().ravel()
                fname_list.extend(utt_ids)
                score_list.extend(scores.tolist())

        assert (
            len(trial_lines) == len(fname_list) == len(score_list)
        ), f"Mismatch: {len(trial_lines)} trials vs {len(fname_list)} scored"

        with open(eval_score_path, "w") as fh:
            for fn, sco, trl in zip(fname_list, score_list, trial_lines):
                _, utt_id, _, src, key = trl.strip().split()
                assert fn == utt_id, f"ID mismatch: {fn} vs {utt_id}"
                fh.write(f"{utt_id} {src} {key} {sco}\n")

        print(f"[AASIST] Scores saved → {eval_score_path}")

        eer, tdcf = calculate_tDCF_EER(
            cm_scores_file=eval_score_path,
            asv_score_file=database_path / self.config["asv_score_path"],
            output_file=output_dir / "tDCF_EER.txt",
        )

        print(f"[AASIST] EER: {eer:.4f}% | min-tDCF: {tdcf:.4f}")
        print(f"[AASIST] Eval time: {(time.time() - start) / 60:.2f} min")
        return eer, tdcf
