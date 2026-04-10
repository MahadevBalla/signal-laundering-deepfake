import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# RawNet2 baseline support is provided inside the `external/aasist` framework
# (see `external/aasist/config/RawNet2_baseline.conf`).
_AASIST_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../external/aasist")
)
sys.path.insert(0, _AASIST_ROOT)

# from evaluation import calculate_tDCF_EER
from main import get_loader, get_model

from src.evaluation.metrics import evaluate_scores
from src.models.hubert_frontend import HuBERTConfig, HuBERTFrontend


class RawNet2Wrapper:
    def __init__(
        self,
        config_path: str,
        data_root: str,
        use_hubert: bool = False,
        hubert_model_name: str = "facebook/hubert-base-ls960",
    ):
        with open(config_path) as f:
            self.config = json.load(f)
        self.config["database_path"] = str(Path(data_root).resolve())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = get_model(self.config["model_config"], self.device)
        self.use_hubert = use_hubert
        self.hubert_frontend = None
        if self.use_hubert:
            self.hubert_frontend = HuBERTFrontend(
                device=self.device,
                cfg=HuBERTConfig(model_name=hubert_model_name),
            )
            print(f"[RawNet2] HuBERT frontend enabled: {hubert_model_name}")

    def load_weights(self, weights_path: str = None):
        path = weights_path or self.config["model_path"]
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        print(f"[RawNet2] Loaded: {path}")

    def evaluate(
        self,
        output_dir: str = "outputs",
        launder_fn=None,
        max_eval: int | None = 500,
    ) -> tuple[float, float]:

        start = time.time()
        print(f"[RawNet2] Device: {self.device}")
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

        # Loader
        _, _, eval_loader = get_loader(database_path, seed=1234, config=self.config)
        dataset = eval_loader.dataset
        if max_eval is not None:
            dataset.list_IDs = dataset.list_IDs[:max_eval]

            eval_loader = DataLoader(
                dataset,
                batch_size=eval_loader.batch_size,
                shuffle=False,
                num_workers=eval_loader.num_workers,
            )
        print(f"[RawNet2] Eval batches: {len(eval_loader)}")

        # Load & filter protocol
        with open(eval_trial_path) as f:
            trial_lines = f.readlines()

        id_set = set(dataset.list_IDs)
        trial_lines = [
            line for line in trial_lines if line.strip().split()[1] in id_set
        ]

        # Scoring
        fname_list, score_list = [], []
        self.model.eval()
        with torch.no_grad():
            for batch_x, utt_ids in tqdm(eval_loader, desc="Scoring", unit="batch"):
                if launder_fn is not None:
                    batch_x = launder_fn(batch_x)
                if self.hubert_frontend is not None:
                    batch_x = self.hubert_frontend(batch_x)
                batch_x = batch_x.to(self.device)
                _, batch_out = self.model(batch_x)
                scores = batch_out[:, 1].cpu().numpy().ravel()
                fname_list.extend(utt_ids)
                score_list.extend(scores.tolist())

        assert len(trial_lines) == len(fname_list) == len(score_list), (
            f"Mismatch: {len(trial_lines)} trials vs {len(fname_list)} scored"
        )

        # Write score file
        with open(eval_score_path, "w") as fh:
            for fn, sco, trl in zip(fname_list, score_list, trial_lines):
                _, utt_id, _, src, key = trl.strip().split()
                assert fn == utt_id, f"ID mismatch: {fn} vs {utt_id}"
                fh.write(f"{utt_id} {src} {key} {sco}\n")

        print(f"[RawNet2] Scores saved → {eval_score_path}")

        # Metrics
        asv_score_file = database_path / self.config["asv_score_path"]
        result = evaluate_scores(eval_score_path, asv_score_file)
        self._last_eval_result = result

        print(f"[RawNet2] EER: {result.eer:.4f}% | min-tDCF: {result.min_tdcf:.4f}")
        print(f"[RawNet2] Eval time: {(time.time() - start) / 60:.2f} min")
        return result.eer, result.min_tdcf


class HuBERTRawNet2Wrapper(RawNet2Wrapper):
    def __init__(
        self,
        config_path: str,
        data_root: str,
        hubert_model_name: str = "facebook/hubert-base-ls960",
    ):
        super().__init__(
            config_path=config_path,
            data_root=data_root,
            use_hubert=True,
            hubert_model_name=hubert_model_name,
        )
